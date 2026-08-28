"""Unit tests for Person 2's standalone Transformer block."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from bench_person2_ffn import accuracy_output, compile_model, mark_inference_step
from profile_person2_gemms import gemm_shapes
from profile_person2_ffn import event_device_time_us

from torch_transformer_benchmark import (
    BaselineTransformerBlock,
    OptimizedTransformerBlock,
    TransformerConfig,
    UserOptimizedTransformer,
)


def assert_or_close(
    testcase: unittest.TestCase,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float = 0.001,
    rtol: float = 0.01,
) -> None:
    testcase.assertEqual(reference.shape, candidate.shape)
    testcase.assertEqual(reference.dtype, candidate.dtype)
    ref = reference.detach().float()
    opt = candidate.detach().float()
    error = (opt - ref).abs()
    passed = torch.isfinite(ref) & torch.isfinite(opt)
    passed &= (error <= atol) | (error <= rtol * ref.abs())
    if not bool(passed.all().item()):
        testcase.fail(
            f"output mismatch: max_abs={error.max().item():.6g}, "
            f"failed={int((~passed).sum().item())}/{passed.numel()}"
        )


class Person2BlockTests(unittest.TestCase):
    @staticmethod
    def make_blocks(
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> tuple[BaselineTransformerBlock, OptimizedTransformerBlock]:
        torch.manual_seed(17)
        baseline = BaselineTransformerBlock(32, 4, 64)
        optimized = OptimizedTransformerBlock(32, 4, 64)
        optimized.load_state_dict(baseline.state_dict(), strict=True)
        return (
            baseline.to(device=device, dtype=dtype).eval(),
            optimized.to(device=device, dtype=dtype).eval(),
        )

    def test_constructor_and_forward_signatures_match(self) -> None:
        self.assertEqual(
            inspect.signature(BaselineTransformerBlock.__init__),
            inspect.signature(OptimizedTransformerBlock.__init__),
        )
        self.assertEqual(
            inspect.signature(BaselineTransformerBlock.forward),
            inspect.signature(OptimizedTransformerBlock.forward),
        )

    def test_state_dict_is_strictly_compatible(self) -> None:
        baseline, optimized = self.make_blocks()
        self.assertEqual(set(baseline.state_dict()), set(optimized.state_dict()))
        optimized.load_state_dict(baseline.state_dict(), strict=True)

    def test_fast_ffn_cache_is_nonpersistent_and_refreshes(self) -> None:
        baseline, optimized = self.make_blocks()
        self.assertNotIn("_ffn_out_weight_nt", optimized.state_dict())
        self.assertTrue(
            torch.equal(
                optimized._ffn_out_weight_nt,
                optimized.ffn_out.weight.detach().t().contiguous(),
            )
        )
        with torch.no_grad():
            baseline.ffn_out.weight.add_(0.125)
            baseline.norm2.weight[0] = 0.75
        optimized.load_state_dict(baseline.state_dict(), strict=True)
        self.assertTrue(
            torch.equal(
                optimized._ffn_out_weight_nt,
                optimized.ffn_out.weight.detach().t().contiguous(),
            )
        )
        self.assertFalse(optimized._norm2_affine_is_identity)

    def test_fast_ffn_cpu_and_build_failure_fall_back(self) -> None:
        _baseline, optimized = self.make_blocks()
        x = torch.randn(2, 5, 32)
        mask = torch.ones(2, 5, dtype=torch.bool)
        self.assertFalse(optimized.prepare_fast_ffn(x, mask))
        self.assertIn("unsupported", optimized.fast_ffn_error or "")

        optimized = self.make_blocks()[1]
        with (
            mock.patch.object(
                optimized, "_supports_fast_ffn", return_value=True
            ),
            mock.patch(
                "person2_ffn_post.load_extension",
                side_effect=RuntimeError("simulated build failure"),
            ),
        ):
            self.assertFalse(optimized.prepare_fast_ffn(x, mask))
        self.assertIn("simulated build failure", optimized.fast_ffn_error or "")
        self.assertFalse(optimized._fast_ffn_enabled)

    def test_fast_ffn_preflight_and_repeated_calls(self) -> None:
        baseline, optimized = self.make_blocks()
        x = torch.randn(2, 5, 32)
        mask = torch.tensor([[True] * 5, [True, True, True, False, False]])

        def masked(update, residual, valid):
            return (update + residual).masked_fill(~valid[:, None], 0)

        with (
            mock.patch.object(
                optimized, "_supports_fast_ffn", return_value=True
            ),
            mock.patch("person2_ffn_post.load_extension"),
            mock.patch(
                "person2_ffn_post.residual_masked", side_effect=masked
            ),
        ):
            self.assertTrue(optimized.prepare_fast_ffn(x, mask))
            with torch.inference_mode():
                reference = baseline._ffn_residual_native(x, mask) if hasattr(
                    baseline, "_ffn_residual_native"
                ) else x + baseline.ffn_out(
                    torch.nn.functional.gelu(
                        baseline.ffn_in(baseline.norm2(x)), approximate="none"
                    )
                )
                reference = reference.masked_fill(~mask[..., None], 0)
                first = optimized._ffn_residual_fast(x, mask)
                second = optimized._ffn_residual_fast(x, mask)
        assert_or_close(self, reference, first)
        self.assertTrue(torch.equal(first, second))
        self.assertNotEqual(first.data_ptr(), second.data_ptr())

    def test_fast_post_custom_ops_have_fake_shapes(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        import person2_ffn_post

        mode = FakeTensorMode()
        update = mode.from_tensor(torch.empty(4, 8))
        residual = mode.from_tensor(torch.empty(4, 8))
        mask = mode.from_tensor(torch.ones(4, dtype=torch.bool))
        self.assertEqual(
            person2_ffn_post.residual_masked(update, residual, mask).shape,
            (4, 8),
        )
        self.assertEqual(
            person2_ffn_post.residual_unmasked(update, residual).shape,
            (4, 8),
        )

    def test_user_model_remains_owned_by_integration(self) -> None:
        config = TransformerConfig(1, 4, 32, 4, 64, 2, False)
        model = UserOptimizedTransformer(config)
        self.assertTrue(
            all(type(layer) is BaselineTransformerBlock for layer in model.layers)
        )

    def test_cpu_full_block_masks_and_causality(self) -> None:
        baseline, optimized = self.make_blocks()
        for causal in (False, True):
            for mask in (
                None,
                torch.tensor(
                    [[True, True, True, True, True], [True, True, False, False, False]]
                ),
            ):
                torch.manual_seed(23)
                x = torch.randn(2, 5, 32)
                with torch.inference_mode():
                    reference = baseline(x, mask, causal)
                    candidate = optimized(x, mask, causal)
                assert_or_close(self, reference, candidate)
                if mask is not None:
                    self.assertTrue(bool((candidate[~mask] == 0).all().item()))

    def test_noncontiguous_input(self) -> None:
        baseline, optimized = self.make_blocks()
        torch.manual_seed(29)
        x = torch.randn(2, 32, 7).transpose(1, 2)
        self.assertFalse(x.is_contiguous())
        mask = torch.ones(2, 7, dtype=torch.bool)
        with torch.inference_mode():
            reference = baseline(x, mask, False)
            candidate = optimized(x, mask, False)
        assert_or_close(self, reference, candidate)

    def test_isolated_ffn_matches_baseline(self) -> None:
        baseline, optimized = self.make_blocks()
        torch.manual_seed(31)
        x = torch.randn(3, 6, 32)
        mask = torch.tensor(
            [
                [True, True, True, True, True, True],
                [True, True, True, True, False, False],
                [True, True, False, False, False, False],
            ]
        )
        with torch.inference_mode():
            reference = x + baseline.ffn_out(
                torch.nn.functional.gelu(
                    baseline.ffn_in(baseline.norm2(x)), approximate="none"
                )
            )
            reference = reference.masked_fill(~mask[..., None], 0)
            candidate = optimized._ffn_residual(x, mask)
        assert_or_close(self, reference, candidate)

    def test_accuracy_output_owns_storage(self) -> None:
        class IdentityWithMask(torch.nn.Module):
            def forward(
                self, value: torch.Tensor, _mask: torch.Tensor
            ) -> torch.Tensor:
                return value

        x = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3, dtype=torch.bool)
        output = accuracy_output(
            IdentityWithMask(), x, mask, torch.device("cpu"), "eager"
        )
        self.assertTrue(torch.equal(output, x))
        self.assertNotEqual(output.data_ptr(), x.data_ptr())

    def test_cuda_graph_step_marker_is_mode_gated(self) -> None:
        with mock.patch.object(
            torch.compiler, "cudagraph_mark_step_begin"
        ) as marker:
            mark_inference_step(torch.device("cuda"), "reduce-overhead")
            marker.assert_called_once_with()
            marker.reset_mock()
            mark_inference_step(torch.device("cuda"), "default")
            marker.assert_not_called()
            mark_inference_step(torch.device("cpu"), "reduce-overhead")
            marker.assert_not_called()

    def test_compile_model_requests_static_fullgraph(self) -> None:
        module = torch.nn.Identity()
        compiled = object()
        with mock.patch.object(
            torch, "compile", return_value=compiled
        ) as compiler:
            self.assertIs(compile_model(module, "default"), compiled)
        compiler.assert_called_once_with(
            module, mode="default", fullgraph=True, dynamic=False
        )

    def test_profiler_uses_cuda_only_time(self) -> None:
        event = SimpleNamespace(
            self_cuda_time_total=7.5,
            self_device_time_total=101.0,
            device_type="CPU",
        )
        self.assertEqual(event_device_time_us(event), 7.5)
        cpu_only = SimpleNamespace(
            self_device_time_total=101.0, device_type="CPU"
        )
        self.assertEqual(event_device_time_us(cpu_only), 0.0)

    def test_article_gemm_shapes_are_derived_from_tokens(self) -> None:
        from bench_person2_ffn import Case

        case = Case(8, 512, 512, 8, 2048, True, 0.2)
        up, down = gemm_shapes(case)
        self.assertEqual((up.m, up.n, up.k), (4096, 2048, 512))
        self.assertEqual((down.m, down.n, down.k), (4096, 512, 2048))
        self.assertEqual(up.flops, down.flops)



@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class Person2CudaTests(unittest.TestCase):
    def test_cuda_fast_ffn_extension_and_compile(self) -> None:
        from torch.utils.cpp_extension import CUDA_HOME

        if CUDA_HOME is None:
            self.skipTest("a CUDA toolkit is unavailable for extension build")
        device = torch.device("cuda")
        baseline, optimized = Person2BlockTests.make_blocks(
            device, torch.float16
        )
        x = torch.randn(2, 16, 32, device=device, dtype=torch.float16)
        mask = torch.tensor(
            [[True] * 16, [True] * 12 + [False] * 4], device=device
        )
        self.assertTrue(
            optimized.prepare_fast_ffn(x, mask), optimized.fast_ffn_error
        )
        with torch.inference_mode():
            reference = x + baseline.ffn_out(
                torch.nn.functional.gelu(
                    baseline.ffn_in(baseline.norm2(x)), approximate="none"
                )
            )
            reference = reference.masked_fill(~mask[..., None], 0)
            candidate = optimized._ffn_residual(x, mask)
        assert_or_close(self, reference, candidate)
        self.assertTrue(bool((candidate[~mask] == 0).all().item()))

        if torch.cuda.get_device_capability()[0] >= 7:
            compiled = torch.compile(
                optimized, mode="default", fullgraph=True, dynamic=False
            )
            with torch.inference_mode():
                assert_or_close(self, reference, compiled(x, mask, False))

    def test_cuda_float32_and_float16(self) -> None:
        device = torch.device("cuda")
        for dtype in (torch.float32, torch.float16):
            baseline, optimized = Person2BlockTests.make_blocks(device, dtype)
            torch.manual_seed(37)
            x = torch.randn(2, 16, 32, device=device, dtype=dtype)
            mask = torch.tensor(
                [[True] * 16, [True] * 12 + [False] * 4], device=device
            )
            with torch.inference_mode():
                reference = baseline(x, mask, True)
                candidate = optimized(x, mask, True)
            assert_or_close(self, reference, candidate)

    def test_cuda_bfloat16_when_supported(self) -> None:
        if not torch.cuda.is_bf16_supported():
            self.skipTest("GPU does not support bfloat16")
        device = torch.device("cuda")
        baseline, optimized = Person2BlockTests.make_blocks(
            device, torch.bfloat16
        )
        x = torch.randn(1, 8, 32, device=device, dtype=torch.bfloat16)
        mask = torch.ones(1, 8, device=device, dtype=torch.bool)
        with torch.inference_mode():
            assert_or_close(
                self,
                baseline(x, mask, False),
                optimized(x, mask, False),
            )

if __name__ == "__main__":
    unittest.main()
