"""Unit tests for Person 2's standalone Transformer block."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from bench_person2_ffn import accuracy_output, compile_model, mark_inference_step
from person2_ffn_fusion import (
    ArticleFusedFFNResidual,
    linear_exact_gelu,
    strict_accuracy,
    triton,
)
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

    def test_article_fusion_has_native_cpu_fallback(self) -> None:
        baseline, optimized = self.make_blocks()
        fused = ArticleFusedFFNResidual(optimized)
        x = torch.randn(2, 5, 32)
        for mask in (
            None,
            torch.tensor([[True] * 5, [True, True, True, False, False]]),
        ):
            with torch.inference_mode():
                reference = x + baseline.ffn_out(
                    torch.nn.functional.gelu(
                        baseline.ffn_in(baseline.norm2(x)), approximate="none"
                    )
                )
                if mask is not None:
                    reference = reference.masked_fill(~mask[..., None], 0)
                candidate = fused(x, mask)
            assert_or_close(self, reference, candidate)
            if mask is not None:
                self.assertTrue(bool((candidate[~mask] == 0).all().item()))

        self.assertFalse(fused.prepare(x, None))
        self.assertIn("unsupported", fused.last_error or "")

    def test_article_fusion_preflight_and_repeated_calls(self) -> None:
        _baseline, optimized = self.make_blocks()
        fused = ArticleFusedFFNResidual(optimized)
        x = torch.randn(2, 5, 32)
        mask = torch.tensor([[True] * 5, [True, True, True, False, False]])

        def up(value, weight, bias):
            return torch.nn.functional.gelu(
                torch.nn.functional.linear(value, weight, bias),
                approximate="none",
            )

        def down(hidden, residual, weight, bias, valid):
            result = residual + torch.nn.functional.linear(hidden, weight, bias)
            return result.masked_fill(~valid[:, None], 0)

        with (
            mock.patch("person2_ffn_fusion.supports_fusion", return_value=True),
            mock.patch("person2_ffn_fusion.linear_exact_gelu", side_effect=up),
            mock.patch("person2_ffn_fusion.down_residual_masked", side_effect=down),
        ):
            self.assertTrue(fused.prepare(x, mask))
            first = fused(x, mask)
            second = fused(x, mask)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool((first[~mask] == 0).all().item()))

    def test_article_fusion_build_failure_falls_back(self) -> None:
        _baseline, optimized = self.make_blocks()
        fused = ArticleFusedFFNResidual(optimized)
        x = torch.randn(2, 5, 32)
        mask = torch.ones(2, 5, dtype=torch.bool)

        def up(value, weight, bias):
            return torch.nn.functional.gelu(
                torch.nn.functional.linear(value, weight, bias),
                approximate="none",
            )

        with (
            mock.patch("person2_ffn_fusion.supports_fusion", return_value=True),
            mock.patch("person2_ffn_fusion.linear_exact_gelu", side_effect=up),
            mock.patch(
                "person2_ffn_fusion.down_residual_masked",
                side_effect=RuntimeError("simulated extension failure"),
            ),
        ):
            self.assertFalse(fused.prepare(x, mask))
        self.assertIn("simulated extension failure", fused.last_error or "")
        with torch.inference_mode():
            self.assertTrue(torch.equal(fused(x, mask), optimized._ffn_residual(x, mask)))

    def test_article_custom_op_has_fake_shape_implementation(self) -> None:
        from torch._subclasses.fake_tensor import FakeTensorMode

        mode = FakeTensorMode()
        x = mode.from_tensor(torch.empty(4, 8))
        weight = mode.from_tensor(torch.empty(16, 8))
        bias = mode.from_tensor(torch.empty(16))
        output = linear_exact_gelu(x, weight, bias)
        self.assertEqual(output.shape, (4, 16))

    def test_strict_accuracy_handles_adversarial_near_zero_values(self) -> None:
        reference = torch.tensor([-1e-4, 0.0, 1e-4, 1.0])
        self.assertTrue(strict_accuracy(reference, reference + 5e-4))
        self.assertFalse(strict_accuracy(reference, reference + 2e-3))



@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class Person2CudaTests(unittest.TestCase):
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

    def test_triton_up_exact_gelu_multi_seed(self) -> None:
        if triton is None or torch.cuda.get_device_capability()[0] < 7:
            self.skipTest("Triton tensor-core path requires compute capability 7+")
        device = torch.device("cuda")
        for seed in (3, 17, 41):
            torch.manual_seed(seed)
            x = torch.randn(64, 32, device=device, dtype=torch.float16)
            weight = torch.randn(64, 32, device=device, dtype=torch.float16)
            bias = torch.randn(64, device=device, dtype=torch.float16)
            with torch.inference_mode():
                reference = torch.nn.functional.gelu(
                    torch.nn.functional.linear(x, weight, bias),
                    approximate="none",
                )
                candidate = linear_exact_gelu(x, weight, bias)
            assert_or_close(self, reference, candidate)


if __name__ == "__main__":
    unittest.main()
