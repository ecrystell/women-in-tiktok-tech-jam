"""Unit tests for Person 2's standalone Transformer block."""

from __future__ import annotations

import inspect
import unittest

import torch

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


if __name__ == "__main__":
    unittest.main()
