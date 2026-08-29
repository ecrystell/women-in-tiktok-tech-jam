"""Unit tests for Person 1's standalone attention implementation."""

from __future__ import annotations

import unittest

try:
    import torch
    import torch.nn.functional as F

    from person1_triton_attention import (
        TritonSelfAttention,
        triton_available,
        triton_op_available,
        triton_scaled_dot_product_attention,
    )
    from torch_transformer_benchmark import BaselineSelfAttention

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    triton_available = lambda: False  # type: ignore[assignment]
    triton_op_available = lambda: False  # type: ignore[assignment]


_CUDA_TRITON_AVAILABLE = bool(
    _TORCH_AVAILABLE and torch.cuda.is_available() and triton_available()
)


def reference_attention(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    valid_token_mask: "torch.Tensor | None" = None,
    causal: bool = False,
) -> "torch.Tensor":
    """Reproduce BaselineSelfAttention's attention-core semantics."""

    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if causal:
        seq_len = q.shape[-2]
        causal_mask = torch.ones(
            (seq_len, seq_len), device=q.device, dtype=torch.bool
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
    if valid_token_mask is not None:
        scores = scores.masked_fill(
            ~valid_token_mask[:, None, None, :], float("-inf")
        )

    probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    output = torch.matmul(probabilities, v)
    if valid_token_mask is not None:
        output = output.masked_fill(~valid_token_mask[:, None, :, None], 0)
    return output


def assert_or_close(
    testcase: unittest.TestCase,
    reference: "torch.Tensor",
    candidate: "torch.Tensor",
    rtol: float = 0.01,
    atol: float = 0.001,
) -> None:
    testcase.assertEqual(reference.shape, candidate.shape)
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    error = (candidate_float - reference_float).abs()
    passed = torch.isfinite(reference_float) & torch.isfinite(candidate_float)
    passed &= (error <= atol) | (error <= rtol * reference_float.abs())
    if not bool(passed.all()):
        testcase.fail(
            "attention mismatch: "
            f"max_abs={error.max().item():.6g}, "
            f"failed={int((~passed).sum().item())}/{passed.numel()}"
        )


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch is not installed")
class Person1AttentionTests(unittest.TestCase):
    @staticmethod
    def _make_mask(batch: int, seq_len: int, device: "torch.device"):
        lengths = torch.tensor(
            [seq_len, max(1, seq_len - 3)][:batch],
            device=device,
            dtype=torch.long,
        )
        if batch > lengths.numel():
            lengths = torch.full(
                (batch,), seq_len, device=device, dtype=torch.long
            )
        positions = torch.arange(seq_len, device=device)[None, :]
        return positions < lengths[:, None]

    def test_cpu_sdpa_fallback_matches_reference(self):
        torch.manual_seed(7)
        q = torch.randn(2, 4, 17, 16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        for causal in (False, True):
            for mask in (None, self._make_mask(2, 17, q.device)):
                candidate = triton_scaled_dot_product_attention(
                    q, k, v, valid_token_mask=mask, causal=causal
                )
                reference = reference_attention(
                    q, k, v, valid_token_mask=mask, causal=causal
                )
                assert_or_close(self, reference, candidate)

    def test_cpu_sdpa_fallback_exactly_zeros_invalid_queries(self):
        torch.manual_seed(9)
        q = torch.randn(2, 2, 11, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mask = self._make_mask(2, 11, q.device)

        for causal in (False, True):
            candidate = triton_scaled_dot_product_attention(
                q, k, v, valid_token_mask=mask, causal=causal
            )
            invalid_queries = ~mask[:, None, :, None]
            self.assertTrue(
                bool((candidate.masked_select(invalid_queries) == 0).all())
            )

    def test_packed_weight_transfer_matches_baseline(self):
        torch.manual_seed(11)
        baseline = BaselineSelfAttention(d_model=32, num_heads=4).eval()
        optimized = TritonSelfAttention(
            d_model=32, num_heads=4, backend="sdpa"
        ).eval()
        optimized.copy_from_baseline(baseline)

        x = torch.randn(2, 13, 32)
        mask = self._make_mask(2, 13, x.device)
        for causal in (False, True):
            reference = baseline(x, mask, causal)
            candidate = optimized(x, mask, causal)
            assert_or_close(self, reference, candidate)

    @unittest.skipUnless(
        _CUDA_TRITON_AVAILABLE,
        "CUDA and Triton are required for custom-kernel tests",
    )
    def test_cuda_forward_shape_and_mask_matrix(self):
        torch.manual_seed(13)
        device = torch.device("cuda")
        for seq_len, head_dim in ((31, 48), (32, 64), (128, 80), (513, 64)):
            q = torch.randn(2, 3, seq_len, head_dim, device=device, dtype=torch.float16)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            mask = self._make_mask(2, seq_len, device)
            for causal, current_mask in (
                (False, None),
                (True, None),
                (False, mask),
                (True, mask),
            ):
                candidate = triton_scaled_dot_product_attention(
                    q, k, v, valid_token_mask=current_mask, causal=causal
                )
                reference = reference_attention(
                    q, k, v, valid_token_mask=current_mask, causal=causal
                )
                assert_or_close(self, reference, candidate, rtol=0.01, atol=0.001)

    @unittest.skipUnless(
        _CUDA_TRITON_AVAILABLE,
        "CUDA and Triton are required for custom-kernel tests",
    )
    def test_cuda_bfloat16_forward(self):
        device = torch.device("cuda")
        q = torch.randn(2, 2, 64, 64, device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        candidate = triton_scaled_dot_product_attention(q, k, v, causal=True)
        reference = reference_attention(q, k, v, causal=True)
        assert_or_close(self, reference, candidate, rtol=0.01, atol=0.001)

    def _compare_gradients(
        self,
        device: "torch.device",
        dtype: "torch.dtype",
        valid_token_mask: "torch.Tensor | None",
        causal: bool,
    ) -> None:
        torch.manual_seed(17)
        shape = (2, 2, 19, 16)
        q_base = torch.randn(shape, device=device, dtype=dtype)
        k_base = torch.randn_like(q_base)
        v_base = torch.randn_like(q_base)
        upstream = torch.randn_like(q_base)

        q_ref = q_base.detach().clone().requires_grad_(True)
        k_ref = k_base.detach().clone().requires_grad_(True)
        v_ref = v_base.detach().clone().requires_grad_(True)
        reference = reference_attention(
            q_ref, k_ref, v_ref, valid_token_mask, causal
        )
        (reference * upstream).sum().backward()

        q_opt = q_base.detach().clone().requires_grad_(True)
        k_opt = k_base.detach().clone().requires_grad_(True)
        v_opt = v_base.detach().clone().requires_grad_(True)
        candidate = triton_scaled_dot_product_attention(
            q_opt, k_opt, v_opt, valid_token_mask, causal
        )
        (candidate * upstream).sum().backward()

        output_rtol = 0.05 if dtype != torch.float32 else 0.01
        output_atol = 0.005 if dtype != torch.float32 else 0.001
        assert_or_close(self, reference, candidate, output_rtol, output_atol)
        for reference_grad, candidate_grad in (
            (q_ref.grad, q_opt.grad),
            (k_ref.grad, k_opt.grad),
            (v_ref.grad, v_opt.grad),
        ):
            self.assertTrue(torch.isfinite(candidate_grad).all())
            assert_or_close(
                self,
                reference_grad,
                candidate_grad,
                rtol=output_rtol,
                atol=output_atol,
            )

        if valid_token_mask is not None:
            invalid_positions = ~valid_token_mask[:, None, :, None]
            self.assertEqual(
                int(q_opt.grad.masked_select(invalid_positions).abs().sum().item()),
                0,
            )
            self.assertEqual(
                int(k_opt.grad.masked_select(invalid_positions).abs().sum().item()),
                0,
            )
            self.assertEqual(
                int(v_opt.grad.masked_select(invalid_positions).abs().sum().item()),
                0,
            )

    def test_cpu_fallback_backward(self):
        mask = self._make_mask(2, 19, torch.device("cpu"))
        self._compare_gradients(torch.device("cpu"), torch.float32, mask, True)

    @unittest.skipUnless(
        _CUDA_TRITON_AVAILABLE,
        "CUDA and Triton are required for custom-kernel tests",
    )
    def test_cuda_custom_backward(self):
        device = torch.device("cuda")
        mask = self._make_mask(2, 19, device)
        self._compare_gradients(device, torch.float16, mask, True)

    @unittest.skipUnless(
        _CUDA_TRITON_AVAILABLE
        and triton_op_available()
        and hasattr(torch, "compile"),
        "torch.compile, torch.library.triton_op, CUDA, and Triton are required",
    )
    def test_torch_compile_smoke(self):
        device = torch.device("cuda")
        module = TritonSelfAttention(64, 2, backend="triton").to(device).eval()
        x = torch.randn(2, 32, 64, device=device, dtype=torch.float16)
        compiled = torch.compile(module, backend="inductor")
        output = compiled(x)
        self.assertEqual(tuple(output.shape), (2, 32, 64))


if __name__ == "__main__":
    unittest.main(verbosity=2)
