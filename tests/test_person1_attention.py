"""Unit tests for Person 1's standalone attention implementation."""

from __future__ import annotations

import unittest
import os
from unittest import mock

try:
    import torch
    import torch.nn.functional as F
    import person1_triton_attention as attention_impl

    from person1_triton_attention import (
        TritonSelfAttention,
        cuda_bfloat16_supported,
        prepare_valid_token_mask,
        sdpa_backend_diagnostics,
        triton_available,
        triton_op_available,
        triton_scaled_dot_product_attention,
    )
    from torch_transformer_benchmark import BaselineSelfAttention

    _TORCH_AVAILABLE = True
except ModuleNotFoundError:
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    attention_impl = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False
    cuda_bfloat16_supported = lambda device=None: False  # type: ignore[assignment]
    prepare_valid_token_mask = lambda mask: mask  # type: ignore[assignment]
    sdpa_backend_diagnostics = lambda *args, **kwargs: {}  # type: ignore[assignment]
    triton_available = lambda: False  # type: ignore[assignment]
    triton_op_available = lambda: False  # type: ignore[assignment]


_CUDA_TRITON_AVAILABLE = bool(
    _TORCH_AVAILABLE and torch.cuda.is_available() and triton_available()
)
_CUDA_BF16_AVAILABLE = bool(
    _TORCH_AVAILABLE
    and torch.cuda.is_available()
    and cuda_bfloat16_supported(torch.device("cuda"))
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

    def test_cpu_sdpa_supports_all_official_head_dimensions(self):
        """SDPA must not be gated by the custom Triton head-dim set."""

        torch.manual_seed(8)
        # (d_model, heads) covers official Dh values 8, 32, 64, 128, and 256.
        for d_model, heads in ((32, 4), (128, 4), (128, 2), (128, 1), (1024, 4)):
            head_dim = d_model // heads
            q = torch.randn(1, heads, 13, head_dim)
            k = torch.randn_like(q)
            v = torch.randn_like(q)
            candidate = triton_scaled_dot_product_attention(q, k, v, causal=True)
            reference = reference_attention(q, k, v, causal=True)
            assert_or_close(self, reference, candidate)

    def test_auto_module_reports_sdpa_for_official_head_dimensions(self):
        for d_model, heads in ((32, 4), (128, 4), (128, 2), (128, 1), (1024, 4)):
            module = TritonSelfAttention(d_model, heads, backend="auto")
            x = torch.randn(1, 5, d_model)
            self.assertEqual(module.selected_backend(x), "sdpa")

    def test_setup_mask_normalization_only_collapses_all_valid_masks(self):
        all_valid = torch.ones(2, 7, dtype=torch.bool)
        self.assertIsNone(prepare_valid_token_mask(all_valid))
        padded = all_valid.clone()
        padded[1, -1] = False
        self.assertIs(prepare_valid_token_mask(padded), padded)

    def test_backend_status_reports_actual_sdpa_or_fallback_reason(self):
        module = TritonSelfAttention(32, 4, backend="sdpa").eval()
        x = torch.randn(1, 5, 32)
        module(x)
        self.assertEqual(module.backend_status(), ("sdpa", None))

        triton_module = TritonSelfAttention(64, 1, backend="triton").eval()
        triton_module(torch.randn(1, 32, 64))
        backend, reason = triton_module.backend_status()
        self.assertEqual(backend, "sdpa-fallback")
        self.assertTrue(reason)

    def test_sdpa_diagnostic_reports_layout_mask_and_backend_candidates(self):
        q = torch.randn(1, 2, 9, 16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        diagnostics = sdpa_backend_diagnostics(
            q, k, v, valid_token_mask=self._make_mask(1, 9, q.device), causal=True
        )
        self.assertEqual(diagnostics["shape"], (1, 2, 9, 16))
        self.assertEqual(diagnostics["mask"], "key-only-bool")
        self.assertEqual(diagnostics["q_stride"], tuple(q.stride()))
        self.assertIn("selected_kernel", diagnostics)
        self.assertIn("math", diagnostics["backends"])

    def test_non_contiguous_qkv_layout_matches_contiguous_reference(self):
        torch.manual_seed(9)
        storage = torch.randn(1, 2, 13, 32)
        q = storage[..., ::2]
        k = torch.randn_like(storage)[..., ::2]
        v = torch.randn_like(storage)[..., ::2]
        self.assertFalse(q.is_contiguous())
        mask = self._make_mask(1, 13, q.device)
        candidate = triton_scaled_dot_product_attention(
            q, k, v, valid_token_mask=mask, causal=True
        )
        reference = reference_attention(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            valid_token_mask=mask,
            causal=True,
        )
        assert_or_close(self, reference, candidate)

    def test_repeated_calls_preserve_previous_output(self):
        module = TritonSelfAttention(32, 4, backend="sdpa").eval()
        x = torch.randn(2, 13, 32)
        first = module(x).clone()
        second = module(x)
        self.assertTrue(torch.equal(first, second))
        self.assertIsNot(first, second)
        self.assertNotEqual(first.data_ptr(), x.data_ptr())
        self.assertNotEqual(second.data_ptr(), x.data_ptr())

    def test_masked_keys_cannot_change_valid_outputs_and_queries_are_zero(self):
        torch.manual_seed(10)
        q = torch.randn(1, 2, 13, 16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mask = torch.zeros(1, 13, dtype=torch.bool)
        mask[:, :7] = True
        baseline = triton_scaled_dot_product_attention(
            q, k, v, valid_token_mask=mask, causal=False
        )
        k_changed = k.clone()
        v_changed = v.clone()
        k_changed[:, :, 7:, :] = 1000.0
        v_changed[:, :, 7:, :] = -1000.0
        changed = triton_scaled_dot_product_attention(
            q, k_changed, v_changed, valid_token_mask=mask, causal=False
        )
        self.assertTrue(torch.equal(baseline[:, :, :7, :], changed[:, :, :7, :]))
        self.assertEqual(float(changed[:, :, 7:, :].abs().sum()), 0.0)

    def test_mask_boundaries_at_attention_tile_edges(self):
        """Check masks ending at and just after the initial tile boundary."""

        torch.manual_seed(12)
        q = torch.randn(1, 2, 65, 16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        for valid_length in (31, 32, 33, 63, 64, 65):
            mask = torch.zeros(1, 65, dtype=torch.bool)
            mask[:, :valid_length] = True
            for causal in (False, True):
                candidate = triton_scaled_dot_product_attention(
                    q, k, v, valid_token_mask=mask, causal=causal
                )
                reference = reference_attention(
                    q, k, v, valid_token_mask=mask, causal=causal
                )
                assert_or_close(self, reference, candidate)

    def test_long_masked_fallback_never_uses_dense_reference(self):
        """An unsupported causal+padding SDPA combination must stay tiled."""

        q = torch.randn(1, 1, 2050, 8)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        mask = torch.zeros(1, 2050, dtype=torch.bool)
        mask[:, :2047] = True
        with mock.patch.object(
            attention_impl.F,
            "scaled_dot_product_attention",
            side_effect=RuntimeError("forced unsupported SDPA combination"),
        ), mock.patch.object(
            attention_impl,
            "_explicit_compatibility_attention",
            side_effect=AssertionError("dense attention must not be used"),
        ):
            candidate = attention_impl.sdpa_scaled_dot_product_attention(
                q, k, v, valid_token_mask=mask, causal=True
            )
        self.assertEqual(tuple(candidate.shape), tuple(q.shape))
        self.assertTrue(torch.isfinite(candidate).all())
        self.assertEqual(float(candidate[:, :, 2047:, :].abs().sum()), 0.0)

    @unittest.skipUnless(
        _CUDA_TRITON_AVAILABLE
        and triton_op_available()
        and os.environ.get("RUN_LONG_ATTENTION_TESTS") == "1",
        "set RUN_LONG_ATTENTION_TESTS=1 with CUDA, Triton, and triton_op",
    )
    def test_cuda_100k_sequence_smoke_is_memory_bounded(self):
        """Exercise the long-sequence tiled core without a dense reference."""

        device = torch.device("cuda")
        q = torch.randn(1, 1, 100_000, 64, device=device, dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        candidate = triton_scaled_dot_product_attention(q, k, v, causal=True)
        self.assertEqual(tuple(candidate.shape), tuple(q.shape))
        self.assertTrue(torch.isfinite(candidate).all())

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

    @unittest.skipUnless(_CUDA_BF16_AVAILABLE, "native CUDA BF16 is unsupported")
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
        shape = (2, 2, 32, 64)
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
            invalid_tokens = (~valid_token_mask)[:, None, :, None]
            self.assertEqual(
                float(k_opt.grad.masked_select(invalid_tokens).abs().sum().item()),
                0.0,
            )
            self.assertEqual(
                float(v_opt.grad.masked_select(invalid_tokens).abs().sum().item()),
                0.0,
            )
            self.assertEqual(
                float(q_opt.grad.masked_select(invalid_tokens).abs().sum().item()),
                0.0,
            )

    def test_cpu_fallback_backward(self):
        mask = self._make_mask(2, 32, torch.device("cpu"))
        self._compare_gradients(torch.device("cpu"), torch.float32, mask, True)

    @unittest.skipUnless(
        _CUDA_TRITON_AVAILABLE,
        "CUDA and Triton are required for custom-kernel tests",
    )
    def test_cuda_custom_backward(self):
        device = torch.device("cuda")
        mask = self._make_mask(2, 32, device)
        self._compare_gradients(device, torch.float16, mask, True)

    @unittest.skipUnless(
        _CUDA_TRITON_AVAILABLE
        and triton_op_available()
        and hasattr(torch, "compile"),
        "torch.compile, torch.library.triton_op, CUDA, and Triton are required",
    )
    def test_torch_compile_smoke(self):
        device = torch.device("cuda")
        baseline = (
            BaselineSelfAttention(64, 2)
            .to(device=device, dtype=torch.float16)
            .eval()
        )
        module = (
            TritonSelfAttention(64, 2, backend="triton")
            .to(device=device, dtype=torch.float16)
            .eval()
        )
        module.copy_from_baseline(baseline)
        x = torch.randn(2, 32, 64, device=device, dtype=torch.float16)
        compiled_baseline = torch.compile(
            baseline,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        compiled_candidate = torch.compile(
            module,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        eager_reference = baseline(x)
        eager_candidate = module(x)
        compiled_reference = compiled_baseline(x)
        compiled_output = compiled_candidate(x)
        self.assertEqual(tuple(compiled_output.shape), (2, 32, 64))
        assert_or_close(self, eager_reference, eager_candidate, rtol=0.01, atol=0.001)
        assert_or_close(
            self, eager_reference, compiled_reference, rtol=0.01, atol=0.001
        )
        assert_or_close(
            self, compiled_reference, compiled_output, rtol=0.01, atol=0.001
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
