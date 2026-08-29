"""Person 1's standalone packed-QKV and Triton attention implementation.

The public API in this module is deliberately independent of the supplied
benchmark harness.  It can therefore be tested and benchmarked in isolation
before Person 3 wires it into ``UserOptimizedTransformer``.

The implementation has two layers:

* ``TritonSelfAttention`` combines Q/K/V projections and output projection.
* ``triton_scaled_dot_product_attention`` computes the attention core.

The attention core uses a tiled online-softmax Triton implementation when the
runtime and shape are supported.  All other cases use PyTorch SDPA, preserving
autograd and CPU portability.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except (ImportError, RuntimeError):
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False

try:
    from torch.library import triton_op, wrap_triton

    _TRITON_OP_AVAILABLE = _TRITON_AVAILABLE and hasattr(torch.library, "triton_op")
except (ImportError, AttributeError):
    triton_op = None  # type: ignore[assignment]
    wrap_triton = None  # type: ignore[assignment]
    _TRITON_OP_AVAILABLE = False


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MAX_CUSTOM_HEAD_DIM = 128
_MAX_CUSTOM_BACKWARD_SEQ_LEN = 32
_CUSTOM_HEAD_DIMS = (64,)


def triton_available() -> bool:
    """Return whether Triton can be imported in this Python environment."""

    return _TRITON_AVAILABLE


def triton_op_available() -> bool:
    """Return whether the structured ``torch.library.triton_op`` API exists."""

    return _TRITON_OP_AVAILABLE


def _validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
) -> Tuple[int, int, int, int]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k, and v must have shape [batch, heads, seq_len, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q, k, and v must have identical shapes")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must have the same dtype")
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError("q, k, and v must use float32, float16, or bfloat16")

    batch, heads, seq_len, head_dim = q.shape
    if valid_token_mask is not None:
        if valid_token_mask.ndim != 2 or valid_token_mask.shape != (batch, seq_len):
            raise ValueError(
                "valid_token_mask must have shape [batch_size, seq_len]"
            )
        if valid_token_mask.device != q.device:
            raise ValueError("valid_token_mask must be on the same device as q")

    return batch, heads, seq_len, head_dim


def _build_sdpa_mask(
    valid_token_mask: Optional[torch.Tensor],
    seq_len: int,
    causal: bool,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if valid_token_mask is None:
        return None

    valid_keys = valid_token_mask.to(device=device, dtype=torch.bool)
    if not causal:
        # True means that the key position is allowed by SDPA's boolean-mask
        # convention.  The singleton dimensions broadcast over heads/queries.
        return valid_keys[:, None, None, :]

    causal_allowed = torch.ones(
        (seq_len, seq_len), device=device, dtype=torch.bool
    ).tril()
    return causal_allowed[None, None, :, :] & valid_keys[:, None, None, :]


def _explicit_compatibility_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> torch.Tensor:
    """Match the reference's explicit FP32-softmax semantics exactly.

    This intentionally remains a small-shape/ BF16 compatibility fallback.
    Fused SDPA and the custom FP32-statistics kernel can make different
    rounding choices than the benchmark's explicit reference, especially for
    BF16 outputs.  Keeping this path narrow protects correctness without
    changing the production dense FP16/FP32 dispatch.
    """

    _, _, seq_len, _ = q.shape
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if causal:
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


def _sdpa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> torch.Tensor:
    """Correct PyTorch fallback matching the benchmark's mask semantics."""

    _, _, seq_len, _ = q.shape
    if (
        q.dtype == torch.bfloat16
        or seq_len < 32
        or q.shape[-1] not in _CUSTOM_HEAD_DIMS
    ):
        return _explicit_compatibility_attention(
            q, k, v, valid_token_mask, causal
        )

    attention_mask = _build_sdpa_mask(
        valid_token_mask=valid_token_mask,
        seq_len=seq_len,
        causal=causal,
        device=q.device,
    )

    if attention_mask is None:
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)

    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        is_causal=False,
    )
    # SDPA's boolean mask describes valid keys.  The benchmark contract also
    # requires invalid query rows to be exactly zero, including their
    # gradients.  Apply that query-side rule after SDPA so the functional API
    # and the module wrapper have identical semantics.
    valid_queries = valid_token_mask.to(device=q.device, dtype=torch.bool)
    return output.masked_fill(~valid_queries[:, None, :, None], 0)


def supports_triton_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
) -> bool:
    """Return whether the custom kernel is eligible for these tensors."""

    if not _TRITON_AVAILABLE or q.device.type != "cuda":
        return False
    if q.dtype not in _SUPPORTED_DTYPES:
        return False
    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        return False
    if q.device != k.device or q.device != v.device:
        return False
    if valid_token_mask is not None:
        if valid_token_mask.ndim != 2:
            return False
        if valid_token_mask.shape != (q.shape[0], q.shape[2]):
            return False
        if valid_token_mask.device != q.device:
            return False
    # SDPA is both faster and more numerically robust for tiny causal tiles;
    # reserve the custom path for the intended BLOCK_M-sized workload.
    return (
        q.shape[2] >= 32
        and q.dtype in (torch.float16, torch.float32)
        and q.shape[-1] in _CUSTOM_HEAD_DIMS
    )


if _TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=2, num_stages=1),
        ],
        key=["D"],
    )
    @triton.jit
    def _attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        out_ptr,
        lse_ptr,
        mask_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_km,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vm,
        stride_vd,
        stride_ob,
        stride_oh,
        stride_om,
        stride_od,
        stride_lseb,
        stride_lseh,
        stride_lsem,
        stride_mb,
        stride_ms,
        B,
        H,
        S,
        D,
        scale,
        IS_CAUSAL: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        scale_f32 = tl.cast(scale, tl.float32)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n_base = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_in_bounds = offs_m < S
        q_valid = q_in_bounds
        if HAS_PADDING:
            q_valid = q_valid & tl.load(
                mask_ptr + pid_b * stride_mb + offs_m * stride_ms,
                mask=q_in_bounds,
                other=0,
            ).to(tl.int1)
        q_ptrs = (
            q_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
        q = tl.load(
            q_ptrs,
            mask=q_in_bounds[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        running_max = tl.full([BLOCK_M], float("-inf"), tl.float32)
        running_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
        accumulator = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + offs_n_base
            key_valid = offs_n < S
            if HAS_PADDING:
                key_valid = key_valid & tl.load(
                    mask_ptr + pid_b * stride_mb + offs_n * stride_ms,
                    mask=offs_n < S,
                    other=0,
                ).to(tl.int1)

            k_ptrs = (
                k_ptr
                + pid_b * stride_kb
                + pid_h * stride_kh
                + offs_n[:, None] * stride_km
                + offs_d[None, :] * stride_kd
            )
            v_ptrs = (
                v_ptr
                + pid_b * stride_vb
                + pid_h * stride_vh
                + offs_n[:, None] * stride_vm
                + offs_d[None, :] * stride_vd
            )
            k = tl.load(
                k_ptrs,
                mask=key_valid[:, None] & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptrs,
                mask=key_valid[:, None] & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)

            scores = tl.dot(
                q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32
            ) * scale_f32
            allowed = q_valid[:, None] & key_valid[None, :]
            if IS_CAUSAL:
                allowed = allowed & (offs_m[:, None] >= offs_n[None, :])
            scores = tl.where(allowed, scores, float("-inf"))

            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            # Invalid query rows are guaranteed to be zero at store time.  Use
            # a finite normalization value for them to avoid -inf - -inf.
            new_max = tl.where(q_valid, new_max, 0.0)
            alpha = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            probabilities = tl.where(allowed, probabilities, 0.0)

            running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
            accumulator = accumulator * alpha[:, None] + tl.dot(
                probabilities,
                v,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            running_max = new_max

        safe_sum = tl.where(running_sum > 0.0, running_sum, 1.0)
        output = accumulator / safe_sum[:, None]
        output = tl.where(q_valid[:, None], output, 0.0)
        output_ptrs = (
            out_ptr
            + pid_b * stride_ob
            + pid_h * stride_oh
            + offs_m[:, None] * stride_om
            + offs_d[None, :] * stride_od
        )
        tl.store(
            output_ptrs,
            output,
            mask=q_in_bounds[:, None] & (offs_d[None, :] < D),
        )

        lse = running_max + tl.log(safe_sum)
        lse = tl.where(q_valid & (running_sum > 0.0), lse, float("-inf"))
        lse_ptrs = (
            lse_ptr
            + pid_b * stride_lseb
            + pid_h * stride_lseh
            + offs_m * stride_lsem
        )
        tl.store(lse_ptrs, lse, mask=q_in_bounds)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=2, num_stages=1),
        ],
        key=["D"],
    )
    @triton.jit
    def _attention_backward_dq_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        dq_ptr,
        mask_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_km,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vm,
        stride_vd,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_dod,
        stride_lseb,
        stride_lseh,
        stride_lsem,
        stride_dqb,
        stride_dqh,
        stride_dqm,
        stride_dqd,
        stride_mb,
        stride_ms,
        B,
        H,
        S,
        D,
        scale,
        IS_CAUSAL: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        scale_f32 = tl.cast(scale, tl.float32)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n_base = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        q_in_bounds = offs_m < S
        q_valid = q_in_bounds
        if HAS_PADDING:
            q_valid = q_valid & tl.load(
                mask_ptr + pid_b * stride_mb + offs_m * stride_ms,
                mask=q_in_bounds,
                other=0,
            ).to(tl.int1)

        q_ptrs = (
            q_ptr
            + pid_b * stride_qb
            + pid_h * stride_qh
            + offs_m[:, None] * stride_qm
            + offs_d[None, :] * stride_qd
        )
        do_ptrs = (
            do_ptr
            + pid_b * stride_dob
            + pid_h * stride_doh
            + offs_m[:, None] * stride_dom
            + offs_d[None, :] * stride_dod
        )
        q = tl.load(
            q_ptrs,
            mask=q_in_bounds[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        do = tl.load(
            do_ptrs,
            mask=q_in_bounds[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        lse = tl.load(
            lse_ptr
            + pid_b * stride_lseb
            + pid_h * stride_lseh
            + offs_m * stride_lsem,
            mask=q_in_bounds,
            other=float("-inf"),
        ).to(tl.float32)
        safe_lse = tl.where(q_valid, lse, 0.0)
        dq = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        for start_n in range(0, S, BLOCK_N):
            offs_n = start_n + offs_n_base
            key_valid = offs_n < S
            if HAS_PADDING:
                key_valid = key_valid & tl.load(
                    mask_ptr + pid_b * stride_mb + offs_n * stride_ms,
                    mask=offs_n < S,
                    other=0,
                ).to(tl.int1)

            k = tl.load(
                k_ptr
                + pid_b * stride_kb
                + pid_h * stride_kh
                + offs_n[:, None] * stride_km
                + offs_d[None, :] * stride_kd,
                mask=key_valid[:, None] & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_ptr
                + pid_b * stride_vb
                + pid_h * stride_vh
                + offs_n[:, None] * stride_vm
                + offs_d[None, :] * stride_vd,
                mask=key_valid[:, None] & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)

            allowed = q_valid[:, None] & key_valid[None, :]
            if IS_CAUSAL:
                allowed = allowed & (offs_m[:, None] >= offs_n[None, :])
            scores = tl.dot(
                q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32
            ) * scale_f32
            scores = tl.where(allowed, scores, float("-inf"))
            probabilities = tl.exp(scores - safe_lse[:, None])
            probabilities = tl.where(allowed, probabilities, 0.0)

            dp = tl.dot(
                do,
                tl.trans(v),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            row_dot = tl.sum(dp * probabilities, axis=1)
            ds = probabilities * (dp - row_dot[:, None])
            dq += tl.dot(
                ds,
                k,
                input_precision="ieee",
                out_dtype=tl.float32,
            ) * scale_f32

        dq_ptrs = (
            dq_ptr
            + pid_b * stride_dqb
            + pid_h * stride_dqh
            + offs_m[:, None] * stride_dqm
            + offs_d[None, :] * stride_dqd
        )
        tl.store(
            dq_ptrs,
            dq,
            mask=q_in_bounds[:, None] & (offs_d[None, :] < D),
        )

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64}, num_warps=2, num_stages=1),
        ],
        key=["D"],
    )
    @triton.jit
    def _attention_backward_dkdv_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        do_ptr,
        lse_ptr,
        dk_ptr,
        dv_ptr,
        mask_ptr,
        stride_qb,
        stride_qh,
        stride_qm,
        stride_qd,
        stride_kb,
        stride_kh,
        stride_km,
        stride_kd,
        stride_vb,
        stride_vh,
        stride_vm,
        stride_vd,
        stride_dob,
        stride_doh,
        stride_dom,
        stride_dod,
        stride_lseb,
        stride_lseh,
        stride_lsem,
        stride_dkb,
        stride_dkh,
        stride_dkm,
        stride_dkd,
        stride_dvb,
        stride_dvh,
        stride_dvm,
        stride_dvd,
        stride_mb,
        stride_ms,
        B,
        H,
        S,
        D,
        scale,
        IS_CAUSAL: tl.constexpr,
        HAS_PADDING: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)
        scale_f32 = tl.cast(scale, tl.float32)

        offs_m_base = tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        key_valid = offs_n < S

        k = tl.load(
            k_ptr
            + pid_b * stride_kb
            + pid_h * stride_kh
            + offs_n[:, None] * stride_km
            + offs_d[None, :] * stride_kd,
            mask=key_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr
            + pid_b * stride_vb
            + pid_h * stride_vh
            + offs_n[:, None] * stride_vm
            + offs_d[None, :] * stride_vd,
            mask=key_valid[:, None] & (offs_d[None, :] < D),
            other=0.0,
        ).to(tl.float32)

        if HAS_PADDING:
            key_valid = key_valid & tl.load(
                mask_ptr + pid_b * stride_mb + offs_n * stride_ms,
                mask=offs_n < S,
                other=0,
            ).to(tl.int1)

        dk = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
        dv = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)

        for start_m in range(0, S, BLOCK_M):
            offs_m = start_m + offs_m_base
            q_in_bounds = offs_m < S
            q_valid = q_in_bounds
            if HAS_PADDING:
                q_valid = q_valid & tl.load(
                    mask_ptr + pid_b * stride_mb + offs_m * stride_ms,
                    mask=q_in_bounds,
                    other=0,
                ).to(tl.int1)
            q = tl.load(
                q_ptr
                + pid_b * stride_qb
                + pid_h * stride_qh
                + offs_m[:, None] * stride_qm
                + offs_d[None, :] * stride_qd,
                mask=q_in_bounds[:, None] & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            do = tl.load(
                do_ptr
                + pid_b * stride_dob
                + pid_h * stride_doh
                + offs_m[:, None] * stride_dom
                + offs_d[None, :] * stride_dod,
                mask=q_in_bounds[:, None] & (offs_d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            lse = tl.load(
                lse_ptr
                + pid_b * stride_lseb
                + pid_h * stride_lseh
                + offs_m * stride_lsem,
                mask=q_in_bounds,
                other=float("-inf"),
            ).to(tl.float32)
            safe_lse = tl.where(q_valid, lse, 0.0)

            allowed = q_valid[:, None] & key_valid[None, :]
            if IS_CAUSAL:
                allowed = allowed & (offs_m[:, None] >= offs_n[None, :])
            scores = tl.dot(
                q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32
            ) * scale_f32
            scores = tl.where(allowed, scores, float("-inf"))
            probabilities = tl.exp(scores - safe_lse[:, None])
            probabilities = tl.where(allowed, probabilities, 0.0)

            dv += tl.dot(
                tl.trans(probabilities),
                do,
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            dp = tl.dot(
                do,
                tl.trans(v),
                input_precision="ieee",
                out_dtype=tl.float32,
            )
            row_dot = tl.sum(dp * probabilities, axis=1)
            ds = probabilities * (dp - row_dot[:, None])
            dk += tl.dot(
                tl.trans(ds),
                q,
                input_precision="ieee",
                out_dtype=tl.float32,
            ) * scale_f32

        dk_ptrs = (
            dk_ptr
            + pid_b * stride_dkb
            + pid_h * stride_dkh
            + offs_n[:, None] * stride_dkm
            + offs_d[None, :] * stride_dkd
        )
        dv_ptrs = (
            dv_ptr
            + pid_b * stride_dvb
            + pid_h * stride_dvh
            + offs_n[:, None] * stride_dvm
            + offs_d[None, :] * stride_dvd
        )
        store_mask = key_valid[:, None] & (offs_d[None, :] < D)
        tl.store(dk_ptrs, dk, mask=store_mask)
        tl.store(dv_ptrs, dv, mask=store_mask)


def _kernel_grid(batch: int, heads: int, seq_len: int):
    """Build a Triton grid callback with only compile-time metadata as input."""

    def grid(meta: dict) -> tuple:
        return (triton.cdiv(seq_len, meta["BLOCK_M"]), heads, batch)

    return grid


def _backward_kv_grid(batch: int, heads: int, seq_len: int):
    """Build the dK/dV grid callback for one program per key tile."""

    def grid(meta: dict) -> tuple:
        return (triton.cdiv(seq_len, meta["BLOCK_N"]), heads, batch)

    return grid


def _launch_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    use_wrapped_triton: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, head_dim = q.shape
    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    logsumexp = torch.empty(
        (batch, heads, seq_len), device=q.device, dtype=torch.float32
    )
    mask_ref = q if valid_token_mask is None else valid_token_mask
    kernel = _attention_forward_kernel
    if use_wrapped_triton:
        kernel = wrap_triton(kernel)

    block_d = 1 << (head_dim - 1).bit_length()
    kernel[_kernel_grid(batch, heads, seq_len)](
        q,
        k,
        v,
        output,
        logsumexp,
        mask_ref,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        *logsumexp.stride(),
        *mask_ref.stride()[-2:],
        B=batch,
        H=heads,
        S=seq_len,
        D=head_dim,
        scale=head_dim**-0.5,
        IS_CAUSAL=causal,
        HAS_PADDING=valid_token_mask is not None,
        BLOCK_D=block_d,
    )
    return output, logsumexp


def _launch_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    use_wrapped_triton: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, heads, seq_len, head_dim = q.shape
    # Masked rows/keys are intentionally not written by the kernels.  Start
    # from zeros so their gradients are deterministic and exactly zero.
    grad_q = torch.zeros_like(q, memory_format=torch.contiguous_format)
    grad_k = torch.zeros_like(k, memory_format=torch.contiguous_format)
    grad_v = torch.zeros_like(v, memory_format=torch.contiguous_format)
    mask_ref = q if valid_token_mask is None else valid_token_mask

    dq_kernel = _attention_backward_dq_kernel
    dkdv_kernel = _attention_backward_dkdv_kernel
    if use_wrapped_triton:
        dq_kernel = wrap_triton(dq_kernel)
        dkdv_kernel = wrap_triton(dkdv_kernel)

    block_d = 1 << (head_dim - 1).bit_length()
    dq_kernel[_kernel_grid(batch, heads, seq_len)](
        q,
        k,
        v,
        grad_output,
        logsumexp,
        grad_q,
        mask_ref,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *grad_output.stride(),
        *logsumexp.stride(),
        *grad_q.stride(),
        *mask_ref.stride()[-2:],
        B=batch,
        H=heads,
        S=seq_len,
        D=head_dim,
        scale=head_dim**-0.5,
        IS_CAUSAL=causal,
        HAS_PADDING=valid_token_mask is not None,
        BLOCK_D=block_d,
    )
    dkdv_kernel[_backward_kv_grid(batch, heads, seq_len)](
        q,
        k,
        v,
        grad_output,
        logsumexp,
        grad_k,
        grad_v,
        mask_ref,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *grad_output.stride(),
        *logsumexp.stride(),
        *grad_k.stride(),
        *grad_v.stride(),
        *mask_ref.stride()[-2:],
        B=batch,
        H=heads,
        S=seq_len,
        D=head_dim,
        scale=head_dim**-0.5,
        IS_CAUSAL=causal,
        HAS_PADDING=valid_token_mask is not None,
        BLOCK_D=block_d,
    )
    return grad_q, grad_k, grad_v


def _sdpa_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Use PyTorch's production autograd for unvalidated multi-tile cases."""

    with torch.enable_grad():
        q_ref = q.detach().requires_grad_(True)
        k_ref = k.detach().requires_grad_(True)
        v_ref = v.detach().requires_grad_(True)
        output = _sdpa_attention(
            q_ref, k_ref, v_ref, valid_token_mask, causal
        )
        return torch.autograd.grad(
            output,
            (q_ref, k_ref, v_ref),
            grad_outputs=grad_output,
            retain_graph=False,
            create_graph=False,
        )


class _TritonAttentionFunction(torch.autograd.Function):
    """Eager/autograd compatibility path when ``triton_op`` is unavailable."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: torch.Tensor,
        causal: bool,
        has_padding: bool,
    ) -> torch.Tensor:
        mask = valid_token_mask if has_padding else None
        output, logsumexp = _launch_forward(q, k, v, mask, causal)
        ctx.save_for_backward(q, k, v, logsumexp, valid_token_mask)
        ctx.causal = causal
        ctx.has_padding = has_padding
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, logsumexp, valid_token_mask = ctx.saved_tensors
        mask = valid_token_mask if ctx.has_padding else None
        if q.shape[2] > _MAX_CUSTOM_BACKWARD_SEQ_LEN:
            grad_q, grad_k, grad_v = _sdpa_backward(
                q, k, v, grad_output, mask, ctx.causal
            )
            return grad_q, grad_k, grad_v, None, None, None
        grad_q, grad_k, grad_v = _launch_backward(
            q,
            k,
            v,
            grad_output,
            logsumexp,
            mask,
            ctx.causal,
        )
        return grad_q, grad_k, grad_v, None, None, None


if _TRITON_OP_AVAILABLE:

    @triton_op("person1::scaled_dot_product_attention", mutates_args={})
    def _triton_attention_op(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: torch.Tensor,
        causal: bool,
        has_padding: bool,
    ) -> torch.Tensor:
        mask = valid_token_mask if has_padding else None
        output, _ = _launch_forward(
            q,
            k,
            v,
            mask,
            causal,
            use_wrapped_triton=True,
        )
        return output

    @_triton_attention_op.register_kernel("cpu")
    def _triton_attention_cpu_fallback(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: torch.Tensor,
        causal: bool,
        has_padding: bool,
    ) -> torch.Tensor:
        mask = valid_token_mask if has_padding else None
        return _sdpa_attention(q, k, v, mask, causal)

    def _triton_attention_setup_context(ctx, inputs, output) -> None:
        q, k, v, valid_token_mask, causal, has_padding = inputs
        ctx.save_for_backward(q, k, v, valid_token_mask)
        ctx.causal = causal
        ctx.has_padding = has_padding

    def _triton_attention_backward(ctx, grad_output):
        q, k, v, valid_token_mask = ctx.saved_tensors
        mask = valid_token_mask if ctx.has_padding else None
        if q.shape[2] > _MAX_CUSTOM_BACKWARD_SEQ_LEN:
            grad_q, grad_k, grad_v = _sdpa_backward(
                q, k, v, grad_output, mask, ctx.causal
            )
            return grad_q, grad_k, grad_v, None, None, None
        # Recompute only the compact log-sum-exp state.  The full attention
        # probability matrix is never saved.
        _, logsumexp = _launch_forward(
            q,
            k,
            v,
            mask,
            ctx.causal,
            use_wrapped_triton=True,
        )
        grad_q, grad_k, grad_v = _launch_backward(
            q,
            k,
            v,
            grad_output,
            logsumexp,
            mask,
            ctx.causal,
            use_wrapped_triton=True,
        )
        return grad_q, grad_k, grad_v, None, None, None

    _triton_attention_op.register_autograd(
        _triton_attention_backward,
        setup_context=_triton_attention_setup_context,
    )


def triton_scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
) -> torch.Tensor:
    """Compute attention with a Triton path and an SDPA fallback.

    ``valid_token_mask`` follows the benchmark convention: it is a boolean
    ``[batch, seq_len]`` right-padding mask.  True entries are valid tokens.
    The function is differentiable with respect to Q/K/V on both paths.
    """

    _validate_attention_inputs(q, k, v, valid_token_mask)
    if valid_token_mask is not None:
        if valid_token_mask.dtype != torch.bool:
            valid_token_mask = valid_token_mask.to(dtype=torch.bool)
        if valid_token_mask.device != q.device:
            valid_token_mask = valid_token_mask.to(device=q.device)

    if not supports_triton_attention(q, k, v, valid_token_mask):
        return _sdpa_attention(q, k, v, valid_token_mask, causal)

    # Kernel arguments always contain a tensor for the mask.  HAS_PADDING keeps
    # the no-padding path from reading the dummy tensor.
    mask_tensor = (
        valid_token_mask.contiguous()
        if valid_token_mask is not None
        else torch.empty((0,), device=q.device, dtype=torch.bool)
    )
    has_padding = valid_token_mask is not None

    if _TRITON_OP_AVAILABLE:
        return _triton_attention_op(
            q,
            k,
            v,
            mask_tensor,
            causal,
            has_padding,
        )
    return _TritonAttentionFunction.apply(
        q,
        k,
        v,
        mask_tensor,
        causal,
        has_padding,
    )


class TritonSelfAttention(nn.Module):
    """Packed-QKV self-attention with selectable SDPA/Triton backend."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if backend not in {"auto", "sdpa", "triton"}:
            raise ValueError("backend must be 'auto', 'sdpa', or 'triton'")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.backend = backend
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    @torch.no_grad()
    def copy_from_baseline(self, source) -> None:
        """Copy weights from a ``BaselineSelfAttention``-compatible module."""

        if self.d_model != source.d_model or self.num_heads != source.num_heads:
            raise ValueError("source and destination attention shapes do not match")
        self.qkv_proj.weight.copy_(
            torch.cat(
                [source.q_proj.weight, source.k_proj.weight, source.v_proj.weight],
                dim=0,
            )
        )
        self.qkv_proj.bias.copy_(
            torch.cat(
                [source.q_proj.bias, source.k_proj.bias, source.v_proj.bias],
                dim=0,
            )
        )
        self.out_proj.load_state_dict(source.out_proj.state_dict())

    def selected_backend(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> str:
        if self.backend in {"auto", "sdpa"}:
            return "sdpa"
        eligible = (
            _TRITON_AVAILABLE
            and x.device.type == "cuda"
            and x.ndim == 3
            and x.shape[1] >= 32
            and x.dtype in (torch.float16, torch.float32)
            and self.head_dim in _CUSTOM_HEAD_DIMS
        )
        if valid_token_mask is not None:
            eligible = eligible and (
                valid_token_mask.shape == x.shape[:2]
                and valid_token_mask.device == x.device
            )
        if self.backend == "triton":
            return "triton" if eligible else "sdpa-fallback"
        return "triton" if eligible else "sdpa"

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        if valid_token_mask is not None:
            if valid_token_mask.shape != (batch, seq_len):
                raise ValueError(
                    "valid_token_mask must have shape [batch_size, seq_len]"
                )
            valid_token_mask = valid_token_mask.to(
                device=x.device, dtype=torch.bool
            )

        qkv = self.qkv_proj(x).view(
            batch, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        if self.backend in {"auto", "sdpa"}:
            context = _sdpa_attention(q, k, v, valid_token_mask, causal)
        else:
            context = triton_scaled_dot_product_attention(
                q,
                k,
                v,
                valid_token_mask=valid_token_mask,
                causal=causal,
            )

        context = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(context)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class PackedQKVSDPAAttention(TritonSelfAttention):
    """Packed-QKV module that always uses PyTorch SDPA for benchmarking."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__(d_model=d_model, num_heads=num_heads, backend="sdpa")


__all__ = [
    "PackedQKVSDPAAttention",
    "TritonSelfAttention",
    "supports_triton_attention",
    "triton_available",
    "triton_op_available",
    "triton_scaled_dot_product_attention",
]
