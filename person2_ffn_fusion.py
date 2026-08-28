"""Inference-only Person 2 FFN fusion candidates for CUDA FP16.

The public Transformer block remains in ``torch_transformer_benchmark.py``.
This module deliberately keeps experimental kernels behind a native PyTorch
fallback so a failed build, unsupported input, or numerical gate cannot make
the standalone block incorrect.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except (ImportError, RuntimeError):
    triton = None
    tl = None


_EXTENSION = None
_EXTENSION_ERROR: Optional[str] = None


def load_down_extension():
    """Build and load the cuBLASLt extension outside timed inference."""
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError(_EXTENSION_ERROR)

    root = Path(__file__).resolve().parent
    build_root = root / "benchmark-results" / "person2-gemm-cuda-build"
    build_lib = build_root / "lib"
    build_temp = build_root / "temp"
    build_lib.mkdir(parents=True, exist_ok=True)
    build_temp.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "setup_person2_ffn_cuda.py"),
        "build_ext",
        "--build-lib",
        str(build_lib),
        "--build-temp",
        str(build_temp),
    ]
    verbose = os.environ.get("PERSON2_CUDA_VERBOSE", "0") == "1"
    result = subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=not verbose,
        text=True,
    )
    if result.returncode != 0:
        _EXTENSION_ERROR = (
            "Person 2 CUDA extension build failed:\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )
        raise RuntimeError(_EXTENSION_ERROR)
    sys.path.insert(0, str(build_lib))
    _EXTENSION = importlib.import_module("person2_ffn_cuda_ext")
    return _EXTENSION


if triton is not None:
    _UP_CONFIGS = [
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=2,
            num_warps=8,
        ),
    ]

    @triton.autotune(configs=_UP_CONFIGS, key=["M", "N", "K"])
    @triton.jit
    def _linear_exact_gelu_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        stride_xm: tl.constexpr,
        stride_xk: tl.constexpr,
        stride_wn: tl.constexpr,
        stride_wk: tl.constexpr,
        stride_om: tl.constexpr,
        stride_on: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            current_k = k_start + offs_k
            x = tl.load(
                x_ptr + offs_m[:, None] * stride_xm + current_k[None, :] * stride_xk,
                mask=(offs_m[:, None] < M) & (current_k[None, :] < K),
                other=0.0,
            )
            weight_t = tl.load(
                weight_ptr
                + offs_n[None, :] * stride_wn
                + current_k[:, None] * stride_wk,
                mask=(offs_n[None, :] < N) & (current_k[:, None] < K),
                other=0.0,
            )
            accumulator += tl.dot(x, weight_t)

        linear = accumulator + tl.load(
            bias_ptr + offs_n[None, :], mask=offs_n[None, :] < N, other=0.0
        )
        # Baseline Linear writes FP16 before F.gelu reads it. Preserve that
        # rounding boundary even though both operations share one kernel.
        rounded = linear.to(tl.float16).to(tl.float32)
        activated = 0.5 * rounded * (1.0 + tl.erf(rounded * 0.7071067811865476))
        tl.store(
            output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            activated,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


@torch.library.custom_op("person2_gemm::linear_exact_gelu", mutates_args=())
def linear_exact_gelu(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    m, k = x.shape
    n = weight.shape[0]
    output = torch.empty((m, n), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(m, meta["BLOCK_M"]) * triton.cdiv(n, meta["BLOCK_N"]),
    )
    _linear_exact_gelu_kernel[grid](
        x,
        weight,
        bias,
        output,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
    )
    return output


@linear_exact_gelu.register_fake
def _linear_exact_gelu_fake(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
) -> torch.Tensor:
    del bias
    return x.new_empty((x.shape[0], weight.shape[0]))


@torch.library.custom_op("person2_gemm::down_residual_masked", mutates_args=())
def down_residual_masked(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(residual)
    load_down_extension().down_residual_masked_out(
        hidden, residual, weight, bias, valid_token_mask, output
    )
    return output


@down_residual_masked.register_fake
def _down_residual_masked_fake(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    del hidden, weight, bias, valid_token_mask
    return torch.empty_like(residual)


@torch.library.custom_op("person2_gemm::down_residual_unmasked", mutates_args=())
def down_residual_unmasked(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(residual)
    load_down_extension().down_residual_unmasked_out(
        hidden, residual, weight, bias, output
    )
    return output


@down_residual_unmasked.register_fake
def _down_residual_unmasked_fake(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    del hidden, weight, bias
    return torch.empty_like(residual)


def supports_fusion(x: torch.Tensor) -> bool:
    return (
        triton is not None
        and x.device.type == "cuda"
        and x.dtype == torch.float16
        and x.ndim == 3
        and x.shape[-1] % 16 == 0
    )


def strict_accuracy(reference: torch.Tensor, candidate: torch.Tensor) -> bool:
    ref = reference.detach().float()
    opt = candidate.detach().float()
    error = (opt - ref).abs()
    passed = torch.isfinite(ref) & torch.isfinite(opt)
    passed &= (error <= 0.001) | (error <= 0.01 * ref.abs())
    return bool(passed.all().item())


class ArticleFusedFFNResidual(nn.Module):
    """Globally apply fused FFN kernels after an explicit correctness gate."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block
        self.enabled = False
        self.last_error: Optional[str] = None

    def _native(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.block._ffn_residual(x, valid_token_mask)

    def _fused(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        residual = x.reshape(batch * seq_len, d_model).contiguous()
        normalized = self.block.norm2(x).reshape(batch * seq_len, d_model)
        hidden = linear_exact_gelu(
            normalized, self.block.ffn_in.weight, self.block.ffn_in.bias
        )
        if valid_token_mask is None:
            output = down_residual_unmasked(
                hidden,
                residual,
                self.block.ffn_out.weight,
                self.block.ffn_out.bias,
            )
        else:
            output = down_residual_masked(
                hidden,
                residual,
                self.block.ffn_out.weight,
                self.block.ffn_out.bias,
                valid_token_mask.reshape(-1).contiguous(),
            )
        return output.reshape(batch, seq_len, d_model)

    def prepare(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> bool:
        """Compile/autotune and validate before the timed or compiled path."""
        self.enabled = False
        self.last_error = None
        if not supports_fusion(x):
            self.last_error = "unsupported device, dtype, rank, or alignment"
            return False
        try:
            with torch.inference_mode():
                reference = self._native(x, valid_token_mask)
                candidate = self._fused(x, valid_token_mask)
            if not strict_accuracy(reference, candidate):
                self.last_error = "strict numerical preflight failed"
                return False
            self.enabled = True
            return True
        except (ImportError, OSError, RuntimeError) as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return False

    def forward(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if self.enabled:
            return self._fused(x, valid_token_mask)
        return self._native(x, valid_token_mask)


class ArticleFusedFullBlock(nn.Module):
    """Full-block safety wrapper; final model assembly remains Person 3's."""

    def __init__(self, block: nn.Module, causal: bool) -> None:
        super().__init__()
        self.block = block
        self.causal = causal
        self.ffn = ArticleFusedFFNResidual(block)

    def prepare(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> bool:
        with torch.inference_mode():
            ffn_input = x + self.block.attention(
                self.block.norm1(x), valid_token_mask, self.causal
            )
        return self.ffn.prepare(ffn_input, valid_token_mask)

    def forward(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = x + self.block.attention(
            self.block.norm1(x), valid_token_mask, self.causal
        )
        return self.ffn(x, valid_token_mask)
