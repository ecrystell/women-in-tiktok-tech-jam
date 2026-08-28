"""Experimental inference-only CUDA fusion for Person 2's FFN output path."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load


_EXTENSION = None


def load_extension():
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    root = Path(__file__).resolve().parent
    build_directory = root / "benchmark-results" / "torch-extensions"
    build_directory.mkdir(parents=True, exist_ok=True)
    _EXTENSION = load(
        name="person2_ffn_cuda_ext",
        sources=[
            str(root / "csrc" / "person2_ffn_cuda.cpp"),
            str(root / "csrc" / "person2_ffn_cuda_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        extra_ldflags=["-lcublasLt"],
        build_directory=str(build_directory),
        verbose=os.environ.get("PERSON2_CUDA_VERBOSE", "0") == "1",
    )
    return _EXTENSION


@torch.library.custom_op("person2::ffn_out_residual", mutates_args=())
def ffn_out_residual(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(residual)
    load_extension().ffn_out_residual_out(
        hidden, residual, weight, bias, valid_token_mask, output
    )
    return output


@ffn_out_residual.register_fake
def _ffn_out_residual_fake(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    del hidden, weight, bias, valid_token_mask
    return torch.empty_like(residual)


class CudaFusedFFNResidual(nn.Module):
    """Use native norm/first GEMM/exact GELU and a fused output projection."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if (
            x.device.type != "cuda"
            or x.dtype != torch.float16
            or valid_token_mask is None
        ):
            return self.block._ffn_residual(x, valid_token_mask)

        batch, seq_len, d_model = x.shape
        residual = x.reshape(batch * seq_len, d_model).contiguous()
        normalized = self.block.norm2(x).reshape(batch * seq_len, d_model)
        hidden = F.gelu(
            self.block.ffn_in(normalized), approximate="none"
        ).contiguous()
        output = ffn_out_residual(
            hidden,
            residual,
            self.block.ffn_out.weight,
            self.block.ffn_out.bias,
            valid_token_mask.reshape(-1).contiguous(),
        )
        return output.reshape(batch, seq_len, d_model)


class CudaFusedFullBlock(nn.Module):
    """Full block wrapper used only for candidate safety benchmarking."""

    def __init__(self, block: nn.Module, causal: bool) -> None:
        super().__init__()
        self.block = block
        self.causal = causal
        self.ffn = CudaFusedFFNResidual(block)

    def forward(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = x + self.block.attention(
            self.block.norm1(x), valid_token_mask, self.causal
        )
        return self.ffn(x, valid_token_mask)
