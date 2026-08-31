"""Inference-only FP16 residual/mask CUDA postprocessing with safe fallback."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

import torch


_EXTENSION = None
_EXTENSION_ERROR: Optional[str] = None


def load_extension():
    """Build and import the extension outside timed inference."""
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError(_EXTENSION_ERROR)

    root = Path(__file__).resolve().parent
    build_root = root / "benchmark-results" / "person2-post-cuda-build"
    build_lib = build_root / "lib"
    build_temp = build_root / "temp"
    build_lib.mkdir(parents=True, exist_ok=True)
    build_temp.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(root / "setup_person2_post_cuda.py"),
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
            "Person 2 CUDA post extension build failed:\n"
            + (result.stdout or "")
            + (result.stderr or "")
        )
        raise RuntimeError(_EXTENSION_ERROR)
    sys.path.insert(0, str(build_lib))
    _EXTENSION = importlib.import_module("person2_post_cuda_ext")
    return _EXTENSION


@torch.library.custom_op("person2_post::residual_masked_v2", mutates_args=())
def residual_masked(
    update: torch.Tensor,
    residual: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(residual)
    load_extension().residual_masked_out(
        update, residual, valid_token_mask, output
    )
    return output


@residual_masked.register_fake
def _residual_masked_fake(
    update: torch.Tensor,
    residual: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    del update, valid_token_mask
    return torch.empty_like(residual)


@torch.library.custom_op("person2_post::residual_unmasked_v2", mutates_args=())
def residual_unmasked(
    update: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty_like(residual)
    load_extension().residual_unmasked_out(update, residual, output)
    return output


@residual_unmasked.register_fake
def _residual_unmasked_fake(
    update: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    del update
    return torch.empty_like(residual)


@torch.library.custom_op(
    "person2_post::exact_gelu_down_masked_v2",
    mutates_args=("preactivation",),
)
def exact_gelu_down_masked(
    preactivation: torch.Tensor,
    weight_nt: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    """Launch native exact GELU/down GEMM and the masked post kernel in C++."""
    return load_extension().exact_gelu_down_masked(
        preactivation, weight_nt, bias, residual, valid_token_mask
    )


@exact_gelu_down_masked.register_fake
def _exact_gelu_down_masked_fake(
    preactivation: torch.Tensor,
    weight_nt: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    valid_token_mask: torch.Tensor,
) -> torch.Tensor:
    del preactivation, weight_nt, bias, valid_token_mask
    return torch.empty_like(residual)


@torch.library.custom_op(
    "person2_post::exact_gelu_down_unmasked_v2",
    mutates_args=("preactivation",),
)
def exact_gelu_down_unmasked(
    preactivation: torch.Tensor,
    weight_nt: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    """Launch native exact GELU/down GEMM and the unmasked post kernel in C++."""
    return load_extension().exact_gelu_down_unmasked(
        preactivation, weight_nt, bias, residual
    )


@exact_gelu_down_unmasked.register_fake
def _exact_gelu_down_unmasked_fake(
    preactivation: torch.Tensor,
    weight_nt: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    del preactivation, weight_nt, bias
    return torch.empty_like(residual)


@torch.library.custom_op(
    "person2_post::identity_layer_norm_ffn_masked_v1",
    mutates_args=(),
)
def identity_layer_norm_ffn_masked(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight_nt: torch.Tensor,
    down_bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Orchestrate the complete identity-affine FFN behind one boundary."""
    return load_extension().identity_layer_norm_ffn_masked(
        residual,
        up_weight,
        up_bias,
        down_weight_nt,
        down_bias,
        valid_token_mask,
        eps,
    )


@identity_layer_norm_ffn_masked.register_fake
def _identity_layer_norm_ffn_masked_fake(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight_nt: torch.Tensor,
    down_bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del (
        up_weight,
        up_bias,
        down_weight_nt,
        down_bias,
        valid_token_mask,
        eps,
    )
    return torch.empty_like(residual)


@torch.library.custom_op(
    "person2_post::identity_layer_norm_ffn_unmasked_v1",
    mutates_args=(),
)
def identity_layer_norm_ffn_unmasked(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight_nt: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Orchestrate the complete unmasked identity-affine FFN."""
    return load_extension().identity_layer_norm_ffn_unmasked(
        residual,
        up_weight,
        up_bias,
        down_weight_nt,
        down_bias,
        eps,
    )


@identity_layer_norm_ffn_unmasked.register_fake
def _identity_layer_norm_ffn_unmasked_fake(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    down_weight_nt: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del up_weight, up_bias, down_weight_nt, down_bias, eps
    return torch.empty_like(residual)


@torch.library.custom_op(
    "person2_post::algebraic_layer_norm_up_gelu_v1",
    mutates_args=(),
)
def algebraic_layer_norm_up_gelu(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    up_weight_row_sum: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run raw up GEMM followed by LayerNorm correction and exact GELU."""
    return load_extension().algebraic_layer_norm_up_gelu(
        residual,
        up_weight,
        up_bias,
        up_weight_row_sum,
        eps,
    )


@algebraic_layer_norm_up_gelu.register_fake
def _algebraic_layer_norm_up_gelu_fake(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    up_weight_row_sum: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del up_bias, up_weight_row_sum, eps
    return residual.new_empty((residual.shape[0], up_weight.shape[0]))


@torch.library.custom_op(
    "person2_post::algebraic_layer_norm_ffn_masked_v1",
    mutates_args=(),
)
def algebraic_layer_norm_ffn_masked(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    up_weight_row_sum: torch.Tensor,
    down_weight_nt: torch.Tensor,
    down_bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the algebraic identity-LayerNorm FFN and masked residual post."""
    return load_extension().algebraic_layer_norm_ffn_masked(
        residual,
        up_weight,
        up_bias,
        up_weight_row_sum,
        down_weight_nt,
        down_bias,
        valid_token_mask,
        eps,
    )


@algebraic_layer_norm_ffn_masked.register_fake
def _algebraic_layer_norm_ffn_masked_fake(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    up_weight_row_sum: torch.Tensor,
    down_weight_nt: torch.Tensor,
    down_bias: torch.Tensor,
    valid_token_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del (
        up_weight,
        up_bias,
        up_weight_row_sum,
        down_weight_nt,
        down_bias,
        valid_token_mask,
        eps,
    )
    return torch.empty_like(residual)


@torch.library.custom_op(
    "person2_post::algebraic_layer_norm_ffn_unmasked_v1",
    mutates_args=(),
)
def algebraic_layer_norm_ffn_unmasked(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    up_weight_row_sum: torch.Tensor,
    down_weight_nt: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the algebraic identity-LayerNorm FFN and unmasked residual post."""
    return load_extension().algebraic_layer_norm_ffn_unmasked(
        residual,
        up_weight,
        up_bias,
        up_weight_row_sum,
        down_weight_nt,
        down_bias,
        eps,
    )


@algebraic_layer_norm_ffn_unmasked.register_fake
def _algebraic_layer_norm_ffn_unmasked_fake(
    residual: torch.Tensor,
    up_weight: torch.Tensor,
    up_bias: torch.Tensor,
    up_weight_row_sum: torch.Tensor,
    down_weight_nt: torch.Tensor,
    down_bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    del (
        up_weight,
        up_bias,
        up_weight_row_sum,
        down_weight_nt,
        down_bias,
        eps,
    )
    return torch.empty_like(residual)
