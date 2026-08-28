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
    "person2_post::gelu_exact_inplace_v1", mutates_args=("hidden",)
)
def gelu_exact_inplace(hidden: torch.Tensor) -> None:
    """Apply erf-based exact GELU to an exclusively owned FP16 temporary."""
    load_extension().gelu_exact_inplace(hidden)


@gelu_exact_inplace.register_fake
def _gelu_exact_inplace_fake(hidden: torch.Tensor) -> None:
    del hidden
