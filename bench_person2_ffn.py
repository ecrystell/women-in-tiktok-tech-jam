#!/usr/bin/env python3
"""Accuracy and latency benchmark for Person 2's standalone block work."""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_transformer_benchmark import (
    BaselineTransformerBlock,
    OptimizedTransformerBlock,
)


SWEEP_SHAPES = (
    # batch, sequence, model, heads, FFN, causal, padding ratio
    (1, 128, 512, 8, 2048, False, 0.0),
    (8, 512, 512, 8, 2048, False, 0.0),
    (8, 512, 512, 8, 2048, True, 0.2),
    (4, 2048, 1024, 16, 4096, False, 0.0),
    (2, 4096, 1024, 16, 4096, True, 0.0),
)

CUDA_GRAPH_MODES = frozenset(("reduce-overhead", "max-autotune"))


class BaselineFFNResidual(nn.Module):
    """Expose only the baseline block's second pre-norm sublayer."""

    def __init__(self, block: BaselineTransformerBlock) -> None:
        super().__init__()
        self.block = block

    def forward(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = x + self.block.ffn_out(
            F.gelu(self.block.ffn_in(self.block.norm2(x)), approximate="none")
        )
        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)
        return x


class OptimizedFFNResidual(nn.Module):
    """Expose the exact FFN path used by OptimizedTransformerBlock."""

    def __init__(self, block: OptimizedTransformerBlock) -> None:
        super().__init__()
        self.block = block

    def forward(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.block._ffn_residual(x, valid_token_mask)


class FullBlock(nn.Module):
    def __init__(self, block: nn.Module, causal: bool) -> None:
        super().__init__()
        self.block = block
        self.causal = causal

    def forward(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.block(x, valid_token_mask, self.causal)


@dataclass(frozen=True)
class Case:
    batch: int
    seq_len: int
    d_model: int
    heads: int
    ffn_dim: int
    causal: bool
    padding_ratio: float

    @property
    def label(self) -> str:
        return (
            f"B={self.batch},S={self.seq_len},D={self.d_model},"
            f"H={self.heads},FFN={self.ffn_dim},causal={self.causal},"
            f"padding={self.padding_ratio:g}"
        )


@dataclass(frozen=True)
class Accuracy:
    passed: bool
    max_abs: float
    max_rel: float
    failed: int
    elements: int


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def make_input(
    case: Case, device: torch.device, dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(
        case.batch,
        case.seq_len,
        case.d_model,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if case.padding_ratio <= 0:
        mask = torch.ones(
            case.batch, case.seq_len, device=device, dtype=torch.bool
        )
        return x, mask

    min_valid = max(1, int(round(case.seq_len * (1.0 - case.padding_ratio))))
    lengths = torch.randint(
        min_valid,
        case.seq_len + 1,
        (case.batch,),
        device=device,
        generator=generator,
    )
    positions = torch.arange(case.seq_len, device=device)[None, :]
    mask = positions < lengths[:, None]
    return x.masked_fill(~mask[..., None], 0), mask


def compare(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    atol: float,
    rtol: float,
) -> Accuracy:
    ref = reference.detach().float()
    opt = candidate.detach().float()
    error = (opt - ref).abs()
    finite = torch.isfinite(ref) & torch.isfinite(opt)
    passed = finite & ((error <= atol) | (error <= rtol * ref.abs()))
    relative = error / ref.abs().clamp_min(1e-12)
    return Accuracy(
        passed=bool(passed.all().item()),
        max_abs=float(error.max().item()),
        max_rel=float(relative.max().item()),
        failed=int((~passed).sum().item()),
        elements=passed.numel(),
    )


def percentile(samples: list[float], q: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def warmup(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    iterations: int,
    device: torch.device,
    mode: str,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            mark_inference_step(device, mode)
            model(x, mask)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_model(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    iterations: int,
    device: torch.device,
    mode: str,
) -> list[float]:
    samples: list[float] = []
    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
            torch.cuda.synchronize(device)
            for index in range(iterations):
                mark_inference_step(device, mode)
                starts[index].record()
                model(x, mask)
                ends[index].record()
            torch.cuda.synchronize(device)
            samples.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, mask)
                samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def compile_model(model: nn.Module, mode: str) -> nn.Module:
    if mode == "eager":
        return model
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return torch.compile(model, mode=mode, fullgraph=True, dynamic=False)


def mark_inference_step(device: torch.device, mode: str) -> None:
    """Mark output lifetime boundaries for TorchInductor CUDA graphs."""
    if device.type != "cuda" or mode not in CUDA_GRAPH_MODES:
        return
    compiler = getattr(torch, "compiler", None)
    marker = getattr(compiler, "cudagraph_mark_step_begin", None)
    if marker is None:
        raise RuntimeError(
            "this PyTorch build lacks torch.compiler.cudagraph_mark_step_begin"
        )
    marker()


def accuracy_output(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    """Return an owned result that survives the next CUDA-graph replay."""
    mark_inference_step(device, mode)
    return model(x, mask).clone()


def make_models(
    case: Case,
    scope: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[nn.Module, nn.Module]:
    baseline_block = BaselineTransformerBlock(
        case.d_model, case.heads, case.ffn_dim
    )
    optimized_block = OptimizedTransformerBlock(
        case.d_model, case.heads, case.ffn_dim
    )
    optimized_block.load_state_dict(baseline_block.state_dict(), strict=True)
    baseline_block = baseline_block.to(device=device, dtype=dtype).eval()
    optimized_block = optimized_block.to(device=device, dtype=dtype).eval()
    optimized_block.prepare_ffn_weights()

    if scope == "isolated":
        return BaselineFFNResidual(baseline_block), OptimizedFFNResidual(
            optimized_block
        )
    return FullBlock(baseline_block, case.causal), FullBlock(
        optimized_block, case.causal
    )


def run_case(
    case: Case,
    scope: str,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> None:
    print(f"\n[{scope} | {mode}] {case.label}")
    try:
        baseline, candidate = make_models(case, scope, device, dtype)
        baseline = compile_model(baseline, mode)
        candidate = compile_model(candidate, mode)
        x, mask = make_input(case, device, dtype, args.seed)

        with torch.inference_mode():
            reference_output = accuracy_output(baseline, x, mask, device, mode)
            candidate_output = accuracy_output(candidate, x, mask, device, mode)
        accuracy = compare(
            reference_output,
            candidate_output,
            atol=args.atol,
            rtol=args.rtol,
        )
        print(
            f"accuracy={'PASS' if accuracy.passed else 'FAIL'} | "
            f"max_abs={accuracy.max_abs:.6g} | max_rel={accuracy.max_rel:.6g} | "
            f"failed={accuracy.failed}/{accuracy.elements}"
        )
        if not accuracy.passed:
            print("timing skipped because strict accuracy validation failed")
            return

        warmup(baseline, x, mask, args.warmup, device, mode)
        warmup(candidate, x, mask, args.warmup, device, mode)
        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        for round_index in range(args.rounds):
            if round_index % 2 == 0:
                baseline_samples.extend(
                    time_model(baseline, x, mask, args.repeats, device, mode)
                )
                candidate_samples.extend(
                    time_model(candidate, x, mask, args.repeats, device, mode)
                )
            else:
                candidate_samples.extend(
                    time_model(candidate, x, mask, args.repeats, device, mode)
                )
                baseline_samples.extend(
                    time_model(baseline, x, mask, args.repeats, device, mode)
                )

        baseline_median = statistics.median(baseline_samples)
        candidate_median = statistics.median(candidate_samples)
        tokens = case.batch * case.seq_len
        print(
            f"baseline median={baseline_median:.4f} ms | "
            f"p90={percentile(baseline_samples, 0.90):.4f} ms | "
            f"throughput={tokens * 1000.0 / baseline_median:.2f} token/s"
        )
        print(
            f"optimized median={candidate_median:.4f} ms | "
            f"p90={percentile(candidate_samples, 0.90):.4f} ms | "
            f"throughput={tokens * 1000.0 / candidate_median:.2f} token/s"
        )
        print(f"speedup={baseline_median / candidate_median:.3f}x")
    except torch.OutOfMemoryError as error:
        print(f"SKIP OOM: {error}")
    except Exception as error:  # Keep a multi-mode sweep running after backend failures.
        print(f"SKIP ERROR {type(error).__name__}: {error}")
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def cases_from_args(args: argparse.Namespace) -> Iterable[Case]:
    if args.sweep:
        return (Case(*shape) for shape in SWEEP_SHAPES)
    return (
        Case(
            args.batch_size,
            args.seq_len,
            args.d_model,
            args.heads,
            args.ffn_dim,
            args.causal,
            args.padding_ratio,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--scope", choices=("isolated", "full", "both"), default="isolated"
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "eager",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="eager",
    )
    parser.add_argument("--all-compile-modes", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.padding_ratio < 1:
        raise ValueError("padding_ratio must be in [0, 1)")
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("SKIP: this GPU does not support bfloat16")
            return 0

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(f"torch={torch.__version__} | device={device} | dtype={dtype}")
    if device.type == "cuda":
        print(
            f"gpu={torch.cuda.get_device_name(device)} | "
            f"capability={torch.cuda.get_device_capability(device)} | "
            f"cuda={torch.version.cuda}"
        )

    scopes = ("isolated", "full") if args.scope == "both" else (args.scope,)
    modes = (
        (
            "eager",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        )
        if args.all_compile_modes
        else (args.compile_mode,)
    )
    cases = tuple(cases_from_args(args))
    for case in cases:
        for scope in scopes:
            for mode in modes:
                run_case(case, scope, mode, device, dtype, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
