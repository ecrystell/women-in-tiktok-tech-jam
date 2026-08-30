#!/usr/bin/env python3
"""Accuracy and latency benchmark for Person 2's standalone block work."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional

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

# Organizer Appendix configurations 1-13. Configuration 14 is intentionally
# excluded: its activation alone exceeds practical T4 memory and its dominant
# 100,000-token attention workload is outside Person 2 ownership.
OFFICIAL_CASES = (
    # batch, sequence, model, heads, FFN, causal, padding ratio, official IDs
    (64, 128, 128, 4, 128, True, 0.0, (1,)),
    (1, 128, 128, 4, 128, True, 0.0, (2,)),
    (4, 128, 128, 4, 128, True, 0.0, (3,)),
    (16, 128, 128, 4, 128, True, 0.0, (4,)),
    (128, 128, 128, 4, 128, True, 0.0, (5,)),
    (10000, 128, 128, 4, 128, True, 0.0, (6,)),
    (64, 128, 32, 4, 32, True, 0.0, (7,)),
    (64, 128, 1024, 4, 1024, True, 0.0, (8,)),
    (64, 128, 128, 1, 128, True, 0.0, (9,)),
    (64, 128, 128, 2, 128, True, 0.0, (10,)),
    (64, 128, 128, 16, 128, True, 0.0, (11,)),
    (64, 32, 128, 4, 128, True, 0.0, (12,)),
    (64, 1024, 128, 4, 128, True, 0.0, (13,)),
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

    def prepare(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> bool:
        return self.block.prepare_fast_ffn(x, valid_token_mask)

    @property
    def prepare_error(self) -> Optional[str]:
        return self.block.fast_ffn_error


class FullBlock(nn.Module):
    def __init__(self, block: nn.Module, causal: bool) -> None:
        super().__init__()
        self.block = block
        self.causal = causal

    def forward(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.block(x, valid_token_mask, self.causal)

    def prepare(
        self, x: torch.Tensor, valid_token_mask: Optional[torch.Tensor]
    ) -> bool:
        if not isinstance(self.block, OptimizedTransformerBlock):
            return False
        with torch.inference_mode():
            ffn_input = x + self.block.attention(
                self.block.norm1(x), valid_token_mask, self.causal
            )
        return self.block.prepare_fast_ffn(ffn_input, valid_token_mask)

    @property
    def prepare_error(self) -> Optional[str]:
        return getattr(self.block, "fast_ffn_error", None)


@dataclass(frozen=True)
class Case:
    batch: int
    seq_len: int
    d_model: int
    heads: int
    ffn_dim: int
    causal: bool
    padding_ratio: float
    official_ids: tuple[int, ...] = ()

    @property
    def tokens(self) -> int:
        return self.batch * self.seq_len

    @property
    def label(self) -> str:
        official = (
            f"official={','.join(str(value) for value in self.official_ids)},"
            if self.official_ids
            else ""
        )
        return (
            f"{official}B={self.batch},S={self.seq_len},D={self.d_model},"
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


class StaticCudaGraphFFN(nn.Module):
    """Diagnostic fixed-pointer replay wrapper; never a production backend."""

    def __init__(
        self,
        model: nn.Module,
        x: torch.Tensor,
        mask: torch.Tensor,
        capture_warmups: int,
    ) -> None:
        super().__init__()
        if not x.is_cuda:
            raise RuntimeError("CUDA-graph replay requires CUDA")
        self.model = model
        self.static_x = x
        self.static_mask = mask
        side_stream = torch.cuda.Stream(device=x.device)
        side_stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(side_stream), torch.inference_mode():
            for _ in range(max(3, capture_warmups)):
                model(x, mask)
        torch.cuda.current_stream(x.device).wait_stream(side_stream)
        torch.cuda.synchronize(x.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.inference_mode():
            self.static_output = model(self.static_x, self.static_mask)
        torch.cuda.synchronize(x.device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.data_ptr() != self.static_x.data_ptr():
            raise RuntimeError("CUDA graph requires the captured input pointer")
        if mask.data_ptr() != self.static_mask.data_ptr():
            raise RuntimeError("CUDA graph requires the captured mask pointer")
        self.graph.replay()
        return self.static_output

    def output_ownership(self) -> tuple[bool, str]:
        with torch.inference_mode():
            first = self(self.static_x, self.static_mask)
            first_pointer = first.data_ptr()
            second = self(self.static_x, self.static_mask)
        if first_pointer == second.data_ptr():
            return False, "replay reuses and may overwrite the previous output buffer"
        return True, "each replay owns distinct output storage"


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
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case": asdict(case),
        "scope": scope,
        "backend": args.backend,
        "mode": mode,
        "process_index": args.process_index,
        "status": "error",
        "eligible": False,
    }
    print(f"\n[{scope} | {args.backend} | {mode}] {case.label}")
    try:
        if args.backend == "cuda-graph" and scope != "isolated":
            raise RuntimeError("CUDA-graph diagnostics support isolated FFN only")
        if args.backend == "cuda-graph" and mode != "eager":
            raise RuntimeError("CUDA-graph diagnostics require eager mode")
        x, mask = make_input(case, device, dtype, args.seed)
        baseline, candidate = make_models(case, scope, device, dtype)
        fast_enabled = False
        prepare = getattr(candidate, "prepare", None)
        if prepare is not None:
            fast_enabled = prepare(x, mask)
            reason = getattr(candidate, "prepare_error", None)
            block = getattr(candidate, "block", None)
            compact_enabled = bool(
                getattr(block, "_compact_ffn_enabled", False)
            )
            compact_reason = getattr(block, "compact_ffn_error", None)
            print(
                f"fast_ffn={'ENABLED' if fast_enabled else 'FALLBACK'}"
                + (f" | reason={reason}" if reason else "")
                + (
                    " | valid_row_compaction=ENABLED"
                    if compact_enabled
                    else ""
                )
                + (
                    f" | compact_reason={compact_reason}"
                    if compact_reason
                    else ""
                )
            )
        baseline = compile_model(baseline, mode)
        graph_owns_outputs = True
        graph_ownership_reason = "not applicable"
        if args.backend == "cuda-graph":
            candidate = StaticCudaGraphFFN(candidate, x, mask, args.warmup)
            graph_owns_outputs, graph_ownership_reason = candidate.output_ownership()
            print(
                "cuda_graph_output_ownership="
                f"{'PASS' if graph_owns_outputs else 'FAIL'} | "
                f"reason={graph_ownership_reason}"
            )
        else:
            candidate = compile_model(candidate, mode)

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
            result.update(
                status="accuracy-failed",
                accuracy=asdict(accuracy),
                graph_output_ownership=graph_owns_outputs,
                graph_ownership_reason=graph_ownership_reason,
                fast_path_enabled=fast_enabled,
            )
            return result

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
        baseline_p90 = percentile(baseline_samples, 0.90)
        candidate_p90 = percentile(candidate_samples, 0.90)
        speedup = baseline_median / candidate_median
        tokens = case.tokens
        print(
            f"baseline median={baseline_median:.4f} ms | "
            f"p90={baseline_p90:.4f} ms | "
            f"throughput={tokens * 1000.0 / baseline_median:.2f} token/s"
        )
        print(
            f"optimized median={candidate_median:.4f} ms | "
            f"p90={candidate_p90:.4f} ms | "
            f"throughput={tokens * 1000.0 / candidate_median:.2f} token/s"
        )
        print(f"speedup={speedup:.3f}x")
        if scope == "isolated":
            median_gate = speedup >= args.isolated_gate
            p90_gate = candidate_p90 <= baseline_p90
        else:
            median_gate = speedup >= args.full_block_gate
            p90_gate = candidate_p90 <= baseline_p90 * args.full_block_p90_limit
        eligible = (
            accuracy.passed
            and median_gate
            and p90_gate
            and graph_owns_outputs
            and fast_enabled
        )
        result.update(
            status="measured",
            eligible=eligible,
            accuracy=asdict(accuracy),
            graph_output_ownership=graph_owns_outputs,
            graph_ownership_reason=graph_ownership_reason,
            fast_path_enabled=fast_enabled,
            baseline_median_ms=baseline_median,
            baseline_p90_ms=baseline_p90,
            baseline_throughput_tokens_s=tokens * 1000.0 / baseline_median,
            optimized_median_ms=candidate_median,
            optimized_p90_ms=candidate_p90,
            optimized_throughput_tokens_s=tokens * 1000.0 / candidate_median,
            speedup=speedup,
            median_gate_passed=median_gate,
            p90_gate_passed=p90_gate,
        )
        print(f"selection_gate={'PASS' if eligible else 'FAIL'}")
        return result
    except torch.OutOfMemoryError as error:
        print(f"SKIP OOM: {error}")
        result.update(status="oom", error=str(error))
        return result
    except Exception as error:  # Keep a multi-mode sweep running after backend failures.
        print(f"SKIP ERROR {type(error).__name__}: {error}")
        result.update(
            status="error", error_type=type(error).__name__, error=str(error)
        )
        return result
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()


def official_cases() -> tuple[Case, ...]:
    return tuple(Case(*shape) for shape in OFFICIAL_CASES)


def official_isolated_cases() -> tuple[Case, ...]:
    """Deduplicate official cases that generate the same Person 2 workload."""
    unique: dict[tuple[int, int, int], Case] = {}
    for case in official_cases():
        key = (case.tokens, case.d_model, case.ffn_dim)
        previous = unique.get(key)
        if previous is None:
            unique[key] = case
        else:
            unique[key] = replace(
                previous,
                official_ids=previous.official_ids + case.official_ids,
            )
    return tuple(unique.values())


def cases_from_args(args: argparse.Namespace, scope: str) -> Iterable[Case]:
    if args.suite == "official":
        cases = official_isolated_cases() if scope == "isolated" else official_cases()
        if args.official_case:
            requested = set(args.official_case)
            cases = tuple(
                case for case in cases if requested.intersection(case.official_ids)
            )
        return cases
    if args.suite == "legacy" or args.sweep:
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
    parser.add_argument(
        "--suite", choices=("custom", "legacy", "official"), default="custom"
    )
    parser.add_argument(
        "--official-case",
        action="append",
        type=int,
        help="limit the official suite to one or more organizer case IDs",
    )
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
        "--backend", choices=("full-op", "cuda-graph"), default="full-op"
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
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--process-index", type=int, default=1)
    parser.add_argument("--isolated-gate", type=float, default=1.005)
    parser.add_argument("--full-block-gate", type=float, default=0.99)
    parser.add_argument("--full-block-p90-limit", type=float, default=1.02)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.sweep and args.suite != "custom":
        raise ValueError("--sweep is a legacy-suite alias and cannot be combined")
    if args.official_case and args.suite != "official":
        raise ValueError("--official-case requires --suite official")
    if args.official_case and not set(args.official_case).issubset(range(1, 14)):
        raise ValueError("official case IDs must be between 1 and 13")
    if args.rounds is None:
        args.rounds = 5 if args.suite == "official" else 3
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
    environment: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "dtype": str(dtype),
        "compile_available": hasattr(torch, "compile"),
    }
    print(f"torch={torch.__version__} | device={device} | dtype={dtype}")
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment.update(
            gpu=properties.name,
            capability=list(torch.cuda.get_device_capability(device)),
            total_memory_bytes=properties.total_memory,
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
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
    results: list[dict[str, Any]] = []
    for scope in scopes:
        cases = tuple(cases_from_args(args, scope))
        for case in cases:
            for mode in modes:
                results.append(run_case(case, scope, mode, device, dtype, args))

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        serializable_args = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        args.json_output.write_text(
            json.dumps(
                {
                    "environment": environment,
                    "arguments": serializable_args,
                    "results": results,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
