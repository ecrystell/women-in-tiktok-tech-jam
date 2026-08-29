#!/usr/bin/env python3
"""Profile Person 2's attention-residual plus ``norm2`` pre-FFN segment.

The control segment is exactly::

    residual = x + attention_update
    normalized = block.norm2(residual)

Use ``--candidate package.module:callable`` to compare a fused experiment.
The callable must have this deliberately small inference-only interface::

    candidate(x, attention_update, norm2) -> (residual, normalized)

It must return tensors with the same shapes and dtypes as the control.  The
default ``reference`` candidate is only an interface/calibration scaffold; it
is not an optimization.  Candidate setup, extension loading, plan selection,
and numerical preflight belong in the callable's construction/import path and
are excluded from the timed loop.

The script records CUDA-event latency, strict output equivalence, CUDA operator
shares for add and LayerNorm, and the kernel names/times captured by
``torch.profiler``.  It is intentionally standalone so a new Person 2 fusion
can be assessed before it changes the optimized block.
"""

from __future__ import annotations

import argparse
import importlib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, cast

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from bench_person2_ffn import (
    Case,
    SWEEP_SHAPES,
    compare,
    make_input,
    percentile,
    resolve_device,
    resolve_dtype,
)
from torch_transformer_benchmark import BaselineTransformerBlock


PreNormCandidate = Callable[
    [torch.Tensor, torch.Tensor, nn.LayerNorm], tuple[torch.Tensor, torch.Tensor]
]


@dataclass(frozen=True)
class TraceSummary:
    """CUDA-only profiler data for one operation, averaged per invocation."""

    cuda_ms: Optional[float]
    add_cuda_ms: Optional[float]
    layer_norm_cuda_ms: Optional[float]
    other_operator_cuda_ms: Optional[float]
    unattributed_cuda_ms: Optional[float]
    kernels: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class OperationResult:
    case: str
    implementation: str
    operation: str
    median_ms: float
    p90_ms: float
    throughput_tokens_per_s: float
    trace: TraceSummary


@dataclass(frozen=True)
class AccuracySummary:
    passed: bool
    max_abs: float
    max_rel: float
    failed: int
    elements: int
    residual_max_abs: float
    residual_max_rel: float
    residual_failed: int
    normalized_max_abs: float
    normalized_max_rel: float
    normalized_failed: int


def reference_candidate(
    x: torch.Tensor,
    attention_update: torch.Tensor,
    norm2: nn.LayerNorm,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the reference result; useful for wiring and profiler calibration."""
    residual = x + attention_update
    return residual, norm2(residual)


def _is_cuda_event(event: object) -> bool:
    return "cuda" in str(getattr(event, "device_type", "")).lower()


def _event_duration_us(event: object) -> float:
    """Read a profiler event duration across supported PyTorch versions."""
    duration_ns = getattr(event, "duration_time_ns", None)
    if duration_ns is not None and float(duration_ns or 0.0) > 0.0:
        return float(duration_ns) / 1000.0
    for attribute in (
        "self_cuda_time_total",
        "self_device_time_total",
        "cuda_time_total",
        "device_time_total",
    ):
        value = getattr(event, attribute, None)
        if value is not None and float(value or 0.0) > 0.0:
            return float(value)
    return 0.0


def _self_cuda_time_us(event: object) -> float:
    """Get self CUDA time without accidentally treating CPU time as device time."""
    value = getattr(event, "self_cuda_time_total", None)
    if value is not None:
        return float(value or 0.0)
    device_type = str(getattr(event, "device_type", "")).lower()
    if "cuda" not in device_type:
        return 0.0
    return float(getattr(event, "self_device_time_total", 0.0) or 0.0)


def _operator_category(name: str) -> str:
    lowered = name.lower()
    if "layer_norm" in lowered or "layernorm" in lowered:
        return "layer_norm"
    if lowered.startswith("aten::add") or "add_kernel" in lowered:
        return "add"
    return "other"


def _trace_operation(
    operation: Callable[[], object],
    device: torch.device,
    iterations: int,
) -> TraceSummary:
    """Capture operator shares and actual CUDA-kernel names when CUDA exists."""
    if iterations <= 0:
        return TraceSummary(None, None, None, None, None, ())

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as captured:
        with torch.inference_mode():
            for _ in range(iterations):
                operation()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type != "cuda":
        return TraceSummary(None, None, None, None, None, ())

    operator_us = defaultdict(float)
    for event in captured.key_averages():
        operator_us[_operator_category(str(getattr(event, "key", "")))] += (
            _self_cuda_time_us(event)
        )

    kernel_us = defaultdict(float)
    for event in captured.events():
        if not _is_cuda_event(event):
            continue
        duration_us = _event_duration_us(event)
        if duration_us <= 0.0:
            continue
        name = str(getattr(event, "name", getattr(event, "key", "unknown")))
        kernel_us[name] += duration_us

    total_kernel_us = sum(kernel_us.values())
    total_operator_us = sum(operator_us.values())
    per_call = 1.0 / iterations / 1000.0
    kernels = tuple(
        (name, duration_us * per_call)
        for name, duration_us in sorted(
            kernel_us.items(), key=lambda item: item[1], reverse=True
        )
    )
    return TraceSummary(
        cuda_ms=total_kernel_us * per_call,
        add_cuda_ms=operator_us["add"] * per_call,
        layer_norm_cuda_ms=operator_us["layer_norm"] * per_call,
        other_operator_cuda_ms=operator_us["other"] * per_call,
        unattributed_cuda_ms=max(0.0, total_kernel_us - total_operator_us)
        * per_call,
        kernels=kernels,
    )


def _time_operation(
    operation: Callable[[], object],
    device: torch.device,
    repeats: int,
) -> list[float]:
    if repeats <= 0:
        raise ValueError("--repeats must be positive")
    samples: list[float] = []
    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            torch.cuda.synchronize(device)
            for index in range(repeats):
                starts[index].record()
                operation()
                ends[index].record()
            torch.cuda.synchronize(device)
            samples.extend(
                start.elapsed_time(end) for start, end in zip(starts, ends)
            )
        else:
            for _ in range(repeats):
                start = time.perf_counter_ns()
                operation()
                samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def _warmup(
    operation: Callable[[], object], device: torch.device, iterations: int
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            operation()
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure_operation(
    *,
    case: Case,
    implementation: str,
    operation_name: str,
    operation: Callable[[], object],
    device: torch.device,
    warmup: int,
    repeats: int,
    rounds: int,
    profile_iterations: int,
) -> OperationResult:
    _warmup(operation, device, warmup)
    samples: list[float] = []
    for _ in range(rounds):
        samples.extend(_time_operation(operation, device, repeats))
    trace = _trace_operation(operation, device, profile_iterations)
    return _result_from_samples(
        case=case,
        implementation=implementation,
        operation_name=operation_name,
        samples=samples,
        trace=trace,
    )


def _result_from_samples(
    *,
    case: Case,
    implementation: str,
    operation_name: str,
    samples: list[float],
    trace: TraceSummary,
) -> OperationResult:
    median_ms = statistics.median(samples)
    return OperationResult(
        case=case.label,
        implementation=implementation,
        operation=operation_name,
        median_ms=median_ms,
        p90_ms=percentile(samples, 0.90),
        throughput_tokens_per_s=(case.batch * case.seq_len * 1000.0 / median_ms),
        trace=trace,
    )


def _measure_pair(
    *,
    case: Case,
    control: Callable[[], object],
    candidate: Callable[[], object],
    candidate_name: str,
    device: torch.device,
    warmup: int,
    repeats: int,
    rounds: int,
    profile_iterations: int,
) -> tuple[OperationResult, OperationResult]:
    """Time control/candidate in alternating rounds to reduce thermal drift."""
    _warmup(control, device, warmup)
    _warmup(candidate, device, warmup)
    control_samples: list[float] = []
    candidate_samples: list[float] = []
    for round_index in range(rounds):
        if round_index % 2 == 0:
            control_samples.extend(_time_operation(control, device, repeats))
            candidate_samples.extend(_time_operation(candidate, device, repeats))
        else:
            candidate_samples.extend(_time_operation(candidate, device, repeats))
            control_samples.extend(_time_operation(control, device, repeats))
    control_trace = _trace_operation(control, device, profile_iterations)
    candidate_trace = _trace_operation(candidate, device, profile_iterations)
    return (
        _result_from_samples(
            case=case,
            implementation="control",
            operation_name="residual_add_plus_norm2",
            samples=control_samples,
            trace=control_trace,
        ),
        _result_from_samples(
            case=case,
            implementation=candidate_name,
            operation_name="residual_add_plus_norm2",
            samples=candidate_samples,
            trace=candidate_trace,
        ),
    )


def _format_ms(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_result(result: OperationResult, max_kernels: int) -> None:
    trace = result.trace
    print(
        f"{result.implementation}/{result.operation}: "
        f"median={result.median_ms:.4f} ms | p90={result.p90_ms:.4f} ms | "
        f"throughput={result.throughput_tokens_per_s:.2f} token/s"
    )
    print(
        "  profiler CUDA: "
        f"total={_format_ms(trace.cuda_ms)} ms | "
        f"add={_format_ms(trace.add_cuda_ms)} ms | "
        f"layer_norm={_format_ms(trace.layer_norm_cuda_ms)} ms | "
        f"other_ops={_format_ms(trace.other_operator_cuda_ms)} ms | "
        f"unattributed={_format_ms(trace.unattributed_cuda_ms)} ms"
    )
    if trace.kernels:
        selected = trace.kernels[:max_kernels]
        print(
            "  CUDA kernels: "
            + "; ".join(f"{name} ({latency:.4f} ms)" for name, latency in selected)
        )
    elif trace.cuda_ms is None:
        print("  CUDA kernels: unavailable (CPU execution)")
    else:
        print("  CUDA kernels: profiler returned no kernel events")


def _coerce_output(
    value: object,
    x: torch.Tensor,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(
            f"{label} must return `(residual, normalized)`, got {type(value).__name__}"
        )
    residual, normalized = value
    if not isinstance(residual, torch.Tensor) or not isinstance(normalized, torch.Tensor):
        raise TypeError(f"{label} must return two torch.Tensor values")
    for name, tensor in (("residual", residual), ("normalized", normalized)):
        if tensor.shape != x.shape:
            raise ValueError(
                f"{label} {name} shape {tuple(tensor.shape)} != input {tuple(x.shape)}"
            )
        if tensor.dtype != x.dtype:
            raise ValueError(
                f"{label} {name} dtype {tensor.dtype} != input dtype {x.dtype}"
            )
        if tensor.device != x.device:
            raise ValueError(
                f"{label} {name} device {tensor.device} != input device {x.device}"
            )
    return residual, normalized


def _compare_outputs(
    reference: tuple[torch.Tensor, torch.Tensor],
    candidate: tuple[torch.Tensor, torch.Tensor],
    atol: float,
    rtol: float,
) -> AccuracySummary:
    residual = compare(reference[0], candidate[0], atol=atol, rtol=rtol)
    normalized = compare(reference[1], candidate[1], atol=atol, rtol=rtol)
    return AccuracySummary(
        passed=residual.passed and normalized.passed,
        max_abs=max(residual.max_abs, normalized.max_abs),
        max_rel=max(residual.max_rel, normalized.max_rel),
        failed=residual.failed + normalized.failed,
        elements=residual.elements + normalized.elements,
        residual_max_abs=residual.max_abs,
        residual_max_rel=residual.max_rel,
        residual_failed=residual.failed,
        normalized_max_abs=normalized.max_abs,
        normalized_max_rel=normalized.max_rel,
        normalized_failed=normalized.failed,
    )


def _resolve_candidate(spec: str) -> tuple[str, PreNormCandidate]:
    if spec == "reference":
        return "reference", reference_candidate
    module_name, separator, attribute_path = spec.partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(
            "--candidate must be `reference` or `package.module:callable`"
        )
    module = importlib.import_module(module_name)
    value: object = module
    for attribute in attribute_path.split("."):
        value = getattr(value, attribute)
    if not callable(value):
        raise TypeError(f"candidate {spec!r} is not callable")
    return spec, cast(PreNormCandidate, value)


def _make_attention_update(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator(device=x.device).manual_seed(seed + 1)
    update = torch.randn(
        x.shape, device=x.device, dtype=x.dtype, generator=generator
    )
    if mask is not None:
        update = update.masked_fill(~mask[..., None], 0)
    return update.contiguous()


def _cases_from_args(args: argparse.Namespace) -> Iterable[Case]:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        default="reference",
        help=(
            "`reference` or a callable `package.module:callable` accepting "
            "(x, attention_update, norm2) and returning (residual, normalized)"
        ),
    )
    parser.add_argument("--candidate-name", help="short label used in reports")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
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
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--max-kernels", type=int, default=12)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("--padding-ratio must be in [0, 1)")
    for name in ("warmup", "repeats", "rounds", "profile_iterations", "max_kernels"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")


def main() -> int:
    args = _parse_args()
    _validate_args(args)
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if dtype == torch.bfloat16 and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            print("SKIP: this GPU does not support bfloat16")
            return 0

    candidate_source, candidate = _resolve_candidate(args.candidate)
    candidate_name = args.candidate_name or candidate_source
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    print(
        f"torch={torch.__version__} | device={device} | dtype={dtype} | "
        f"candidate={candidate_name}"
    )
    if device.type == "cuda":
        print(
            f"gpu={torch.cuda.get_device_name(device)} | "
            f"capability={torch.cuda.get_device_capability(device)} | "
            f"cuda={torch.version.cuda}"
        )
        if torch.cuda.get_device_capability(device) != (7, 5):
            print("NOTE: this is not the Tesla T4 (sm_75) acceptance configuration")
    else:
        print("NOTE: CUDA profiler/kernel timing is unavailable on CPU")
    if candidate_source == "reference":
        print("NOTE: candidate=reference is a calibration scaffold, not an optimization")

    results: list[OperationResult] = []
    accuracies: list[dict[str, object]] = []
    for case in _cases_from_args(args):
        print(f"\n[{case.label}]")
        block = BaselineTransformerBlock(
            case.d_model, case.heads, case.ffn_dim
        ).to(device=device, dtype=dtype).eval()
        x, mask = make_input(case, device, dtype, args.seed)
        attention_update = _make_attention_update(x, mask, args.seed)
        control = lambda: reference_candidate(x, attention_update, block.norm2)

        try:
            with torch.inference_mode():
                reference_output = _coerce_output(control(), x, "control")
                candidate_output = _coerce_output(
                    candidate(x, attention_update, block.norm2), x, candidate_name
                )
            accuracy = _compare_outputs(
                reference_output, candidate_output, args.atol, args.rtol
            )
            print(
                f"accuracy={'PASS' if accuracy.passed else 'FAIL'} | "
                f"max_abs={accuracy.max_abs:.6g} | max_rel={accuracy.max_rel:.6g} | "
                f"failed={accuracy.failed}/{accuracy.elements}"
            )
            print(
                "  residual: "
                f"max_abs={accuracy.residual_max_abs:.6g} | "
                f"max_rel={accuracy.residual_max_rel:.6g} | "
                f"failed={accuracy.residual_failed} | normalized: "
                f"max_abs={accuracy.normalized_max_abs:.6g} | "
                f"max_rel={accuracy.normalized_max_rel:.6g} | "
                f"failed={accuracy.normalized_failed}"
            )
        except Exception as error:
            accuracy = None
            print(f"candidate setup/correctness error: {type(error).__name__}: {error}")

        residual = reference_output[0] if accuracy is not None else x + attention_update
        control_add = lambda: x + attention_update
        control_norm = lambda: block.norm2(residual)
        add_result = _measure_operation(
            case=case,
            implementation="control",
            operation_name="residual_add_only",
            operation=control_add,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            rounds=args.rounds,
            profile_iterations=args.profile_iterations,
        )
        norm_result = _measure_operation(
            case=case,
            implementation="control",
            operation_name="norm2_only",
            operation=control_norm,
            device=device,
            warmup=args.warmup,
            repeats=args.repeats,
            rounds=args.rounds,
            profile_iterations=args.profile_iterations,
        )

        if accuracy is not None:
            accuracies.append({"case": case.label, **asdict(accuracy)})
        if accuracy is None or not accuracy.passed:
            control_result = _measure_operation(
                case=case,
                implementation="control",
                operation_name="residual_add_plus_norm2",
                operation=control,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                rounds=args.rounds,
                profile_iterations=args.profile_iterations,
            )
            results.extend((control_result, add_result, norm_result))
            _print_result(control_result, args.max_kernels)
            _print_result(add_result, args.max_kernels)
            _print_result(norm_result, args.max_kernels)
            print("candidate timing skipped because strict correctness did not pass")
        else:
            candidate_operation = lambda: candidate(x, attention_update, block.norm2)
            control_result, candidate_result = _measure_pair(
                case=case,
                control=control,
                candidate=candidate_operation,
                candidate_name=candidate_name,
                device=device,
                warmup=args.warmup,
                repeats=args.repeats,
                rounds=args.rounds,
                profile_iterations=args.profile_iterations,
            )
            results.extend((control_result, add_result, norm_result, candidate_result))
            _print_result(control_result, args.max_kernels)
            _print_result(add_result, args.max_kernels)
            _print_result(norm_result, args.max_kernels)
            _print_result(candidate_result, args.max_kernels)
            speedup = control_result.median_ms / candidate_result.median_ms
            if candidate_source == "reference":
                print(f"reference calibration ratio={speedup:.3f}x (not a speed claim)")
            else:
                print(f"candidate speedup={speedup:.3f}x")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(
                {
                    "candidate": candidate_name,
                    "device": str(device),
                    "dtype": str(dtype),
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "results": [asdict(result) for result in results],
                    "accuracy": accuracies,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
