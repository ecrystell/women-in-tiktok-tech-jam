#!/usr/bin/env python3
"""Profile Person 2's isolated FFN path and estimate its speedup ceiling."""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.profiler import ProfilerActivity, profile

from bench_person2_ffn import (
    Case,
    SWEEP_SHAPES,
    compile_model,
    make_input,
    make_models,
    mark_inference_step,
    resolve_device,
    resolve_dtype,
    time_model,
    warmup,
)


TARGET_SPEEDUP = 1.05


@dataclass(frozen=True)
class ProfileResult:
    case: str
    model: str
    mode: str
    median_ms: float
    profiled_device_ms: float
    launch_gap_upper_bound_ms: float
    gemm_percent: float
    layer_norm_percent: float
    gelu_percent: float
    residual_mask_percent: float
    copy_layout_percent: float
    other_percent: float
    non_gemm_percent: float
    amdahl_upper_bound: float
    target_feasible: bool


def event_device_time_us(event: object) -> float:
    value = getattr(event, "self_device_time_total", None)
    if value is None:
        value = getattr(event, "self_cuda_time_total", 0.0)
    return float(value or 0.0)


def categorize(key: str) -> str:
    name = key.lower()
    if any(token in name for token in ("addmm", "gemm", "cublas", "aten::mm")):
        return "gemm"
    if "layer_norm" in name or "layernorm" in name:
        return "layer_norm"
    if "gelu" in name:
        return "gelu"
    if any(
        token in name
        for token in ("masked_fill", "aten::add", "bitwise_not", "where")
    ):
        return "residual_mask"
    if any(
        token in name
        for token in ("copy", "clone", "contiguous", "transpose", "view")
    ):
        return "copy_layout"
    return "other"


def profile_model(
    model: torch.nn.Module,
    model_name: str,
    case: Case,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
    args: argparse.Namespace,
) -> ProfileResult:
    x, mask = make_input(case, device, dtype, args.seed)
    warmup(model, x, mask, args.warmup, device, mode)
    latency_samples = time_model(
        model, x, mask, args.repeats, device, mode
    )

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
            for _ in range(args.profile_iterations):
                mark_inference_step(device, mode)
                model(x, mask)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    category_us = {
        "gemm": 0.0,
        "layer_norm": 0.0,
        "gelu": 0.0,
        "residual_mask": 0.0,
        "copy_layout": 0.0,
        "other": 0.0,
    }
    for event in captured.key_averages():
        category_us[categorize(event.key)] += event_device_time_us(event)

    total_us = sum(category_us.values())
    if total_us <= 0:
        raise RuntimeError("profiler did not record device activity")

    per_call_device_ms = total_us / args.profile_iterations / 1000.0
    median_ms = statistics.median(latency_samples)
    gap_ms = max(0.0, median_ms - per_call_device_ms)
    gemm_fraction = category_us["gemm"] / total_us
    non_gemm_fraction = 1.0 - gemm_fraction
    amdahl_upper_bound = float("inf") if gemm_fraction == 0 else 1.0 / gemm_fraction

    def percent(category: str) -> float:
        return 100.0 * category_us[category] / total_us

    return ProfileResult(
        case=case.label,
        model=model_name,
        mode=mode,
        median_ms=median_ms,
        profiled_device_ms=per_call_device_ms,
        launch_gap_upper_bound_ms=gap_ms,
        gemm_percent=percent("gemm"),
        layer_norm_percent=percent("layer_norm"),
        gelu_percent=percent("gelu"),
        residual_mask_percent=percent("residual_mask"),
        copy_layout_percent=percent("copy_layout"),
        other_percent=percent("other"),
        non_gemm_percent=100.0 * non_gemm_fraction,
        amdahl_upper_bound=amdahl_upper_bound,
        target_feasible=amdahl_upper_bound >= TARGET_SPEEDUP,
    )


def print_result(result: ProfileResult) -> None:
    print(f"\n[{result.model} | {result.mode}] {result.case}")
    print(
        f"median={result.median_ms:.4f} ms | "
        f"profiled_device={result.profiled_device_ms:.4f} ms | "
        f"launch_gap_upper_bound={result.launch_gap_upper_bound_ms:.4f} ms"
    )
    print(
        f"gemm={result.gemm_percent:.2f}% | "
        f"layer_norm={result.layer_norm_percent:.2f}% | "
        f"gelu={result.gelu_percent:.2f}% | "
        f"residual_mask={result.residual_mask_percent:.2f}% | "
        f"copy_layout={result.copy_layout_percent:.2f}% | "
        f"other={result.other_percent:.2f}%"
    )
    print(
        f"non_gemm={result.non_gemm_percent:.2f}% | "
        f"amdahl_upper_bound={result.amdahl_upper_bound:.3f}x | "
        f"universal_1.05x_feasible={'YES' if result.target_feasible else 'NO'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--mode",
        choices=(
            "eager",
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        ),
        default="eager",
    )
    parser.add_argument(
        "--models", choices=("baseline", "optimized", "both"), default="both"
    )
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--profile-iterations", type=int, default=20)
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    if device.type != "cuda":
        raise RuntimeError("the feasibility profile requires CUDA")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("SKIP: this GPU does not support bfloat16")
        return 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(
        f"torch={torch.__version__} | cuda={torch.version.cuda} | "
        f"gpu={torch.cuda.get_device_name(device)} | dtype={dtype} | "
        f"mode={args.mode}"
    )

    results: list[ProfileResult] = []
    for case in cases_from_args(args):
        baseline, optimized = make_models(case, "isolated", device, dtype)
        selected = []
        if args.models in ("baseline", "both"):
            selected.append(("baseline", compile_model(baseline, args.mode)))
        if args.models in ("optimized", "both"):
            selected.append(("optimized", compile_model(optimized, args.mode)))
        for model_name, model in selected:
            result = profile_model(
                model, model_name, case, args.mode, device, dtype, args
            )
            results.append(result)
            print_result(result)
        torch.cuda.empty_cache()

    baseline_results = [result for result in results if result.model == "baseline"]
    if baseline_results:
        universal = all(result.target_feasible for result in baseline_results)
        print(
            "\nPROFILE GATE: "
            + ("PASS" if universal else "FAIL")
            + " | theoretical 1.05x target "
            + ("remains feasible" if universal else "is impossible from non-GEMM removal alone")
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps([asdict(result) for result in results], indent=2),
            encoding="utf-8",
        )
        print(f"wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
