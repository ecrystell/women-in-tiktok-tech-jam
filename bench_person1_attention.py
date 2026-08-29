"""Standalone benchmark for Person 1's attention implementations.

Examples:

    python bench_person1_attention.py --device cuda --dtype float16
    python bench_person1_attention.py --device cuda --dtype float16 --causal
    python bench_person1_attention.py --device cuda --dtype float16 \
        --padding-ratio 0.25 --mode both
    python bench_person1_attention.py --device cuda --dtype float16 --sweep

The benchmark intentionally does not modify ``torch_transformer_benchmark.py``.
It compares complete attention modules so packed-QKV projection and output
layout costs are visible alongside the attention-core speedup.
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    torch = None  # type: ignore[assignment]

if torch is not None:
    from person1_triton_attention import (
        PackedQKVSDPAAttention,
        TritonSelfAttention,
    )
    from torch_transformer_benchmark import BaselineSelfAttention


@dataclass(frozen=True)
class BenchmarkConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    dtype: torch.dtype
    device: torch.device
    causal: bool
    padding_ratio: float

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_heads

    def validate(self) -> None:
        if self.batch_size <= 0 or self.seq_len <= 0:
            raise ValueError("batch_size and seq_len must be positive")
        if self.d_model <= 0 or self.num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= self.padding_ratio < 1.0:
            raise ValueError("padding_ratio must be in [0, 1)")


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def make_inputs(
    config: BenchmarkConfig,
    seed: int,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    generator = torch.Generator(device=config.device)
    generator.manual_seed(seed)
    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        device=config.device,
        dtype=config.dtype,
        generator=generator,
    )
    if config.padding_ratio <= 0.0:
        return x, None

    min_valid = max(1, int(round(config.seq_len * (1.0 - config.padding_ratio))))
    lengths = torch.randint(
        min_valid,
        config.seq_len + 1,
        (config.batch_size,),
        device=config.device,
        generator=generator,
    )
    positions = torch.arange(config.seq_len, device=config.device)[None, :]
    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)
    return x, valid_token_mask


def compare_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    rtol: float = 0.01,
    atol: float = 0.001,
) -> Tuple[bool, float, float, int]:
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    error = (candidate_float - reference_float).abs()
    relative = error / reference_float.abs().clamp_min(1e-12)
    finite = torch.isfinite(reference_float) & torch.isfinite(candidate_float)
    passed = finite & ((error <= atol) | (error <= rtol * reference_float.abs()))
    return (
        bool(passed.all()),
        float(error.max().item()),
        float(relative.max().item()),
        int((~passed).sum().item()),
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    warmup: int,
    repeats: int,
) -> List[float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(x, valid_token_mask, causal)
    synchronize(x.device)

    samples: List[float] = []
    with torch.inference_mode():
        if x.device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            for index in range(repeats):
                starts[index].record()
                model(x, valid_token_mask, causal)
                ends[index].record()
            synchronize(x.device)
            samples.extend(
                start.elapsed_time(end)
                for start, end in zip(starts, ends)
            )
        else:
            for _ in range(repeats):
                start = time.perf_counter_ns()
                model(x, valid_token_mask, causal)
                end = time.perf_counter_ns()
                samples.append((end - start) / 1e6)
    return samples


def timed_backward(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    warmup: int,
    repeats: int,
    upstream: torch.Tensor,
) -> Tuple[List[float], torch.Tensor]:
    parameters = list(model.parameters())
    for parameter in parameters:
        parameter.requires_grad_(False)
    input_tensor = x.detach().clone().requires_grad_(True)

    def run_once() -> None:
        input_tensor.grad = None
        output = model(input_tensor, valid_token_mask, causal)
        (output * upstream).sum().backward()

    for _ in range(warmup):
        run_once()
    synchronize(x.device)

    samples: List[float] = []
    if x.device.type == "cuda":
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
        for index in range(repeats):
            starts[index].record()
            run_once()
            ends[index].record()
        synchronize(x.device)
        samples.extend(
            start.elapsed_time(end) for start, end in zip(starts, ends)
        )
    else:
        for _ in range(repeats):
            start = time.perf_counter_ns()
            run_once()
            end = time.perf_counter_ns()
            samples.append((end - start) / 1e6)
    return samples, input_tensor.grad.detach().clone()


def summarize(samples: Sequence[float]) -> str:
    ordered = sorted(samples)
    p90_position = (len(ordered) - 1) * 0.90
    lower = math.floor(p90_position)
    upper = math.ceil(p90_position)
    if lower == upper:
        p90 = ordered[lower]
    else:
        weight = p90_position - lower
        p90 = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return (
        f"median={statistics.median(samples):.4f} ms | "
        f"p90={p90:.4f} ms | min={min(samples):.4f} ms"
    )


def build_models(config: BenchmarkConfig) -> Dict[str, torch.nn.Module]:
    baseline = BaselineSelfAttention(
        config.d_model, config.num_heads
    ).to(device=config.device, dtype=config.dtype).eval()
    packed_sdpa = PackedQKVSDPAAttention(
        config.d_model, config.num_heads
    ).to(device=config.device, dtype=config.dtype).eval()
    triton_model = TritonSelfAttention(
        config.d_model, config.num_heads, backend="triton"
    ).to(device=config.device, dtype=config.dtype).eval()
    packed_sdpa.copy_from_baseline(baseline)
    triton_model.copy_from_baseline(baseline)
    return {
        "baseline": baseline,
        "packed-sdpa": packed_sdpa,
        "triton": triton_model,
    }


def maybe_compile(
    models: Dict[str, torch.nn.Module],
    enabled: bool,
    mode: str,
) -> Dict[str, torch.nn.Module]:
    if not enabled:
        return models
    if not hasattr(torch, "compile"):
        raise RuntimeError("this PyTorch build does not provide torch.compile")
    return {
        name: torch.compile(model, backend="inductor", mode=mode)
        for name, model in models.items()
    }


def run_profile(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> None:
    if not hasattr(torch, "profiler"):
        print("[profile] torch.profiler is unavailable")
        return
    activities = [torch.profiler.ProfilerActivity.CPU]
    if x.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
    ) as profiler:
        with torch.inference_mode():
            model(x, valid_token_mask, causal)
    sort_key = "self_cuda_time_total" if x.device.type == "cuda" else "self_cpu_time_total"
    print(
        profiler.key_averages().table(
            sort_by=sort_key,
            row_limit=20,
        )
    )


def run_case(args: argparse.Namespace, batch_size: int, seq_len: int) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but it is unavailable")
    config = BenchmarkConfig(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        dtype=resolve_dtype(args.dtype),
        device=device,
        causal=args.causal,
        padding_ratio=args.padding_ratio,
    )
    config.validate()

    if args.sweep and device.type == "cpu" and seq_len >= 2048:
        print(f"[warning] large CPU case may be very slow: B={batch_size}, S={seq_len}")

    torch.manual_seed(args.seed)
    x, valid_token_mask = make_inputs(config, args.seed)
    upstream = torch.randn_like(x)
    models = maybe_compile(
        build_models(config),
        enabled=args.compile,
        mode=args.compile_mode,
    )

    reference_model = models["baseline"]
    with torch.inference_mode():
        reference_output = reference_model(x, valid_token_mask, config.causal)

    print(
        f"\n=== B={batch_size}, S={seq_len}, D={config.d_model}, "
        f"H={config.num_heads}, Dh={config.head_dim}, dtype={args.dtype}, "
        f"causal={config.causal}, padding={config.padding_ratio:.2f}, "
        f"device={device} ==="
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    results: Dict[str, List[float]] = {}
    gradient_results: Dict[str, torch.Tensor] = {}
    for name, model in models.items():
        with torch.inference_mode():
            candidate_output = model(x, valid_token_mask, config.causal)
        passed, max_abs, max_rel, failed = compare_outputs(
            reference_output, candidate_output
        )
        backend = (
            model.selected_backend(x, valid_token_mask)
            if hasattr(model, "selected_backend")
            else "baseline"
        )
        print(
            f"{name:12s}: accuracy={'PASS' if passed else 'FAIL'} | "
            f"backend={backend} | max_abs={max_abs:.6g} | "
            f"max_rel={max_rel:.6g} | failed={failed}"
        )

        if args.mode in {"forward", "both"}:
            results[name] = timed_forward(
                model,
                x,
                valid_token_mask,
                config.causal,
                args.warmup,
                args.repeats,
            )
            print(f"  forward:  {summarize(results[name])}")

        if args.mode in {"backward", "both"}:
            backward_samples, input_gradient = timed_backward(
                model,
                x,
                valid_token_mask,
                config.causal,
                args.warmup,
                args.repeats,
                upstream,
            )
            gradient_results[name] = input_gradient
            results[f"{name}-backward"] = backward_samples
            print(f"  backward: {summarize(backward_samples)}")

    if args.mode in {"backward", "both"}:
        baseline_gradient = gradient_results["baseline"]
        for name, gradient in gradient_results.items():
            passed, max_abs, max_rel, failed = compare_outputs(
                baseline_gradient,
                gradient,
                rtol=0.05 if config.dtype != torch.float32 else 0.01,
                atol=0.005 if config.dtype != torch.float32 else 0.001,
            )
            print(
                f"{name:12s} gradient: {'PASS' if passed else 'FAIL'} | "
                f"max_abs={max_abs:.6g} | max_rel={max_rel:.6g} | failed={failed}"
            )

    if args.mode in {"forward", "both"} and "baseline" in results:
        baseline_median = statistics.median(results["baseline"])
        for name in ("packed-sdpa", "triton"):
            if name in results:
                speedup = baseline_median / statistics.median(results[name])
                print(f"  speedup {name:12s}: {speedup:.3f}x")

    if args.profile:
        print("\n=== Triton implementation profile ===")
        run_profile(models["triton"], x, valid_token_mask, config.causal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark baseline, packed SDPA, and Triton attention"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="float16"
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument(
        "--mode", choices=("forward", "backward", "both"), default="forward"
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="max-autotune",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="run B in {1, 8} and S in {32, 128, 512, 2048, 4096}",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if torch is None:
        raise SystemExit(
            "PyTorch is required. Install a CUDA-compatible PyTorch environment "
            "before running this benchmark."
        )
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    if args.sweep:
        for batch_size in (1, 8):
            for seq_len in (32, 128, 512, 2048, 4096):
                run_case(args, batch_size, seq_len)
    else:
        run_case(args, args.batch_size, args.seq_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
