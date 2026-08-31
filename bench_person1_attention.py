"""Standalone benchmark for Person 1's attention implementations.

Examples:

    python bench_person1_attention.py --device cuda --dtype float16
    python bench_person1_attention.py --device cuda --dtype float16 --causal
    python bench_person1_attention.py --device cuda --dtype float16 \
        --padding-ratio 0.25 --mode both
    python bench_person1_attention.py --device cuda --dtype float16 --sweep
    python bench_person1_attention.py --official-case 13 --device cuda
    python bench_person1_attention.py --official-case 14 --attention-core \
        --allow-long-sequence --device cuda

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
        cuda_bfloat16_supported,
        prepare_valid_token_mask,
        sdpa_scaled_dot_product_attention,
        sdpa_backend_diagnostics,
        triton_scaled_dot_product_attention,
        triton_scaled_dot_product_attention_with_status,
    )
    from torch_transformer_benchmark import BaselineSelfAttention


OFFICIAL_CASES = {
    1: (64, 128, 4, 128),
    2: (1, 128, 4, 128),
    3: (4, 128, 4, 128),
    4: (16, 128, 4, 128),
    5: (128, 128, 4, 128),
    6: (10000, 128, 4, 128),
    7: (64, 32, 4, 128),
    8: (64, 1024, 4, 128),
    9: (64, 128, 1, 128),
    10: (64, 128, 2, 128),
    11: (64, 128, 16, 128),
    12: (64, 128, 4, 32),
    13: (64, 128, 4, 1024),
    14: (32, 1024, 16, 100000),
}
_DENSE_BENCHMARK_MAX_SEQ_LEN = 8192


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


def timed_call(
    call: Callable[[], torch.Tensor],
    device: torch.device,
    warmup: int,
    repeats: int,
) -> Tuple[List[float], torch.Tensor]:
    """Time an already-constructed attention call without data generation."""

    with torch.inference_mode():
        output = None
        for _ in range(warmup):
            output = call()
    synchronize(device)

    samples: List[float] = []
    with torch.inference_mode():
        if device.type == "cuda":
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
            for index in range(repeats):
                starts[index].record()
                output = call()
                ends[index].record()
            synchronize(device)
            samples.extend(
                start.elapsed_time(end)
                for start, end in zip(starts, ends)
            )
        else:
            for _ in range(repeats):
                start = time.perf_counter_ns()
                output = call()
                end = time.perf_counter_ns()
                samples.append((end - start) / 1e6)
    if output is None:
        raise RuntimeError("benchmark call produced no output")
    return samples, output.detach()


def timed_forward(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
    warmup: int,
    repeats: int,
) -> List[float]:
    samples, _ = timed_call(
        lambda: model(x, valid_token_mask, causal),
        x.device,
        warmup,
        repeats,
    )
    return samples


def backend_report(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
) -> str:
    """Report the backend used by the preceding eager call when available."""

    if hasattr(model, "backend_status"):
        actual, reason = model.backend_status()
        if actual != "uninitialized":
            if reason:
                return f"{actual} | fallback_reason={reason}"
            return actual
        # The compiled wrapper deliberately does not mutate Python module
        # state during graph capture/execution.  Do not report an eligibility
        # prediction as proof that a Triton kernel ran.
        planned = model.selected_backend(x, valid_token_mask)
        return (
            "unverified | "
            f"planned={planned} | compiled_backend_telemetry_unavailable"
        )
    return "baseline"


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
        name: torch.compile(
            model,
            backend="inductor",
            mode=mode,
            fullgraph=True,
            dynamic=False,
        )
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


def run_sdpa_diagnostic(
    model: torch.nn.Module,
    x: torch.Tensor,
    valid_token_mask: Optional[torch.Tensor],
    causal: bool,
) -> None:
    """Print SDPA eligibility and layout metadata outside benchmark timing."""

    if not hasattr(model, "qkv_proj"):
        return
    try:
        batch, seq_len, _ = x.shape
        num_heads = int(model.num_heads)
        head_dim = int(model.head_dim)
        with torch.inference_mode():
            qkv = model.qkv_proj(x).view(
                batch, seq_len, 3, num_heads, head_dim
            )
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        diagnostics = sdpa_backend_diagnostics(
            q, k, v, valid_token_mask=valid_token_mask, causal=causal
        )
    except (AttributeError, RuntimeError, ValueError) as error:
        print(f"[sdpa-diagnostic] unavailable: {type(error).__name__}: {error}")
        return
    backend_summary = ", ".join(
        f"{name}={status}"
        for name, status in diagnostics["backends"].items()
    )
    print(
        "[sdpa-diagnostic] "
        f"shape={diagnostics['shape']} dtype={diagnostics['dtype']} "
        f"mask={diagnostics['mask']} causal={diagnostics['causal']} "
        f"q_stride={diagnostics['q_stride']} "
        f"k_stride={diagnostics['k_stride']} "
        f"v_stride={diagnostics['v_stride']} "
        f"selected_kernel={diagnostics['selected_kernel']} "
        f"backends={backend_summary}"
    )


def make_core_inputs(
    config: BenchmarkConfig,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Create preprojected Q/K/V for a memory-safe long-sequence smoke run."""

    generator = torch.Generator(device=config.device)
    generator.manual_seed(seed)
    shape = (
        config.batch_size,
        config.num_heads,
        config.seq_len,
        config.head_dim,
    )
    q = torch.randn(*shape, device=config.device, dtype=config.dtype, generator=generator)
    k = torch.randn(*shape, device=config.device, dtype=config.dtype, generator=generator)
    v = torch.randn(*shape, device=config.device, dtype=config.dtype, generator=generator)
    if config.padding_ratio <= 0.0:
        return q, k, v, None

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
    return q, k, v, valid_token_mask


def run_attention_core_case(args: argparse.Namespace, config: BenchmarkConfig) -> None:
    """Benchmark only the memory-safe attention core for long sequences.

    The full official Shape #14 also requires streaming QKV projection, which
    is outside this standalone Person 1 attention-core benchmark.  Refuse a
    case whose preprojected Q/K/V tensors cannot fit instead of risking an
    accidental OOM.
    """

    if args.mode != "forward":
        raise ValueError("long attention-core benchmarking supports forward mode only")

    element_size = torch.tensor([], dtype=config.dtype).element_size()
    required_bytes = (
        4
        * config.batch_size
        * config.num_heads
        * config.seq_len
        * config.head_dim
        * element_size
    )
    if config.device.type == "cuda":
        free_bytes, _ = torch.cuda.mem_get_info(config.device)
        if required_bytes > int(free_bytes * 0.70):
            print(
                f"[skip] core tensors need approximately {required_bytes / 2**30:.2f} GiB "
                f"but only {free_bytes / 2**30:.2f} GiB is free; use a reduced "
                "batch/head configuration for the long-sequence smoke run"
            )
            return
    elif required_bytes > 2**30:
        print(
            f"[skip] core tensors need approximately {required_bytes / 2**30:.2f} GiB "
            "on CPU; use a reduced batch/head configuration for the long-sequence "
            "smoke run"
        )
        return

    q, k, v, valid_token_mask = make_core_inputs(config, args.seed)
    valid_token_mask = prepare_valid_token_mask(valid_token_mask)

    def sdpa_call() -> torch.Tensor:
        return sdpa_scaled_dot_product_attention(
            q, k, v, valid_token_mask=valid_token_mask, causal=config.causal
        )

    def triton_call() -> torch.Tensor:
        return triton_scaled_dot_product_attention(
            q, k, v, valid_token_mask=valid_token_mask, causal=config.causal
        )

    print(
        f"\n=== attention core: B={config.batch_size}, S={config.seq_len}, "
        f"D={config.d_model}, H={config.num_heads}, Dh={config.head_dim}, "
        f"dtype={args.dtype}, causal={config.causal}, device={config.device} ==="
    )
    if config.device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(config.device)}")

    _, triton_backend, triton_reason = triton_scaled_dot_product_attention_with_status(
        q,
        k,
        v,
        valid_token_mask=valid_token_mask,
        causal=config.causal,
    )
    sdpa_samples, sdpa_output = timed_call(
        sdpa_call, config.device, args.warmup, args.repeats
    )
    triton_samples, triton_output = timed_call(
        triton_call, config.device, args.warmup, args.repeats
    )
    finite = bool(torch.isfinite(triton_output).all())
    passed, max_abs, max_rel, failed = compare_outputs(
        sdpa_output, triton_output, rtol=0.01, atol=0.001
    )
    print(f"sdpa-core : {summarize(sdpa_samples)} | backend=sdpa")
    print(
        f"triton-core: {summarize(triton_samples)} | backend={triton_backend} | "
        f"finite={'PASS' if finite else 'FAIL'} | vs_sdpa={'PASS' if passed else 'FAIL'} | "
        f"max_abs={max_abs:.6g} | max_rel={max_rel:.6g} | failed={failed}"
    )
    if triton_reason:
        print(f"  triton fallback reason: {triton_reason}")
    print(
        f"  speedup triton-core vs sdpa-core: "
        f"{statistics.median(sdpa_samples) / statistics.median(triton_samples):.3f}x"
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

    if (
        config.dtype == torch.bfloat16
        and config.device.type == "cuda"
        and not cuda_bfloat16_supported(config.device)
    ):
        print(
            f"[unsupported] native BF16 is not reported for {config.device}; "
            "skipping instead of benchmarking emulated BF16"
        )
        return

    if seq_len > _DENSE_BENCHMARK_MAX_SEQ_LEN:
        if not args.allow_long_sequence:
            raise ValueError(
                f"S={seq_len} exceeds the dense benchmark safety limit "
                f"({_DENSE_BENCHMARK_MAX_SEQ_LEN}); pass --allow-long-sequence "
                "and --attention-core to run only memory-safe core timings"
            )
        if not args.attention_core:
            raise ValueError(
                "long-sequence cases must use --attention-core so the dense "
                "baseline and full-module QKV allocation are not attempted"
            )
        run_attention_core_case(args, config)
        return

    if args.sweep and device.type == "cpu" and seq_len >= 2048:
        print(f"[warning] large CPU case may be very slow: B={batch_size}, S={seq_len}")

    torch.manual_seed(args.seed)
    x, valid_token_mask = make_inputs(config, args.seed)
    valid_token_mask = prepare_valid_token_mask(valid_token_mask)
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
        backend = backend_report(model, x, valid_token_mask)
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
        if "packed-sdpa" in results and "triton" in results:
            packed_median = statistics.median(results["packed-sdpa"])
            triton_median = statistics.median(results["triton"])
            print(
                f"  speedup triton vs packed-sdpa: "
                f"{packed_median / triton_median:.3f}x"
            )

    if args.profile:
        run_sdpa_diagnostic(models["packed-sdpa"], x, valid_token_mask, config.causal)
        print("\n=== Packed SDPA implementation profile ===")
        run_profile(models["packed-sdpa"], x, valid_token_mask, config.causal)
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
        "--official-case",
        type=int,
        choices=tuple(OFFICIAL_CASES),
        help="use one organizer appendix case; CLI dtype/device/padding override it",
    )
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
        "--attention-core",
        action="store_true",
        help="for long sequences, benchmark preprojected Q/K/V only",
    )
    parser.add_argument(
        "--allow-long-sequence",
        action="store_true",
        help="explicitly permit guarded long-sequence benchmarking",
    )
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

    if args.official_case is not None:
        batch_size, d_model, heads, seq_len = OFFICIAL_CASES[args.official_case]
        args.d_model = d_model
        args.heads = heads
        args.causal = True
        run_case(args, batch_size, seq_len)
    elif args.sweep:
        for batch_size in (1, 8):
            for seq_len in (32, 128, 512, 2048, 4096):
                run_case(args, batch_size, seq_len)
    else:
        run_case(args, args.batch_size, args.seq_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
