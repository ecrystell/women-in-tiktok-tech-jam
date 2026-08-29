#!/usr/bin/env python3
"""Benchmark the optional fixed-shape TensorRT Person 2 FFN experiment.

This is intentionally a standalone, eager-only measurement tool.  It compares
the TensorRT engine's *FFN update* with the same native PyTorch operation on a
contiguous FP16 ``[batch_size * seq_len, d_model]`` residual input whose
LayerNorm affine parameters are identity.  Engine construction, TensorRT tactic
selection, stream-context setup, and strict preflight happen before timing.

By default the benchmark stops at the update produced by
``LayerNorm -> Linear -> exact GELU -> Linear``.  ``--append-residual-mask``
explicitly appends the existing Person 2 residual/mask CUDA postprocessing to
both sides, so that postprocessing is never accidentally attributed to the
TensorRT FFN engine.

Examples (run in a CUDA/TensorRT-enabled Colab environment)::

    python bench_person2_tensorrt.py --batch-size 8 --seq-len 512
    python bench_person2_tensorrt.py --append-residual-mask --padding-ratio .2 \
        --json benchmark-results/person2-tensorrt.json

An unavailable or unsupported TensorRT installation is reported as ``SKIP``
and exits successfully, allowing an experiment notebook to continue with the
native Person 2 lane.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FFNParameters:
    """Immutable tensors for one fixed-shape identity-LayerNorm FFN."""

    norm_weight: torch.Tensor
    norm_bias: torch.Tensor
    up_weight: torch.Tensor
    up_bias: torch.Tensor
    down_weight: torch.Tensor
    down_bias: torch.Tensor
    eps: float


@dataclass(frozen=True)
class Accuracy:
    passed: bool
    max_abs_error: float
    max_rel_error: float
    failed_elements: int
    total_elements: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument(
        "--append-residual-mask",
        action="store_true",
        help="append the existing Person 2 residual/mask post to both runners",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--workspace-mib", type=int, default=256)
    parser.add_argument(
        "--allow-non-t4",
        action="store_true",
        help="allow setup on a non-SM75 CUDA GPU (T4 is the acceptance target)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        metavar="PATH",
        help="optionally write the complete result record as JSON",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ("batch_size", "seq_len", "d_model", "ffn_dim", "warmup", "repeats", "rounds"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError("--padding-ratio must be in [0, 1)")
    if args.eps <= 0.0 or not math.isfinite(args.eps):
        raise ValueError("--eps must be finite and positive")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError("--atol and --rtol must be nonnegative")
    if args.workspace_mib <= 0:
        raise ValueError("--workspace-mib must be positive")


def environment_record() -> dict[str, Any]:
    """Collect environment facts without importing optional TensorRT bindings."""

    record: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "compile_available": hasattr(torch, "compile"),
        "compile_mode": "eager_only_tensorrt_experiment",
    }
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        record.update(
            {
                "device": f"cuda:{device}",
                "gpu": torch.cuda.get_device_name(device),
                "capability": list(torch.cuda.get_device_capability(device)),
                "total_memory_bytes": int(properties.total_memory),
                "driver": getattr(torch.cuda, "driver_version", None),
            }
        )
    return record


def make_parameters(
    d_model: int,
    ffn_dim: int,
    device: torch.device,
    seed: int,
    eps: float,
) -> FFNParameters:
    """Create deterministic contiguous FP16 parameters with identity norm."""

    generator = torch.Generator(device=device).manual_seed(seed)

    def random_tensor(shape: tuple[int, ...], scale: float) -> torch.Tensor:
        return (
            torch.randn(shape, device=device, dtype=torch.float16, generator=generator)
            .mul_(scale)
            .contiguous()
        )

    return FFNParameters(
        norm_weight=torch.ones(d_model, device=device, dtype=torch.float16),
        norm_bias=torch.zeros(d_model, device=device, dtype=torch.float16),
        up_weight=random_tensor((ffn_dim, d_model), 1.0 / math.sqrt(d_model)),
        up_bias=random_tensor((ffn_dim,), 0.02),
        down_weight=random_tensor((d_model, ffn_dim), 1.0 / math.sqrt(ffn_dim)),
        down_bias=random_tensor((d_model,), 0.02),
        eps=float(eps),
    )


def make_residual(
    tokens: int, d_model: int, device: torch.device, seed: int
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(
        (tokens, d_model),
        device=device,
        dtype=torch.float16,
        generator=generator,
    ).contiguous()


def make_mask(tokens: int, padding_ratio: float, device: torch.device) -> Optional[torch.Tensor]:
    """Make a right-padded flattened token mask, only when padding is requested."""

    if padding_ratio == 0.0:
        return None
    valid = max(1, int(round(tokens * (1.0 - padding_ratio))))
    return (torch.arange(tokens, device=device) < valid).contiguous()


def native_update_into(
    residual: torch.Tensor, output: torch.Tensor, parameters: FFNParameters
) -> torch.Tensor:
    """Native exact-GELU FFN, writing the final update into a setup buffer."""

    normalized = F.layer_norm(
        residual,
        (residual.shape[-1],),
        parameters.norm_weight,
        parameters.norm_bias,
        parameters.eps,
    )
    hidden = F.gelu(
        F.linear(normalized, parameters.up_weight, parameters.up_bias),
        approximate="none",
    )
    # ``out=`` gives native PyTorch the same preallocated final-update contract
    # as TensorRT's ``run_into``.  LayerNorm, up GEMM, and exact GELU remain
    # ordinary ATen operations rather than an emulated TensorRT implementation.
    torch.addmm(
        parameters.down_bias,
        hidden,
        parameters.down_weight.t(),
        out=output,
    )
    return output


def strict_compare(
    reference: torch.Tensor, candidate: torch.Tensor, atol: float, rtol: float
) -> Accuracy:
    ref = reference.detach().float()
    got = candidate.detach().float()
    error = (got - ref).abs()
    finite = torch.isfinite(ref) & torch.isfinite(got)
    passed = finite & ((error <= atol) | (error <= rtol * ref.abs()))
    relative = torch.where(
        ref.abs() > 0,
        error / ref.abs(),
        torch.zeros_like(error),
    )
    return Accuracy(
        passed=bool(passed.all().item()),
        max_abs_error=float(error.max().item()),
        max_rel_error=float(relative.max().item()),
        failed_elements=int((~passed).sum().item()),
        total_elements=int(passed.numel()),
    )


def percentile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def time_runner(runner: Callable[[], torch.Tensor], repeats: int) -> list[float]:
    """Measure one eager runner using CUDA events, excluding setup and preflight."""

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    torch.cuda.synchronize()
    for index in range(repeats):
        starts[index].record()
        runner()
        ends[index].record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def timing_record(samples: list[float], tokens: int) -> dict[str, Any]:
    median = statistics.median(samples)
    return {
        "samples_ms": samples,
        "sample_count": len(samples),
        "median_ms": median,
        "p90_ms": percentile(samples, 0.90),
        "throughput_tokens_per_second": tokens * 1000.0 / median,
    }


def make_postprocessor(
    append_post: bool,
    residual: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any]]:
    """Return an optional, deliberately external residual/mask postprocessor."""

    if not append_post:
        return (lambda update: update), {"enabled": False, "kind": "none"}

    # This import and extension build are setup work.  They intentionally occur
    # only for the explicit post-inclusive mode and never inside timing loops.
    import person2_ffn_post

    person2_ffn_post.load_extension()
    if mask is None:
        return (
            lambda update: person2_ffn_post.residual_unmasked(update, residual),
            {"enabled": True, "kind": "person2_post.residual_unmasked"},
        )
    return (
        lambda update: person2_ffn_post.residual_masked(update, residual, mask),
        {"enabled": True, "kind": "person2_post.residual_masked"},
    )


def engine_record(engine: Any, preflight: Any) -> dict[str, Any]:
    """Serialize only public engine/key data suitable for a result artifact."""

    return {
        "prepared": True,
        "is_preflighted": bool(engine.is_preflighted),
        "key": asdict(engine.key),
        "preflight": asdict(preflight),
    }


def print_timing(label: str, values: dict[str, Any]) -> None:
    print(
        f"{label}: median={values['median_ms']:.4f} ms | "
        f"p90={values['p90_ms']:.4f} ms | "
        f"throughput={values['throughput_tokens_per_second']:.2f} token/s"
    )


def write_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote JSON result: {path}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    environment = environment_record()
    tokens = args.batch_size * args.seq_len
    result: dict[str, Any] = {
        "status": "skipped",
        "environment": environment,
        "configuration": {
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "tokens": tokens,
            "d_model": args.d_model,
            "ffn_dim": args.ffn_dim,
            "dtype": "float16",
            "identity_norm": True,
            "eps": args.eps,
            "append_residual_mask": args.append_residual_mask,
            "padding_ratio": args.padding_ratio,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "rounds": args.rounds,
            "atol": args.atol,
            "rtol": args.rtol,
            "workspace_bytes": args.workspace_mib * 1024 * 1024,
            "t4_only": not args.allow_non_t4,
        },
    }
    print("environment=" + json.dumps(environment, sort_keys=True))
    if not torch.cuda.is_available():
        result["reason"] = "CUDA is unavailable; TensorRT fixed-shape FFN requires CUDA FP16"
        print("SKIP: " + result["reason"])
        return result

    # Importing this module is safe: it has no import-time TensorRT/CUDA-Python
    # side effects.  ``probe_tensorrt`` performs the first optional import.
    from person2_tensorrt import (
        TensorRTFFNError,
        prepare_fixed_shape_ffn,
        probe_tensorrt,
    )

    availability = probe_tensorrt()
    result["tensorrt_availability"] = asdict(availability)
    print("tensorrt=" + json.dumps(result["tensorrt_availability"], sort_keys=True))
    if not availability.available:
        result["reason"] = availability.reason or "TensorRT is unavailable"
        print("SKIP: " + result["reason"])
        return result

    device = torch.device("cuda", torch.cuda.current_device())
    parameters = make_parameters(
        args.d_model, args.ffn_dim, device, args.seed + 1, args.eps
    )
    residual = make_residual(tokens, args.d_model, device, args.seed)
    mask = make_mask(tokens, args.padding_ratio, device)
    native_output = torch.empty_like(residual)
    tensorrt_output = torch.empty_like(residual)

    try:
        with torch.inference_mode():
            engine = prepare_fixed_shape_ffn(
                residual,
                parameters.norm_weight,
                parameters.norm_bias,
                parameters.up_weight,
                parameters.up_bias,
                parameters.down_weight,
                parameters.down_bias,
                parameters.eps,
                workspace_bytes=args.workspace_mib * 1024 * 1024,
                t4_only=not args.allow_non_t4,
            )
            # Record a tolerance-explicit preflight.  The cached engine already
            # passed its default preflight; this second setup-only call makes the
            # selected command's exact tolerance visible in the result record.
            preflight = engine.strict_preflight(
                residual,
                parameters.norm_weight,
                parameters.norm_bias,
                parameters.up_weight,
                parameters.up_bias,
                parameters.down_weight,
                parameters.down_bias,
                atol=args.atol,
                rtol=args.rtol,
            )
            engine.prepare_stream()
            postprocess, post_record = make_postprocessor(
                args.append_residual_mask, residual, mask
            )
    except TensorRTFFNError as error:
        result["reason"] = f"{type(error).__name__}: {error}"
        print("SKIP: " + result["reason"])
        return result
    except (ImportError, OSError, RuntimeError) as error:
        # A post-extension build error is still a useful outcome for a standalone
        # experiment and must not make a TensorRT-unavailable lane look successful.
        result["reason"] = f"{type(error).__name__}: {error}"
        print("SKIP: " + result["reason"])
        return result

    result["engine"] = engine_record(engine, preflight)
    result["postprocessing"] = post_record
    print("engine=" + json.dumps(result["engine"], sort_keys=True))
    print("postprocessing=" + json.dumps(post_record, sort_keys=True))

    def native_runner() -> torch.Tensor:
        return postprocess(native_update_into(residual, native_output, parameters))

    def tensorrt_runner() -> torch.Tensor:
        return postprocess(engine.run_into(residual, tensorrt_output))

    with torch.inference_mode():
        reference_update = native_update_into(residual, native_output, parameters).clone()
        candidate_update = engine.run_into(residual, tensorrt_output).clone()
        torch.cuda.synchronize()
    update_accuracy = strict_compare(reference_update, candidate_update, args.atol, args.rtol)
    result["update_accuracy"] = asdict(update_accuracy)
    print("update_accuracy=" + json.dumps(result["update_accuracy"], sort_keys=True))
    if not update_accuracy.passed:
        result["status"] = "failed_accuracy"
        print("FAIL: update strict accuracy failed; timing skipped")
        return result

    if args.append_residual_mask:
        with torch.inference_mode():
            reference_output = native_runner().clone()
            candidate_output = tensorrt_runner().clone()
            torch.cuda.synchronize()
        output_accuracy = strict_compare(reference_output, candidate_output, args.atol, args.rtol)
        result["post_accuracy"] = asdict(output_accuracy)
        print("post_accuracy=" + json.dumps(result["post_accuracy"], sort_keys=True))
        if not output_accuracy.passed:
            result["status"] = "failed_accuracy"
            print("FAIL: post-inclusive strict accuracy failed; timing skipped")
            return result

    with torch.inference_mode():
        for _ in range(args.warmup):
            native_runner()
            tensorrt_runner()
        torch.cuda.synchronize()

        native_samples: list[float] = []
        tensorrt_samples: list[float] = []
        for round_index in range(args.rounds):
            # Alternate order every round to reduce thermal/cache/order bias.
            if round_index % 2 == 0:
                native_samples.extend(time_runner(native_runner, args.repeats))
                tensorrt_samples.extend(time_runner(tensorrt_runner, args.repeats))
            else:
                tensorrt_samples.extend(time_runner(tensorrt_runner, args.repeats))
                native_samples.extend(time_runner(native_runner, args.repeats))

    native_timing = timing_record(native_samples, tokens)
    tensorrt_timing = timing_record(tensorrt_samples, tokens)
    result["timing"] = {
        "native_pytorch": native_timing,
        "tensorrt": tensorrt_timing,
        "speedup": native_timing["median_ms"] / tensorrt_timing["median_ms"],
    }
    result["status"] = "passed"
    print_timing("native PyTorch", native_timing)
    print_timing("TensorRT", tensorrt_timing)
    print(f"speedup={result['timing']['speedup']:.3f}x")
    return result


def main() -> int:
    args = parse_args()
    validate_args(args)
    result = run(args)
    if args.json is not None:
        write_json(args.json, result)
    # TensorRT is deliberately optional, so unavailable/unsupported/error lanes
    # return success for notebook sweeps.  Accuracy status remains in JSON/stdout.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
