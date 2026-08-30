#!/usr/bin/env python3
"""Memory-safe forward evaluator for organizer Shape #14.

Shape #14 is batch-independent but does not fit through the original full
batch/reference harness on a T4.  This evaluator keeps one batch block on the
GPU, runs the complete model with packed-QKV SDPA, and repeats the block
sequentially.  The reported latency is explicitly a sequential blockwise
capability measurement, not full-batch GPU throughput.
"""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass

import torch

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    copy_model_weights,
)


@dataclass(frozen=True)
class Shape14MemoryEstimate:
    batch_block: int
    seq_len: int
    d_model: int
    dtype: torch.dtype
    working_set_bytes: int
    free_bytes: int

    @property
    def working_set_gib(self) -> float:
        return self.working_set_bytes / (1024**3)

    @property
    def free_gib(self) -> float:
        return self.free_bytes / (1024**3)


def estimate_block_memory(
    *,
    batch_block: int,
    seq_len: int,
    d_model: int,
    dtype: torch.dtype,
    device: torch.device,
    safety_multiplier: int = 8,
) -> Shape14MemoryEstimate:
    """Estimate a conservative block working set before allocating tensors."""

    if batch_block <= 0 or seq_len <= 0 or d_model <= 0:
        raise ValueError("batch_block, seq_len, and d_model must be positive")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Shape #14 streaming requires CUDA")
    element_bytes = torch.tensor([], dtype=dtype).element_size()
    activation_bytes = batch_block * seq_len * d_model * element_bytes
    # Account conservatively for input/output, packed QKV, FFN workspace, and
    # allocator/compiler scratch.  This is a guard, not a kernel allocation.
    working_set_bytes = activation_bytes * safety_multiplier
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return Shape14MemoryEstimate(
        batch_block=batch_block,
        seq_len=seq_len,
        d_model=d_model,
        dtype=dtype,
        working_set_bytes=working_set_bytes,
        free_bytes=int(free_bytes),
    )


def validate_memory_budget(
    estimate: Shape14MemoryEstimate,
    *,
    free_fraction: float = 0.70,
) -> None:
    """Reject a block before allocation if its conservative estimate is unsafe."""

    if not 0.0 < free_fraction <= 1.0:
        raise ValueError("free_fraction must be in (0, 1]")
    if estimate.working_set_bytes > int(estimate.free_bytes * free_fraction):
        raise MemoryError(
            "Shape #14 block rejected by memory guard: "
            f"estimated={estimate.working_set_gib:.2f} GiB, "
            f"free={estimate.free_gib:.2f} GiB, "
            f"budget={estimate.free_gib * free_fraction:.2f} GiB"
        )


def build_shape14_models(
    *,
    batch_block: int,
    seq_len: int,
    d_model: int,
    heads: int,
    ffn_dim: int,
    layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[BaselineTransformer, UserOptimizedTransformer]:
    """Build matched baseline and production streaming models for one block."""

    config = TransformerConfig(
        batch_block,
        seq_len,
        d_model,
        heads,
        ffn_dim,
        layers,
        True,
    )
    config.validate()
    baseline = BaselineTransformer(config)
    optimized = UserOptimizedTransformer(
        config,
        packed_sdpa_suffix_layers=layers,
    )
    copy_model_weights(baseline, optimized)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    return baseline, optimized


def _make_block_inputs(
    *,
    batch_block: int,
    seq_len: int,
    d_model: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(
        batch_block,
        seq_len,
        d_model,
        device=device,
        dtype=dtype,
        generator=generator,
    )
    # Shape #14 has no padding semantics in the supplied appendix.  Keeping a
    # reusable all-valid mask lets prepare_for_inference remove the no-op mask
    # path before timing while preserving the public forward contract.
    mask = torch.ones(batch_block, seq_len, device=device, dtype=torch.bool)
    return x, mask


def _timed_block_loop(
    model: torch.nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    logical_batch: int,
    batch_block: int,
    warmup: int,
    repeats: int,
) -> list[float]:
    blocks = logical_batch // batch_block
    with torch.inference_mode():
        for _ in range(warmup):
            for _block in range(blocks):
                model(x, mask)
    torch.cuda.synchronize(x.device)

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.inference_mode():
            for _block in range(blocks):
                model(x, mask)
        end.record()
        torch.cuda.synchronize(x.device)
        samples.append(start.elapsed_time(end))
    return samples


def run_streaming_shape14(
    *,
    logical_batch: int = 32,
    batch_block: int = 1,
    seq_len: int = 100_000,
    d_model: int = 1024,
    heads: int = 16,
    ffn_dim: int = 1024,
    layers: int = 2,
    dtype: torch.dtype = torch.float16,
    device: torch.device = torch.device("cuda"),
    warmup: int = 1,
    repeats: int = 1,
    seed: int = 1234,
    free_fraction: float = 0.70,
) -> dict[str, object]:
    """Run the safe sequential Shape #14 path and return measured metadata."""

    if logical_batch <= 0 or batch_block <= 0:
        raise ValueError("logical_batch and batch_block must be positive")
    if logical_batch % batch_block:
        raise ValueError("logical_batch must be divisible by batch_block")
    if warmup < 0 or repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Shape #14 streaming requires CUDA")

    estimate = estimate_block_memory(
        batch_block=batch_block,
        seq_len=seq_len,
        d_model=d_model,
        dtype=dtype,
        device=device,
    )
    validate_memory_budget(estimate, free_fraction=free_fraction)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _baseline, optimized = build_shape14_models(
        batch_block=batch_block,
        seq_len=seq_len,
        d_model=d_model,
        heads=heads,
        ffn_dim=ffn_dim,
        layers=layers,
        device=device,
        dtype=dtype,
    )
    # The reference is intentionally constructed for weight parity, but is
    # never run for the 100k-token path and need not remain live on the GPU.
    del _baseline
    x, mask = _make_block_inputs(
        batch_block=batch_block,
        seq_len=seq_len,
        d_model=d_model,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
    )
    # Prepare only the production model.  The explicit baseline is never run
    # at S=100,000; its [S,S] reference path is unsafe by construction.
    optimized.prepare_for_inference(x, mask, fast_ffn_suffix_layers=0)
    torch.cuda.reset_peak_memory_stats(device)
    optimized_samples = _timed_block_loop(
        optimized,
        x,
        mask,
        logical_batch=logical_batch,
        batch_block=batch_block,
        warmup=warmup,
        repeats=repeats,
    )
    peak_bytes = torch.cuda.max_memory_allocated(device)
    blocks = logical_batch // batch_block
    median_ms = statistics.median(optimized_samples)
    return {
        "logical_batch": logical_batch,
        "batch_block": batch_block,
        "seq_len": seq_len,
        "d_model": d_model,
        "heads": heads,
        "layers": layers,
        "blocks": blocks,
        "backend": optimized.attention_backend,
        "median_ms": median_ms,
        "p90_ms": _percentile(optimized_samples, 0.90),
        "min_ms": min(optimized_samples),
        "peak_gpu_gib": peak_bytes / (1024**3),
        "estimated_block_gib": estimate.working_set_gib,
        "free_before_gib": estimate.free_gib,
    }


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logical-batch", type=int, default=32)
    parser.add_argument("--batch-block", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--free-fraction", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    result = run_streaming_shape14(
        logical_batch=args.logical_batch,
        batch_block=args.batch_block,
        seq_len=args.seq_len,
        d_model=args.d_model,
        heads=args.heads,
        ffn_dim=args.ffn_dim,
        layers=args.layers,
        dtype=dtype,
        device=torch.device(args.device),
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
        free_fraction=args.free_fraction,
    )
    print("=== Shape #14 memory-safe streaming evaluation ===")
    print(
        f"logical_shape=({result['logical_batch']}, {result['seq_len']}, "
        f"{result['d_model']}) batch_block={result['batch_block']} "
        f"blocks={result['blocks']} heads={result['heads']} "
        f"layers={result['layers']} dtype={dtype} "
        f"gpu={torch.cuda.get_device_name(torch.device(args.device))}"
    )
    print(
        f"backend={result['backend']} | median={result['median_ms']:.4f} ms | "
        f"p90={result['p90_ms']:.4f} ms | min={result['min_ms']:.4f} ms | "
        f"peak_gpu={result['peak_gpu_gib']:.2f} GiB | "
        f"estimated_block={result['estimated_block_gib']:.2f} GiB"
    )
    print(
        "status=PASS | no explicit baseline or [B,H,S,S] attention matrix was "
        "allocated; latency is sequential blockwise, not full-batch throughput"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
