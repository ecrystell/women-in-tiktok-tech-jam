"""Measured, setup-time attention dispatch for the integrated benchmark.

The dispatcher deliberately contains no timing, synchronization, or tensor
inspection in the model's timed ``forward`` path.  A plan is selected before
the model is constructed from an exact workload key and a table of externally
validated measurements.  Unknown keys use the native reference attention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple

import torch


PaddingKind = Literal["none", "padded"]
AttentionBackend = Literal["native", "packed-sdpa"]


@dataclass(frozen=True)
class AttentionDispatchKey:
    """All runtime properties that can change a safe attention decision."""

    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    dtype: str
    causal: bool
    padding: PaddingKind
    device_type: str
    device_index: int
    compute_capability: Optional[Tuple[int, int]]
    torch_version: str


@dataclass(frozen=True)
class DispatchMeasurement:
    """One independently measured candidate for one exact dispatch key."""

    key: AttentionDispatchKey
    suffix_layers: int
    correctness_passed: bool
    process_speedups: Tuple[float, ...]
    baseline_p90_ms: Tuple[float, ...]
    optimized_p90_ms: Tuple[float, ...]

    @property
    def median_speedup(self) -> float:
        ordered = sorted(self.process_speedups)
        if not ordered:
            return 0.0
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0

    @property
    def passes_gate(self) -> bool:
        if not self.correctness_passed or not self.process_speedups:
            return False
        return (
            all(speedup > 1.0 for speedup in self.process_speedups)
            and self.median_speedup >= 1.02
            and len(self.baseline_p90_ms) == len(self.optimized_p90_ms)
            and all(
                optimized <= 1.02 * baseline
                for baseline, optimized in zip(
                    self.baseline_p90_ms, self.optimized_p90_ms
                )
            )
        )


@dataclass(frozen=True)
class AttentionDispatchPlan:
    """Immutable plan consumed while constructing ``UserOptimizedTransformer``."""

    key: AttentionDispatchKey
    backend: AttentionBackend
    packed_sdpa_suffix_layers: int
    reason: str

    @property
    def label(self) -> str:
        if self.backend == "native":
            return "native"
        return f"packed-sdpa-suffix:{self.packed_sdpa_suffix_layers}"


def make_dispatch_key(
    *,
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_heads: int,
    dtype: torch.dtype,
    causal: bool,
    valid_token_mask: Optional[torch.Tensor],
    device: torch.device,
) -> AttentionDispatchKey:
    """Create a key during setup, before compilation and timing.

    Checking whether a supplied mask is all-valid is intentionally done here,
    never from the timed model forward.  The returned key records the semantic
    result rather than retaining the mask tensor.
    """

    if valid_token_mask is None:
        padding: PaddingKind = "none"
    else:
        if valid_token_mask.ndim != 2 or tuple(valid_token_mask.shape) != (
            batch_size,
            seq_len,
        ):
            raise ValueError("valid_token_mask shape does not match dispatch key")
        padding = "none" if bool(valid_token_mask.all().item()) else "padded"

    capability: Optional[Tuple[int, int]] = None
    if device.type == "cuda":
        capability = tuple(torch.cuda.get_device_capability(device))

    index = -1 if device.index is None else device.index
    torch_version = torch.__version__.split("+")[0]
    return AttentionDispatchKey(
        batch_size=batch_size,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=num_heads,
        dtype=str(dtype),
        causal=bool(causal),
        padding=padding,
        device_type=device.type,
        device_index=index,
        compute_capability=capability,
        torch_version=torch_version,
    )


def select_attention_plan(
    key: AttentionDispatchKey,
    *,
    num_layers: int,
    measurements: Iterable[DispatchMeasurement] = (),
    manual_suffix_layers: Optional[int] = None,
) -> AttentionDispatchPlan:
    """Select a plan without measuring or inspecting runtime tensors.

    ``manual_suffix_layers`` is for controlled experiments only.  Automatic
    production selection accepts only measurements that clear correctness,
    median-speed, and p90 gates for the exact key.
    """

    if not 0 <= num_layers:
        raise ValueError("num_layers must be nonnegative")
    if manual_suffix_layers is not None:
        if not 0 <= manual_suffix_layers <= num_layers:
            raise ValueError("manual_suffix_layers must be between 0 and num_layers")
        if manual_suffix_layers == 0:
            return AttentionDispatchPlan(
                key, "native", 0, "manual native override"
            )
        return AttentionDispatchPlan(
            key,
            "packed-sdpa",
            manual_suffix_layers,
            "manual experiment override; not an automatic dispatch claim",
        )

    candidates = [
        measurement
        for measurement in measurements
        if measurement.key == key
        and 0 < measurement.suffix_layers <= num_layers
        and measurement.passes_gate
    ]
    if not candidates:
        return AttentionDispatchPlan(
            key,
            "native",
            0,
            "no exact measured candidate cleared correctness, median, and p90 gates",
        )

    selected = max(
        candidates,
        key=lambda measurement: (
            measurement.median_speedup,
            measurement.suffix_layers,
        ),
    )
    return AttentionDispatchPlan(
        key,
        "packed-sdpa",
        selected.suffix_layers,
        f"exact measured candidate; median speedup={selected.median_speedup:.3f}x",
    )


def historical_t4_measurements() -> Tuple[DispatchMeasurement, ...]:
    """Return exact integrated T4 measurements that cleared every gate.

    These entries are deliberately narrow: the dispatch key includes the
    complete workload, device capability, and PyTorch version, so a result is
    never generalized to a nearby official shape without independent data.
    """

    key = AttentionDispatchKey(
        batch_size=8,
        seq_len=512,
        d_model=512,
        num_heads=8,
        dtype="torch.float16",
        causal=False,
        padding="none",
        device_type="cuda",
        device_index=0,
        compute_capability=(7, 5),
        torch_version="2.11.0",
    )
    shape3_key = AttentionDispatchKey(
        batch_size=4,
        seq_len=128,
        d_model=128,
        num_heads=4,
        dtype="torch.float16",
        causal=True,
        padding="none",
        device_type="cuda",
        device_index=0,
        compute_capability=(7, 5),
        torch_version="2.11.0",
    )
    shape4_key = AttentionDispatchKey(
        batch_size=16,
        seq_len=128,
        d_model=128,
        num_heads=4,
        dtype="torch.float16",
        causal=True,
        padding="none",
        device_type="cuda",
        device_index=0,
        compute_capability=(7, 5),
        torch_version="2.11.0",
    )
    return (
        DispatchMeasurement(
            key=key,
            suffix_layers=1,
            correctness_passed=True,
            process_speedups=(1.333, 1.325, 1.327),
            baseline_p90_ms=(27.8528, 28.3276, 29.0056),
            optimized_p90_ms=(21.0168, 21.4778, 21.9393),
        ),
        DispatchMeasurement(
            key=shape3_key,
            suffix_layers=1,
            correctness_passed=True,
            process_speedups=(1.497, 1.435, 1.481),
            baseline_p90_ms=(3.4001, 4.5934, 3.4165),
            optimized_p90_ms=(2.1672, 3.6061, 2.3229),
        ),
        DispatchMeasurement(
            key=shape4_key,
            suffix_layers=1,
            correctness_passed=True,
            process_speedups=(1.448, 1.460, 1.447),
            baseline_p90_ms=(5.9553, 3.3555, 4.4701),
            optimized_p90_ms=(4.1014, 2.4126, 3.7307),
        ),
    )


__all__ = [
    "AttentionDispatchKey",
    "AttentionDispatchPlan",
    "DispatchMeasurement",
    "historical_t4_measurements",
    "make_dispatch_key",
    "select_attention_plan",
]
