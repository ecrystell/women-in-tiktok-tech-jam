#!/usr/bin/env python3
"""Run the integrated Transformer validation matrix in isolated processes."""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Case:
    batch: int
    seq_len: int
    d_model: int
    heads: int
    ffn_dim: int
    layers: int
    causal: bool
    padding: float

    @property
    def label(self) -> str:
        mode = "causal" if self.causal else "noncausal"
        return f"B{self.batch} S{self.seq_len} D{self.d_model} {mode} P{self.padding:g}"


@dataclass(frozen=True)
class RunResult:
    correct: bool
    baseline_median_ms: float
    baseline_p90_ms: float
    optimized_median_ms: float
    optimized_p90_ms: float
    speedup: float


CASES = (
    Case(1, 128, 512, 8, 2048, 6, False, 0.0),
    Case(8, 512, 512, 8, 2048, 6, False, 0.0),
    Case(8, 512, 512, 8, 2048, 6, True, 0.2),
    Case(2, 2048, 512, 8, 2048, 6, False, 0.0),
    Case(1, 4096, 512, 8, 2048, 6, True, 0.25),
)


def parse_timing(output: str, label: str) -> tuple[float, float]:
    match = re.search(
        rf"^{label}\s*: median=([0-9.]+) ms .*? p90=([0-9.]+) ms",
        output,
        flags=re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"missing {label} timing")
    return float(match.group(1)), float(match.group(2))


def parse_result(output: str) -> RunResult:
    baseline_median, baseline_p90 = parse_timing(output, "baseline")
    optimized_median, optimized_p90 = parse_timing(output, "optimized")
    speedup_match = re.search(
        r"^speedup\s*: ([0-9.]+)x based on median latency$",
        output,
        flags=re.MULTILINE,
    )
    if speedup_match is None:
        raise ValueError("missing speedup result")
    return RunResult(
        correct="summary: PASS" in output,
        baseline_median_ms=baseline_median,
        baseline_p90_ms=baseline_p90,
        optimized_median_ms=optimized_median,
        optimized_p90_ms=optimized_p90,
        speedup=float(speedup_match.group(1)),
    )


def build_command(
    case: Case,
    device: str,
    dtype: str,
    warmup: int,
    repeats: int,
    rounds: int,
) -> list[str]:
    command = [
        sys.executable,
        "torch_transformer_benchmark.py",
        "--batch-size",
        str(case.batch),
        "--seq-len",
        str(case.seq_len),
        "--d-model",
        str(case.d_model),
        "--heads",
        str(case.heads),
        "--ffn-dim",
        str(case.ffn_dim),
        "--layers",
        str(case.layers),
        "--device",
        device,
        "--dtype",
        dtype,
        "--padding-ratio",
        str(case.padding),
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--benchmark-rounds",
        str(rounds),
    ]
    if case.causal:
        command.append("--causal")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--processes", type=int)
    parser.add_argument(
        "--skip-long",
        action="store_true",
        help="skip the B1 S4096 memory-pressure case",
    )
    args = parser.parse_args()
    if args.processes is not None and args.processes <= 0:
        parser.error("--processes must be positive")
    return args


def main() -> int:
    args = parse_args()
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    dtype = args.dtype
    if dtype == "auto":
        dtype = "float16" if device.startswith("cuda") else "float32"

    final_mode = args.mode == "final"
    processes = args.processes or (3 if final_mode else 1)
    warmup, repeats, rounds = (20, 100, 3) if final_mode else (5, 20, 1)
    cases = CASES[:-1] if args.skip_long else CASES

    print(
        f"mode={args.mode} device={device} dtype={dtype} processes={processes} "
        f"warmup={warmup} repeats={repeats} rounds={rounds}"
    )
    correctness_failed = False
    execution_failed = False

    for case in cases:
        results: list[RunResult] = []
        print(f"\n[{case.label}]")
        command = build_command(case, device, dtype, warmup, repeats, rounds)
        for process_index in range(processes):
            completed = subprocess.run(command, capture_output=True, text=True)
            if completed.returncode != 0:
                execution_failed = True
                print(f"process {process_index + 1}: ERROR ({completed.returncode})")
                print(completed.stdout)
                print(completed.stderr)
                continue
            try:
                result = parse_result(completed.stdout)
            except ValueError as error:
                execution_failed = True
                print(f"process {process_index + 1}: ERROR ({error})")
                print(completed.stdout)
                continue

            results.append(result)
            correctness_failed |= not result.correct
            print(
                f"process {process_index + 1}: "
                f"{'PASS' if result.correct else 'FAIL'} | "
                f"median={result.optimized_median_ms:.4f} ms | "
                f"p90={result.optimized_p90_ms:.4f} ms | "
                f"speedup={result.speedup:.3f}x"
            )

        if len(results) == processes:
            median_speedup = statistics.median(result.speedup for result in results)
            repeatable = (
                all(result.correct and result.speedup > 1.0 for result in results)
                and median_speedup >= 1.02
                and all(
                    result.optimized_p90_ms <= 1.02 * result.baseline_p90_ms
                    for result in results
                )
            )
            verdict = "REPEATABLE" if repeatable else "SHAPE-DEPENDENT"
            print(f"verdict={verdict} median_process_speedup={median_speedup:.3f}x")

    if execution_failed:
        return 2
    return 1 if correctness_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
