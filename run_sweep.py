#!/usr/bin/env python3
"""Run the integrated Transformer validation matrix in isolated processes."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Case:
    case_id: int
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
        prefix = f"ID{self.case_id} " if self.case_id else ""
        return (
            f"{prefix}B{self.batch} S{self.seq_len} D{self.d_model} "
            f"H{self.heads} L{self.layers} {mode} P{self.padding:g}"
        )


@dataclass(frozen=True)
class RunResult:
    correct: bool
    baseline_median_ms: float
    baseline_p90_ms: float
    optimized_median_ms: float
    optimized_p90_ms: float
    speedup: float


@dataclass(frozen=True)
class Candidate:
    packed_sdpa_suffix_layers: int
    fast_ffn_suffix_layers: int

    @property
    def label(self) -> str:
        return (
            f"packed={self.packed_sdpa_suffix_layers},"
            f"fast_ffn={self.fast_ffn_suffix_layers}"
        )


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate: Candidate
    results: tuple[RunResult, ...]
    rejected_reason: str | None = None

    @property
    def accepted(self) -> bool:
        if self.rejected_reason is not None or not self.results:
            return False
        median_speedup = statistics.median(result.speedup for result in self.results)
        return (
            all(result.correct and result.speedup > 1.0 for result in self.results)
            and median_speedup >= 1.02
            and all(
                result.optimized_p90_ms <= 1.02 * result.baseline_p90_ms
                for result in self.results
            )
        )

    @property
    def median_latency_ms(self) -> float:
        return statistics.median(
            result.optimized_median_ms for result in self.results
        )

    @property
    def median_speedup(self) -> float:
        return statistics.median(result.speedup for result in self.results)


INTEGRATION_CASES = (
    Case(0, 1, 128, 512, 8, 2048, 6, False, 0.0),
    Case(0, 8, 512, 512, 8, 2048, 6, False, 0.0),
    Case(0, 8, 512, 512, 8, 2048, 6, True, 0.2),
    Case(0, 2, 2048, 512, 8, 2048, 6, False, 0.0),
    Case(0, 1, 4096, 512, 8, 2048, 6, True, 0.25),
)

# Organizer Appendix configurations. qkv_dim maps to d_model and no padding
# was specified, so these cases use an all-valid mask.
OFFICIAL_CASES = (
    Case(1, 64, 128, 128, 4, 128, 4, True, 0.0),
    Case(2, 1, 128, 128, 4, 128, 4, True, 0.0),
    Case(3, 4, 128, 128, 4, 128, 4, True, 0.0),
    Case(4, 16, 128, 128, 4, 128, 4, True, 0.0),
    Case(5, 128, 128, 128, 4, 128, 4, True, 0.0),
    Case(6, 10000, 128, 128, 4, 128, 4, True, 0.0),
    Case(7, 64, 128, 32, 4, 32, 4, True, 0.0),
    Case(8, 64, 128, 1024, 4, 1024, 4, True, 0.0),
    Case(9, 64, 128, 128, 1, 128, 4, True, 0.0),
    Case(10, 64, 128, 128, 2, 128, 4, True, 0.0),
    Case(11, 64, 128, 128, 16, 128, 4, True, 0.0),
    Case(12, 64, 32, 128, 4, 128, 4, True, 0.0),
    Case(13, 64, 1024, 128, 4, 128, 4, True, 0.0),
    Case(14, 32, 100000, 1024, 16, 1024, 2, True, 0.0),
)


def shape14_memory_summary(case: Case, dtype: str) -> str:
    """Estimate unavoidable full tensors in the original ID 14 harness."""
    element_bytes = 2 if dtype == "float16" else 4
    gib = 1024**3
    activation_elements = case.batch * case.seq_len * case.d_model
    score_elements = case.batch * case.heads * case.seq_len * case.seq_len
    input_gib = activation_elements * element_bytes / gib
    input_output_gib = 2 * input_gib
    scores_gib = score_elements * element_bytes / gib
    fp32_probabilities_gib = score_elements * 4 / gib
    return (
        f"input={input_gib:.2f} GiB, input+output={input_output_gib:.2f} GiB, "
        f"explicit scores={scores_gib:.2f} GiB, "
        f"FP32 softmax probabilities={fp32_probabilities_gib:.2f} GiB"
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
    fast_ffn_suffix_layers: int,
    packed_sdpa_suffix_layers: int,
    atol: float = 0.001,
    rtol: float = 0.01,
    accuracy_trials: int = 20,
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
        "--accuracy-trials",
        str(accuracy_trials),
        "--fast-ffn-suffix-layers",
        str(fast_ffn_suffix_layers),
        "--packed-sdpa-suffix-layers",
        str(packed_sdpa_suffix_layers),
        "--atol",
        str(atol),
        "--rtol",
        str(rtol),
    ]
    if case.causal:
        command.append("--causal")
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("integration", "official"), default="integration"
    )
    parser.add_argument(
        "--case-id",
        type=int,
        help="run one official case ID instead of the complete selected suite",
    )
    parser.add_argument(
        "--allow-unsafe-shape14",
        action="store_true",
        help="allow the original full-allocation harness to attempt official ID 14",
    )
    parser.add_argument("--mode", choices=("smoke", "final"), default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32"),
        default="auto",
    )
    parser.add_argument("--processes", type=int)
    parser.add_argument("--fast-ffn-suffix-layers", type=int, default=0)
    parser.add_argument("--packed-sdpa-suffix-layers", type=int, default=0)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help=(
            "sweep packed-attention depth, then FFN depth for one official case"
        ),
    )
    parser.add_argument(
        "--policy-json",
        help="optionally write the accepted calibration result as JSON",
    )
    parser.add_argument(
        "--skip-long",
        action="store_true",
        help="skip the B1 S4096 memory-pressure case",
    )
    args = parser.parse_args()
    if args.processes is not None and args.processes <= 0:
        parser.error("--processes must be positive")
    if args.fast_ffn_suffix_layers < 0:
        parser.error("--fast-ffn-suffix-layers must be nonnegative")
    if args.packed_sdpa_suffix_layers < 0:
        parser.error("--packed-sdpa-suffix-layers must be nonnegative")
    if args.case_id is not None and args.suite != "official":
        parser.error("--case-id requires --suite official")
    if args.case_id is not None and not 1 <= args.case_id <= len(OFFICIAL_CASES):
        parser.error("--case-id must be between 1 and 14")
    if args.calibrate and (args.suite != "official" or args.case_id is None):
        parser.error("--calibrate requires --suite official and --case-id")
    if args.calibrate and args.case_id == 14:
        parser.error("ID 14 uses bench_shape14_blockwise.py, not --calibrate")
    if args.calibrate and (
        args.fast_ffn_suffix_layers or args.packed_sdpa_suffix_layers
    ):
        parser.error("--calibrate chooses suffix depths; do not provide depth flags")
    if args.calibrate and args.processes is not None and args.processes != 3:
        parser.error("--calibrate requires exactly three independent processes")
    if args.policy_json and not args.calibrate:
        parser.error("--policy-json requires --calibrate")
    return args


def failed_summary(output: str) -> str | None:
    match = re.search(
        r"^summary: FAIL .*? failed=([0-9]+/[0-9]+)$",
        output,
        flags=re.MULTILINE,
    )
    return match.group(1) if match is not None else None


def evaluate_candidate(
    case: Case,
    candidate: Candidate,
    *,
    device: str,
    dtype: str,
    processes: int,
    warmup: int,
    repeats: int,
    rounds: int,
) -> CandidateEvaluation:
    """Run one candidate in isolated processes and retain strict failures."""
    results: list[RunResult] = []
    command = build_command(
        case,
        device,
        dtype,
        warmup,
        repeats,
        rounds,
        candidate.fast_ffn_suffix_layers,
        candidate.packed_sdpa_suffix_layers,
    )
    for process_index in range(processes):
        completed = subprocess.run(command, capture_output=True, text=True)
        failures = failed_summary(completed.stdout)
        if failures is not None:
            return CandidateEvaluation(
                candidate,
                tuple(results),
                f"strict correctness failed ({failures})",
            )
        if completed.returncode != 0:
            return CandidateEvaluation(
                candidate,
                tuple(results),
                f"process {process_index + 1} exited {completed.returncode}",
            )
        try:
            results.append(parse_result(completed.stdout))
        except ValueError as error:
            return CandidateEvaluation(candidate, tuple(results), str(error))
    return CandidateEvaluation(candidate, tuple(results))


def print_evaluation(evaluation: CandidateEvaluation) -> None:
    if evaluation.rejected_reason is not None:
        print(f"{evaluation.candidate.label}: REJECT | {evaluation.rejected_reason}")
        return
    p90_gate = all(
        result.optimized_p90_ms <= 1.02 * result.baseline_p90_ms
        for result in evaluation.results
    )
    verdict = "ACCEPT" if evaluation.accepted else "REVIEW"
    print(
        f"{evaluation.candidate.label}: {verdict} | "
        f"median={evaluation.median_latency_ms:.4f} ms | "
        f"speedup={evaluation.median_speedup:.3f}x | "
        f"p90_gate={'PASS' if p90_gate else 'FAIL'}"
    )


def calibrate_case(
    case: Case,
    *,
    device: str,
    dtype: str,
    processes: int,
    warmup: int,
    repeats: int,
    rounds: int,
) -> CandidateEvaluation | None:
    """Calibrate attention first, then FFN against the best safe attention."""
    print(f"\n[{case.label}] attention calibration")
    attention_evaluations = []
    for depth in range(case.layers + 1):
        evaluation = evaluate_candidate(
            case,
            Candidate(depth, 0),
            device=device,
            dtype=dtype,
            processes=processes,
            warmup=warmup,
            repeats=repeats,
            rounds=rounds,
        )
        attention_evaluations.append(evaluation)
        print_evaluation(evaluation)

    accepted_attention = [
        evaluation for evaluation in attention_evaluations if evaluation.accepted
    ]
    if not accepted_attention:
        print("no attention candidate cleared all gates")
        return None
    # Each benchmark invocation measures its own paired baseline. Selecting by
    # speedup is therefore less sensitive to thermal drift between candidates
    # than comparing absolute candidate latency from separate processes.
    attention_winner = max(
        accepted_attention, key=lambda evaluation: evaluation.median_speedup
    )

    packed_depth = attention_winner.candidate.packed_sdpa_suffix_layers
    print(f"\n[{case.label}] FFN calibration with packed={packed_depth}")
    ffn_evaluations = []
    for depth in range(case.layers + 1):
        evaluation = evaluate_candidate(
            case,
            Candidate(packed_depth, depth),
            device=device,
            dtype=dtype,
            processes=processes,
            warmup=warmup,
            repeats=repeats,
            rounds=rounds,
        )
        ffn_evaluations.append(evaluation)
        print_evaluation(evaluation)

    accepted_ffn = [
        evaluation for evaluation in ffn_evaluations if evaluation.accepted
    ]
    if not accepted_ffn:
        print("no combined candidate cleared all gates")
        return None
    winner = max(accepted_ffn, key=lambda evaluation: evaluation.median_speedup)
    print(f"\nselected {winner.candidate.label}")
    return winner


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

    final_mode = args.mode == "final" or args.calibrate
    processes = args.processes or (3 if final_mode else 1)
    warmup, repeats, rounds = (20, 100, 3) if final_mode else (5, 20, 1)
    cases = OFFICIAL_CASES if args.suite == "official" else INTEGRATION_CASES
    if args.case_id is not None:
        cases = tuple(case for case in cases if case.case_id == args.case_id)
    elif args.skip_long:
        cases = cases[:-1]

    if any(case.case_id == 14 for case in cases) and not args.allow_unsafe_shape14:
        shape14 = OFFICIAL_CASES[-1]
        print(
            "official ID 14 is blocked in the original harness: its full input, "
            "reference, and explicit attention scores cannot fit on a T4. "
            "Run IDs 1-13 with --skip-long; use --allow-unsafe-shape14 only "
            "on a judge environment designed for that allocation."
        )
        print(f"ID 14 {dtype} memory floor: {shape14_memory_summary(shape14, dtype)}")
        return 2

    if args.calibrate:
        case = cases[0]
        print(
            f"suite=official mode=calibrate device={device} dtype={dtype} "
            f"processes={processes} warmup={warmup} repeats={repeats} "
            f"rounds={rounds} accuracy_trials=20 atol=0.001 rtol=0.01"
        )
        winner = calibrate_case(
            case,
            device=device,
            dtype=dtype,
            processes=processes,
            warmup=warmup,
            repeats=repeats,
            rounds=rounds,
        )
        if winner is None:
            return 1
        if args.policy_json:
            policy = {
                "case_id": case.case_id,
                "shape": {
                    "batch_size": case.batch,
                    "seq_len": case.seq_len,
                    "d_model": case.d_model,
                    "num_heads": case.heads,
                    "ffn_dim": case.ffn_dim,
                    "num_layers": case.layers,
                    "causal": case.causal,
                    "padding": case.padding,
                    "dtype": dtype,
                },
                "packed_sdpa_suffix_layers": (
                    winner.candidate.packed_sdpa_suffix_layers
                ),
                "fast_ffn_suffix_layers": winner.candidate.fast_ffn_suffix_layers,
                "median_speedup": winner.median_speedup,
                "median_latency_ms": winner.median_latency_ms,
            }
            with open(args.policy_json, "w", encoding="utf-8") as policy_file:
                json.dump(policy, policy_file, indent=2)
                policy_file.write("\n")
            print(f"wrote policy candidate to {args.policy_json}")
        return 0

    print(
        f"suite={args.suite} mode={args.mode} device={device} dtype={dtype} "
        f"processes={processes} "
        f"warmup={warmup} repeats={repeats} rounds={rounds} "
        f"fast_ffn_suffix_layers={args.fast_ffn_suffix_layers} "
        f"packed_sdpa_suffix_layers={args.packed_sdpa_suffix_layers}"
    )
    correctness_failed = False
    execution_failed = False

    for case in cases:
        results: list[RunResult] = []
        print(f"\n[{case.label}]")
        command = build_command(
            case,
            device,
            dtype,
            warmup,
            repeats,
            rounds,
            args.fast_ffn_suffix_layers,
            args.packed_sdpa_suffix_layers,
        )
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
            if not final_mode:
                verdict = "SMOKE-PASS" if repeatable else "SMOKE-REVIEW"
            else:
                verdict = "REPEATABLE" if repeatable else "SHAPE-DEPENDENT"
            print(f"verdict={verdict} median_process_speedup={median_speedup:.3f}x")

    if execution_failed:
        return 2
    return 1 if correctness_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
