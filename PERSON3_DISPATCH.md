# Person 3 Dispatcher and Shape #14 Experiment

## Scope

This branch is `codex/person3-dispatch-shape14`, isolated from the integration
branch. It adds measured setup-time dispatch, official-shape sweep reporting,
and a memory-safe Shape #14 evaluator. The reference model and Person 2 code
remain unchanged in behavior; their existing integration source is included
only so the dispatcher can be exercised end to end.

## Dispatch policy

The dispatch key includes batch size, sequence length, model width, head count,
dtype, causal mode, padding kind, device, CUDA capability, and PyTorch version.
Automatic selection requires an exact key and all of these gates:

- every correctness trial passes the benchmark OR rule;
- every process is faster than 1.0x;
- median process speedup is at least 1.02x;
- every optimized p90 is no more than 1.02 times the baseline p90.

If no measured candidate matches, the backend is native. A packed suffix is
automatically selected only for the exact measured key; manual suffixes remain
available for experiments. Triton is not auto-selected.

The validated automatic T4 entries are:

- B8/S512/D512/H8, FP16, non-causal, unpadded: one-layer packed SDPA,
  1.327x median process speedup.
- Official Shape 2 (B1/S128/D128/H4), FP16, causal, unpadded: one-layer
  packed SDPA, 1.363x median process speedup.
- Official Shape 3 (B4/S128/D128/H4), FP16, causal, unpadded: one-layer
  packed SDPA, 1.481x median process speedup.
- Official Shape 4 (B16/S128/D128/H4), FP16, causal, unpadded: one-layer
  packed SDPA, 1.448x median process speedup.
- Official Shape 12 (B64/S32/D128/H4), FP16, causal, unpadded: one-layer
  packed SDPA, 1.480x median process speedup.

These results are not generalized to nearby shapes, padding modes, devices,
or PyTorch versions.

An earlier Shape 2 experiment had p90 variability, but the later strict
20-trial, three-process run cleared correctness, median, and paired-baseline
p90 gates and supersedes it. Shape 7 (B64/S128/D32/H4) passed strict
correctness at every packed depth, but every packed candidate missed the 2%
relative improvement gate, so it retains native attention.

## Shape #14 safety path

The organizer's Shape #14 is B32/S100000/D1024/H16/L2. The original harness is
unsafe because the explicit attention reference would require a quadratic
score tensor. `bench_shape14_streaming.py` instead:

- allocates one batch block at a time;
- uses packed-QKV SDPA for the complete model;
- repeats blocks sequentially for the logical batch;
- performs a conservative free-memory preflight;
- never runs or times the explicit baseline at S=100000.

The reported value is sequential blockwise capability latency, not full-batch
throughput. On the T4, the full-width B1 smoke passed in 4.17 s at 1.57 GiB
peak allocation. The logical official B32 workload also passed with batch
block 1 in 152.72 s at 1.57 GiB peak allocation; no `[B,H,S,S]` matrix was
allocated.

The combined branch repeated the full B32/S100000/D1024/H16/L2 capability run
in 161.07 s with 1.57 GiB peak allocation. This is a finite/no-OOM result, not
full-shape reference correctness. At the representative B2/S4096/D1024/H16/L2
shape, packed SDPA in both layers failed the stricter internal gate by one of
4,194,304 elements (`max_abs=0.0078125`), but passed the organizer's
`atol=0.002`, `rtol=0.02` OR criterion with zero failed elements. Under those
judge-equivalent tolerances it measured 344.2802 ms baseline versus 21.8125 ms
optimized, a 15.784x sequential blockwise speedup. Report both the strict
failure and organizer-equivalent pass; do not claim strict Shape 14 accuracy.

## Validation status

The updated T4 checkout passes 54 tests with one expected opt-in skip. The
official CUDA smoke sweep passed strict correctness for IDs 1–13; automatic
dispatch selected packed SDPA only for IDs 2, 3, 4, and 12, with native fallback
for every other official shape. ID 9 was correctness-safe but received
`SMOKE-REVIEW` because of p90 timing variability. Dedicated three-process final
runs for IDs 3, 4, and 12 passed correctness, repeatability, and p90 gates.
The original full-batch ID 14 harness remains blocked; use the batch-blocked
evaluator for its memory-safe result.

No benchmark timing is emitted after strict accuracy failure. Generated timing
results are not written to tracked files.
