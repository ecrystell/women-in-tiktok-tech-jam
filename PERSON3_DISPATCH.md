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

If no measured candidate matches, the backend is native. A packed suffix can
be requested only as an explicit experiment. Triton is not auto-selected.

The only historical automatic candidate is the exact Tesla T4 B8/S512/D512/H8
FP16 non-causal unpadded workload on PyTorch 2.11.0, where a one-layer packed
SDPA suffix passed the gate at 1.327x median process speedup. It is intentionally
not generalized to the official appendix shapes.

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
throughput. A reduced local smoke (`B1/S100000/D64/H1/L1`, FP16) passed on an
RTX 4050 with 0.09 GiB peak GPU allocation. A full official T4 claim still
requires execution on the acceptance environment.

## Validation status

The isolated integration suite passes the CPU and available CUDA tests,
including reduced Shape #14 exact correctness and the streaming smoke. The
official CUDA sweep passed IDs 1–5 and 7–13 with native fallback. ID 6 was
stopped by the new full-batch reference memory guard on the local RTX 4050;
this is a safety result, not a correctness or speedup result. Run the sweep on
the T4, or add a separately validated batch-blocked evaluator, before claiming
ID 6 performance.

No benchmark timing is emitted after strict accuracy failure. Generated timing
results are not written to tracked files.
