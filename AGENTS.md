# Hackathon Agent Context

## Main goal

Optimize the provided PyTorch Transformer benchmark for the target GPU while
preserving the reference output within the benchmark's numerical tolerance.
The team has three members and approximately 3–5 days. The preferred strategy
is a reliable PyTorch optimization ladder:

1. PyTorch scaled dot-product attention (SDPA).
2. Packed QKV projection.
3. `torch.compile` and shape-aware dispatch.
4. Custom Triton/CUDA kernels only when profiling proves they are worthwhile.

Do not make a custom kernel a prerequisite for a correct submission.

## Benchmark and correctness contract

The main benchmark is `torch_transformer_benchmark.py`.

- The public API of `UserOptimizedTransformer.forward` must remain unchanged.
- Inputs and outputs have shape `[batch_size, seq_len, d_model]`.
- The baseline uses pre-norm residual blocks, explicit multi-head attention,
  FP32 softmax input for numerical stability, exact GELU, and final zeroing of
  invalid token positions.
- Causal masking prevents attending to future positions.
- `valid_token_mask` masks invalid key positions and invalid query outputs.
- The benchmark compares every output element using an OR condition:
  `abs_error <= atol` or `abs_error <= rtol * abs(reference)`.
- The organizer-updated script currently defaults to `atol=0.002` and
  `rtol=0.02`; the Person 1 standalone tests continue to use the stricter
  `atol=0.001` and `rtol=0.01` contract for local validation.
- Timing uses CUDA events after warmup. Do not include random-data generation
  in performance measurements.
- Keep dtype, matmul precision, and TF32 settings identical for reference and
  optimized models.

## Baseline profiler evidence

The initial unchanged-baseline profile reported approximately:

- `bmm`: 13.56% CPU total;
- `masked_fill`: 11.81% CPU total;
- `_softmax`: 5.22% CPU total;
- `addmm`: 52.65% CPU total across 36 calls;
- `copy` plus `clone`: approximately 27% CPU total.

The `bmm` + `masked_fill` + `_softmax` entries total approximately 30.6%,
supporting the SDPA direction. With the default six-layer configuration, 18 of
the 36 `addmm` calls are the separate Q/K/V projections. Packed QKV should
replace those 18 calls with 6 packed projections, removing 12 calls while
leaving the 6 output projections and 12 FFN projections unchanged. This is an
expected reduction, not a measured end-to-end result until the optimized
attention is integrated and benchmarked.

The screenshot is CPU-side profiling. It identifies operator and launch
overhead but does not establish GPU latency, DRAM bandwidth, or FlashAttention
backend selection. Use CUDA events or a profiler trace with CUDA activity for
the final performance claim. Watch tensor copies and layout conversions because
they can erase the benefit of packed QKV.

The organizer-updated `torch_transformer_benchmark.py` is a baseline harness:
the duplicate in-file `FastSelfAttention` implementation has been removed.
Person 1's optimized attention remains in the standalone package, and Person 3
owns wiring it into `UserOptimizedTransformer`.

## Organizer appendix: required test shapes

In this table, `QKV Dim` maps to the benchmark's `d_model` argument. Cases 1–13
are ordinary correctness/performance cases. Case 14 is an extreme stress case:
the explicit baseline would need `32 * 16 * 100000^2 = 5.12e12` score
elements, or about 20.48 TB in FP32 before other tensors. Do not run it
blindly on the RTX 4050 or include it in an ordinary full-baseline timing
sweep. Treat it as a capability/dispatch case with an explicit memory guard,
an analytical resource note, or a separately approved long-context strategy.

| # | Batch | QKV Dim | Heads | Seq Len | Layers | Causal | FFN Dim |
|--:|------:|--------:|------:|--------:|-------:|:------:|--------:|
| 1 | 64 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 2 | 1 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 3 | 4 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 4 | 16 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 5 | 128 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 6 | 10000 | 128 | 4 | 128 | 4 | TRUE | 128 |
| 7 | 64 | 32 | 4 | 128 | 4 | TRUE | 32 |
| 8 | 64 | 1024 | 4 | 128 | 4 | TRUE | 1024 |
| 9 | 64 | 128 | 1 | 128 | 4 | TRUE | 128 |
| 10 | 64 | 128 | 2 | 128 | 4 | TRUE | 128 |
| 11 | 64 | 128 | 16 | 128 | 4 | TRUE | 128 |
| 12 | 64 | 128 | 4 | 32 | 4 | TRUE | 128 |
| 13 | 64 | 128 | 4 | 1024 | 4 | TRUE | 128 |
| 14 | 32 | 1024 | 16 | 100000 | 2 | TRUE | 1024 |

## Team ownership

### Person 1 — self-attention pipeline

Own the standalone `TritonSelfAttention` package in
`person1_triton_attention.py`:

- packed QKV projection and baseline-compatible weight transfer;
- PyTorch SDPA as the guaranteed and default path;
- causal and right-padding masks;
- exact zeroing of invalid query/output tokens;
- standalone correctness tests and attention-only benchmarks.

`PackedQKVSDPAAttention` is the explicit packed-QKV SDPA implementation.
`TritonSelfAttention(backend="auto")` also selects SDPA unconditionally for
the current safe submission path. SDPA is available for all official head
dimensions (8, 32, 64, 128, and 256); `_CUSTOM_HEAD_DIMS` restricts only the
experimental Triton path. The custom online-softmax Triton kernel is opt-in
with `backend="triton"`, requires both Triton and the structured
`torch.library.triton_op` API, and falls back to SDPA when runtime, shape,
dtype, mask, or compatibility conditions are unsupported. FP32 is retained
for numerically sensitive online-softmax statistics. Causal masked cases that
cannot use a compact SDPA mask use a tiled PyTorch online-softmax fallback;
the dense compatibility reference is bounded to small sequences and is never
used for Shape #14.

Current Person 1 branch: `person1/attention-sdpa`.

Current Person 1 source commit: `18348ba Harden Person 1 attention diagnostics and validation`.
This integration branch includes the Person 1 hardening updates in its merge
history.

The implementation adds `triton_scaled_dot_product_attention`,
`TritonSelfAttention`, `PackedQKVSDPAAttention`,
`tests/test_person1_attention.py`, and `bench_person1_attention.py` without
modifying `UserOptimizedTransformer`; Person 3 owns dispatch and final
assembly.

The RTX 4050 Laptop GPU validation environment is
`C:\Users\zj040\AppData\Local\Python\pythoncore-3.14-64\python.exe`
with PyTorch 2.13.0+cu126 and Triton installed. The expanded suite currently
passes 11/11 tests, with only the long test gated by default; the gated
100,000-token smoke test also passes. Representative official-shape runs
pass strict accuracy. Packed SDPA remains the production choice: on official
Shape #10, packed SDPA measured 0.1869 ms versus custom Triton 0.5386 ms, and
on the reduced 100,000-token core smoke, SDPA measured 91.6388 ms versus
Triton 4824.5186 ms.

The same package was independently checked on the Colab acceptance-style
environment: Tesla T4 (compute capability 7.5), PyTorch 2.11.0+cu128, and
Triton with `torch.library.triton_op` available. The full discovered suite
passed 10 tests with 1 long test skipped by its default gate; the explicitly
enabled 100,000-token Triton smoke test passed in 7.648 seconds. T4 forward
benchmarks measured packed-QKV SDPA at 4.572x faster than the explicit
baseline for Shape #1 and 17.949x faster for Shape #13. On Shape #10, custom
Triton was correct but slower than packed SDPA (1.8558 ms versus 0.2231 ms).
For a reduced `B=1, H=1, S=100000, Dh=64` causal core, SDPA measured
115.7148 ms and Triton 2970.7415 ms; both were finite and passed comparison.
Therefore `backend="auto"` must remain SDPA-first, with Triton retained as an
optional correctness-validated experiment rather than the production default.

The attention-only benchmark also exposes the organizer appendix through
`--official-case`. Shape #14 is guarded: ordinary dense baseline and
full-module QKV benchmarks are refused, while `--attention-core
--allow-long-sequence` can run a preprojected, memory-budgeted long-sequence
smoke test without materializing an attention matrix.

Recent Person 1 hardening records the backend used by the most recent eager
forward, including an explicit SDPA fallback reason when an opt-in Triton
call cannot run. `prepare_valid_token_mask()` collapses only proven all-valid
masks during setup, so the timed/compiled forward path performs no reduction
or host synchronization for that decision. `sdpa_backend_diagnostics()` can
be run outside timing to record Q/K/V strides, mask form, dtype, and scoped
Flash/Efficient/Math SDPA eligibility; an accepted diagnostic is evidence of
eligibility, not a claim about the automatic kernel selected by the runtime.
The benchmark's `--profile` mode profiles packed SDPA separately and reports
the selected CUDA operator alongside the eligibility probe.
CUDA BF16 benchmarks are skipped unless
`torch.cuda.is_bf16_supported(including_emulation=False)` reports native
support. Fixed-shape compile tests use `fullgraph=True` and `dynamic=False`
and compare compiled baseline and candidate outputs independently. Compiled
backend telemetry is reported as unverified unless a profiler or runtime
diagnostic proves the selected implementation.

### Person 2 — FFN, LayerNorm, and residuals

Own FFN and normalization optimization. Start with `torch.compile`, exact
`F.gelu(..., approximate="none")`, contiguous layouts, and residual/LayerNorm
fusion. Do not assume that a hand-written single kernel for both large Linear
GEMMs is faster than optimized PyTorch/cuBLAS. Custom Triton is a stretch path.

Current Person 2 branch: `person2/ffn-fullop-t4`.

Current Person 2 core commit: `d6bfab0 Add standalone Person 2 FFN optimization`.

Current Person 2 validated Task 2 head: `c319ab4` (source implementation
`16cf403`, followed by documentation and explicit CUDA coverage).

The standalone, baseline-weight-compatible `OptimizedTransformerBlock`, focused
unit tests, and `bench_person2_ffn.py` are implemented. Final
`UserOptimizedTransformer` assembly remains owned by Person 3. The implementation
uses exact GELU and a token-major FFN view; no custom Triton kernel was added.

Detailed Person 2 benchmark history, profiler evidence, and rejected
variants now live in `PERSON2_EXPERIMENTS.md`. Keep `AGENTS.md` focused on
the active handoff.

The current validation branch is `person2/ffn-fullop-t4`. Source commit
`16cf403` extends the strongest balanced implementation with a single
inference-only custom-op boundary for the complete identity-affine LayerNorm
and FFN sequence. PyTorch/cuBLAS still execute both GEMMs and ATen still
executes exact GELU; the custom boundary removes Python/dispatcher transitions
and finishes with the existing 128-bit residual/mask kernel. Nonidentity
LayerNorm parameters, unsupported devices/dtypes/layouts, build failures, and
failed numerical preflight retain the native fallback.

The NeurIPS run-length-tokenization follow-up adds a strictly guarded valid-row
compaction path. Setup caches row indices only when the exact supplied mask has
padding; timed inference may use them only for that same mask object and tensor
version. It computes the FFN only for valid rows and scatters into a fixed-shape
zero tensor. Changed masks, unpadded inputs, compilation, unsupported devices,
and failed numerical preflight use the dense implementation.

On the Colab Tesla T4 (`sm_75`, 14.56 GiB), PyTorch 2.11.0+cu128 and CUDA
12.8, the guarded candidate passed all 19 CUDA/CPU tests. Padded-case isolated
speedups were 1.048x, 1.040x, and 1.035x in three processes with zero failed
elements. One complete three-process restored-candidate run cleared the 1.005x
median threshold on all five shapes, but a later independent run measured
0.980x on the shortest shape in one process; one short p90 and the short
full-block case also regressed. Therefore there is no repeatable universal
1.005x or full-block safety claim. Exact measurements and rejected experiments
are in `PERSON2_EXPERIMENTS.md`.

The all-valid-mask bypass was later retested at `ce762be`. It passed all 20 T4
tests, but failed the universal gate: short-shape isolated speedups were
0.960x, 0.984x, and 1.086x across three processes, with two p90 regressions.
The primary-sampling short full-block result was 0.969x and its p90 regressed
6.4%; compiled short measured 0.888x. The optimization remains reverted.
Current validated source commit: `fd7b8d7`. `UserOptimizedTransformer` and
Person 3 dispatch remain unchanged.

The final Task 2 head `c319ab4` passed all 20 tests on the Colab T4, including
explicit real-CUDA masked and unmasked full-op execution.
Across three independent five-shape isolated FP16 sweeps, its lowest speedup
was 1.023x; all 15 p90 comparisons improved and every accuracy comparison had
zero failed elements. A reduced-sampling full-block sweep produced one 0.858x
medium outlier. Primary-sampling diagnosis did not reproduce it: three medium
full-block processes measured 1.016x, 1.011x, and 1.013x with improved p90,
while the short primary result was 1.037x. The outlier remains disclosed in
`PERSON2_EXPERIMENTS.md`; the isolated universal 1.005x claim is validated,
and full-block evidence is reported with that variance caveat.

### Person 3 — integration, profiling, and dispatch

Current integration branch: `person3/integrate-person1`.

Person 1 source `bbd0cc8` (branch tip `cfee3c1`) and validated Person 2 tip
`f50ef57` are both contained by integration merge `99c1830` (Person 2 entered
at merge `f6d897a`). The T4-validated source-code tree is `d57502e`; the final
Person 1 ancestry merge did not change implementation files. The combined
local suite passed 28 tests with six hardware/toolkit skips, and the strict CPU
harness smoke passed two trials with zero failed elements.

Combined T4 validation used a Tesla T4 (15,360 MiB), driver 580.82.07,
PyTorch 2.11.0+cu128, CUDA 12.8, and FP16. All 28 tests passed, including the
Person 1 Triton CUDA paths and Person 2 custom-op paths. The strict CUDA harness
smoke passed two trials with zero failed elements (`0.8313 ms` baseline median,
`0.8337 ms` optimized median); because `UserOptimizedTransformer` remains
unchanged, this noisy `0.997x` result is a safety check rather than a speedup
claim.

The representative attention matrix covered sequence lengths 128 and 4096,
causal and non-causal execution, and unpadded and 25%-padded masks. Packed SDPA
passed every strict comparison with zero failed elements and improved median
latency in all eight cases (`1.341x` to `12.254x`). Triton also passed strict
correctness but ranged from `0.119x` to `2.208x`; the long-sequence cases
regressed substantially. Keep packed SDPA as the production attention path and
Triton as an explicit experiment only; do not dispatch to Triton by default.

T4 attention timings below use batch 1, model width 512, 8 heads, FP16, 5
warmups, and 20 CUDA-event repetitions. Throughput is tokens/second; all rows
had zero failed elements at `atol=0.001`, `rtol=0.01`.

| Seq / mode / padding | Baseline median / p90 (ms) | Packed SDPA median / p90 (ms), speedup, tok/s | Triton median / p90 (ms), speedup, tok/s | Max abs error |
| --- | ---: | ---: | ---: | ---: |
| 128 / non-causal / none | 0.7108 / 0.9181 | 0.1943 / 0.2898, 3.658x, 658,775 | 0.3677 / 0.4176, 1.933x, 348,110 | 0.00012207 |
| 128 / causal / none | 0.4928 / 0.5958 | 0.1559 / 0.1851, 3.160x, 821,039 | 0.5359 / 0.5638, 0.920x, 238,851 | 0.000488281 |
| 128 / non-causal / 25% | 0.5181 / 0.5572 | 0.3864 / 0.4241, 1.341x, 331,263 | 0.4608 / 0.5389, 1.124x, 277,778 | 0.00012207 |
| 128 / causal / 25% | 1.1022 / 1.2813 | 0.7260 / 0.9109, 1.518x, 176,309 | 0.4992 / 0.5428, 2.208x, 256,410 | 0.000488281 |
| 4096 / non-causal / none | 17.0902 / 19.9334 | 3.1335 / 3.2119, 5.454x, 1,307,165 | 30.5717 / 30.5827, 0.559x, 133,980 | 0.0000610352 |
| 4096 / causal / none | 22.8837 / 25.7587 | 1.8675 / 1.9661, 12.254x, 2,193,307 | 39.5864 / 39.6987, 0.578x, 103,470 | 0.000488281 |
| 4096 / non-causal / 25% | 22.2259 / 23.4281 | 4.3809 / 4.4221, 5.073x, 934,968 | 186.1327 / 186.6188, 0.119x, 22,006 | 0.0000610352 |
| 4096 / causal / 25% | 27.8329 / 30.0444 | 5.7226 / 5.7611, 4.864x, 715,759 | 219.6946 / 220.4763, 0.127x, 18,644 | 0.000488281 |

Own `UserOptimizedTransformer`, weight-copy integration, benchmark-driven
dispatch, profiling, README/reproducibility, and the final demo. Dispatch keys
should include at least:

`(batch_size, seq_len, d_model, num_heads, dtype, causal, padding)`.

Do not use fixed sequence-length thresholds without measurements. The current
benchmark passes a mask even when all tokens are valid, so investigate whether
the no-padding path can avoid an unnecessary attention mask while preserving
the required forward signature.

## Integration rules

- Keep `BaselineSelfAttention` and all reference behavior unchanged.
- `TritonSelfAttention` must expose the same forward signature as the baseline.
- Packed QKV weights must be copied in Q/K/V row order.
- Avoid CPU/GPU synchronization, unnecessary tensor-to-scalar conversions,
  and data-dependent Python control flow inside compiled forward paths.
- Keep a correct fallback path for unsupported shapes, masks, dtypes, or
  numerical failures.
- Never claim a backend, speedup, or memory improvement without benchmark or
  profiler evidence.
- Prefer small, isolated changes and document why each optimization exists.

## Validation expectations

Before merging, test all available combinations of:

- short and long sequence lengths;
- small and large batch sizes;
- model/head dimensions;
- causal and non-causal attention;
- padded and unpadded masks;
- float32, float16, and bfloat16 where supported.

Record median latency, p90 latency, speedup, throughput, max absolute error,
max relative error, failed elements, GPU name, PyTorch version, dtype, and
compile mode. Report both eager-vs-eager and compiled-vs-compiled comparisons
when possible.

## Git collaboration

- Do not commit directly to `main`.
- Each person works on a named branch and commits focused changes.
- Person 1 owns attention definitions; Person 2 owns FFN/normalization
  definitions; Person 3 owns model assembly, dispatch, and benchmark/report
  integration.
- Before editing, check `git status` and preserve unrelated changes.
- Pull or fetch before handoff. Merge through a pull request or cherry-pick a
  reviewed commit into the integration branch.
- Avoid committing generated benchmark outputs, caches, or environment files.
- If the monolithic benchmark file becomes conflict-prone, extract
  `attention_kernels.py`, `ffn_kernels.py`, and `dispatch.py`, leaving the
  benchmark file as a stable harness.

## Suggested execution order

1. Establish baseline numbers for every supplied configuration.
2. Implement and validate SDPA.
3. Add packed QKV and validate weight equivalence.
4. Compile and benchmark each relevant mode.
5. Add shape/mask dispatch based on measured results.
6. Attempt one custom kernel only for a demonstrated remaining bottleneck.
7. Freeze code, run regression tests, and prepare the reproducibility report
   and 2–3 minute demo.
