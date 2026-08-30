# Hackathon Agent Context

## Main goal

Optimize the provided PyTorch Transformer benchmark for the target GPU while
preserving the reference output within the benchmark's numerical tolerance.
The team has three members. The preferred strategy is a reliable PyTorch
optimization ladder:

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
- The organizer harness defaults to `atol=0.002` and `rtol=0.02`. Use the
  stricter `atol=0.001`, `rtol=0.01` gate for local candidate screening, then
  report judge-equivalent results with the organizer defaults explicitly.
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
the current safe submission path. The custom online-softmax Triton kernel is
opt-in with `backend="triton"` and falls back to SDPA when Triton, the shape,
dtype, mask, or compatibility conditions are unsupported. FP32 is retained
for numerically sensitive online-softmax statistics.

Current Person 1 branch: `person1/attention-sdpa`.

Current Person 1 source commit: `bbd0cc8 Harden Triton attention fallbacks and dispatch`.
The shared context on that branch is updated in `cfee3c1`.

The implementation adds `triton_scaled_dot_product_attention`,
`TritonSelfAttention`, `PackedQKVSDPAAttention`,
`tests/test_person1_attention.py`, and `bench_person1_attention.py` without
modifying `UserOptimizedTransformer`; Person 3 owns dispatch and final
assembly. Validation on the RTX 4050 Laptop GPU passed all 8 standalone tests
with PyTorch 2.13.0+cu126 and Triton 3.7.1. Packed SDPA is currently faster
than the custom Triton path at the default attention shape, so SDPA remains
the production choice.

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

Current assembly candidate branch: `person3/optimize-transformer-dispatch`.
The current strict-safe candidate uses baseline attention with Person 2's
guarded optimized FFN block. It refreshes nonpersistent FFN state, preserves
the public forward signature, and prepares extension-backed FFN paths and
exact-mask caches before compilation and timing. Unsupported paths retain
native fallbacks. Person 1's packed SDPA and Triton implementations remain
available as standalone experiments, but are not selected by the production
full-model path because of the T4 failure documented below.

Local CPU validation of the assembly passed all 36 tests with eight expected
CUDA/Triton skips. Strict non-causal/unpadded and causal/25%-padded harness
smokes each passed with zero failed elements. These CPU timings are not GPU
speedup evidence.

The first end-to-end T4 smoke rejected the candidate before timing. All four
tested shapes prepared the fast FFN in 6/6 layers, but strict five-trial
accuracy failed after six combined layers. Failed elements ranged from 1,190
of 327,680 for B1/S128 to 10,712 of 10,485,760 for causal padded B8/S512;
maximum absolute error ranged from 0.0078125 to 0.00976562. Standalone component
correctness therefore does not establish full-model correctness. Use
`diagnose_integration.py` to separate attention drift, FFN drift, and their
interaction before adding dispatch. Do not benchmark on failure or merge the
candidate into the shared integration branch until strict T4 accuracy passes.

The first fallback smoke then selected baseline attention plus fast FFN in all
six layers. B1/S128 passed five strict trials with exact output, but performance
was not repeatable: the direct run measured 0.987x with worse p90 while the
sweep process measured 1.015x. B8/S512 unpadded, B8/S512 causal with 20%
padding, and B2/S2048 unpadded all failed strict accuracy before timing, with
maximum absolute errors from 0.0078125 and thousands of failed elements across
five trials. The next diagnostic must test partial FFN layer selection; the
six-layer FFN fallback is not a valid universal production dispatch.

The B8/S512 layer-selection diagnostic found that the first four individual
fast FFN layers failed strict full-model accuracy. Layers 5 and 6 passed
individually, and the final two-layer suffix passed together with zero failed
elements for seed 1234; a final three-layer suffix failed 23 elements. This is
candidate evidence, not dispatch evidence, until the two-layer suffix passes
all benchmark accuracy trials and repeated timing. The main benchmark now
defaults to zero fast FFN layers and exposes `--fast-ffn-suffix-layers` only
for controlled validation. Do not change the safe default until a shape has
repeatable strict-correct speedup evidence.

The next exact-safe candidate implements the requested all-valid-mask bypass.
Preparation checks the fixed mask outside timing and caches only its exact
object identity and tensor version when every token is valid. Eager forward
then treats only that unchanged mask as `None`, removing no-op attention,
block-output, and final-output masking. Cloned, changed, padded, unprepared,
and compiled masks retain the original path. Accuracy validation now includes
the exact prepared benchmark input so the path being timed must pass strict
comparison. Local unpadded and padded harness checks were exact; T4 correctness
and performance were subsequently validated at candidate commit `7d87e79`.

The transformer-level all-valid-mask bypass passed three independent eager
FP16 processes on a Tesla T4 for B8/S512/D512/H8/FFN2048/L6, non-causal and
unpadded, using 20 warmups, 100 repeats, and three alternating rounds per
process. All 37,748,736 checked output elements were bit-exact. Median speedups
were 1.220x, 1.218x, and 1.213x (median-of-processes 1.218x). Optimized p90
latencies were 22.7702 ms, 23.1190 ms, and 23.5960 ms versus baseline p90
latencies of 27.6974 ms, 28.0642 ms, and 28.6723 ms, so every process cleared
the correctness, median-speedup, and p90 gates. This validates dispatch for
that exact all-valid shape family; padded, causal, short, long, and other batch
or model shapes still require separate measurements before broader claims.

The next accuracy-recovery experiment adds
`--packed-sdpa-suffix-layers N`, defaulting to zero. It constructs Person 1's
canonical `PackedQKVSDPAAttention` only in the final N transformer layers so
its FP16 differences have fewer later residual blocks in which to accumulate.
Weight transfer still packs rows in exact `[Q; K; V]` order, the FFN experiment
remains disabled independently, and invalid suffix counts are rejected. Start
T4 validation with N=1 and increase only after five-trial strict correctness;
this is an experimental control, not a production dispatch or speedup claim.

Initial T4 suffix validation used B8/S512/D512/H8/FFN2048/L6, non-causal,
unpadded FP16 with the all-valid-mask bypass and native FFN. N=1 passed 20
random trials plus the exact prepared case with zero failed elements across
44,040,192 comparisons; its one-round smoke measured 1.336x end-to-end. N=2
is rejected: 16 of 20 random trials and the prepared case failed, totaling 45
failed elements across the same comparison count with maximum absolute error
0.0078125. Performance was correctly skipped. N=1 proceeded to three-process
timing; do not dispatch N=2 for this shape.

The subsequent three-process primary-sampling run validated N=1 for that exact
shape. All 132,120,576 comparisons passed, and median speedups were 1.333x,
1.325x, and 1.327x (median-of-processes 1.327x). Optimized p90 latencies were
21.0168 ms, 21.4778 ms, and 21.9393 ms versus baseline p90 latencies of
27.8528 ms, 28.3276 ms, and 29.0056 ms. Every process therefore cleared the
correctness, speedup, and p90 gates. The next isolated experiment may combine
this one-layer attention suffix with one final fast FFN layer; broader FFN or
attention suffixes remain disabled.

That combined one-layer attention plus one-layer fast-FFN experiment also
passed all 132,120,576 comparisons in three primary-sampling processes.
Speedups were 1.329x, 1.326x, and 1.324x (median-of-processes 1.326x), and all
p90 latencies improved against baseline. This does not improve on the 1.327x
packed-only median and is therefore rejected as a performance dispatch for
this shape. Keep the native FFN with the validated mask bypass plus final-layer
packed SDPA; the fast FFN remains available only as an explicit experiment.

The final eager full-integration matrix then validated the selected native-FFN
candidate across all five target configurations. Each shape ran in three
independent processes with 20 warmups, 100 repeats, and three alternating
rounds; all 15 processes passed strict correctness and the median/p90 gate.

| Full-model configuration | Process speedups | Median-of-processes |
| --- | --- | ---: |
| B1/S128, non-causal, unpadded | 1.480x, 1.492x, 1.462x | 1.480x |
| B8/S512, non-causal, unpadded | 1.334x, 1.325x, 1.324x | 1.325x |
| B8/S512, causal, 20% padding | 1.098x, 1.093x, 1.098x | 1.098x |
| B2/S2048, non-causal, unpadded | 1.452x, 1.448x, 1.450x | 1.450x |
| B1/S4096, causal, 25% padding | 1.137x, 1.145x, 1.141x | 1.141x |

The unweighted geometric mean of the five median-of-process speedups is
1.289x, equivalent to a 22.4% aggregate latency reduction under that summary
method. Report the complete 1.098x-1.480x range alongside this aggregate;
there is no competition-provided rule that defines one official cross-shape
score.

The organizer Appendix later supplied 14 all-causal configurations. They use
`qkv_dim` as `d_model`, mostly D=FFN=128 and L=4, and vary batch, width, heads,
and sequence length. `run_sweep.py --suite official --skip-long` now runs IDs
1-13 directly; `--case-id N` isolates one case. These shapes replace the
earlier five-case integration matrix as the actual optimization target. Dtype
was not specified by the Appendix and must be reported explicitly.

Official ID 2 (`B1/S128/D128/H4/FFN128/L4`, causal, FP16) established a strict
attention-depth boundary on the T4. Packed suffix depth 1 passed 20 randomized
trials and the prepared benchmark input with zero failed elements. Depths 2,
3, and 4 failed 1, 5, and 25 of 344,064 comparisons respectively at
`atol=0.001`, `rtol=0.01`, so they are rejected. In three independent paired
runs, depth 1 measured end-to-end speedups of 1.383x, 1.363x, and 1.335x; its
optimized median beat depth 0 by 1.122x, 1.052x, and 1.038x. One p90 comparison
against depth 0 regressed, but every optimized p90 remained faster than its
paired original baseline. FFN depth remains uncalibrated for this shape, so no
automatic production policy is enabled yet.

`run_sweep.py --suite official --case-id N --calibrate` now performs the
strict evidence workflow for IDs 1-13. It runs exactly three isolated
processes with 20 warmups, 100 repeats, and three rounds; rejects numerical
failures without aborting the remaining sweep; calibrates packed attention
before FFN; and applies the 1.02x median-speedup and paired-p90 gates. Optional
`--policy-json PATH` writes the winning candidate for review. Generated policy
reports are evidence artifacts and must not be committed as production
dispatch until the corresponding T4 run has been reviewed.

The isolated `person3/experiment-narrow-ffn` integration tested Person 2's
official narrow-FFN source through `99fc19e` against ID 2's winning one-layer
packed-attention suffix. All six strict comparisons passed with zero failed
elements. Fast FFN depth 1 did not pass the performance gate: versus native
FFN it won only the first paired median, lost the next two, and its
median-of-process optimized latency was 2.4019 ms versus 2.3764 ms. Process 3
p90 regressed from 2.5807 ms to 4.0306 ms. Retain native FFN for ID 2. The
narrow branch's 1D masked-residual grid remains relevant to ID 6 and other
workloads above 65,535 token rows, but ID 2's all-valid-mask bypass does not
exercise that fix.

Official ID 7 (`B64/S128/D32/H4/FFN32/L4`, causal, FP16) passed strict
correctness at every packed-attention suffix depth. Depth 0 measured 3.7509 ms;
depths 1 through 4 measured 3.7639, 3.7704, 3.7770, and 3.7848 ms. Their
median speedups differed from depth 0 by less than 0.3%, far below the 2%
relative-selection gate, so packed attention is rejected for this shape. The
first calibration incorrectly reported depth 4 because attention selection
used absolute baseline speedup rather than the relative-control gate already
used for FFN. The gate is now applied symmetrically. FFN must be recalibrated
with packed depth 0 before ID 7 receives a final policy.

Official ID 14 is B32/S100000/D1024/H16/FFN1024/L2. In FP16, a full input is
6.10 GiB and input plus output is 12.21 GiB, while explicit scores require
9536.74 GiB before the baseline's FP32 softmax probabilities. The ordinary
harness blocks this case by default on safety grounds. The separate
`bench_shape14_blockwise.py` harness splits the mathematically independent
batch dimension and uses a query-tiled reference whose score working set is
`[batch_block, heads, query_block, seq_len]`; every query still evaluates and
masks all S keys, preserving the reference softmax domain. Its timing is a
sequential blockwise projection and must not be reported as full-batch GPU
utilization. A reduced B2/S1024/D128/H4/L2 T4 validation passed strict
correctness and measured a 17.408x blockwise speedup. At B2/S4096/D1024/H16,
packing QKV in both layers failed one of 4,194,304 output elements with maximum
absolute error 0.0078125, so that plan is rejected before timing. The harness
now exposes `--attention-plan separate-then-packed` to preserve baseline
Q/K/V projection rounding in layer 1 while retaining an SDPA core in both
layers and packed QKV in the final layer. Full ID 14 correctness and
performance remain unmeasured.

The B2/S4096/D1024/H16 `separate-all` follow-up failed the same single element
with the same 0.0078125 maximum absolute error as `packed-all`. This isolates
the remaining drift to the fused SDPA core rather than packed projection
rounding. The next diagnostic is `--attention-plan separate-triton-all`, which
keeps baseline projection arithmetic and uses Person 1's online-softmax Triton
core with FP32 normalization statistics. The organizer requires 20 warmups;
the shape-14 harness now defaults to 20, and reduced warmup counts are
diagnostic-only rather than reportable performance evidence.

`separate-triton-all` also failed at B2/S4096/D1024/H16: two of 4,194,304
elements failed with maximum absolute error 0.0078125. The root cause is now
isolated to attention-core arithmetic. The reference rounds FP16 scores before
FP32 softmax and casts probabilities to FP16 before the value matmul; fused
SDPA and the Triton online-softmax kernel use different intermediate precision
and reduction order. `--attention-plan explicit-tiled-all` is the exact
recovery candidate: it preserves the reference operator order while bounding
score memory by query tiles. `--candidate-query-block` independently tunes the
candidate tile size, defaulting to 64 versus the reference harness's 16.

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

## Isolated Person 3 dispatcher experiment

This worktree is the isolated branch `codex/person3-dispatch-shape14`, based on
the locally available `origin/person3/integrate-person1` ref. It must not push
or modify `main`, `person3/integrate-person1`, or Person 1's branch.

The production-safe policy is setup-time, exact-key dispatch:

1. Unknown or unvalidated keys use native attention.
2. Packed-QKV SDPA is selected automatically only when an exact correctness,
   median-speed, and p90 gate has been recorded for the full key.
3. Triton remains opt-in and is never selected by this dispatcher.
4. Timing is skipped when strict correctness fails.
5. Official Shape #14 is evaluated only by the batch-blocked streaming runner;
   the original full-batch reference harness is blocked before allocation.

`person3_dispatch.py` owns the immutable workload key, measured candidate gate,
and plan selection. `torch_transformer_benchmark.py` performs the selection
before model construction and reports `attention_backend` and the dispatch
reason. `run_sweep.py` covers official IDs 1–14, reports the selected backend,
and marks a hardware memory refusal as `RESOURCE-BLOCKED` rather than timing an
unsafe run.

`bench_shape14_streaming.py` keeps one block on the GPU and repeats it
sequentially for the logical batch. It never runs the explicit baseline at
S=100,000 and does not claim that sequential latency is full-batch throughput.
The reduced exact and opt-in full 100k smoke tests live in
`tests/test_person3_integration.py`.

Local validation on the RTX 4050 Laptop GPU with PyTorch 2.13.0+cu126:

- 21 isolated integration tests passed; the 100k test is opt-in.
- The reduced CUDA Shape #14 exact test and streaming smoke passed.
- A reduced B1/S100000/D64/H1/L1 FP16 run passed with 0.09 GiB peak GPU
  allocation and no `[B,H,S,S]` matrix.
- Official IDs 1–5 and 7–13 passed strict smoke validation with native
  fallback. ID 6 was resource-blocked by the local 4050's full-batch reference
  memory guard; rerun on the T4 or use a blockwise evaluator before making an
  official ID 6 performance claim.

The prevalidated automatic table entries are exact T4 keys on PyTorch 2.11.0,
compute capability 7.5: B8/S512/D512/H8 FP16 non-causal unpadded with a 1.327x
median speedup, official Shape 2 (B1/S128/D128/H4) FP16 causal unpadded with a
1.363x median speedup, official Shape 3 (B4/S128/D128/H4) FP16 causal unpadded with a
1.481x median speedup, official Shape 4 (B16/S128/D128/H4) FP16 causal
unpadded with a 1.448x median speedup, and official Shape 12
(B64/S32/D128/H4) FP16 causal unpadded with a 1.480x median speedup. Each uses
a one-layer packed-SDPA suffix, passed strict correctness and three-process
timing gates, and is not generalized to nearby shapes, padding modes, devices,
or versions.

The later strict 20-trial Shape 2 run superseded an earlier p90-variable
experiment and cleared every production gate. Shape 7
(B64/S128/D32/H4) passed strict correctness at every packed depth, but all
packed candidates missed the 2% relative-improvement gate and remain native.

The updated T4 checkout passed strict correctness for the full official 1–13
smoke sweep. Automatic dispatch selected packed SDPA only for IDs 2, 3, 4, and
12; all other official IDs used native fallback. ID 9 was correctness-safe but
received `SMOKE-REVIEW` because of p90 timing variability. The dedicated
final-mode three-process runs for IDs 3, 4, and 12 also passed. The full-batch
ID 14 harness remains blocked; its validated path is the batch-blocked
streaming evaluator.

Recommended commands:

```text
python run_sweep.py --suite official --skip-long --device cuda --dtype float16 --processes 1 --mode smoke
python bench_shape14_streaming.py --logical-batch 1 --batch-block 1 --seq-len 100000 --d-model 64 --heads 1 --ffn-dim 64 --layers 1 --dtype float16 --device cuda --warmup 0 --repeats 1
set RUN_SHAPE14_100K=1
python -m unittest tests.test_person3_integration -v
```
