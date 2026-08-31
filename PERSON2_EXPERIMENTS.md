# Person 2 Optimization Experiments

This report contains the detailed FFN, LayerNorm, residual, compiler, and
custom-kernel experiments removed from `AGENTS.md`. The benchmark contract is
unchanged throughout: exact GELU, fixed `[batch, sequence, model]` outputs,
strict `atol=0.001` / `rtol=0.01`, and exact zeroing of invalid token rows.

## Environments

- Laptop: NVIDIA GeForce MX350 (`sm_61`, 2 GiB), Python 3.10, PyTorch
  2.12.1+cu126. TorchInductor compilation is unsupported because its Triton
  backend requires compute capability 7.0 or newer.
- Colab: Tesla T4 (`sm_75`, 14.56 GiB), Python 3.13.15, PyTorch
  2.11.0+cu128, CUDA 12.8, driver 580.82.07.
- Timing uses CUDA events after warmup. Random-data generation, compilation,
  extension builds, and autotuning are outside measured inference latency.

## Reliable PyTorch baseline

Commit `d6bfab0` added the standalone, baseline-weight-compatible
`OptimizedTransformerBlock`, focused unit tests, and `bench_person2_ffn.py`.
It retained exact GELU and a token-major FFN view while leaving
`UserOptimizedTransformer` unchanged for Person 3.

The initial MX350 eager isolated speedups were 0.982x, 1.006x, 0.995x,
1.086x, and 1.069x in sweep order. Four full-block shapes completed; the
`(2,4096,1024)` attention case exceeded 2 GiB. The first T4 validation at
`b4fd2b9` passed all eight tests and all five strict accuracy checks, but its
eager isolated speedups were 0.988x, 1.001x, 1.003x, 1.013x, and 1.000x.

## Compiler and profiler pass

Branch `person2/ffn-fusion-t4` corrected CUDA-graph output ownership, added
step markers, static full-graph compilation, and CUDA-only profiling. Global
compiler-only isolated speedups were:

| Mode | Five T4 speedups | 1.05x gate |
| --- | --- | --- |
| `default` | 0.996x, 1.000x, 1.000x, 1.000x, 1.000x | FAIL |
| `reduce-overhead` | 0.999x, 1.001x, 1.012x, 1.000x, 1.000x | FAIL |
| `max-autotune-no-cudagraphs` | 0.994x, 1.010x, 1.020x, 1.000x, 1.004x | FAIL |

The eager CUDA profile attributed approximately 68-87% of time to the two
GEMMs, 3-7% to LayerNorm, 3-12% to exact GELU, and 3-5% to residual/masking.
The non-GEMM Amdahl bounds ranged from 1.155x to 1.477x, but eliminating only
LayerNorm could not satisfy the former 1.05x target on every shape.

## cuBLASLt residual fusion

Fusing the residual through cuBLASLt `beta=1` reordered FP16 arithmetic and
failed 3 of 65,536 elements (`max_abs=0.0078125`). The strict-valid `beta=0`
version followed the down GEMM with a residual/mask CUDA kernel. Its five eager
speedups were 1.047x, 1.052x, 1.054x, 1.026x, and 1.006x, so it missed the
former 1.05x gate. Algorithm autotuning caused a one-element short-case failure
and did not close the gap. The candidate was rejected.

## Article-inspired GEMM pass

Branch `person2/ffn-gemm-t4` separately measured the generated up and down
workloads. On `(1,128,512)`, the T4 medians were 0.0470 ms for raw up GEMM,
0.0726 ms for up plus exact GELU, 0.1918 ms for raw down GEMM, and 0.1966 ms
for down plus residual. A Triton up-projection with FP32 accumulation and
erf-based GELU was strict-valid but regressed to 0.8925 ms from a 0.4691 ms
baseline. The retained cuBLASLt down path could not compensate, so the
candidate was removed in `ee3ab0d`.

## Layout, split-K, and identity-affine experiments

- `person2/ffn-prepack-t4`: prepacking both operands changed cuBLAS kernel
  selection. The short up product's NN kernel was slower than native TN
  (0.0793 vs 0.0712 ms including exact GELU); down improved only from 0.1915
  to 0.1905 ms. First-shape end-to-end speedup was 1.011x under the old 1.05x
  gate. Rejected.
- Split-K down projection reduced the raw short median from 0.1901 to 0.1311
  ms. FP16 partial reduction changed accumulation order and failed strict
  near-zero cases; FP32 partials passed but regressed to 0.2519 ms. A reverse
  FP16 reduction passed eager seeds but failed one compiled element. Rejected.
- `person2/ffn-ln-identity-t4`: eliding identity LayerNorm affine work passed
  16 T4 tests. Its eager speedups were 1.001x, 1.013x, 1.011x, 1.012x, and
  1.002x. Rejected under the former 1.05x gate and removed from final source.

## Universal 1.01x pass

Branch `person2/ffn-101-t4` tested a guarded FP16 CUDA path with native exact
GELU, a prepacked down operand, combined C++ launch orchestration, and
vectorized residual/mask postprocessing. All experimental suites passed 18
tests, including state transfer, fallbacks, output ownership, masks, and
static full-graph tracing.

The strongest balanced candidate used native up projection, ATen's in-place
exact GELU on its owned temporary, prepacked down projection, and the row-tiled
post kernel for every supplied mask. Three eager processes measured:

| Process | Five isolated speedups | 1.01x result |
| --- | --- | --- |
| 1 | 1.083x, 1.019x, 1.024x, 1.050x, 1.028x | PASS |
| 2 | 1.081x, 1.0104x, 1.018x, 1.049x, 1.036x | PASS |
| 3 | 1.081x, 1.006x, 1.024x, 1.036x, 1.025x | FAIL |

All 15 p90 comparisons improved and all outputs had zero failed elements. The
candidate was rejected only because the every-process threshold was 1.01x.

Additional variants from this pass were rejected:

- A custom erf GELU was slower on both medium shapes (0.998x and 1.000x).
- An in-place residual-output variant produced worse latency and p90.
- Combined native GELU/down/post orchestration improved launch gaps but still
  reached only 1.007x in one medium-shape process before later refinements.
- `reduce-overhead` measured 0.950x, 0.960x, 0.962x, 1.020x, and 1.001x.
- A custom FP32-Welford LayerNorm was correct but measured 0.980x and 0.995x
  on the medium shapes.
- Prepacking the up operand improved the medium shapes but moved the bottleneck
  to 1024-wide cases: one process had an unrounded 1.0095x result, another had
  a 1.006x long-case result, and one short-case p90 regressed.

The source was restored to the reliable implementation in `2a37c5b`; all
experimental commits remain recoverable in branch history.

## NeurIPS 2024 run-length tokenization assessment

Choudhury et al., *Don't Look Twice: Faster Video Transformers with Run-Length
Tokenization*, obtain speedups by detecting temporally repeated video patches,
removing redundant input tokens before inference, and packing variable-length
examples. The useful systems lesson is to eliminate padded or redundant rows
before expensive transformer work rather than masking them after computation.

Direct run-length tokenization is outside Person 2's contract: this benchmark
provides already embedded tokens, requires the exact fixed-shape output for
every position, and does not permit content-dependent token removal or learned
run-length encodings. A bounded compatible experiment may compact only rows
already marked invalid by `valid_token_mask`, run the FFN on valid rows, and
scatter exact zeros back. It must include compaction/scatter overhead, preserve
all unpadded behavior, and be rejected if the dynamic indexing cost outweighs
the saved GEMM work.

Source: https://proceedings.neurips.cc/paper_files/paper/2024/file/3181db351fd3ced43cd589b0b572675d-Paper-Conference.pdf

## Universal 1.005x pass and valid-row compaction

Branch `person2/ffn-1005-t4` restored the strongest balanced candidate from
commit `452e028`. The restored source passed 18 T4 tests. Its first new set of
three independent eager processes produced:

| Process | Five isolated FP16 speedups | Median 1.005x gate |
| --- | --- | --- |
| 1 | 1.082x, 1.009x, 1.028x, 1.051x, 1.050x | PASS |
| 2 | 1.060x, 1.021x, 1.025x, 1.050x, 1.049x | PASS |
| 3 | 1.082x, 1.008x, 1.013x, 1.048x, 1.047x | PASS |

All 15 accuracy checks had zero failed elements. One process-2 short-case p90
was 0.4508 ms versus 0.4468 ms for baseline, so the stronger every-p90 gate
did not pass. The five full-block speedups were 0.944x, 1.021x, 1.023x,
1.006x, and 1.002x. All were strict-correct, but the short case failed the
0.99x median and 2% p90 safety requirements.

The paper's compatible systems lesson was then implemented as valid-row
compaction. Preparation performs the synchronizing `nonzero` and strict
accuracy check outside timed inference. The fast path is guarded by the exact
mask object's identity and tensor version; it gathers valid residual rows,
runs per-row LayerNorm and the exact-GELU FFN, and scatters into an exactly
zeroed fixed-shape result. Unpadded inputs and changed or mutated masks retain
the dense path, while compiled execution deliberately avoids dynamic
compaction.

This candidate passed 19 T4 tests, including CUDA extension execution,
full-graph compile fallback, repeated output ownership, exact invalid zeroing,
mask mutation/replacement guards, FP16, FP32, and the runtime-supported BF16
test. On the padded `(8,512,512,2048)` case it first measured 1.045x. A later
three-process sweep measured 1.048x, 1.040x, and 1.035x on that case, always
with zero failed elements and improved p90.

The same later sweep exposed that the universal claim was not repeatable:

| Process | Five isolated FP16 speedups | Median 1.005x gate |
| --- | --- | --- |
| 1 | 0.980x, 1.037x, 1.048x, 1.054x, 1.053x | FAIL |
| 2 | 1.082x, 1.021x, 1.040x, 1.053x, 1.052x | PASS |
| 3 | 1.081x, 1.029x, 1.035x, 1.052x, 1.051x | PASS |

The failing short process measured baseline at an unusually low 0.2469 ms
and optimized at 0.2519 ms; its p90 also regressed. The result is retained as
real variance rather than discarded as an outlier. Consequently the restored
implementation and useful padded-row enhancement are delivered without a
repeatable universal 1.005x claim.

## All-valid-mask bypass rejection

A final semantic experiment cached a proven all-true mask and selected the
unmasked post kernel only for that exact mask identity and tensor version.
Changed or mutated masks fell back, and compiled execution deliberately used
the dense masked path. Commit `ce762be` was initially reverted by `fd7b8d7`
when the first Colab session exhausted its GPU quota.

The experiment was subsequently rerun on a fresh Tesla T4 session using pinned
`fd7b8d7` control and `ce762be` candidate clones. The candidate passed all 20
tests in 117.786 seconds, including real CUDA extension execution, static
full-graph compile fallback, output ownership, mask guards, exact zeroing,
FP16, FP32, and runtime-supported BF16.

The initial screen was strict-correct and measured 1.084x on the short shape
and 1.037x on the medium unpadded shape. Its absolute short optimized median
was 0.2369 ms versus 0.2520 ms for the pinned control in the preceding run.
The required independent-process sweep did not reproduce that result:

| Process | Five isolated FP16 speedups | Median 1.005x gate |
| --- | --- | --- |
| 1 | 0.960x, 1.028x, 1.046x, 1.053x, 1.052x | FAIL |
| 2 | 0.984x, 1.032x, 1.055x, 1.053x, 1.052x | FAIL |
| 3 | 1.086x, 1.032x, 1.034x, 1.045x, 1.051x | FAIL (p90) |

All 15 outputs had zero failed elements. Process 2 short p90 regressed from
0.2744 to 0.3852 ms, and process 3 short p90 regressed from 0.2747 to
0.3668 ms. The candidate therefore failed both the median and p90 gates.

The reduced-sampling full-block sweep remained strict-correct and measured
0.992x, 0.998x, 1.010x, 1.007x, and 1.004x. A primary-sampling rerun of the
short full block measured 0.969x, with p90 increasing from 0.8045 to
0.8563 ms (6.4%). Static `default` compilation stayed correct but measured
0.888x on the short isolated case. The official harness smoke test passed with
zero error; its unchanged `UserOptimizedTransformer` measured 0.918x in that
noisy small-shape run and remains Person 3's responsibility.

The all-valid-mask bypass remains reverted at source commit `fd7b8d7`. It is
rejected because its saved launch/mask work is smaller than short-shape
variance and does not satisfy repeatable isolated or full-block safety gates.

## Task 2 full-op orchestration pass

Branch `person2/ffn-fullop-t4` revisited the fused-FFN assignment without
replacing either cuBLAS GEMM. Source commit `16cf403` places identity-affine
LayerNorm, the native up projection, exact in-place ATen GELU, the native down
projection, and the existing vectorized residual/mask post kernel behind one
inference-only C++ custom-op boundary. This preserves operation order and
removes Python and custom-op transitions between the stages. Nonidentity
LayerNorm parameters and unsupported environments retain the native fallback.

On Tesla T4 (`sm_75`, 15,360 MiB), driver 580.82.07, PyTorch 2.11.0+cu128,
and CUDA 12.8, the source checkpoint passed all 19 tests. This included actual
CUDA extension execution, strict state loading, fake/meta tracing, static
full-graph compilation, masks, repeated output ownership, exact invalid
zeroing, FP16, FP32 fallback, and runtime-supported BF16 fallback.
Final head `c319ab4` added explicit real-CUDA masked and unmasked full-op
coverage and passed all 20 tests in 8.846 seconds with the extension cached.

The five-shape isolated FP16 gate used seed 1234, `atol=0.001`, `rtol=0.01`,
20 warmups, 100 CUDA-event repetitions, five alternating rounds, and three
independent processes:

| Process | Five isolated speedups | Lowest | p90 / accuracy |
| --- | --- | --- | --- |
| 1 | 1.184x, 1.023x, 1.052x, 1.035x, 1.053x | 1.023x | all improved / zero failures |
| 2 | 1.086x, 1.026x, 1.047x, 1.053x, 1.038x | 1.026x | all improved / zero failures |
| 3 | 1.084x, 1.023x, 1.034x, 1.051x, 1.051x | 1.023x | all improved / zero failures |

The repeatable universal 1.005x isolated gate therefore passed. The smallest
margin was still 1.023x, and all 15 optimized p90 measurements were lower.

The reduced-sampling full-block sweep measured 1.004x, 0.858x, 1.004x,
1.008x, and 1.001x. The medium unpadded result was inconsistent with its
isolated result and failed the 0.99x safety threshold; the short p90 was also
noisy. Primary-sampling repeats were therefore run instead of discarding the
observation. The short full block measured 1.037x with p90 improving from
1.2606 to 1.1983 ms. Three independent primary-sampling medium full-block
processes measured 1.016x, 1.011x, and 1.013x, each with lower p90 and zero
failed elements. The 0.858x observation remains reported as sampling variance,
so the full-block result carries this caveat even though primary repeats pass.

Static `default` compilation remained strict-correct on the short isolated
case and measured 1.035x. The official harness smoke test retained the
unchanged `UserOptimizedTransformer.forward`, passed with zero error, and
measured 1.013x in that run. The algebraic identity-LayerNorm/up-projection
reformulation was also screened: it had zero strict failures on short and
medium shapes, but was not promoted because it changes floating-point
evaluation and would require a new fused statistics/correction kernel. It
remains a separately gated future experiment rather than part of this result.

## Official narrow-shape retuning

Branch `person2/official-narrow-ffn-t4` retuned the validated full-op control
for the nine unique FFN workloads represented by official configurations
1–13. Configuration 14 was excluded from the T4 performance gate because its
activation alone is impractical on the target GPU and its dominant
long-sequence attention cost is outside Person 2 ownership. No attention,
causal-mask, `UserOptimizedTransformer`, or dispatch code was changed.

The benchmark now supports `--suite official`, original official case IDs,
explicit backend selection, process identifiers, and JSON metrics. Duplicate
attention configurations share an isolated FFN result, while full-block checks
retain their original batch, sequence, head, causal, and mask settings. The
profiler separately reports LayerNorm, both GEMMs, exact GELU, residual/mask,
approximate launch count, and CPU enqueue gaps.

### CUDA launch-limit correction

The first T4 screen exposed an implementation defect at the two largest token
counts: the masked half2/half8 post kernels placed one token row in `grid.y`,
which exceeds CUDA's 65,535 `grid.y` limit. Source commit `99fc19e` replaces
that launch with a 1D vector grid and derives the row from the linear vector
index. This preserves FP16 residual-add order and mask semantics while making
the 65,536- and 1,280,000-row workloads legal. A real-CUDA regression test now
covers more than 65,535 rows.

The corrected source passed all 23 tests in 105.922 seconds on Tesla T4
(`sm_75`, 15,360 MiB), driver 580.82.07, PyTorch 2.11.0+cu128, and CUDA 12.8.
Coverage includes strict state loading, unchanged signatures, all-valid and
padded masks, exact invalid zeroing, non-contiguous fallback, parameter
invalidation, repeat ownership, FP16, FP32/BF16 fallback, static compilation,
CUDA-graph lifecycle, and actual CUDA extension execution.

### Isolated eager results

Each process used FP16, seed 1234, `atol=0.001`, `rtol=0.01`, 20 warmups,
100 CUDA-event repetitions, and five alternating rounds. All 27 accuracy
comparisons had zero failed elements.

| Official IDs | `(M,D,FFN)` | Process speedups | p90 gate | Recommendation |
| --- | --- | --- | --- | --- |
| 2 | `(128,128,128)` | 1.215x / 1.210x / 1.261x | PASS | eager full-op |
| 3 | `(512,128,128)` | 1.228x / 1.230x / 1.235x | PASS | eager full-op |
| 4, 12 | `(2048,128,128)` | 1.222x / 1.286x / 1.210x | FAIL in processes 1 and 3 | native |
| 1, 9, 10, 11 | `(8192,128,128)` | 1.129x / 1.128x / 1.235x | PASS | eager full-op |
| 5 | `(16384,128,128)` | 1.270x / 1.278x / 1.260x | PASS | eager full-op |
| 13 | `(65536,128,128)` | 1.249x / 1.245x / 1.248x | PASS | eager full-op |
| 6 | `(1280000,128,128)` | 1.271x / 1.267x / 1.261x | PASS | eager full-op |
| 7 | `(8192,32,32)` | 1.219x / 1.250x / 1.164x | PASS | eager full-op |
| 8 | `(8192,1024,1024)` | 1.141x / 1.135x / 1.141x | PASS | eager full-op |

For the rejected 2,048-row tuple, process 1 p90 increased from 0.221936 to
0.292051 ms and process 3 p90 increased from 0.236006 to 0.290906 ms. Its
medians clear 1.005x, but the acceptance contract requires every process to
clear both median and p90 gates.

### Profiles and backend screens

The full-op reduced the profiled CUDA launch count from approximately eight to
five. Across the official suite, baseline GEMM share ranged from 13.46% to
63.77%, LayerNorm from 9.53% to 68.60%, GELU from 3.40% to 8.85%, and
residual/mask from 6.44% to 21.66%. Representative baseline-to-full-op CPU
enqueue gaps were 0.2233 to 0.1814 ms at `(128,128,128)`, 0.2312 to 0.1585 ms
at `(512,128,128)`, and 0.2393 to 0.1509 ms at `(8192,32,32)`. Larger cases
were increasingly GEMM-bound but still benefited from the reduced residual
stage and launch count.

Symmetric `default` and `reduce-overhead` compilation failed the isolated
screen on both the smallest and unstable tuples. `max-autotune-no-cudagraphs`
also failed the smallest tuple, but the 2,048-row tuple measured 1.060x,
1.054x, and 1.113x with passing p90. That rescue was rejected because compiled
full-block checks for official IDs 4 and 12 measured only 0.909x and 0.896x.

Fixed-shape CUDA-graph core replay measured 1.136x for the 8,192-row tuple and
10.423x for the 128-row tuple, with zero numerical failures. It was not
eligible: replay reused and overwrote the same output storage, so preserving
previous outputs would require a timed clone or copy. Whole-model graph capture
and ownership management remain Person 3 responsibilities.

### Causal full-block safety and final allowlist

Strict eager full-block speedups for official configurations 1–13 were:
1.011x, 1.003x, 0.992x, 0.911x, 1.033x, 1.037x, 1.009x, 1.025x,
0.994x, 1.017x, 1.010x, 1.005x, and 1.007x. Every output had zero failed
elements. Configuration 4 failed the 0.99x median gate and increased p90 by
about 9.6%; the other configurations passed both full-block safety limits.
Because configurations 4 and 12 share the Person 2 dispatch key
`(2048,128,128,FP16,all-valid,eager)`, the tuple is conservatively excluded.

The measured eager FP16 all-valid allowlist is therefore:

- `(128,128,128)`
- `(512,128,128)`
- `(8192,32,32)`
- `(8192,128,128)`
- `(8192,1024,1024)`
- `(16384,128,128)`
- `(65536,128,128)`
- `(1280000,128,128)`

Use native execution for `(2048,128,128)`, padded or unsupported masks,
FP32, BF16, non-contiguous inputs, nonidentity LayerNorm parameters, compile
fallbacks, or failed extension/preflight cases. FP32 and runtime-supported
BF16 smoke checks passed through the native fallback with zero error. The
official harness signature/accuracy smoke passed two of two trials with zero
failed elements. No runtime dispatch was added; Person 3 must revalidate this
allowlist in the integrated official harness before consuming it.

## Algebraic LayerNorm/GEMM correction experiment

Branch `person2/ln-gemm-correction-t4` tested a directives-derived algebraic
reformulation for identity-affine LayerNorm. For an up-projection weight
matrix `W`, it computes the native FP16 raw GEMM first and applies
`(xW - mean(x) * row_sum(W)) * rstd(x)` in a CUDA epilogue before adding the
bias and evaluating exact erf GELU. The linear result is explicitly rounded
to FP16 before GELU, LayerNorm mean/variance remain FP32, both native cuBLAS
GEMMs are retained, and the validated residual/mask post kernel remains the
last stage. The experiment is inference-only and guarded by strict
intermediate and final numerical preflights; it is disabled by default.

Source commit `d4d247f` introduced the custom operation. A pure warp-per-row
schedule at `9eccbba` reduced the correction grid by eight times and removed
block-wide synchronization, but paired testing showed that one topology was
not suitable for every dimension. Final experimental commit `7eaee3e` selects
the CUDA reduction topology inside the kernel wrapper: block-per-row for
moderate `D=128` workloads and warp-per-row for `D=32` or at least 65,536
rows. This is kernel launch tuning, not `UserOptimizedTransformer` dispatch.

Commit `7fa364c` added `--control-backend full-op` so the candidate and the
validated full-op run are prepared, alternated, and timed in the same process.
Earlier cross-process subtraction was discarded after it produced a spurious
`0.644x` point from process-level Colab variance.

On Tesla T4 (`sm_75`, 15,360 MiB), driver 580.82.07, PyTorch 2.11.0+cu128,
and CUDA 12.8, the hybrid source passed all 27 Person 2 tests in 102.049
seconds. Coverage included real extension execution, strict state loading,
parameter invalidation, exact invalid-token zeroing, repeated output
ownership, non-contiguous and nonidentity fallbacks, FP16, FP32, runtime BF16,
static compile fallback, and the alternating full-op control.

Paired FP16 timing used seed 1234, `atol=0.001`, `rtol=0.01`, 20 warmups,
100 CUDA-event repetitions, five alternating rounds, and three independent
processes. Across the 21 `D=128` comparisons, every candidate median improved:
the range was `1.0166x` to `1.5263x`, with zero failed elements. Twenty of 21
p90 comparisons improved; the exception was `(M,D,FFN)=(2048,128,128)` in
process 1, where p90 increased from 0.207664 to 0.245328 ms despite a 1.0619x
median speedup. The `D=32` tuple measured 1.0974x, 1.1492x, and 0.9379x and
therefore was not repeatable. The `D=1024` screen measured 0.9455x with a p90
regression and was also rejected.

The experiment does not pass the project-wide strict gate, so no full-block
speedup claim or production allowlist is made and `AGENTS.md` is unchanged.
The validated full-op implementation remains the best Person 2 handoff. The
algebraic path and its paired benchmark remain on the experimental branch as
evidence that the launch topology is promising for `D=128`, but Person 3 must
not dispatch to it without a fresh integrated gate.
