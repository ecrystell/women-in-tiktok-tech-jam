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
