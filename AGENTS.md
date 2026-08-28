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
- The script defaults to `atol=0.001` and `rtol=0.01`; use these stricter
  values for local validation even if external instructions allow looser ones.
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

Own the standalone `FastSelfAttention` implementation:

- packed QKV projection;
- SDPA attention as the guaranteed path;
- causal and padding masks;
- zeroing invalid query tokens;
- baseline-compatible weight transfer;
- attention accuracy and latency measurements.

Custom Triton FlashAttention with online softmax is a stretch path only. FP32
should be used for numerically sensitive online-softmax statistics; do not
force every matrix operation to FP32 without measurements.

Current Person 1 branch: `person1/attention-sdpa`.

Current Person 1 commit: `85cfc14 Add packed QKV SDPA attention module`.

The current implementation adds `FastSelfAttention` but intentionally does not
modify `UserOptimizedTransformer` integration. Person 3 owns final assembly.

### Person 2 — FFN, LayerNorm, and residuals

Own FFN and normalization optimization. Start with `torch.compile`, exact
`F.gelu(..., approximate="none")`, contiguous layouts, and residual/LayerNorm
fusion. Do not assume that a hand-written single kernel for both large Linear
GEMMs is faster than optimized PyTorch/cuBLAS. Custom Triton is a stretch path.

Current Person 2 branch: `person2/ffn-ln-identity-t4`.

Current Person 2 core commit: `d6bfab0 Add standalone Person 2 FFN optimization`.

The standalone, baseline-weight-compatible `OptimizedTransformerBlock`, focused
unit tests, and `bench_person2_ffn.py` are implemented. Final
`UserOptimizedTransformer` assembly remains owned by Person 3. The implementation
uses exact GELU and a token-major FFN view; no custom Triton kernel was added.

The newer-GPU validation target was commit `b4fd2b9`. It ran in Colab with
Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, driver 580.82.07, and a Tesla
T4 (`sm_75`, 14.56 GiB). The reproducible notebook is named
`Person2_T4_Validation_b4fd2b9.ipynb`. It uses `atol=0.001`, `rtol=0.01`, seed
1234, CUDA-event timing, and excludes compile/autotune setup through warmup.
All eight unit tests passed, including strict weight transfer, signatures,
non-contiguous inputs, masks, causality, FP32, FP16, and runtime-supported BF16.

All five isolated FP16 shapes passed with `max_abs=0`, `max_rel=0`, and zero
failed elements. Each entry below is baseline median/p90 to optimized
median/p90, followed by optimized throughput and median speedup. Sampling used
20 warmups, 100 repetitions, and three alternating rounds.

| Shape in `run_sweep.py` order | Mode | Latency (ms) | Optimized token/s | Speedup |
| --- | --- | ---: | ---: | ---: |
| `(1, 128, 512)` | eager | 0.2273/0.2632 to 0.2300/0.2724 | 556,599 | 0.988x |
| `(8, 512, 512)` | eager | 1.0848/1.0998 to 1.0834/1.0875 | 3,780,718 | 1.001x |
| `(8, 512, 512)`, causal, 20% padding | eager | 1.0732/1.1427 to 1.0701/1.0834 | 3,827,579 | 1.003x |
| `(4, 2048, 1024)` | eager | 7.3932/7.4356 to 7.2991/7.4339 | 1,122,330 | 1.013x |
| `(2, 4096, 1024)`, causal | eager | 7.4342/7.5763 to 7.4341/7.5756 | 1,101,943 | 1.000x |
| `(1, 128, 512)` | `default` | 0.4050/0.4657 to 0.4322/0.4956 | 296,132 | 0.937x |
| `(8, 512, 512)` | `default` | 1.0035/1.2742 to 0.9899/1.0042 | 4,137,909 | 1.014x |
| `(8, 512, 512)`, causal, 20% padding | `default` | 0.9894/1.1166 to 0.9900/0.9933 | 4,137,374 | 0.999x |
| `(4, 2048, 1024)` | `default` | 7.2508/7.4015 to 7.2505/7.4014 | 1,129,856 | 1.000x |
| `(2, 4096, 1024)`, causal | `default` | 7.4010/7.5558 to 7.4015/7.5581 | 1,106,807 | 1.000x |

The five full-block FP16 eager cases also had zero error. With five warmups,
ten repetitions, and three rounds, their median speedups were 0.991x, 0.965x,
1.001x, 1.014x, and 0.998x in sweep order. The former MX350 OOM case
`(2, 4096, 1024)` completed on the T4 at 121.2859/121.8447 ms baseline versus
121.4752/121.8742 ms optimized (median/p90, 67,438 optimized token/s).

Representative causal, 20%-padded `(8, 512, 512)` checks also passed with zero
error. FP32 measured 1.000x in eager and `default`; BF16 measured 1.000x eager.
Although the runtime reports BF16 support, TorchInductor warns that the T4 does
not support native BF16 compilation and skips that compilation, so the observed
0.997x `default` result is not claimed as compiled performance.

The universal-speedup follow-up branch is `person2/ffn-fusion-t4`. Commit
`76a1d73` added static full-graph compilation, CUDA-graph output ownership and
step markers, `max-autotune-no-cudagraphs`, focused tests, and the T4 profiler.
Commit `181521e` corrected the profiler to count CUDA-only self time. The
expanded local/T4 unit suite has 12 tests. `reduce-overhead` now executes all
five shapes correctly instead of failing on overwritten CUDA-graph outputs.

With 20 warmups, 100 repetitions, and five alternating rounds, compiler-only
median speedups in sweep order were:

| Global mode | Five isolated FP16 speedups | Universal 1.05x |
| --- | --- | --- |
| `default` | 0.996x, 1.000x, 1.000x, 1.000x, 1.000x | FAIL |
| `reduce-overhead` | 0.999x, 1.001x, 1.012x, 1.000x, 1.000x | FAIL |
| `max-autotune-no-cudagraphs` | 0.994x, 1.010x, 1.020x, 1.000x, 1.004x | FAIL |

The corrected eager baseline profile reported the following CUDA-time shares.
The Amdahl bound assumes every non-GEMM operation could be eliminated, so it is
an upper bound rather than an expected result.

| Shape | GEMM | LayerNorm | exact GELU | residual/mask | non-GEMM | Amdahl bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `(1, 128, 512)` | 86.59% | 3.58% | 2.65% | 3.91% | 13.41% | 1.155x |
| `(8, 512, 512)` | 68.07% | 6.90% | 12.02% | 5.47% | 31.93% | 1.469x |
| `(8, 512, 512)`, causal, padded | 67.71% | 6.93% | 12.27% | 5.44% | 32.29% | 1.477x |
| `(4, 2048, 1024)` | 82.18% | 3.28% | 6.92% | 3.28% | 17.82% | 1.217x |
| `(2, 4096, 1024)`, causal | 82.48% | 3.28% | 6.75% | 3.24% | 17.52% | 1.212x |

A bounded CUDA/cuBLASLt experiment was attempted and rejected. Fusing the
residual through cuBLASLt `beta=1` reordered FP16 arithmetic and failed 3 of
65,536 elements (`max_abs=0.0078125`). Preserving baseline order by fusing only
the residual add and final mask produced exact-zero error, but its five eager
speedups were only 1.047x, 1.052x, 1.054x, 1.026x, and 1.006x. One-time
cuBLASLt algorithm autotuning introduced a one-element strict failure on the
short case and still did not make every other case 1.05x. Even eliminating all
measured LayerNorm time would cap the worst strict-valid case near 1.04x, so the
LayerNorm extension gate was closed. The rejected extension is absent from the
final tree; its experimental commits remain in branch history.

The official harness smoke test preserved the public forward signature and
passed two strict trials with zero of 32,768 elements failing. No universal
Person 2 optimization met the required 1.05x threshold, so the standalone
PyTorch block remains the accepted implementation and no kernel speedup is
claimed. Person 3 may use the measured table for dispatch, but Person 2 does not
add shape dispatch or change `UserOptimizedTransformer`.

The article-inspired GEMM follow-up branch is `person2/ffn-gemm-t4`, created
from `ff54b21`. Commit `0d2d2f5` adds the retained, standalone up/down GEMM
profiler. Candidate commit `bb62a8d` was evaluated on the same Colab Tesla T4
with PyTorch 2.11.0+cu128 and CUDA 12.8. All 19 candidate tests passed on the
T4, including multi-seed exact-GELU equivalence, custom-op fallbacks, repeated
calls, masks, compilation tracing, and output ownership.

The article's useful methodology was applied by timing each concrete workload
independently before an end-to-end sweep. For the first and most
launch-sensitive isolated FP16 shape, `(1, 128, 512)`, the generated workloads
were up `[128,512] x [512,2048]` and down `[128,2048] x [2048,512]`. Baseline
median/p90 measurements were 0.0470/0.0533 ms for the raw up GEMM,
0.0726/0.0821 ms for up plus exact GELU, 0.1918/0.1925 ms for the raw down GEMM,
and 0.1966/0.1970 ms for down plus residual.

The Triton FP16 tensor-core up projection used FP32 accumulation, an explicit
FP16 rounding boundary, and FP32 erf-based exact GELU. It passed the strict
accuracy contract but measured 0.676896/0.681984 ms versus
0.075216/0.103744 ms for PyTorch, about nine times slower. The strict-order
cuBLASLt down projection plus residual/mask measured 0.198496/0.198656 ms versus
0.196240/0.196608 ms for PyTorch. Of eight returned cuBLASLt algorithms, only
algorithms 0 and 7 passed multi-seed strict validation; algorithm 0 measured
0.1981/0.1986 ms and algorithm 7 measured 0.2158/0.2166 ms. Algorithms 1-6 each
failed one element with `max_abs=0.00195312`.

The combined isolated candidate retained zero failed elements
(`max_abs=0.00195312`) but regressed from 0.4691/0.5110 ms to
0.8925/1.0622 ms, a 0.526x median speedup. This fails both the universal 1.05x
median gate and the no-worse-p90 gate on the first required shape. Because both
experimental component replacements were already slower than PyTorch on that
shape, no global candidate could satisfy the every-shape requirement. Per the
early-stop rule, the remaining multi-process/full-sweep timing was not run and
no speedup is claimed.

Commit `ee3ab0d` removes the rejected Triton/cuBLASLt candidate, build files,
and candidate-only tests while retaining the reusable per-GEMM profiler and
the reliable standalone PyTorch block. The retained 13-test suite passes both
locally and on the T4 across CPU and available CUDA FP32, FP16, and BF16. Final
T4 strict-zero-error eager smokes measured 0.999x for FP32 and 1.000x for BF16.
The official one-layer FP16 harness passed two trials with zero of 131,072
elements failing; its reduced-sample median was 1.026x and is a safety smoke,
not a performance claim. The preserved Colab artifact is named
`Person2_T4_Article_GEMM_Gate_ee3ab0d.ipynb`. FP8/FP4 and integer quantization
remain out of scope because they do not satisfy the T4 and strict FP16
correctness contract. `UserOptimizedTransformer` and Person 3 dispatch remain
unchanged.

The prepacked-layout branch `person2/ffn-prepack-t4` changed cuBLAS kernels on
both GPUs but did not clear the T4 gate: its first isolated shape measured only
1.011x. The prepacked up product selected an NN kernel but was slower than the
native TN kernel (0.0793 versus 0.0712 ms including exact GELU); the prepacked
down product improved only from 0.1915 to 0.1905 ms. The candidate is rejected.

A split-K down-projection reduced the raw short-shape median from 0.1901 to
0.1311 ms (about 31%). FP16 partial reduction changed accumulation order and
caused strict near-zero failures on multiple seeds; FP32 partials passed but
regressed to 0.2519 ms. A reverse FP16 reduction order passed seven eager seed
checks, but the equally compiled comparison still failed one element and
measured 0.990x. Split-K is rejected.

The final bounded experiment on `person2/ffn-ln-identity-t4` elided LayerNorm's
affine work only when scale was exactly one and bias exactly zero, with native
fallback for non-identity weights and unsupported modes. Candidate commit
`a855739` passed all 16 T4 tests, including fullgraph compilation and randomized
non-identity fallback. One fresh-process isolated FP16 sweep used 20 warmups,
100 repetitions, and five alternating rounds. Every shape had zero error and
its p90 improved, but median speedups were only 1.001x, 1.013x, 1.011x, 1.012x,
and 1.002x in sweep order. The universal 1.05x gate therefore failed in the
first process; the remaining two processes and full-block gate were not run.
The identity-affine candidate is removed from the final source tree, and no
speedup is claimed.

### Person 3 — integration, profiling, and dispatch

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
- `FastSelfAttention` must expose the same forward signature as the baseline.
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
