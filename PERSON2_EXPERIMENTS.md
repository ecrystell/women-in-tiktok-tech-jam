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
