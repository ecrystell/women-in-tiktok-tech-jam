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

Current Person 2 branch: `person2/ffn-layernorm`.

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

`reduce-overhead` and `max-autotune` could not be timed: alternating baseline
and candidate calls access a CUDA-graph output overwritten by a subsequent run.
The exact backend error is retained in the notebook. `max-autotune` also reports
that the T4 has too few SMs for `max_autotune_gemm`. No speedup is claimed for
either mode. The official harness smoke test preserved the public forward
signature and passed two strict trials with zero of 32,768 elements failing.
Results remain shape-dependent and essentially neutral, so there is no blanket
speedup claim and the profiler-evidence gate for a custom Triton fusion remains.

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
