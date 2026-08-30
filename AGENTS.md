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

Current best official-shape branch: `person2/official-narrow-ffn-t4`.
Validated source commit: `99fc19e`. It preserves the standalone
`OptimizedTransformerBlock`, exact GELU, FP32 LayerNorm statistics, state-dict
layout, and native fallbacks. `UserOptimizedTransformer`, attention, causal
masking, and Person 3 dispatch remain unchanged.

The official retuning found and fixed a CUDA launch-limit defect in the masked
residual kernel: using one grid row per token failed above 65,535 rows. The
corrected 1D vector grid supports the official 65,536- and 1,280,000-token FFN
workloads and passed all 23 CPU/CUDA tests on the Colab Tesla T4 (`sm_75`),
PyTorch 2.11.0+cu128, and CUDA 12.8.

The eager full-op is the measured recommendation for eight official FP16,
all-valid FFN tuples. Their worst three-process isolated speedup was 1.128x;
all eligible p90 measurements improved and every accuracy comparison had zero
failed elements. The `(M,D,FFN)=(2048,128,128)` tuple is excluded: two isolated
p90 measurements regressed and official configuration 4 measured 0.911x in
the causal full-block safety run. Recommend native execution for that tuple.
This is a measured allowlist, not a universal speedup or runtime dispatch.

Compiled modes did not provide a safe global replacement. A
`max-autotune-no-cudagraphs` screen improved the excluded isolated tuple but
regressed its full blocks to 0.909x and 0.896x. CUDA-graph replay reduced core
latency but reused output storage, so it failed repeated-output ownership and
was rejected. Person 3 may consume the exact allowlist after integrated
official-harness validation. Full profiles, per-process measurements, and
rejected variants are in `PERSON2_EXPERIMENTS.md`.

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
