# Transformer GPU Optimization Technical Report

## Problem and Solution

The project optimizes the supplied PyTorch Transformer inference benchmark for
an NVIDIA Tesla T4. The final implementation preserves the reference model's
pre-norm residual structure, exact GELU, causal and padding semantics, final
invalid-token zeroing, public forward signature, and elementwise OR correctness
rule.

The organizer-controlled baseline, random input generation, correctness logic,
CUDA-event timing, CLI, and `main()` remain unchanged. The submission replaces
the designated `UserOptimizedTransformer` implementation and customizes weight
transfer for packed parameters.

## Environment

- GPU: NVIDIA Tesla T4, compute capability 7.5, 15,360 MiB.
- Driver observed during final validation: 580.82.07.
- Software: Python, PyTorch 2.11.0+cu128, CUDA 12.8.
- Precision: FP16 inference for reported GPU measurements.
- Runtime: Google Colab Linux/Jupyter; CPU, RAM, and disk allocations varied by
  Colab session and were not treated as fixed optimization targets.

## Profiling and Design

Initial profiling identified separate Q/K/V projections, explicit attention
scores, masking, softmax, copies, and launch overhead as significant costs. The
team used a measured optimization ladder:

1. Replace explicit attention with PyTorch scaled dot-product attention.
2. Pack Q/K/V projections to reduce three linear launches to one.
3. Remove no-op all-valid masking after setup-time validation.
4. Use contiguous token-major FFN inputs while preserving exact GELU.
5. Select optimized layers only for exact shapes that pass correctness, median,
   and p90 gates; otherwise retain the native implementation.
6. Use batch-blocked, memory-efficient attention for the extreme long-sequence
   workload.

Packed weights are transferred by concatenating baseline projection rows in
exact `[Q; K; V]` order. Runtime guards require inference mode, CUDA FP16, the
validated PyTorch/T4 environment, an all-valid mask, and an exact measured
shape. This prevents an optimization validated for one workload from being
silently generalized to another.

## Evaluated Implementations

The production model uses packed-QKV PyTorch SDPA only for measured official
configurations and uses reference-order native attention elsewhere. The final
FFN uses a token-major PyTorch path with exact GELU.

The repository also contains two substantial experimental paths:

- A tiled online-softmax Triton attention implementation with FP32 running
  statistics. It passed selected correctness tests but was less consistently
  fast than packed PyTorch SDPA on the T4, so it remains opt-in.
- An inference-only FFN/LayerNorm custom-operation boundary backed by PyTorch,
  cuBLAS, and a vectorized CUDA residual/mask kernel. Standalone shapes improved,
  but full-model timing and numerical accumulation were not universally stable,
  so it is not selected by the canonical production model.

Retaining these as experiments while excluding them from automatic production
dispatch was a deliberate correctness and reproducibility decision.

## Correctness and Timing Method

For each directly runnable official shape, baseline and optimized models use
identical weights, inputs, masks, dtype, matmul settings, and seeds. The
organizer accepts an output element when either its absolute error is at most
0.002 or its relative error is at most 0.02. Development candidates were first
screened with the stricter 0.001/0.01 rule.

CUDA latency excludes random-data generation and is measured with CUDA events
after warmup. Final internal validation used 20 warmups, 100 repetitions, three
alternating timing rounds, and three independent Python processes per shape.
Candidates that failed correctness were not timed or dispatched.

## Results

### Official IDs 1-13

The canonical T4 FP16 matrix completed 39 of 39 strict process comparisons with
zero failed comparisons. Median-of-process speedups ranged from 1.090x to
1.471x. The unweighted geometric mean across the 13 official shapes was 1.272x.
Official ID 12 was the fastest validated result at 1.471x, while ID 8 was the
lowest at 1.090x. Every shape improved median and paired p90 latency under the
team's acceptance gate.

Automatic dispatch selected a one-layer packed-SDPA suffix for official IDs 2,
3, 4, and 12. The remaining shapes used native attention plus the safe model
layout and all-valid-mask behavior.

Repeated judge-contract ID 2 examples produced 1.424x and 1.522x speedups with
zero failed elements. These examples illustrate normal Colab timing variance;
the report uses multi-process medians rather than selecting the best run.

### Official ID 14

Official ID 14 is B32/S100000/D1024/H16/FFN1024/L2. Its explicit FP16 attention
scores alone require roughly 9.3 TiB, before the baseline's FP32 softmax
probabilities. Therefore the organizer's unchanged dense reference cannot run
this shape on a T4.

The Shape 14 evaluator processes mathematically independent batch blocks and
uses packed SDPA without allocating `[B,H,S,S]`. A reduced
B2/S4096/D1024/H16/L2 comparison passed the organizer's 0.002/0.02 contract and
measured approximately 15.4x speedup. The full logical B32/S100000 optimized
workload completed in approximately 153-161 seconds with about 1.57 GiB peak
GPU allocation. The full-size measurement demonstrates bounded-memory
execution only; it is not reported as a speedup or full-reference correctness
claim.

## AI, Tools, APIs, and Libraries

Development tools included Visual Studio Code, Google Colab, Jupyter, Git,
GitHub, OpenAI Codex, PyTorch profiler, CUDA events, and Python unit tests.
Codex assisted with profiling interpretation, implementation, debugging,
experiment design, regression tests, integration review, and documentation.

APIs and frameworks included Python, PyTorch tensor/neural-network/CUDA APIs,
`scaled_dot_product_attention`, `torch.compile` experiments, the PyTorch
profiler, CUDA event timing, Triton, and PyTorch C++/CUDA custom-operator APIs.
The production benchmark requires only PyTorch.

## Datasets and Assets

No external dataset is used. Validation inputs are deterministic seeded random
tensors generated by the organizer's benchmark. Project assets are the supplied
PyTorch benchmark, the official shape appendix, the problem statement, and the
team-authored source, tests, profiling tools, and reports.

## Limitations and Next Steps

- The automatic policy is intentionally hardware-, version-, dtype-, mask-, and
  shape-specific.
- Full official ID 14 cannot be compared with the supplied dense reference on
  the target GPU.
- Shape 14 currently uses sequential batch blocks.
- Further work should calibrate broader packed-attention depths under the judge
  contract, tune larger Shape 14 blocks, and profile wide/extreme-batch cases
  before introducing additional custom kernels.
