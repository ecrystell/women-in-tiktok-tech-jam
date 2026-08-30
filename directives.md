# Project Optimization Directives

## Purpose

This file converts the project brief, supplied research papers, and supplied
framework documentation into engineering rules for the entire Transformer
optimization project. It applies to attention, FFN/normalization, model
integration, dispatch, profiling, validation, and reporting.

The sources provide hypotheses and techniques, not automatic performance
claims. The repository benchmark and measurements on the target GPU remain the
authority for correctness and speed.

## Source register

### Primary project and research sources

1. **Hackathon problem statement, "Implement a GPU Kernel for a Transformer
   Layer"** (user-supplied brief, updated 27 August 2026). This defines the
   fixed Transformer workload, correctness requirement, varied test shapes,
   hardware-specific optimization scope, and required technical report.
2. **NVIDIA, ["How to Optimize Transformer-Based Models for Low-Precision
   Training"](https://developer.nvidia.com/blog/how-to-optimize-transformer-based-models-for-low-precision-training/).**
   This motivates deriving the exact `M x K x N` GEMMs from model shapes,
   benchmarking each projection separately, and distinguishing realistic
   end-to-end cost from kernel-only cost.
3. **Choudhury et al., ["Don't Look Twice: Faster Video Transformers with
   Run-Length Tokenization"](https://proceedings.neurips.cc/paper_files/paper/2024/file/3181db351fd3ced43cd589b0b572675d-Paper-Conference.pdf),
   NeurIPS 2024.** This motivates avoiding work on redundant tokens and keeping
   token-reduction overhead small enough to preserve wall-clock gains.
4. **Mukherjee and Cha, ["GPU-Accelerated Optimization of Transformer-Based
   Neural Networks for Real-Time Inference"](https://arxiv.org/pdf/2603.28708).**
   This motivates hybrid precision, FP32 treatment of numerically sensitive
   reductions, graph-level optimization, and systematic latency/throughput/
   accuracy measurement. Its reported results are from other models and GPUs,
   so they are not performance evidence for this project.
5. **NVIDIA Transformer Engine,
   ["Performance Optimizations"](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/advanced_optimizations.html).**
   This supplies general lessons about fused operations, avoiding repeated
   conversions, layout-aware execution, and caching reusable representations.
   The duplicated link supplied in the project discussion is one source.

### Supplied implementation references

- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile)
- [PyTorch CUDA-graph step marker](https://docs.pytorch.org/docs/main/generated/torch.compiler.cudagraph_mark_step_begin.html)
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [NVIDIA cuBLAS/cuBLASLt](https://docs.nvidia.com/cuda/cublas/index.html)
- [Triton exact `erf`](https://triton-lang.org/main/python-api/generated/triton.language.erf.html)
- [TensorRT exact `GELU_ERF`](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/namespacenvinfer1.html)
- [TensorRT performance and fusion guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/optimization.html)
- [Transformer Engine platform guidance](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/)
- [NVIDIA T4 specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [PyTorch release installation matrix](https://pytorch.org/get-started/previous-versions/)
- [PyTorch CUDA architecture policy](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)

## 1. Non-negotiable correctness contract

Correctness takes precedence over speed for every subsystem and backend.

- Do not change `UserOptimizedTransformer.forward`, input/output shapes, public
  constructor behavior, parameter names, or state-dict layout.
- Preserve the baseline's pre-norm residual order, causal behavior, padding
  semantics, and final zeroing of invalid query tokens.
- Preserve exact GELU: `F.gelu(..., approximate="none")`, or an equivalent
  erf-based expression that passes the elementwise benchmark. Do not substitute
  a tanh GELU epilogue.
- Keep numerically sensitive statistics in FP32 where the baseline does:
  attention softmax input and LayerNorm statistics. Lower precision may be used
  for eligible linear algebra only after strict validation.
- Use the repository's stricter comparison contract: `atol=0.001`,
  `rtol=0.01`, with each element passing when its absolute error satisfies the
  absolute **or** relative threshold.
- Reference and candidate must use identical dtype, TF32 settings, matmul
  precision, model weights, inputs, masks, and random seed.
- A candidate that fails correctness is not timed or dispatched. Unsupported
  devices, dtypes, layouts, masks, training mode, compilation, build failures,
  and numerical preflight failures must use a correct native fallback.

### 1.1 Official appendix test shapes

The organizer appendix defines the required workload matrix below. `QKV Dim`
maps to the benchmark's `d_model` argument. The appendix specifies `causal =
TRUE` for every case but does not specify a dtype or padding ratio; run and
report the matrix explicitly for each supported dtype, using an all-valid mask
when no padding ratio is otherwise supplied.

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

Official IDs 1–13 are direct full-model cases. ID 14 is not an ordinary
baseline timing case: its explicit score tensor has
`32 * 16 * 100000^2 = 5.12e12` elements, requiring about 9.31 TiB in FP16 or
18.63 TiB in FP32 before softmax and other tensors. Guard it in the ordinary
harness and validate it only through a mathematically equivalent tiled,
batch-blocked harness. Such timing must be labeled sequential blockwise
projection rather than full-batch GPU utilization.

The official matrix is dominated by narrow, launch-sensitive configurations:
most cases have `d_model = ffn_dim = 128`, while head dimensions range across
8, 32, 64, 128, and 256. Triton head-dimension eligibility must not restrict
PyTorch SDPA eligibility. Benchmark packed QKV depth per shape because
rounding drift can accumulate across the four causal layers.

## 2. Target-hardware policy

The acceptance GPU is the Tesla T4 (`sm_75`). Optimize for what that GPU and the
installed runtime actually support.

- FP16 Tensor Core GEMMs with FP32 accumulation are the primary mixed-precision
  path.
- FP8, MXFP8, and NVFP4 results from Hopper, Ada, or Blackwell are inspiration,
  not available T4 strategies. Do not add Transformer Engine or a quantized
  format as a required dependency.
- BF16 runs only when the runtime reports hardware support. Otherwise report it
  as unsupported; do not emulate it and call the result a native benchmark.
- Integer quantization is outside the default path because it is unlikely to
  satisfy strict elementwise equivalence without a separate calibration and
  acceptance case.
- Never transfer a speedup claim from B300, H100, A100, RTX 3090, or a CPU
  comparison to the T4. Re-measure the exact repository shapes on the T4.

## 3. Shared optimization ladder

Use this order for project decisions:

1. Establish strict baseline correctness and CUDA-event latency for every test
   configuration.
2. Profile CUDA activity and derive the exact workloads before changing code.
3. Prefer library kernels and remove unnecessary launches, copies, conversions,
   and materialized intermediates around them.
4. Test `torch.compile` with fixed, documented lifecycle rules.
5. Add guarded custom Triton/CUDA or optional TensorRT only for a measured
   bottleneck with an Amdahl bound large enough to matter.
6. Add dispatch only from repeatable per-configuration measurements.
7. Integrate subsystems and re-run full-model gates; isolated wins do not imply
   end-to-end wins.
8. Retain the simplest validated implementation when a candidate does not
   clear its gate.

## 4. Workload derivation and profiling

Apply the NVIDIA GEMM article's shape-first method to inference.

- Flatten tokens as `M = batch_size * seq_len` when the memory layout permits.
- Record these forward GEMMs independently for each supplied configuration:
  packed QKV `[M,D] x [D,3D]`, attention output `[M,D] x [D,D]`, FFN up
  `[M,D] x [D,FFN]`, and FFN down `[M,FFN] x [FFN,D]`.
- For each GEMM, record dimensions, input/weight/output layouts, alignment,
  dtype, selected kernel name, latency, and achieved throughput. Compare
  alternate cuBLASLt plans outside the timed region.
- Separately measure attention score/value matmuls, masking, softmax,
  LayerNorm, exact GELU, residual additions, invalid-token zeroing, copies,
  layout conversions, and launch gaps.
- Use `torch.profiler` with CPU and CUDA activities for attribution and CUDA
  events for latency. CPU-side percentages alone do not prove GPU bottlenecks.
- Calculate an Amdahl upper bound before writing a custom kernel. Stop an
  experiment when eliminating the entire measured target still cannot clear
  the acceptance gate.
- Profile the generated kernels or backend dispatch. A requested backend that
  silently falls back is not evidence that the backend ran.

## 5. Person 1: attention directives

- Keep packed Q/K/V projection in Q, K, V row order and preserve baseline
  weight-transfer compatibility.
- Use PyTorch scaled-dot-product attention as the guaranteed path. It can avoid
  materializing the full score/probability matrices and can select an efficient
  backend for the installed GPU/runtime.
- Keep SDPA eligibility separate from custom Triton eligibility. The official
  matrix exercises head dimensions 8, 32, 64, 128, and 256; a Triton kernel
  restricted to one proven head dimension must not force the other cases onto
  explicit quadratic attention.
- Support causal and padding masks together. Invalid key positions must not be
  attended to, and invalid query outputs must be exactly zero.
- Avoid constructing an attention mask for a proven no-padding case only when
  the proof is outside timing, safe against mask mutation/replacement, and the
  no-mask path passes identical correctness tests.
- Treat custom Triton FlashAttention as an optional measured candidate. Use
  FP32 online-softmax maxima/sums, handle partially filled tiles explicitly,
  and retain SDPA fallback for unsupported head sizes, layouts, dtypes, or
  compilation states.
- Benchmark short and long sequences separately because attention changes from
  launch/overhead-sensitive to quadratic compute/memory pressure as sequence
  length grows.
- For official ID 14, use streaming/tiled attention with online FP32 softmax
  statistics and batch blocking. The dense baseline must be memory-guarded;
  do not claim full-batch timing from sequential blockwise validation.

## 6. Person 2: FFN, LayerNorm, and residual directives

- Preserve token-major `[B*S,D]` operation without an unnecessary copy; accept
  non-contiguous public inputs through a correct fallback or one measured
  conversion.
- Leave the two large linear projections on PyTorch/cuBLAS unless independent
  GEMM measurements prove another backend faster at the exact shapes.
- Focus fusion work on memory-bound boundaries: pre-norm residual + LayerNorm,
  bias/activation boundaries, and down-projection residual + invalid-row mask.
  Do not force both GEMMs into a hand-written kernel merely to reduce launches.
- Keep FP32 LayerNorm statistics and exact erf GELU semantics. cuBLASLt GELU
  epilogues that implement the tanh approximation are disallowed.
- A custom up-projection may fuse bias and exact erf GELU only if it explicitly
  reproduces the baseline's intermediate rounding and passes adversarial and
  multi-seed checks.
- A down-projection epilogue may use cuBLASLt `beta`/bias or a following fused
  residual-mask kernel only when its arithmetic order passes strict validation.
- Cache backend plans by device, runtime/backend version, dtype, matrix
  dimensions, layout, alignment, and relevant parameter versions. Do not
  allocate, synchronize, build, autotune, inspect masks, or choose tactics
  inside timed inference.

## 7. Token reduction and padding directives

The RLT paper's transferable principle is to avoid work for genuinely redundant
tokens while keeping discovery/packing overhead below the saved compute. Its
video-specific tokenization and learned run-length encoding are not compatible
with this fixed benchmark contract.

- Do not remove, merge, approximate, or reorder valid tokens.
- Valid-row FFN compaction is permissible only for already-invalid padded rows,
  because their required final outputs are zero.
- Cache compacted indices only when tied to the exact mask object, device,
  shape, and tensor version. A changed or unverified mask must use the dense
  path.
- Include index creation, gather, scatter, and fixed-output initialization in
  the realistic cost model even if immutable setup is legitimately cached.
- Do not assume fewer arithmetic operations mean lower wall-clock time. Reject
  compaction when irregular memory movement, synchronization, or variance
  erases the benefit.
- For attention, packed variable-length execution requires semantically correct
  sequence boundaries; never let tokens from different examples attend to one
  another.

## 8. Person 3: integration and dispatch directives

- Person 3 owns `UserOptimizedTransformer`, strict weight copying, subsystem
  assembly, dispatch, official-harness validation, and final reporting.
- Integrate validated standalone attention and FFN blocks before introducing
  dispatch. Confirm signatures and strict state-dict loading after every merge.
- At minimum, key measured decisions by
  `(batch_size, seq_len, d_model, num_heads, dtype, causal, padding)` plus device
  capability and backend/runtime version where results can differ.
- Do not use a fixed sequence threshold, all-valid-mask shortcut, compiled mode,
  or custom kernel without measurements for each region it serves.
- Use one global candidate when the project requires a universal gate. Use
  shape-aware dispatch only when the rules permit it and every selected branch
  has independent correctness, latency, and fallback evidence.
- Keep dispatch free of device synchronization and data-dependent tensor-to-
  scalar reads in the timed path. Any cached decision must be invalidated when
  its inputs or parameters change.
- Re-profile the assembled model. Kernel savings can be hidden by attention,
  Python/dispatcher overhead, extra copies, or interaction with CUDA graphs.

## 9. Compilation and CUDA-graph directives

- Compare baseline and candidate with identical `torch.compile` settings.
- Compile with fixed shapes using `fullgraph=True` and `dynamic=False` when the
  benchmark configuration permits it; record graph breaks or backend failures.
- Treat compilation, autotuning, engine building, and graph capture as setup,
  never inference latency.
- Clone or consume compiled outputs before another CUDA-graph replay can reuse
  their storage. Test repeated-call output ownership explicitly.
- Call `torch.compiler.cudagraph_mark_step_begin()` before each logical timed
  inference step when CUDA-graph lifetime detection needs that boundary.
- Test eager, `default`, `reduce-overhead`, and a runtime-supported
  non-CUDA-graph autotune mode independently. A compilation failure is recorded
  as unavailable, not converted into a compiled speedup claim.
- Custom operations need fake/meta implementations only when compilation can
  safely trace them; otherwise deliberately select the documented compile
  fallback.

## 10. Optional TensorRT and custom-backend directives

- TensorRT is an optional eager inference experiment, not a repository runtime
  dependency. Install it only in the external validation environment and
  record its resolved version.
- Use fixed-shape engines for fixed benchmark shapes. Engine building and
  tactic selection occur before warmup/timing, and engine caches include device,
  TensorRT/CUDA versions, shape, dtype, normalization settings, and parameter
  versions.
- Use TensorRT `GELU_ERF` when exact GELU is required. A backend's tanh GELU is
  not interchangeable.
- Verify stream use, pointer binding, output ownership, CUDA-graph compatibility,
  and strict multi-seed numerical equivalence before timing.
- Any build, import, binding, capture, or numerical failure makes the optional
  lane unavailable and selects the native fallback.
- Custom CUDA/Triton paths follow the same rules: inference-only guards where
  applicable, current CUDA stream, no timed allocation or synchronization,
  explicit boundary handling, and a native fallback.

## 11. Benchmark and acceptance protocol

Every result table must identify the exact commit and environment and make
setup costs and sampling parameters explicit.

- Record GPU name/capability/VRAM, driver, PyTorch, CUDA, optional backend
  versions, dtype, compile/engine mode, shape, causal flag, and padding state.
- Use fixed inputs and seed, correctness before timing, CUDA-event timing,
  sufficient warmups, alternating baseline/candidate rounds, and independent
  processes for final claims.
- Record median, p90, speedup, throughput, maximum absolute error, maximum
  relative error, and failed element count. Preserve failures and OOMs rather
  than silently dropping them.
- Benchmark isolated attention, isolated FFN/pre-norm, one block, and the full
  model. Clearly label kernel-only, isolated-subsystem, and end-to-end numbers.
- Include mask/no-mask, padded/unpadded, causal/non-causal, contiguous/non-
  contiguous, repeated-call, FP32, FP16, and runtime-supported BF16 coverage.
- Report eager-vs-eager and compiled-vs-compiled comparisons. Do not compare a
  compiled candidate against an eager baseline as the sole optimization claim.
- A median win does not excuse a p90 regression when the active gate constrains
  tail latency. A single unstable process must be disclosed and investigated.
- If the current task defines a numerical or speed gate, that gate overrides
  aspirational article results. Failed candidates remain experiments, not
  production dispatch choices.

## 12. Evidence and reporting rules

- State whether a result is measured, inferred from profiling, a theoretical
  upper bound, or reported by an external source.
- Never claim FlashAttention, TensorRT, Triton, cuBLASLt, Tensor Cores, FP8, or a
  particular kernel ran without dispatch/profiler evidence.
- Keep rejected methods and raw measurements in the experiment log; keep this
  file focused on durable project decisions.
- Do not commit raw profiler traces, compiled extensions, TensorRT engines,
  generated benchmark outputs, caches, virtual environments, or credentials.
- The final README/report must include setup, exact commands, environment,
  team contributions, representative correctness/latency tables, limitations,
  and a reproducible demo path.

## 13. What the sources do not justify

- The NVIDIA low-precision article does not justify FP8/FP4 on a T4 or promise
  that a faster isolated GEMM improves this full model.
- The RLT paper does not authorize changing valid tokens, model semantics, or
  positional encoding in a fixed-output benchmark.
- The TensorRT paper's CPU-relative and non-T4 results do not establish a T4
  speedup, strict elementwise equivalence, or superiority to tuned PyTorch.
- Transformer Engine examples for newer architectures do not make Transformer
  Engine a T4 dependency or prove its fused layers preserve this state dict and
  forward contract.
- A profiler share does not itself prove an optimization; it only bounds the
  opportunity. A reduction in kernel count, FLOPs, or memory traffic is not a
  speedup until CUDA-event measurements show one.

## 14. Project decision checklist

Before accepting any optimization, answer all of the following:

1. Does it preserve the public API, state dict, exact mask/causal behavior,
   exact invalid-token zeroing, and strict numerical contract?
2. Did profiling identify this operation on the target GPU and establish a
   sufficient Amdahl bound?
3. Were all setup, allocation, inspection, compilation, and autotuning costs
   classified honestly?
4. Was the requested backend proven to execute rather than silently fall back?
5. Was it tested over every shape and semantic mode it will serve, including
   repeated calls and fallbacks?
6. Does isolated improvement survive block and full-model measurement without
   unacceptable median or p90 regression?
7. Is the dispatch decision measurable, cache-safe, synchronization-free, and
   reversible?
8. Are the commit, environment, commands, measurements, failures, and
   limitations documented reproducibly?

If any required answer is no, keep the candidate experimental and retain the
last validated implementation.
