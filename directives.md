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
6. **Panopoulos et al., ["Exploring the Performance and Efficiency of
   Transformer Models for NLP on Mobile Devices"](https://arxiv.org/pdf/2306.11426).**
   This shows that accelerator coverage, accuracy, and latency depend on the
   model, device, delegate, and precision; a nominal accelerator is not
   automatically the fastest or most accurate execution path.
7. **Kluska et al., ["QAttn: Efficient GPU Kernels for Mixed-precision Vision
   Transformers"](https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Kluska_QAttn_Efficient_GPU_Kernels_for_Mixed-precision_Vision_Transformers_CVPRW_2024_paper.html),
   CVPR Workshops 2024.** This supplies useful Triton integration, tile-boundary,
   mixed-precision-attention, and quantization-overhead lessons. Its accuracy
   criterion, A100 platform, ViT workloads, and INT8 semantics differ from this
   project.
8. **Du et al., ["Improving Computation and Memory Efficiency for Real-world
   Transformer Inference on GPUs"](https://doi.org/10.1145/3617689), ACM TACO
   2023.** This motivates valid-block attention, dense valid-token FFN execution,
   block-organized layouts, and reusable memory planning for variable-length
   inputs, while explicitly exposing indexing and layout-switch overhead.
9. **Mittal and Vaishay, ["A Survey of Techniques for Optimizing Deep Learning
   on GPUs"](https://doi.org/10.1016/j.sysarc.2019.101635), Journal of Systems
   Architecture 2019.** This provides the broader GPU principles behind tiling,
   batching, coalescing, bank-conflict avoidance, fusion, occupancy, low-precision
   conversion cost, and hardware-aware sparsity.
10. **Gerami and Duraiswami, ["Transformer Based Linear Attention with
    Optimized GPU Kernel Implementation"](https://arxiv.org/pdf/2510.21956).**
    This demonstrates algebra/implementation co-design for data reuse and
    coalesced CUDA execution. Linear attention changes the model's attention
    equation, so only its kernel-engineering lessons transfer to this project.
11. **Team synthesis, "High-Performance Acceleration Strategies for PyTorch
    Transformer Optimization"** (user-supplied local report, 12 pages). This
    collects the project's profiler observations and proposes SDPA backend
    testing, compiler tuning, CUDA Graphs, and normalization fusion. Treat its
    expected reductions, dispatch diagrams, and kernel sketches as hypotheses:
    only repository code, backend traces, and T4 measurements can promote them
    to project directives or performance claims.

The supplied local `2603.28708v1.pdf` is the same work as source 4. It is not
counted as a separate source or as independent corroboration.

### Supplied implementation references

- [PyTorch `torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile)
- [PyTorch high-performance SDPA tutorial](https://docs.pytorch.org/tutorials/intermediate/scaled_dot_product_attention_tutorial.html)
- [PyTorch SDPA backend controls](https://docs.pytorch.org/docs/stable/nn.attention.html)
- [PyTorch compiler programming model](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/compile/programming_model.html)
- [PyTorch CUDA Graph Trees](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html)
- [PyTorch CUDA-graph step marker](https://docs.pytorch.org/docs/main/generated/torch.compiler.cudagraph_mark_step_begin.html)
- [PyTorch profiler](https://docs.pytorch.org/docs/stable/profiler.html)
- [Triton LayerNorm tutorial](https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html)
- [PyTorch normalization-fusion analysis](https://pytorch.org/blog/towards-free-normalization-fusing-normalization-into-gemm-and-attention-kernels/)
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

### Backend coverage and deployment

- Verify that the intended backend executes every operation in the candidate
  graph. Partial delegation can insert transfers or fallbacks that dominate
  latency, and a successfully loaded model does not prove full acceleration.
- Treat device, model shape, precision, and backend version as one performance
  configuration. There is no presumed globally optimal accelerator or library
  setting.
- Validate elementwise output on the target device, not only through a host
  interpreter. A backend may be fast while producing unacceptable target-device
  accuracy.
- When measuring a thermally constrained or shared device, document idle/cooldown
  policy and competing load. For the Colab T4, record runtime resets and process
  isolation instead of assuming a stable cloud allocation.

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

### Tile, batch, and layout selection

- Select tile size and work assignment jointly with batch size and exact
  `M/K/N`; a tile that maximizes reuse can reduce thread-level parallelism, and
  a tile that maximizes occupancy can increase traffic or idle lanes.
- Include awkward and boundary dimensions in tuning. Odd sequence lengths and
  dimensions that are not multiples of a tensor-core tile can change masking,
  wasted work, and the winning kernel.
- Require global-memory accesses to be coalesced and inspect shared-memory bank
  conflicts. Transposing or padding a shared-memory tile is useful only when its
  extra instructions and storage are cheaper than the conflicts removed.
- Balance registers and shared memory against occupancy. Keeping intermediates
  on-chip is beneficial only while enough warps remain resident to hide latency.
- Autotune only a bounded set of legal candidates during setup, then cache the
  winning plan for the complete configuration. Do not infer a T4 tile from an
  A100, V100, A6000, or mobile result.
- Reducing FLOPs through sparsity or compaction is insufficient when irregular
  addressing loses dense GEMM tiling and coalescing. Measure realized latency,
  not the nominal operation count.

## 5. Person 1: attention directives

- Keep packed Q/K/V projection in Q, K, V row order and preserve baseline
  weight-transfer compatibility.
- Use PyTorch scaled-dot-product attention as the guaranteed path. It can avoid
  materializing the full score/probability matrices and can select an efficient
  backend for the installed GPU/runtime.
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

### SDPA backend verification

- Use `torch.nn.attention.sdpa_kernel` as a scoped diagnostic context to isolate
  eligible SDPA implementations. Do not globally disable backends in submission
  code or let one benchmark case contaminate later cases.
- Do not assume that `attn_mask=None` implies FlashAttention, or that every
  explicit boolean/additive mask disables it. Backend eligibility depends on
  device, dtype, head dimension, strides, causal mode, mask form, and installed
  PyTorch/CUDA versions. Capture warnings and prove the selected kernel with a
  profiler or dispatch trace for every claimed configuration.
- Preserve SDPA boolean semantics: `True` means the position participates in
  attention, which is the inverse of `MultiheadAttention.key_padding_mask`.
  Test the conversion with padded queries and keys, not only an all-valid mask.
- Never evaluate `valid_token_mask.all().item()` in the compiled or timed
  forward path. Establish an all-valid fact from immutable host metadata or a
  cache guarded by mask identity, version, shape, strides, and device; otherwise
  pass the mask through the validated general path.
- Treat Q/K/V head views as a layout contract. Record strides and backend
  eligibility before inserting `.contiguous()`; count any required copy in
  end-to-end latency rather than attributing only the SDPA kernel time.

### Exact-attention boundary

- The benchmark fixes scaled softmax attention. Linear attention, sparse
  attention, learned token pruning, approximate softmax, or another attention
  kernel function changes the model and cannot enter the submission path even
  if it performs well on downstream model-quality metrics.
- Transfer the linear-attention paper's implementation principles instead:
  factor repeated work where algebraically exact, organize adjacent threads for
  adjacent memory, maximize reuse from registers/shared memory, and minimize
  intermediate global-memory round trips.
- Compare a reformulation against SDPA/FlashAttention at the repository's actual
  sequence and head dimensions. An asymptotic crossover reported for very long
  sequences on an A6000 is not a T4 crossover measurement.
- Keep normalization and denominator handling numerically stable and explicitly
  test causal/padding tile boundaries. Algebraic equivalence alone does not
  establish floating-point equivalence or correct masking.
- QAttn's pattern of low-precision score computation followed by FP32 softmax is
  an experiment only. INT8 Q/K/V or outputs require strict elementwise preflight;
  downstream top-1 or mIOU preservation is not the benchmark contract.

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

### LayerNorm and GEMM fusion gate

- Distinguish a row reduction from a GEMM tile. LayerNorm needs the complete
  hidden row to compute mean and variance, while a high-throughput GEMM divides
  the output over two-dimensional `M x N` tiles. A naive GEMM epilogue cannot
  normalize a row that is split across column tiles without communication,
  redundant reductions, a second stage, or an altered tiling strategy.
- Do not transfer the PyTorch Lazy Pre-Norm algebra to this benchmark. That
  method relies on affine-free RMSNorm being a row-wise scale and explicitly
  does not apply to mean-subtracting LayerNorm or general affine parameters.
- A dedicated fused residual-add plus LayerNorm candidate must accumulate its
  mean and variance in FP32, reproduce the baseline epsilon and population-
  variance convention, apply current gamma/beta values, and specify ownership
  of both the updated residual and normalized output. Writing either public
  input in place is disallowed unless the caller's ownership contract permits
  it and repeated-call tests prove safety.
- Size a row kernel from feature bytes, register use, shared memory, and active
  warps rather than copying the Triton tutorial's limit as a universal rule.
  Test dimensions immediately below, at, and above each implementation guard;
  large power-of-two padding can reduce occupancy or make the kernel illegal.
- Compare three boundaries independently: native LayerNorm; a standalone
  fused add/LayerNorm kernel; and the validated full FFN custom-op boundary.
  Record launch count, bytes moved, occupancy/resource limits, and block/full-
  model latency. Fewer launches or HBM passes are not sufficient acceptance.

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

### Valid-block execution

- For padded attention, valid mini-block scheduling is permitted only when it
  computes the same scaled-softmax result as the dense baseline. Cache the block
  schedule from immutable mask metadata outside timing and invalidate it on
  object, version, shape, stride, device, causal-mode, or block-size changes.
- Choose block granularity from a measured trade-off: smaller blocks reduce
  padding FLOPs but enlarge index structures and worsen locality; larger blocks
  preserve regular tensor-core work but compute more invalid elements.
- Avoid atomics when each output tile can have one owner. If accumulation cannot
  be uniquely owned, measure contention and preserve the baseline reduction
  order closely enough to pass strict correctness.
- Treat transitions between dense valid-token FFN layout and block-padded
  attention layout as first-class kernels. Count index construction, gather,
  scatter, transpose, padding, and layout-switch latency unless safely reused
  under the exact cache guards above.
- The current fixed-shape benchmark does not justify a dynamic chunk allocator
  in timed inference. Reuse PyTorch's allocator and fixed buffers unless memory
  profiling across changing shapes proves allocation churn or peak memory is a
  limiting bottleneck.

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

### Graph and compiler diagnostics

- Use `fullgraph=True` during qualification to turn graph breaks into explicit
  failures. Inspect `TORCH_LOGS=graph_breaks`, `TORCH_LOGS=perf_hints`, or a
  `tlparse` trace; a successful `torch.compile` call does not prove a single
  optimized graph or CUDA Graph execution.
- Keep tensor-to-scalar reads (`.item()`, `int(tensor)`, `float(tensor)`), direct
  pointer inspection, logging, exception-driven fallback, and tensor-dependent
  Python branches outside the compiled forward. Prefer static Python metadata;
  use a traceable control-flow operator only when both branches preserve the
  exact model contract and are independently validated.
- Query `torch._inductor.list_mode_options()` for the installed runtime and use
  documented `torch.compile(options=...)` controls where possible. Private
  `torch._inductor.config` names are version-sensitive. `max_autotune`,
  `shape_padding`, `epilogue_fusion`, and coordinate-descent searches are
  candidates, not a bundle that is assumed faster.
- For every compiler option, record compile/autotune time separately, inspect
  generated kernels or profiler events, and remeasure peak memory. Shape
  padding can improve Tensor Core alignment while increasing FLOPs and memory;
  epilogue fusion is useful only when the selected template actually supports
  the exact activation and arithmetic order.
- Raw CUDA Graph replay requires stable kernel arguments, dependencies, and
  memory addresses. Include any input staging copy and output ownership copy in
  realistic latency unless the caller already owns fixed buffers. Reject a
  graph that mutates public inputs or silently overwrites a previously returned
  tensor.
- Prefer one exact capture for each fixed official shape. Sequence bucketing is
  permitted only if internal padding, mask construction, extra compute,
  unpadding, and output shape all preserve semantics and the complete measured
  path wins. CUDA Graph Trees reduce recapture overhead; they do not remove the
  need for static-address and output-lifetime discipline.

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
- For quantized or mixed-precision experiments, report conversion/calibration
  cost separately and include quantize/dequantize kernels in end-to-end latency.
  A prequantized kernel-only result may explain a bottleneck but is not a model
  speedup.
- Compare operation coverage as well as timing: record which graph segments ran
  in PyTorch, compiled CUDA/Triton, cuBLASLt, TensorRT, or fallback code.
- Add tile-adversarial cases around expected block multiples when validating a
  custom kernel, even if the official five shapes do not expose every boundary.

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
- Mobile delegate results do not predict T4 behavior, and their suggestions to
  replace GELU or LayerNorm are forbidden by this benchmark's exact semantics.
- QAttn's A100 INT8 throughput and downstream accuracy do not establish strict
  elementwise correctness, a T4 speedup, or permission to quantize this model.
- The valid-block and chunk-allocation results for variable-length BERT serving
  do not prove a gain for fixed benchmark shapes; their index, layout-switch,
  and allocation costs must be included and can produce negative cases.
- The GPU survey does not imply that tiling, pruning, batching, fusion, or lower
  precision is independently beneficial. These techniques interact and sparse
  execution can be slower than dense execution on a highly parallel GPU.
- The linear-attention paper does not authorize replacing exact softmax
  attention. Its reported A6000 results at much longer sequences are neither a
  T4 performance claim nor numerical equivalence evidence.
- The supplied acceleration report's operator percentages are CPU-side
  attribution. SDPA may replace those operators, but it does not thereby prove
  that exactly 30.59% of end-to-end latency is eliminated. Likewise, reducing
  36 `addmm` calls to 24 is an expected launch-count change, not a measured
  model speedup.
- The report does not prove that an explicit mask always rejects FlashAttention
  or that `attn_mask=None` always selects it. Verify the exact T4 dispatch; do
  not label a path FlashAttention from source structure alone.
- The report's suggested Inductor settings and quoted coordinate-descent gains
  are not portable guarantees. Private configuration keys, search spaces,
  compilation cost, generated code, and winning kernels change by PyTorch,
  Triton, CUDA, GPU, and matrix shape.
- Exact GELU is not automatically fused merely because compilation succeeds.
  Inspect generated code and reject any tanh approximation or changed
  intermediate rounding, even if a fused epilogue benchmarks faster.
- The report's persistent add/LayerNorm pseudocode mutates a residual buffer and
  omits production guards, launch geometry, resource limits, stream/error
  handling, output-lifetime rules, and adversarial numerical validation. It is
  a design sketch, not submission-ready code.
- The PyTorch normalization-fusion article's Lazy Pre-Norm result is specific
  to affine-free RMSNorm and explicitly excludes LayerNorm. It does not justify
  replacing this benchmark's FP32-statistics LayerNorm or changing its state
  dict to create a fusion opportunity.
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
9. Does the measured graph actually execute the intended backend for every
   operation, without hidden host/device transfer or native fallback?
10. If the method changes precision, sparsity, token count, or attention
    algebra, is it still exactly within the benchmark contract? If not, has it
    been kept out of the submission path?

If any required answer is no, keep the candidate experimental and retain the
last validated implementation.
