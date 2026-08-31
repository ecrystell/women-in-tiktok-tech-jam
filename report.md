# KernelKraft: Adaptive Transformer Acceleration on NVIDIA T4

## Executive summary

KernelKraft accelerates the organizer-provided PyTorch Transformer inference
benchmark while preserving its observable behavior. The task was not simply to
make a similar model faster: the optimized model had to retain the public
`UserOptimizedTransformer.forward` interface, pre-norm residual structure,
state-dict compatibility, causal and padding semantics, FP32-stabilized
attention normalization, exact GELU, output shape, and exact zeroing of invalid
tokens. Every output element was checked using the benchmark's absolute-or-
relative error rule before a candidate could be timed.

The final self-contained implementation in
[`torch_transformer_benchmark.py`](torch_transformer_benchmark.py) completed
all 39 strict official ID 1–13 process comparisons with zero failed elements.
Median-of-process speedups ranged from **1.090x to 1.471x**, with an unweighted
geometric mean of **1.272x**. The implementation uses a measured one-layer
packed-QKV PyTorch SDPA suffix for four exact T4 workloads, removes proven
all-valid mask work, and otherwise falls back to reference-order native
execution. Experimental Triton, CUDA-extension, compiler, CUDA Graph, and
TensorRT paths remain available for engineering evidence but are not required
by the submission.

This conservative design matters for latency-sensitive search, ranking, and
recommendation systems. It improves throughput on the widely deployed NVIDIA
T4 without changing model weights or requiring a specialized production
runtime. It also demonstrates a practical systems lesson: the fastest
standalone kernel is not automatically the safest end-to-end implementation.
KernelKraft promotes an optimization only after strict correctness, median
latency, p90 latency, and repeated-process evidence agree.

## Problem contract and engineering approach

The reference model uses explicit multi-head attention, pre-norm residual
blocks, two-layer feed-forward networks, and a final LayerNorm. Its masks have
two distinct responsibilities: invalid keys cannot be attended to, and invalid
query outputs must be exactly zero. Causal execution must also prevent every
query from observing future keys. These constraints made several common
shortcuts—approximate GELU, approximate attention, unrestricted quantization,
or changed normalization—invalid even when they appeared faster.

Initial profiling showed that `addmm` accounted for about 52.65% of reported
CPU total time across 36 calls. Explicit `bmm`, `masked_fill`, and `_softmax`
together accounted for about 30.6%, while copies and clones added substantial
overhead. The profile was CPU-side attribution rather than proof of GPU kernel
time, but it supplied testable hypotheses:

1. Replace the explicit attention core with PyTorch scaled dot-product
   attention (SDPA).
2. Pack separate Q, K, and V projections into one `[Q; K; V]` linear
   projection to reduce launches.
3. Avoid mask operations only when setup has proven that the exact mask is
   all-valid.
4. Keep FFN data token-major and target launch/memory boundaries without
   replacing efficient cuBLAS GEMMs unnecessarily.
5. Calibrate full-model layer depth and dispatch per exact workload because
   FP16 differences accumulate through residual layers.
6. Retain native fallbacks whenever a device, dtype, shape, mask, layout, or
   correctness guard is not satisfied.

This profile-measure-accept cycle shaped the project more than any single
kernel technique. A candidate with fewer launches or FLOPs was treated as a
hypothesis until CUDA-event measurements proved a repeatable full-path win.

## Environment and measurement protocol

### Development and validation environments

| Purpose | Compute environment | Software | CPU, RAM, and disk treatment |
| --- | --- | --- | --- |
| Final acceptance | Google Colab Linux/Jupyter; NVIDIA Tesla T4, compute capability 7.5 (`sm_75`), 15,360 MiB VRAM; driver 580.82.07 | Python 3.13.15, PyTorch 2.11.0+cu128, CUDA 12.8, FP16 inference | Colab CPU model/count, host RAM, and ephemeral `/content` disk allocation varied by session and were not preserved as fixed benchmark variables. Source clones and build artifacts lived on the session-local ephemeral disk. |
| Person 1 development | NVIDIA RTX 4050 Laptop GPU | PyTorch 2.13.0+cu126 with Triton | Used for attention correctness, fallback, and long-sequence development; final speed claims were rerun on the T4. |
| Person 2 development | NVIDIA GeForce MX350, `sm_61`, 2 GiB VRAM | Python 3.10, PyTorch 2.12.1+cu126 | Used for CPU and limited CUDA smoke tests. Its 2 GiB memory and pre-Triton architecture made it unsuitable for final compiler and long-attention acceptance. |

The target GPU, driver, framework, CUDA version, and precision were recorded
with each final run. We do not report invented Colab CPU, RAM, or disk values:
those resources are dynamically assigned, while all reportable performance
claims were GPU inference measurements on the fixed T4 target.

### Correctness and timing

Baseline and optimized models used identical weights, inputs, masks, random
seeds, dtype, matmul settings, and TF32 settings. Candidate development used
the stricter acceptance rule

`abs_error <= 0.001 OR abs_error <= 0.01 * abs(reference)`.

The organizer-facing CLI retains five accuracy trials and the executable
defaults `atol=0.002`, `rtol=0.02`. Passing the stricter internal rule provides
additional margin but is not presented as a change to the organizer's contract.

Final T4 timing used seed 1234, CUDA events, 20 untimed warmups, 100 inference
repetitions, three alternating baseline/candidate rounds, and three independent
Python processes per official shape. Alternating order reduces bias from clock
and thermal drift. Random input generation, compilation, extension builds,
autotuning, engine construction, mask inspection, and cache preparation were
excluded from inference latency. We recorded median and p90 latency and refused
to time candidates that failed correctness.

## Optimization journey

The team divided ownership by subsystem but evaluated the assembled model as a
single numerical pipeline. The tables below distinguish a branch-local result
from a feature that reached the canonical benchmark.

### Person 1 — attention pipeline

| Branch or candidate | Technical idea | Evidence | Decision |
| --- | --- | --- | --- |
| Baseline profiling | Measure separate Q/K/V projections, explicit score matrix, masking, FP32 softmax, and copies. | Attention operators represented a material launch and compute opportunity; 18 of 36 default `addmm` calls were Q/K/V projections. | Motivated packed projection and SDPA; the CPU profile was not itself a speedup claim. |
| `person1/attention-sdpa` packed path | Concatenate baseline weights and biases in exact `[Q; K; V]` row order, project once, reshape by head, and call `F.scaled_dot_product_attention`. | Strict weight-transfer, causal, mask, layout, repeated-output, dtype, and compile tests passed. Representative T4 subsystem speedups ranged from 1.341x to 12.254x across eight sequence/mask cases. | **Selected primitive.** Person 3 later restricted it to measured suffix depths and exact runtime keys. |
| Triton online-softmax path | Tile Q/K/V, maintain FP32 online maximum and normalization statistics, and handle causal/padding boundaries without materializing the full score matrix. | Strict correctness passed for supported cases, but T4 performance ranged from 0.119x to 2.208x. Long padded cases were much slower than SDPA; a 100,000-token core smoke was finite and memory-bounded but also much slower. | Retained as an opt-in research implementation; never the default. |
| Mask and backend hardening at `18348ba` | Normalize proven all-valid masks during setup, report actual fallback reasons, cover every official head dimension, and bound long masked fallbacks. | The T4 suite passed with the gated 100,000-token smoke separately enabled. Diagnostics distinguished backend eligibility from proof that a particular kernel executed. | Integrated as validation and diagnostics; production still prefers SDPA. |
| Shape 14 attention plans | Compare packed projections, separate projections with SDPA, Triton online softmax, and explicit query tiling. | Fused attention cores differed from the strict reference by one or two FP16 elements at a reduced D1024 case, although the packed path passed the organizer's looser contract. | Use separate Shape 14 evaluators and disclose the contract boundary; do not generalize the strict claim. |

The crucial integration finding was that packed SDPA could be accurate as a
standalone module yet fail after several FP16 residual layers. Person 3
therefore calibrated suffix depth rather than replacing attention in every
layer.

### Person 2 — LayerNorm, FFN, and residuals

Person 2 explored the widest range of kernel boundaries. Full measurements and
failure cases live in
[`PERSON2_EXPERIMENTS.md`](PERSON2_EXPERIMENTS.md); the condensed table records
why each branch was or was not promoted.

| Branch or candidate | Technical idea | Evidence | Decision |
| --- | --- | --- | --- |
| `person2/ffn-layernorm` | Token-major `[B*S,D]` view, exact `GELU(approximate="none")`, contiguous-friendly expressions, and symmetric `torch.compile` modes. | Correct on CPU/T4, but initial eager speedups were approximately 0.988x–1.013x and no compiler mode produced a blanket win. | Token-major/exact-GELU design retained; compilation alone rejected as universal dispatch. |
| `person2/ffn-fusion-t4` | Repair CUDA Graph output lifetime, add step markers, profile both GEMMs and non-GEMM work, and compare `default`, `reduce-overhead`, and non-graph autotune modes. | The two GEMMs consumed roughly 68–87% of eager time. Global compiler speedups did not meet the original 1.05x gate. | Profiling framework retained; compiler-only candidate rejected. |
| `person2/ffn-gemm-t4` | Keep cuBLAS GEMMs, experiment with cuBLASLt down-projection/residual epilogues, an exact-erf Triton up path, and split-K reduction. | A strict-valid `beta=0` down path reached 1.006x–1.054x across the old sweep but missed the former universal gate. Other variants changed accumulation order, failed elements near zero, or regressed latency. | Arithmetic-order-sensitive variants rejected. |
| `person2/ffn-ln-identity-t4`, `ffn-prepack-t4` | Remove identity affine work, prepack stable weights, and reduce repeated preparation. | Identity-affine elision produced about 1.001x–1.013x; prepacking and 1024-wide specialization had unstable sub-gate processes or p90 regressions. | Kept as experiments, not automatic paths. |
| `person2/ffn-101-t4`, `ffn-1005-t4` | Vectorized CUDA residual/mask postprocessing, valid-row compaction for padding, and a cached all-valid-mask bypass. | Several runs improved, but the shortest shape fell to 1.006x under a 1.01x gate and later all-valid runs measured 0.960x/0.984x in two processes. Full-block and p90 regressions were reproducible. | Universal claims rejected; mask bypass reverted in the standalone control. |
| `person2/ffn-fullop-t4` at `f50ef57` | Place identity LayerNorm, native up GEMM, exact in-place ATen GELU, native down GEMM, and vectorized residual/mask completion behind one inference-only C++ custom-op boundary. | Twenty T4 tests passed. Across three five-shape wide FP16 sweeps, the lowest isolated speedup was 1.023x, all 15 p90 values improved, and every output passed. Full-block sampling had one disclosed 0.858x outlier that did not reproduce in primary reruns. | **Strongest standalone Person 2 result**, but not selected by the canonical full model because official-shape integration did not establish a safe global advantage. |
| `person2/tensorrt-ffn-t4` | Fuse residual plus FP32-statistics LayerNorm and build an optional fixed-shape TensorRT 11 engine using exact `GELU_ERF`. | Pre-norm fusion lost on medium screens or p90. TensorRT built and ran, but failed 2,079 of 65,536 strict elements with max absolute error 0.00683594. | Both candidates rejected; TensorRT remains optional and is not a dependency. |
| `person2/official-narrow-ffn-t4` | Retune full-op for the nine official FFN tuples, fix CUDA's 65,535 `grid.y` launch limit, and screen compiler/CUDA Graph backends. | Eight tuples formed an isolated eager allowlist, but the 2,048-row tuple failed p90 and official configuration 4 measured only 0.911x full-block. Graph replay overwrote prior outputs; preserving ownership would require a timed copy. | Useful benchmark and launch fix retained on the branch; no Person 2 runtime dispatch entered the canonical model. |
| `person2/ln-gemm-correction-t4` | Algebraically apply identity LayerNorm statistics after a raw up GEMM, with dimension-specific warp/block launch topology. | D128 paired medians improved 1.0166x–1.5263x, but one p90 regressed; D32 was nonrepeatable and D1024 regressed. | Promising D128 experiment, rejected by the project-wide gate. |
| `person2/output-only-layernorm-t4` | Replace general LayerNorm with output-only FP16 CUDA kernels using FP32 two-pass statistics and dimension-specific row reductions. | All 29 tests and all accuracy comparisons passed. Seven tuples improved strongly, but one short process measured 0.956x, one isolated p90 failed, and four full-block p90 checks exceeded the safety limit. | Rejected as a universal backend; `f50ef57` remains the standalone handoff. |

The final canonical model deliberately uses the native token-major FFN with
exact ATen GELU. This is not an omission of Person 2's work: it is the outcome
of integrating that work under narrower official shapes, multi-layer numerical
accumulation, and tail-latency requirements.

### Person 3 — integration, profiling, and dispatch

| Branch or candidate | Technical idea | Evidence | Decision |
| --- | --- | --- | --- |
| `person3/integrate-person1` | Merge standalone Person 1 and Person 2 work while preserving APIs, weight transfer, and independent fallbacks. | Component tests passed, but combining packed attention and fast FFN across six layers produced thousands of strict failures on several earlier wide cases. | Diagnosed subsystem interaction before timing; rejected blanket integration. |
| `person3/optimize-transformer-dispatch` | Cache exact all-valid masks, calibrate suffix depth, and dispatch by shape, dtype, device, causal mode, and padding. | All-valid-mask elimination measured a 1.218x median on the earlier B8/S512 case. One packed suffix layer measured 1.327x; two layers failed strict accumulation. Adding one fast-FFN layer did not beat packed-only. | Selected mask strategy and shallow packed suffix; native FFN retained. |
| `person3/experiment-narrow-ffn` | Test Person 2's official narrow FFN behind the winning attention suffix. | All outputs passed, but the median optimized latency was 2.4019 ms versus 2.3764 ms native, and one p90 reached 4.0306 ms. | Rejected for official narrow integration. |
| `person3/integrate-shape14-dispatch` | Add guarded query/batch-blocked long-context evaluation and compare packed, separate, Triton, and explicit-tiled plans. | Reduced judge-contract timing was about 15.4x faster. Full logical B32/S100000 completed in 153–161 seconds with about 1.57 GiB peak allocation, but no full dense reference could run. | Retained as a separate bounded-memory evaluator, not a canonical full-size speedup claim. |
| `person3/official-benchmark-integration` and hardening | Rebuild from the organizer template, make the optimized model self-contained, calibrate IDs 1–13, and guard unchanged baseline/CLI sections. | The canonical T4 matrix passed 39/39 strict process comparisons with a 1.272x geometric-mean speedup. | **Selected release implementation.** |

## Final selected architecture

The release implementation is intentionally self-contained. It imports no
Triton package, TensorRT engine, Person 2 extension, or external dispatcher.
For a call to `UserOptimizedTransformer.forward(x, valid_token_mask)`, it uses
the following path:

1. **Classify the mask safely.** `None` is already all-valid. A supplied Boolean
   mask is reduced only when it has the correct device, rank, and shape. The
   result is cached only for that exact tensor object and version; changed,
   cloned, mutated, padded, or inference-mode tensors without version tracking
   do not reuse an unsafe decision. Warmup performs this work before timing.
2. **Choose only an exact measured attention candidate.** Official IDs 2, 3,
   4, and 12 construct packed attention only in the final Transformer layer.
   Execution additionally requires CUDA FP16 inference, no gradients, an
   all-valid mask, PyTorch 2.11, and device capability `(7,5)`. All other
   configurations use `BaselineSelfAttention` in every layer.
3. **Preserve weight and arithmetic contracts.** Packed projection parameters
   are copied in `[Q; K; V]` row order. If the packed path is not active, the
   module splits those parameters and reproduces separate reference-order
   linear projections. Padded masks always take the native semantic path.
4. **Run the safe FFN.** LayerNorm output is reshaped to token-major
   `[B*S,D]`, processed by the two native PyTorch linear layers with
   `F.gelu(..., approximate="none")`, reshaped back, added to the residual,
   and masked. There is no custom CUDA FFN dependency in production.
5. **Normalize and zero invalid outputs.** The final LayerNorm and invalid-token
   zeroing remain consistent with the reference.

The shipped contributions are therefore precise: Person 1 supplied the packed
SDPA implementation and attention evidence; Person 2 supplied the safe
token-major exact-GELU FFN design and extensive boundary evidence; Person 3
supplied model integration, mask handling, numerical-depth calibration,
runtime guards, exact dispatch, canonical harness preservation, and Shape 14
evaluators.

## Final results

| Result | Scope and protocol | Outcome | Claim boundary |
| --- | --- | --- | --- |
| Canonical official IDs 1–13 | End-to-end T4 FP16; 13 shapes × 3 processes; strict 0.001/0.01; 20 warmups, 100 repeats, 3 alternating rounds | 39/39 process comparisons passed; zero failed elements; median-of-process speedups 1.090x–1.471x; geometric mean 1.272x | Primary release result |
| Automatic packed dispatch | End-to-end official matrix | IDs 2, 3, 4, and 12 use one packed-SDPA suffix layer; all other IDs use native attention | Exact T4/PyTorch/dtype/mask keys only |
| Canonical test suite | T4 validation at the canonical integration checkpoint | 59-test suite completed, including 11 expected hardware/toolkit skips; organizer-style checks passed | Functional and compatibility evidence, not latency |
| Person 1 packed SDPA | Attention-only T4 matrix, B1/D512/H8, S128 and S4096, causal/non-causal, padded/unpadded | Zero strict failures; 1.341x–12.254x median speedup | Standalone subsystem result; not equivalent to replacing all model layers |
| Person 2 full-op | Isolated wide-shape FP16 FFN, three five-shape processes | Lowest speedup 1.023x; all 15 p90 measurements improved; zero failed elements | Standalone result at `f50ef57`; not shipped in the canonical model |
| Shape 14 reduced comparison | B2/S4096/D1024/H16/L2 under organizer 0.002/0.02 contract | Approximately 15.4x faster with zero contract failures | Reduced, blockwise comparison; not the full official batch/sequence claim |
| Shape 14 full logical execution | B32/S100000/D1024/H16/L2 optimized evaluator only | Completed in approximately 153–161 s with about 1.57 GiB peak GPU allocation | Demonstrates bounded memory only; no dense reference, full-size correctness, or speedup claim |

The official ID 1–13 result is the number that represents KernelKraft as a
whole. The larger attention-only and reduced Shape 14 values explain technical
opportunity but are deliberately not mixed into the end-to-end aggregate.

## Why the result is useful beyond the benchmark

For research and algorithm evaluation, KernelKraft shows how kernel-level
rounding and reduction order can accumulate through residual networks, and why
component equivalence must be checked end to end. For ML systems and
infrastructure, it demonstrates versioned dispatch keys, safe fallbacks,
output-ownership tests, untimed setup, and separation of optional backends from
production requirements. For search and recommendation workloads, it targets
the practical latency and throughput constraints of causal Transformer blocks
on an accessible inference GPU. For open-source adoption, the release path is
plain PyTorch, reproducible, and does not require building experimental CUDA or
Triton code.

The project aligns with the judging criteria as follows:

| Criterion | Evidence in KernelKraft |
| --- | --- |
| Technical Execution | Baseline-compatible state transfer, guarded dispatch, exact mask semantics, 59-test canonical suite, repeated CUDA-event validation, and correct native fallbacks |
| Innovation & Problem Insight | Packed suffix-depth calibration, object/version-safe mask elimination, explicit rejection of locally fast but globally unsafe kernels, and bounded-memory long-context evaluation |
| Impact & Relevance | Up to 1.471x official end-to-end speedup and a 1.272x geometric mean on a commonly deployed T4, applicable to latency-sensitive Transformer services |
| Feasibility & Practicality | PyTorch-only release dependency, fixed and auditable dispatch table, no timed compilation/autotuning, and automatic fallback outside validated conditions |
| Presentation & Communication | Separate labels for end-to-end, subsystem, reduced-shape, theoretical, and bounded-memory results; rejected candidates and limitations remain visible |

## Development tools, APIs, libraries, data, and AI

### Development tools

The team used Visual Studio Code for source development, Google Colab and
Jupyter for T4 execution, Git and GitHub for isolated branches and integration,
PowerShell for local automation, Python `unittest` for regression coverage,
PyTorch Profiler for operator/CUDA attribution, and CUDA events for latency
measurement. Branch isolation let each subsystem evolve without silently
changing the organizer harness.

### APIs, libraries, and frameworks

The implementation and experiments used Python; PyTorch `torch`, `nn`,
functional, CUDA, SDPA, profiler, `torch.compile`, `torch.library` custom-op,
and C++/CUDA extension APIs; NVIDIA CUDA C++; Triton for online-softmax and
fusion experiments; and optional TensorRT for a fixed-shape exact-GELU engine
experiment. The **final submission requires only PyTorch**. It calls no hosted
model or external inference API at runtime.

### AI-assisted engineering

OpenAI Codex was the only AI engineering tool credited by the project. It
assisted with research synthesis, implementation proposals, debugging,
benchmark design, regression-test generation, profiler interpretation, branch
integration review, and documentation. Codex was not part of the Transformer
inference runtime and no OpenAI API is called by the submission. AI suggestions
were treated as hypotheses: team-owned deterministic tests, strict numerical
comparisons, target-GPU profiling, independent processes, and code review
decided whether a change was accepted. Failed AI-inspired approaches are
documented rather than removed from the engineering record.

### Datasets and assets

KernelKraft uses no external dataset and performs no model training. Inputs are
deterministic seeded random tensors generated by the organizer benchmark.
Assets consist of the supplied PyTorch benchmark, the official shape appendix,
the problem statement, research papers summarized in
[`directives.md`](directives.md), and team-authored modules, kernels, tests,
profiling scripts, experiment logs, and reproducibility documentation.

## Limitations and next steps

- Production dispatch is intentionally specific to T4 `sm_75`, PyTorch 2.11,
  CUDA FP16 inference, all-valid masks, and exact measured shapes. Nearby
  workloads use a correct native fallback rather than an extrapolated policy.
- The canonical model does not use Person 2's extension-backed full-op despite
  strong standalone results; official integrated timing did not justify it.
- The custom Triton attention kernel is correctness-tested but not competitive
  with packed SDPA across the T4 matrix.
- Colab CPU, RAM, and ephemeral disk allocations vary between sessions and are
  not part of the fixed performance claim.
- Official Shape 14 cannot be compared against the supplied dense reference on
  a T4 because the attention matrix would require multiple terabytes. Its full
  optimized run is a bounded-memory capability result, not a speedup claim.
- Future work should re-calibrate dispatch on additional GPU architectures,
  evaluate whole-model CUDA Graph capture with explicit output ownership, and
  revisit promising D128 FFN kernels only with a fresh integrated p90 gate.

## Reproducibility

The primary submission and official sweep are documented in
[`README.md`](README.md). The canonical strict workflow runs IDs 1–13 through
`run_sweep.py` with the T4 FP16 environment above; `--contract judge` selects
the organizer's executable numerical defaults. Shape 14 must use the separate
blockwise or streaming evaluator and must never invoke the dense reference at
full size. Generated results, profiler traces, compiled extensions, TensorRT
engines, caches, virtual environments, and credentials are intentionally not
committed.
