# Transformer GPU Optimization Project Report

## How the solution addresses the problem statement

The project optimizes the supplied PyTorch Transformer inference benchmark for
an NVIDIA Tesla T4 while preserving the reference model's output contract. The
work focuses on reducing attention overhead and memory traffic without changing
the public `UserOptimizedTransformer.forward` interface or the Transformer
semantics.

Implemented and evaluated approaches include packed Q/K/V projection, PyTorch
scaled dot-product attention (SDPA), an optional tiled online-softmax Triton
attention kernel, all-valid-mask bypass, guarded FFN/LayerNorm experiments, and
benchmark-driven layer/shape dispatch. Unsupported or numerically unsafe paths
retain a native PyTorch fallback. Correctness is checked element by element
before performance is measured, and failed candidates are not used for speedup
claims.

Testing uses the organizer's causal Transformer shape matrix, including varied
batch sizes, sequence lengths, model widths, head counts, and layer counts.
CUDA-event measurements record median and p90 latency after warmup. Extreme
long-sequence work is evaluated separately with memory-bounded batch and query
tiling because the dense attention score matrix cannot fit in GPU memory.

For Person 2, the official narrow FFNs are dominated less by GEMM throughput
than by dispatcher, allocation, kernel-launch, and memory-traffic overhead. The
native sublayer independently launches LayerNorm, the up projection, exact
GELU, the down projection, residual addition, and `masked_fill`. The optimized
path first flattens `[B,S,D]` to a zero-copy token-major `[M,D]` view, then
crosses Python once through an inference-only C++ custom operator. Inside that
boundary, ATen still supplies FP32-statistics LayerNorm and cuBLAS supplies both
FP16 GEMMs, preserving their tuned kernels; exact `gelu_(..., "none")` runs
in-place to avoid another activation allocation. The down-projection weight is
cached in contiguous `[FFN,D]` form, and a final CUDA kernel uses 128-bit
vectorized loads/stores to combine FP16 residual addition with invalid-row
zeroing, eliminating the separate residual and mask passes. This reduced the
profiled launch sequence from approximately eight CUDA launches to five and
removed intermediate dispatcher transitions without changing numerical order.
Extension compilation, weight packing, and strict numerical preflight occur
before timing, while guards fall back to native PyTorch for training, changed
parameters, non-contiguous layouts, unsupported dtypes, compilation, or failed
validation. Retuning also replaced the masked kernel's row-indexed 2D launch
with a 1D vector grid, avoiding CUDA's 65,535 `grid.y` limit for the 65,536- and
1,280,000-token official workloads. On the Tesla T4, eight of nine official
FP16 FFN tuples passed three-process median and p90 gates with zero failed
elements; the lowest accepted isolated speedup was 1.128x. The
`(M,D,FFN)=(2048,128,128)` tuple remains on the native path because its p90 and
causal full-block results were not repeatable, so the result is a measured
allowlist rather than an unsupported universal-speedup claim.

For the attention portion, I replaced the separate Q/K/V projection path with
a packed QKV projection followed by PyTorch scaled dot-product attention, while
preserving the baseline weights, output projection, causal semantics, padding
masks, and invalid-token zeroing. I also implemented and evaluated an optional
Triton tiled online-softmax backend with FP32 running statistics and guarded
fallbacks; Tesla T4 measurements showed that packed-QKV SDPA was the reliable
production choice, so custom Triton execution remains opt-in unless it passes
strict correctness and timing gates. Exact-key dispatch keeps validated
one-layer packed-SDPA cases on the fast path and sends unsupported or
numerically unstable shapes to the native fallback, while the long-sequence
Shape #14 evaluator uses memory-safe blockwise attention rather than a dense
quadratic score matrix. This work used Visual Studio Code, OpenAI Codex,
Google Colab/Jupyter, Git/GitHub, Python, PyTorch tensor/neural-network/CUDA
APIs, `scaled_dot_product_attention`, CUDA events, `torch.profiler`,
`torch.compile`, Triton, and `unittest`; no external dataset or application
API was used beyond the organizer-provided benchmark, official shape appendix,
and deterministic seeded tensors.

## Development tools used

- Visual Studio Code
- OpenAI Codex for AI-assisted code analysis, implementation, and review
- Google Colab with an NVIDIA Tesla T4 GPU
- Jupyter notebooks
- Git and GitHub for branch-based collaboration and integration
- PyTorch profiler and CUDA events for profiling and latency measurement
- pytest for automated correctness and integration tests

## APIs used

- PyTorch neural-network, tensor, and CUDA APIs
- PyTorch `scaled_dot_product_attention` API
- PyTorch `torch.compile` API for optional compilation experiments
- PyTorch profiler and CUDA Event timing APIs
- Triton language and kernel-launch APIs
- PyTorch C++/CUDA extension and custom-operator APIs for guarded FFN experiments

No external data, mapping, payment, social-media, or other application API is
required by the solution.

## Libraries and frameworks used

- Python
- PyTorch
- Triton
- CUDA and PyTorch C++/CUDA extensions
- pytest

The project does not currently depend on Hugging Face Transformers, TensorFlow,
scikit-learn, or pandas.

## Datasets and assets used

No external dataset is used. Correctness and performance inputs are
deterministic, seeded random tensors generated by the supplied PyTorch
benchmark. Project assets and references include:

- The organizer-provided `torch_transformer_benchmark.py` benchmark
- The organizer's official Transformer test-shape appendix
- The project problem statement and supplied optimization references
- Team-authored source code, tests, profiling scripts, and benchmark runners
