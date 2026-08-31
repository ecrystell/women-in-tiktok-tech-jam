# KernelKraft

This project optimizes the organizer-provided PyTorch Transformer inference
benchmark for an NVIDIA Tesla T4 while preserving its public API, model
semantics, and elementwise correctness contract.

The final `torch_transformer_benchmark.py` is self-contained. The organizer's
baseline, random-case generation, correctness checks, CUDA-event timing, CLI,
and `main()` are unchanged. The submission replaces only the optimized model
and the weight-transfer behavior required by its packed parameters.

## Optimizations

- Packed Query, Key, and Value projection in `[Q; K; V]` row order.
- PyTorch scaled dot-product attention (SDPA) on measured T4 FP16 shapes.
- Exact shape-aware dispatch with native fallback for unvalidated workloads.
- All-valid-mask elimination with tensor identity and version guards.
- Contiguous token-major FFN execution with exact GELU.
- Memory-safe batch-blocked evaluation for the 100,000-token Shape 14 case.

Experimental Triton attention and extension-backed FFN implementations remain
in the repository as supporting engineering work. They are not prerequisites
for the final model because T4 measurements did not justify universal dispatch.

## Repository Layout

- `torch_transformer_benchmark.py`: self-contained primary submission.
- `run_sweep.py`: official-shape orchestration and summary reporting.
- `bench_shape14_blockwise.py`: reference-comparable Shape 14 validation.
- `bench_shape14_streaming.py`: full-size bounded-memory Shape 14 execution.
- `person1_triton_attention.py`: standalone SDPA and Triton attention work.
- `person2_ffn_block.py`, `person2_ffn_post.py`, `csrc/`: standalone FFN work.
- `tests/`: attention, FFN, integration, dispatch, and harness-contract tests.
- `report.md`: technical report and validated results.

## Setup

The validated target environment was:

- NVIDIA Tesla T4, compute capability 7.5, 15,360 MiB;
- PyTorch 2.11.0+cu128 and CUDA 12.8;
- Python on a Google Colab Linux runtime;
- FP16 inference.

The primary submission requires only PyTorch. In a CUDA-enabled environment:

```bash
git clone https://github.com/ecrystell/women-in-tiktok-tech-jam.git
cd women-in-tiktok-tech-jam
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Triton and a CUDA compiler are optional and are needed only for the standalone
experimental paths.

## Reproduce the Benchmark

Run the organizer benchmark directly for one official configuration:

```bash
python torch_transformer_benchmark.py \
  --device cuda \
  --dtype float16 \
  --batch-size 1 \
  --seq-len 128 \
  --d-model 128 \
  --heads 4 \
  --ffn-dim 128 \
  --layers 4 \
  --causal
```

Run official IDs 1-13 with the organizer's executable correctness contract,
20 warmups, 100 repetitions, and three alternating rounds:

```bash
python run_sweep.py \
  --suite official \
  --skip-long \
  --mode final \
  --device cuda \
  --dtype float16 \
  --contract judge \
  --processes 1
```

Use `--processes 3` for the team's repeatability protocol. Use
`--contract strict` for the internal `atol=0.001`, `rtol=0.01`, 20-trial gate.

The supplied dense reference cannot run official ID 14 on a T4 because its
attention matrices require multiple tebibytes. Reproduce the reduced
reference comparison and the full optimized capability separately:

```bash
python bench_shape14_blockwise.py \
  --device cuda --dtype float16 \
  --logical-batch 2 --batch-block 1 \
  --seq-len 4096 --d-model 1024 --heads 16 \
  --ffn-dim 1024 --layers 2 \
  --attention-plan packed-all \
  --query-block 16 --candidate-query-block 64 \
  --warmup 20 --repeats 100 \
  --atol 0.002 --rtol 0.02

python bench_shape14_streaming.py \
  --device cuda --dtype float16 \
  --logical-batch 32 --batch-block 1 \
  --seq-len 100000 --d-model 1024 --heads 16 \
  --ffn-dim 1024 --layers 2 \
  --warmup 0 --repeats 1
```

Run the automated tests with:

```bash
python -m unittest discover -s tests -q
```

## Validated Results

On the Tesla T4, the canonical official ID 1-13 matrix passed all 39 strict
three-process comparisons. Median-of-process speedups ranged from 1.090x to
1.471x, with an unweighted geometric mean of 1.272x. Automatic packed SDPA was
selected only for official IDs 2, 3, 4, and 12; every other shape retained the
measured native-attention fallback.

The reduced Shape 14 comparison passed the organizer's tolerance and measured
approximately 15.4x speedup. The full logical B32/S100000 optimized workload
completed in approximately 153-161 seconds with about 1.57 GiB peak GPU
allocation. The full result demonstrates bounded-memory execution, not a
speedup against the infeasible dense reference.

## Limitations and Future Work

- Dispatch is intentionally limited to exact validated T4/PyTorch/FP16 keys.
- The full Shape 14 reference cannot execute on a T4, so full-size correctness
  is inferred from reduced reference-comparable validation and reported
  separately from the memory-capability run.
- Shape 14 currently processes batch blocks sequentially rather than maximizing
  full-batch utilization.
- Additional profiling may justify broader packed-attention dispatch, larger
  Shape 14 blocks, or a production fused normalization/FFN path.

## Team Contributions

- Zhao Jin (Person 1): packed-QKV SDPA, optional Triton attention, masks, Shape 14 optimization and attention
  validation.
- Yeo Su Gar (Person 2): FFN, LayerNorm, residual fusion experiments, CUDA post-processing, documentation,
  and standalone validation.
- Tiffany Heng (Person 3): final model assembly, weight transfer, shape dispatch, integration, official sweeps,
  and video.

The team used OpenAI Codex for AI-assisted analysis, implementation, debugging,
test generation, benchmark design, and code review. Final correctness and
performance decisions were based on measured T4 results rather than generated
claims.
