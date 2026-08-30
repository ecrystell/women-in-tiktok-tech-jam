#!/usr/bin/env python3
"""Memory-bounded correctness and timing harness for official shape ID 14.

The logical batch is split into independent batch blocks. Within each block,
the reference attention tiles query rows but evaluates every key in each
query's causal prefix, preserving the supplied attention formula without
materializing an [B, H, S, S] tensor.
"""

from __future__ import annotations

import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F

from person1_triton_attention import triton_scaled_dot_product_attention
from torch_transformer_benchmark import (
    BaselineSelfAttention,
    BaselineTransformer,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    percentile,
)


class QueryTiledSelfAttention(BaselineSelfAttention):
    """Reference attention that bounds score memory by tiling query rows."""

    def __init__(self, d_model: int, num_heads: int, query_block: int) -> None:
        super().__init__(d_model, num_heads)
        if query_block <= 0:
            raise ValueError("query_block must be positive")
        self.query_block = query_block

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        context = torch.empty_like(q)

        for start in range(0, seq_len, self.query_block):
            stop = min(start + self.query_block, seq_len)
            scores = torch.matmul(
                q[:, :, start:stop], k.transpose(-2, -1)
            ) * self.scale

            if causal:
                query_positions = torch.arange(start, stop, device=x.device)
                key_positions = torch.arange(seq_len, device=x.device)
                future = key_positions[None, :] > query_positions[:, None]
                scores = scores.masked_fill(future, float("-inf"))

            if valid_token_mask is not None:
                invalid_keys = ~valid_token_mask[:, None, None, :]
                scores = scores.masked_fill(invalid_keys, float("-inf"))

            probabilities = torch.softmax(scores.float(), dim=-1).to(x.dtype)
            context[:, :, start:stop] = torch.matmul(
                probabilities, v
            )

        merged = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(merged)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class SeparateQKVSDPAAttention(BaselineSelfAttention):
    """Memory-efficient SDPA core with baseline Q/K/V projection rounding."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if valid_token_mask is None:
            context = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        else:
            allowed = valid_token_mask[:, None, None, :]
            if causal:
                allowed = allowed & torch.ones(
                    seq_len, seq_len, device=x.device, dtype=torch.bool
                ).tril()[None, None]
            context = F.scaled_dot_product_attention(
                q, k, v, attn_mask=allowed, is_causal=False
            )

        merged = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(merged)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class SeparateQKVTritonAttention(BaselineSelfAttention):
    """Baseline projections with the FP32-statistics online-softmax core."""

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        context = triton_scaled_dot_product_attention(
            q,
            k,
            v,
            valid_token_mask=valid_token_mask,
            causal=causal,
        )
        merged = context.transpose(1, 2).contiguous().view(
            batch, seq_len, self.d_model
        )
        output = self.out_proj(merged)
        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)
        return output


class QueryTiledReferenceTransformer(BaselineTransformer):
    """Baseline-equivalent transformer using query-tiled reference attention."""

    def __init__(self, config: TransformerConfig, query_block: int) -> None:
        super().__init__(config)
        for layer in self.layers:
            layer.attention = QueryTiledSelfAttention(
                config.d_model, config.num_heads, query_block
            )


def elapsed_samples(
    model: nn.Module,
    x: torch.Tensor,
    mask: torch.Tensor,
    repeats: int,
) -> list[float]:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(repeats)]
    torch.cuda.synchronize(x.device)
    with torch.inference_mode():
        for index in range(repeats):
            starts[index].record()
            model(x, mask)
            ends[index].record()
    torch.cuda.synchronize(x.device)
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--logical-batch", type=int, default=32)
    parser.add_argument("--batch-block", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--d-model", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--ffn-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--query-block", type=int, default=16)
    parser.add_argument(
        "--attention-plan",
        choices=(
            "packed-all",
            "separate-all",
            "separate-then-packed",
            "separate-triton-all",
            "explicit-tiled-all",
        ),
        default="packed-all",
        help="choose projection rounding while retaining an SDPA attention core",
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--candidate-query-block",
        type=int,
        default=64,
        help="query tile used by the exact explicit-tiled candidate",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--atol", type=float, default=0.001)
    parser.add_argument("--rtol", type=float, default=0.01)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.logical_batch <= 0 or args.batch_block <= 0:
        raise ValueError("logical-batch and batch-block must be positive")
    if args.logical_batch % args.batch_block:
        raise ValueError("logical-batch must be divisible by batch-block")
    if args.warmup < 0 or args.repeats <= 0:
        raise ValueError("warmup must be nonnegative and repeats must be positive")
    if args.candidate_query_block <= 0:
        raise ValueError("candidate-query-block must be positive")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("shape-14 blockwise validation requires CUDA")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    config = TransformerConfig(
        args.batch_block,
        args.seq_len,
        args.d_model,
        args.heads,
        args.ffn_dim,
        args.layers,
        True,
    )
    config.validate()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    baseline = QueryTiledReferenceTransformer(config, args.query_block)
    optimized = UserOptimizedTransformer(
        config, packed_sdpa_suffix_layers=config.num_layers
    )
    if args.attention_plan == "explicit-tiled-all":
        for index in range(config.num_layers):
            optimized.layers[index].attention = QueryTiledSelfAttention(
                config.d_model,
                config.num_heads,
                args.candidate_query_block,
            )
        separate_layers = ()
    elif args.attention_plan == "separate-triton-all":
        for index in range(config.num_layers):
            optimized.layers[index].attention = SeparateQKVTritonAttention(
                config.d_model, config.num_heads
            )
        separate_layers = ()
    elif args.attention_plan == "separate-all":
        separate_layers = range(config.num_layers)
    elif args.attention_plan == "separate-then-packed":
        separate_layers = range(max(0, config.num_layers - 1))
    else:
        separate_layers = ()
    for index in separate_layers:
        optimized.layers[index].attention = SeparateQKVSDPAAttention(
            config.d_model, config.num_heads
        )
    copy_model_weights(baseline, optimized)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 100000)
    x = torch.randn(
        args.batch_block,
        args.seq_len,
        args.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    mask = torch.ones(
        args.batch_block, args.seq_len, device=device, dtype=torch.bool
    )
    optimized.prepare_for_inference(x, mask, fast_ffn_suffix_layers=0)

    with torch.inference_mode():
        reference = baseline(x, mask)
        candidate = optimized(x, mask)
    accuracy = compare_outputs(reference, candidate, args.rtol, args.atol)
    print("=== Official ID 14 blockwise validation ===")
    print(
        f"logical_shape=({args.logical_batch}, {args.seq_len}, {args.d_model}) "
        f"block_shape={tuple(x.shape)} query_block={args.query_block} "
        f"candidate_query_block={args.candidate_query_block} "
        f"attention_plan={args.attention_plan} dtype={dtype} "
        f"gpu={torch.cuda.get_device_name(device)}"
    )
    print(
        f"accuracy={'PASS' if accuracy.passed else 'FAIL'} | "
        f"max_abs={accuracy.max_abs_error:.6g} | "
        f"max_rel={accuracy.max_relative_error:.6g} | "
        f"failed={accuracy.failed_elements}/{accuracy.total_elements}"
    )
    if not accuracy.passed:
        print("timing skipped because strict block correctness failed")
        return 2

    with torch.inference_mode():
        for _ in range(args.warmup):
            baseline(x, mask)
            optimized(x, mask)
    baseline_ms = elapsed_samples(baseline, x, mask, args.repeats)
    optimized_ms = elapsed_samples(optimized, x, mask, args.repeats)
    blocks = args.logical_batch // args.batch_block
    baseline_median = statistics.median(baseline_ms) * blocks
    optimized_median = statistics.median(optimized_ms) * blocks
    baseline_p90 = percentile(baseline_ms, 0.9) * blocks
    optimized_p90 = percentile(optimized_ms, 0.9) * blocks
    print(
        "timing is a sequential blockwise projection for the logical batch; "
        "it is not equivalent to full-batch GPU utilization"
    )
    print(f"baseline : median={baseline_median:.4f} ms | p90={baseline_p90:.4f} ms")
    print(f"optimized: median={optimized_median:.4f} ms | p90={optimized_p90:.4f} ms")
    print(f"speedup  : {baseline_median / optimized_median:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
