#!/usr/bin/env python3
"""Isolate full-model numerical drift between attention and FFN candidates."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn

from person1_triton_attention import PackedQKVSDPAAttention
from torch_transformer_benchmark import (
    BaselineTransformer,
    OptimizedTransformerBlock,
    TransformerConfig,
    UserOptimizedTransformer,
    compare_outputs,
    copy_model_weights,
    generate_random_case,
    resolve_device,
    resolve_dtype,
)


class FFNOnlyTransformer(BaselineTransformer):
    """Use baseline attention with Person 2's guarded FFN implementation."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__(config)
        self.layers = nn.ModuleList(
            [
                OptimizedTransformerBlock(
                    config.d_model, config.num_heads, config.ffn_dim
                )
                for _ in range(config.num_layers)
            ]
        )

    def prepare_for_inference(
        self,
        x: torch.Tensor,
        valid_token_mask: torch.Tensor,
    ) -> int:
        prepared_layers = 0
        with torch.inference_mode():
            for layer in self.layers:
                x = x + layer.attention(
                    layer.norm1(x), valid_token_mask, self.config.causal
                )
                if layer.prepare_fast_ffn(x, valid_token_mask):
                    prepared_layers += 1
                x = layer._ffn_residual(x, valid_token_mask)
        return prepared_layers


def make_sdpa_prefix_model(
    baseline: BaselineTransformer,
    packed_layers: int,
    device: torch.device,
    dtype: torch.dtype,
) -> UserOptimizedTransformer:
    """Build a native-FFN model with packed SDPA in the first N layers."""
    model = UserOptimizedTransformer(baseline.config)
    copy_model_weights(baseline, model)
    model = model.to(device=device, dtype=dtype).eval()

    for index in range(packed_layers):
        source_attention = baseline.layers[index].attention
        packed_attention = PackedQKVSDPAAttention(
            baseline.config.d_model, baseline.config.num_heads
        ).to(device=device, dtype=dtype)
        packed_attention.copy_from_baseline(source_attention)
        model.layers[index].attention = packed_attention.eval()
    return model


def print_comparison(
    label: str,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    rtol: float,
    atol: float,
) -> bool:
    result = compare_outputs(reference, candidate, rtol=rtol, atol=atol)
    status = "PASS" if result.passed else "FAIL"
    print(
        f"{label:<24} {status} | max_abs={result.max_abs_error:.6g} | "
        f"max_rel={result.max_relative_error:.6g} | "
        f"failed={result.failed_elements}/{result.total_elements}"
    )
    return result.passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--ffn-dim", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--padding-ratio", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.001)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    config = TransformerConfig(
        args.batch_size,
        args.seq_len,
        args.d_model,
        args.heads,
        args.ffn_dim,
        args.layers,
        args.causal,
    )
    config.validate()

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    baseline = BaselineTransformer(config)
    ffn_only = FFNOnlyTransformer(config)
    copy_model_weights(baseline, ffn_only)

    baseline = baseline.to(device=device, dtype=dtype).eval()
    ffn_only = ffn_only.to(device=device, dtype=dtype).eval()
    packed_native = make_sdpa_prefix_model(
        baseline, config.num_layers, device, dtype
    )
    combined = make_sdpa_prefix_model(
        baseline, config.num_layers, device, dtype
    )
    x, valid_mask = generate_random_case(
        config,
        device,
        dtype,
        args.seed,
        args.padding_ratio,
        1.0,
    )

    ffn_prepared = ffn_only.prepare_for_inference(x, valid_mask)
    combined_prepared = combined.prepare_for_inference(x, valid_mask)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "n/a"
    print(f"device={device} dtype={dtype} gpu={gpu_name}")
    print(
        f"prepared_ffn_layers: ffn_only={ffn_prepared}/{config.num_layers}, "
        f"combined={combined_prepared}/{config.num_layers}"
    )

    prefix_models = [
        make_sdpa_prefix_model(baseline, packed_layers, device, dtype)
        for packed_layers in range(config.num_layers + 1)
    ]
    with torch.inference_mode():
        reference = baseline(x, valid_mask)
        print_comparison(
            "packed SDPA + native FFN",
            reference,
            packed_native(x, valid_mask),
            args.rtol,
            args.atol,
        )
        print_comparison(
            "baseline attn + fast FFN",
            reference,
            ffn_only(x, valid_mask),
            args.rtol,
            args.atol,
        )
        print_comparison(
            "packed SDPA + fast FFN",
            reference,
            combined(x, valid_mask),
            args.rtol,
            args.atol,
        )

        print("\nPacked-SDPA prefix with native FFN:")
        for packed_layers, prefix_model in enumerate(prefix_models):
            print_comparison(
                f"first {packed_layers}/{config.num_layers} layers",
                reference,
                prefix_model(x, valid_mask),
                args.rtol,
                args.atol,
            )
    # A diagnostic failure is expected when isolating an invalid candidate;
    # reserve nonzero status for setup/runtime errors so Colab prints all rows.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
