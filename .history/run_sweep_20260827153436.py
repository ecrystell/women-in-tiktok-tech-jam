import subprocess

test_shapes = [
    # (batch_size, seq_len, d_model, heads, ffn_dim, layers, causal, padding)
    (1, 128, 512, 8, 2048, 6, False, 0.0),    # Short sequence / latency bound
    (8, 512, 512, 8, 2048, 6, False, 0.0),    # Standard NLP
    (8, 512, 512, 8, 2048, 6, True, 0.2),     # Causal + Padding
    (4, 2048, 1024, 16, 4096, 6, False, 0.0), # Long sequence / memory bound
    (2, 4096, 1024, 16, 4096, 6, True, 0.0),  # Extreme sequence length
]

print(f"{'Shape (B, S, D)':<25} | {'Causal':<6} | {'Status':<6}")
print("-" * 50)

for B, S, D, H, FFN, L, causal, pad in test_shapes:
    cmd = [
        "python", "torch_transformer_benchmark.py",
        "--batch-size", str(B),
        "--seq-len", str(S),
        "--d-model", str(D),
        "--heads", str(H),
        "--ffn-dim", str(FFN),
        "--layers", str(L),
        "--dtype", "float16",
        "--padding-ratio", str(pad),
    ]
    if causal:
        cmd.append("--causal")

    res = subprocess.run(cmd, capture_output=True, text=True)
    status = "PASS" if "summary: PASS" in res.stdout else "FAIL"
    shape_str = f"({B}, {S}, {D})"
    print(f"{shape_str:<25} | {str(causal):<6} | {status:<6}")