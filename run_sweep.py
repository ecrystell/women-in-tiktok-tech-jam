import subprocess
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing fast sweep on device: {device}")

# Core evaluation shapes
test_shapes = [
    # (batch_size, seq_len, d_model, heads, ffn_dim, layers, causal, padding)
    (1, 128, 512, 8, 2048, 6, False, 0.0),    # Short sequence / latency bound
    (8, 512, 512, 8, 2048, 6, False, 0.0),    # Standard NLP
    (8, 512, 512, 8, 2048, 6, True, 0.2),     # Causal + Padding
    (2, 2048, 512, 8, 2048, 6, False, 0.0),   # Long sequence / memory bound
]

print(f"\n{'Shape (B, S, D)':<20} | {'Causal':<6} | {'Pad':<5} | {'Accuracy':<8} | {'Speedup'}")
print("-" * 65)

for B, S, D, H, FFN, L, causal, pad in test_shapes:
    cmd = [
        "python", "torch_transformer_benchmark.py",
        "--batch-size", str(B),
        "--seq-len", str(S),
        "--d-model", str(D),
        "--heads", str(H),
        "--ffn-dim", str(FFN),
        "--layers", str(L),
        "--device", device,
        "--dtype", "float16" if device == "cuda" else "float32",
        "--padding-ratio", str(pad),
        "--warmup", "5",         # Reduced warmup for fast verification
        "--repeats", "20",       # Reduced repeats for fast verification
        "--benchmark-rounds", "1"
    ]
    if causal:
        cmd.append("--causal")

    res = subprocess.run(cmd, capture_output=True, text=True)

    # Parse status and speedup from stdout
    status = "PASS" if "summary: PASS" in res.stdout else "FAIL"
    speedup = "1.000x"
    for line in res.stdout.splitlines():
        if "speedup" in line and "median" in line:
            speedup = line.split(":")[-1].strip()

    shape_str = f"({B}, {S}, {D})"
    print(f"{shape_str:<20} | {str(causal):<6} | {pad:<5.1f} | {status:<8} | {speedup}")
