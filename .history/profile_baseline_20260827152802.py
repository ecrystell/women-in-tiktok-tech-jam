import torch
from torch.profiler import profile, ProfilerActivity
from torch_transformer_benchmark import TransformerConfig, BaselineTransformer, generate_random_case

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

print(f"Running profiler on device: {device} ({dtype})...")

config = TransformerConfig(
    batch_size=8, seq_len=512, d_model=512,
    num_heads=8, ffn_dim=2048, num_layers=6, causal=False
)

model = BaselineTransformer(config).to(device=device, dtype=dtype).eval()
x, mask = generate_random_case(config, device, dtype, seed=42, padding_ratio=0.0, input_scale=1.0)

# 1. GPU Warmup
with torch.inference_mode():
    for _ in range(5):
        model(x, mask)
if device.type == "cuda":
    torch.cuda.synchronize()

# 2. Fast Profile (Only active execution, no heavy memory/trace overhead)
activities = [ProfilerActivity.CPU]
if device.type == "cuda":
    activities.append(ProfilerActivity.CUDA)

with profile(activities=activities) as prof:
    with torch.inference_mode():
        model(x, mask)

# 3. Print Top 10 Bottlenecks Directly
sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
print("\n" + "=" * 60)
print("TOP 10 BASELINE BOTTLENECKS (Pass these to Person 1 & Person 2):")
print("=" * 60)
print(prof.key_averages().table(sort_by=sort_key, row_limit=10))