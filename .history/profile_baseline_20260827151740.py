import torch
from torch.profiler import profile, record_function, ProfilerActivity
from torch_transformer_benchmark import TransformerConfig, BaselineTransformer, generate_random_case

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

# Standard test configuration
config = TransformerConfig(
    batch_size=8, seq_len=512, d_model=512,
    num_heads=8, ffn_dim=2048, num_layers=6, causal=False
)

model = BaselineTransformer(config).to(device=device, dtype=dtype).eval()
x, mask = generate_random_case(config, device, dtype, seed=42, padding_ratio=0.0, input_scale=1.0)

# Warmup GPU
for _ in range(10):
    with torch.inference_mode():
        model(x, mask)
if device.type == "cuda":
    torch.cuda.synchronize()

# Profile operations
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if device.type == "cuda" else [ProfilerActivity.CPU],
    record_shapes=True,
    profile_memory=True,
) as prof:
    with torch.inference_mode():
        with record_function("transformer_forward"):
            model(x, mask)

print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=15))

# Export Chrome trace for Devpost documentation
prof.export_chrome_trace("baseline_trace.json")
print("Trace saved to baseline_trace.json (View at ui.perfetto.dev)")