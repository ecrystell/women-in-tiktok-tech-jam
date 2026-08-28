#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace {

void validate_common(
    const torch::Tensor& update,
    const torch::Tensor& residual,
    const torch::Tensor& output) {
  TORCH_CHECK(
      update.is_cuda() && residual.is_cuda() && output.is_cuda(),
      "update, residual, and output must be CUDA tensors");
  TORCH_CHECK(
      update.scalar_type() == torch::kFloat16 &&
          residual.scalar_type() == torch::kFloat16 &&
          output.scalar_type() == torch::kFloat16,
      "update, residual, and output must be float16");
  TORCH_CHECK(
      update.is_contiguous() && residual.is_contiguous() &&
          output.is_contiguous(),
      "update, residual, and output must be contiguous");
  TORCH_CHECK(
      update.dim() == 2 && residual.dim() == 2 && output.dim() == 2,
      "update, residual, and output must be matrices");
  TORCH_CHECK(
      update.sizes() == residual.sizes() && output.sizes() == residual.sizes(),
      "update, residual, and output shapes must match");
  TORCH_CHECK(update.size(1) % 2 == 0, "hidden dimension must be even");
}

__global__ void residual_masked_half2(
    const half2* update,
    const half2* residual,
    const bool* valid_token_mask,
    half2* output,
    int64_t rows,
    int64_t pairs_per_row) {
  const int64_t row = blockIdx.y;
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows && column < pairs_per_row) {
    const int64_t index = row * pairs_per_row + column;
    output[index] = valid_token_mask[row]
        ? __hadd2(update[index], residual[index])
        : __float2half2_rn(0.0f);
  }
}

__global__ void residual_unmasked_half2(
    const half2* update,
    const half2* residual,
    half2* output,
    int64_t pairs) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < pairs) {
    output[index] = __hadd2(update[index], residual[index]);
  }
}

}  // namespace

void residual_masked_cuda(
    const torch::Tensor& update,
    const torch::Tensor& residual,
    const torch::Tensor& valid_token_mask,
    torch::Tensor& output) {
  validate_common(update, residual, output);
  TORCH_CHECK(
      valid_token_mask.is_cuda() &&
          valid_token_mask.scalar_type() == torch::kBool &&
          valid_token_mask.is_contiguous(),
      "mask must be a contiguous CUDA bool tensor");
  TORCH_CHECK(
      valid_token_mask.numel() == update.size(0),
      "mask token count must match update rows");
  const c10::cuda::CUDAGuard guard(update.device());
  const int device = update.get_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  const int64_t rows = update.size(0);
  const int64_t pairs_per_row = update.size(1) / 2;
  constexpr int threads = 256;
  const dim3 blocks(
      static_cast<unsigned int>((pairs_per_row + threads - 1) / threads),
      static_cast<unsigned int>(rows));
  residual_masked_half2<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half2*>(update.data_ptr<at::Half>()),
      reinterpret_cast<const half2*>(residual.data_ptr<at::Half>()),
      valid_token_mask.data_ptr<bool>(),
      reinterpret_cast<half2*>(output.data_ptr<at::Half>()),
      rows,
      pairs_per_row);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void residual_unmasked_cuda(
    const torch::Tensor& update,
    const torch::Tensor& residual,
    torch::Tensor& output) {
  validate_common(update, residual, output);
  const c10::cuda::CUDAGuard guard(update.device());
  const int device = update.get_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  const int64_t pairs = update.numel() / 2;
  constexpr int threads = 256;
  const int blocks = static_cast<int>((pairs + threads - 1) / threads);
  residual_unmasked_half2<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half2*>(update.data_ptr<at::Half>()),
      reinterpret_cast<const half2*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<half2*>(output.data_ptr<at::Half>()),
      pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
