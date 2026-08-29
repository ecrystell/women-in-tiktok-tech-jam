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

union alignas(16) Half8 {
  int4 packed;
  half2 pairs[4];
};

struct WelfordData {
  float mean;
  float m2;
  int count;
};

__device__ __forceinline__ WelfordData combine_welford(
    WelfordData left,
    const WelfordData right) {
  if (right.count == 0) {
    return left;
  }
  if (left.count == 0) {
    return right;
  }
  const int count = left.count + right.count;
  const float delta = right.mean - left.mean;
  left.mean += delta * static_cast<float>(right.count) /
      static_cast<float>(count);
  left.m2 += right.m2 + delta * delta *
      static_cast<float>(left.count) * static_cast<float>(right.count) /
      static_cast<float>(count);
  left.count = count;
  return left;
}

__global__ void attention_residual_layer_norm_identity_half(
    const half* residual,
    const half* attention_update,
    half* attention_residual,
    half* normalized,
    int64_t rows,
    int64_t columns,
    float eps) {
  const int64_t row = blockIdx.x;
  const int thread = threadIdx.x;
  WelfordData local{0.0f, 0.0f, 0};

  if (row < rows) {
    const int64_t offset = row * columns;
    for (int64_t column = thread; column < columns; column += blockDim.x) {
      // The baseline produces the half-precision residual first, then uses
      // that rounded value as LayerNorm's input.  Preserve that order before
      // accumulating the statistics in FP32.
      const half value = __hadd(
          residual[offset + column], attention_update[offset + column]);
      attention_residual[offset + column] = value;
      const float fp32_value = __half2float(value);
      ++local.count;
      const float delta = fp32_value - local.mean;
      local.mean += delta / static_cast<float>(local.count);
      local.m2 += delta * (fp32_value - local.mean);
    }
  }

  __shared__ float shared_mean[256];
  __shared__ float shared_m2[256];
  __shared__ int shared_count[256];
  shared_mean[thread] = local.mean;
  shared_m2[thread] = local.m2;
  shared_count[thread] = local.count;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (thread < stride) {
      WelfordData left{
          shared_mean[thread], shared_m2[thread], shared_count[thread]};
      const WelfordData right{
          shared_mean[thread + stride],
          shared_m2[thread + stride],
          shared_count[thread + stride]};
      left = combine_welford(left, right);
      shared_mean[thread] = left.mean;
      shared_m2[thread] = left.m2;
      shared_count[thread] = left.count;
    }
    __syncthreads();
  }

  if (row < rows) {
    const int64_t offset = row * columns;
    const float mean = shared_mean[0];
    const float inv_std = rsqrtf(
        shared_m2[0] / static_cast<float>(shared_count[0]) + eps);
    for (int64_t column = thread; column < columns; column += blockDim.x) {
      normalized[offset + column] = __float2half_rn(
          (__half2float(attention_residual[offset + column]) - mean) * inv_std);
    }
  }
}

__global__ void residual_masked_half8(
    const int4* update,
    const int4* residual,
    const bool* valid_token_mask,
    int4* output,
    int64_t rows,
    int64_t vectors_per_row) {
  const int64_t row = blockIdx.y;
  const int64_t column =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row < rows && column < vectors_per_row) {
    const int64_t index = row * vectors_per_row + column;
    Half8 result;
    if (valid_token_mask[row]) {
      Half8 left;
      Half8 right;
      left.packed = update[index];
      right.packed = residual[index];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        result.pairs[pair] = __hadd2(left.pairs[pair], right.pairs[pair]);
      }
    } else {
      result.packed = make_int4(0, 0, 0, 0);
    }
    output[index] = result.packed;
  }
}

__global__ void residual_unmasked_half8(
    const int4* update,
    const int4* residual,
    int4* output,
    int64_t vectors) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < vectors) {
    Half8 left;
    Half8 right;
    Half8 result;
    left.packed = update[index];
    right.packed = residual[index];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      result.pairs[pair] = __hadd2(left.pairs[pair], right.pairs[pair]);
    }
    output[index] = result.packed;
  }
}

void validate_attention_residual_layer_norm(
    const torch::Tensor& residual,
    const torch::Tensor& attention_update,
    const torch::Tensor& attention_residual,
    const torch::Tensor& normalized,
    double eps) {
  TORCH_CHECK(
      residual.is_cuda() && attention_update.is_cuda() &&
          attention_residual.is_cuda() && normalized.is_cuda(),
      "attention residual inputs and outputs must be CUDA tensors");
  TORCH_CHECK(
      residual.scalar_type() == torch::kFloat16 &&
          attention_update.scalar_type() == torch::kFloat16 &&
          attention_residual.scalar_type() == torch::kFloat16 &&
          normalized.scalar_type() == torch::kFloat16,
      "attention residual inputs and outputs must be float16");
  TORCH_CHECK(
      residual.is_contiguous() && attention_update.is_contiguous() &&
          attention_residual.is_contiguous() && normalized.is_contiguous(),
      "attention residual inputs and outputs must be contiguous");
  TORCH_CHECK(
      residual.dim() == 2 && attention_update.dim() == 2 &&
          attention_residual.dim() == 2 && normalized.dim() == 2,
      "attention residual inputs and outputs must be matrices");
  TORCH_CHECK(
      residual.sizes() == attention_update.sizes() &&
          residual.sizes() == attention_residual.sizes() &&
          residual.sizes() == normalized.sizes(),
      "attention residual input and output shapes must match");
  TORCH_CHECK(
      residual.size(1) > 0 && residual.size(1) <= 1024,
      "identity LayerNorm fusion supports hidden dimensions from 1 through 1024");
  TORCH_CHECK(eps > 0.0, "LayerNorm eps must be positive");
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
  if (update.size(1) % 8 == 0) {
    const int64_t vectors_per_row = update.size(1) / 8;
    constexpr int vector_threads = 128;
    const dim3 vector_blocks(
        static_cast<unsigned int>(
            (vectors_per_row + vector_threads - 1) / vector_threads),
        static_cast<unsigned int>(rows));
    residual_masked_half8<<<vector_blocks, vector_threads, 0, stream>>>(
        reinterpret_cast<const int4*>(update.data_ptr<at::Half>()),
        reinterpret_cast<const int4*>(residual.data_ptr<at::Half>()),
        valid_token_mask.data_ptr<bool>(),
        reinterpret_cast<int4*>(output.data_ptr<at::Half>()),
        rows,
        vectors_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
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
  if (update.numel() % 8 == 0) {
    const int64_t vectors = update.numel() / 8;
    constexpr int vector_threads = 256;
    const int vector_blocks =
        static_cast<int>((vectors + vector_threads - 1) / vector_threads);
    residual_unmasked_half8<<<vector_blocks, vector_threads, 0, stream>>>(
        reinterpret_cast<const int4*>(update.data_ptr<at::Half>()),
        reinterpret_cast<const int4*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<int4*>(output.data_ptr<at::Half>()),
        vectors);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
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

void attention_residual_layer_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& attention_update,
    double eps,
    torch::Tensor& attention_residual,
    torch::Tensor& normalized) {
  validate_attention_residual_layer_norm(
      residual, attention_update, attention_residual, normalized, eps);
  const c10::cuda::CUDAGuard guard(residual.device());
  const int device = residual.get_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  constexpr int threads = 256;
  const int64_t rows = residual.size(0);
  const int64_t columns = residual.size(1);
  attention_residual_layer_norm_identity_half<<<rows, threads, 0, stream>>>(
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(attention_update.data_ptr<at::Half>()),
      reinterpret_cast<half*>(attention_residual.data_ptr<at::Half>()),
      reinterpret_cast<half*>(normalized.data_ptr<at::Half>()),
      rows,
      columns,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
