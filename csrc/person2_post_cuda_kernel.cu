#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace {

__inline__ __device__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__inline__ __device__ float block_sum(float value, float* warp_totals) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  value = warp_sum(value);
  if (lane == 0) {
    warp_totals[warp] = value;
  }
  __syncthreads();

  const int warps = (blockDim.x + 31) / 32;
  value = threadIdx.x < warps ? warp_totals[lane] : 0.0f;
  if (warp == 0) {
    value = warp_sum(value);
  }
  if (threadIdx.x == 0) {
    warp_totals[0] = value;
  }
  __syncthreads();
  return warp_totals[0];
}

__global__ void layer_norm_correct_exact_gelu(
    half* raw_projection,
    const half* residual,
    const float* weight_row_sum,
    const half* bias,
    int64_t rows,
    int64_t hidden,
    int64_t ffn,
    float eps) {
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float warp_totals[32];
  float local_sum = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    local_sum += __half2float(residual[row * hidden + column]);
  }
  const float mean = block_sum(local_sum, warp_totals) / hidden;

  float local_squared_difference = 0.0f;
  for (int64_t column = threadIdx.x; column < hidden; column += blockDim.x) {
    const float centered =
        __half2float(residual[row * hidden + column]) - mean;
    local_squared_difference += centered * centered;
  }
  const float variance =
      block_sum(local_squared_difference, warp_totals) / hidden;
  const float rstd = rsqrtf(variance + eps);

  for (int64_t column = threadIdx.x; column < ffn; column += blockDim.x) {
    const int64_t index = row * ffn + column;
    const float raw = __half2float(raw_projection[index]);
    const float corrected =
        (raw - mean * weight_row_sum[column]) * rstd +
        __half2float(bias[column]);
    // Match the baseline Linear -> GELU boundary: the linear result is first
    // rounded to FP16, then exact erf GELU is evaluated in FP32 opmath.
    const half rounded = __float2half_rn(corrected);
    const float value = __half2float(rounded);
    const float activated =
        0.5f * value * erfcf(-value * 0.70710678118654752440f);
    raw_projection[index] = __float2half_rn(activated);
  }
}

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
    int64_t pairs,
    int64_t pairs_per_row) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < pairs) {
    const int64_t row = index / pairs_per_row;
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

__global__ void residual_masked_half8(
    const int4* update,
    const int4* residual,
    const bool* valid_token_mask,
    int4* output,
    int64_t vectors,
    int64_t vectors_per_row) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < vectors) {
    const int64_t row = index / vectors_per_row;
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

}  // namespace

void layer_norm_correct_exact_gelu_cuda(
    torch::Tensor& raw_projection,
    const torch::Tensor& residual,
    const torch::Tensor& weight_row_sum,
    const torch::Tensor& bias,
    double eps) {
  TORCH_CHECK(
      raw_projection.is_cuda() && residual.is_cuda() &&
          weight_row_sum.is_cuda() && bias.is_cuda(),
      "projection, residual, weight sum, and bias must be CUDA tensors");
  TORCH_CHECK(
      raw_projection.scalar_type() == torch::kFloat16 &&
          residual.scalar_type() == torch::kFloat16 &&
          bias.scalar_type() == torch::kFloat16 &&
          weight_row_sum.scalar_type() == torch::kFloat32,
      "projection/residual/bias must be FP16 and weight sum must be FP32");
  TORCH_CHECK(
      raw_projection.is_contiguous() && residual.is_contiguous() &&
          weight_row_sum.is_contiguous() && bias.is_contiguous(),
      "all correction tensors must be contiguous");
  TORCH_CHECK(
      raw_projection.dim() == 2 && residual.dim() == 2 &&
          weight_row_sum.dim() == 1 && bias.dim() == 1,
      "correction expects two matrices and two vectors");
  TORCH_CHECK(
      raw_projection.size(0) == residual.size(0),
      "projection and residual row counts must match");
  TORCH_CHECK(
      raw_projection.size(1) == weight_row_sum.numel() &&
          raw_projection.size(1) == bias.numel(),
      "projection width must match the weight-sum and bias vectors");
  TORCH_CHECK(
      raw_projection.size(0) > 0 && residual.size(1) > 0 &&
          raw_projection.size(1) > 0,
      "correction tensors must be non-empty");
  TORCH_CHECK(eps > 0.0, "LayerNorm epsilon must be positive");

  const c10::cuda::CUDAGuard guard(residual.device());
  const int device = residual.get_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  const int64_t largest_dimension =
      std::max(residual.size(1), raw_projection.size(1));
  int threads = 32;
  while (threads < largest_dimension && threads < 256) {
    threads *= 2;
  }
  const int64_t rows = residual.size(0);
  layer_norm_correct_exact_gelu<<<rows, threads, 0, stream>>>(
      reinterpret_cast<half*>(raw_projection.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      weight_row_sum.data_ptr<float>(),
      reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
      rows,
      residual.size(1),
      raw_projection.size(1),
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

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
    const int64_t vectors = rows * vectors_per_row;
    constexpr int vector_threads = 256;
    const int64_t vector_blocks =
        (vectors + vector_threads - 1) / vector_threads;
    residual_masked_half8<<<vector_blocks, vector_threads, 0, stream>>>(
        reinterpret_cast<const int4*>(update.data_ptr<at::Half>()),
        reinterpret_cast<const int4*>(residual.data_ptr<at::Half>()),
        valid_token_mask.data_ptr<bool>(),
        reinterpret_cast<int4*>(output.data_ptr<at::Half>()),
        vectors,
        vectors_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return;
  }
  const int64_t pairs = rows * pairs_per_row;
  constexpr int threads = 256;
  const int64_t blocks = (pairs + threads - 1) / threads;
  residual_masked_half2<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<const half2*>(update.data_ptr<at::Half>()),
      reinterpret_cast<const half2*>(residual.data_ptr<at::Half>()),
      valid_token_mask.data_ptr<bool>(),
      reinterpret_cast<half2*>(output.data_ptr<at::Half>()),
      pairs,
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
