#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

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

template <int Hidden>
__global__ void output_only_layer_norm_warp_rows(
    const half2* input,
    half2* output,
    int64_t rows,
    float eps) {
  constexpr int warps_per_block = 8;
  constexpr int pairs_per_row = Hidden / 2;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * warps_per_block + warp;
  if (row >= rows) {
    return;
  }

  const half2* row_input = input + row * pairs_per_row;
  half2* row_output = output + row * pairs_per_row;
  float local_sum = 0.0f;
  for (int pair = lane; pair < pairs_per_row; pair += 32) {
    const float2 values = __half22float2(row_input[pair]);
    local_sum += values.x + values.y;
  }
  const float mean =
      __shfl_sync(0xffffffff, warp_sum(local_sum), 0) / Hidden;

  float local_squared_difference = 0.0f;
  for (int pair = lane; pair < pairs_per_row; pair += 32) {
    const float2 values = __half22float2(row_input[pair]);
    const float centered_x = values.x - mean;
    const float centered_y = values.y - mean;
    local_squared_difference +=
        centered_x * centered_x + centered_y * centered_y;
  }
  const float variance = __shfl_sync(
      0xffffffff, warp_sum(local_squared_difference), 0) / Hidden;
  const float rstd = rsqrtf(variance + eps);

  for (int pair = lane; pair < pairs_per_row; pair += 32) {
    const float2 values = __half22float2(row_input[pair]);
    row_output[pair] = __floats2half2_rn(
        (values.x - mean) * rstd,
        (values.y - mean) * rstd);
  }
}

template <int Hidden>
__global__ void output_only_layer_norm_block_rows(
    const half2* input,
    half2* output,
    int64_t rows,
    float eps) {
  constexpr int pairs_per_row = Hidden / 2;
  const int64_t row = static_cast<int64_t>(blockIdx.x);
  if (row >= rows) {
    return;
  }

  __shared__ float warp_totals[32];
  const half2* row_input = input + row * pairs_per_row;
  half2* row_output = output + row * pairs_per_row;
  float local_sum = 0.0f;
  for (int pair = threadIdx.x; pair < pairs_per_row; pair += blockDim.x) {
    const float2 values = __half22float2(row_input[pair]);
    local_sum += values.x + values.y;
  }
  const float mean = block_sum(local_sum, warp_totals) / Hidden;

  float local_squared_difference = 0.0f;
  for (int pair = threadIdx.x; pair < pairs_per_row; pair += blockDim.x) {
    const float2 values = __half22float2(row_input[pair]);
    const float centered_x = values.x - mean;
    const float centered_y = values.y - mean;
    local_squared_difference +=
        centered_x * centered_x + centered_y * centered_y;
  }
  const float variance =
      block_sum(local_squared_difference, warp_totals) / Hidden;
  const float rstd = rsqrtf(variance + eps);

  for (int pair = threadIdx.x; pair < pairs_per_row; pair += blockDim.x) {
    const float2 values = __half22float2(row_input[pair]);
    row_output[pair] = __floats2half2_rn(
        (values.x - mean) * rstd,
        (values.y - mean) * rstd);
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

void output_only_layer_norm_cuda(
    const torch::Tensor& input,
    torch::Tensor& output,
    double eps) {
  TORCH_CHECK(
      input.is_cuda() && output.is_cuda(),
      "input and output must be CUDA tensors");
  TORCH_CHECK(
      input.scalar_type() == torch::kFloat16 &&
          output.scalar_type() == torch::kFloat16,
      "input and output must be float16");
  TORCH_CHECK(
      input.is_contiguous() && output.is_contiguous(),
      "input and output must be contiguous");
  TORCH_CHECK(
      input.dim() == 2 && output.dim() == 2 && input.sizes() == output.sizes(),
      "input and output must be equally shaped matrices");
  TORCH_CHECK(input.size(0) > 0, "LayerNorm input must have at least one row");
  TORCH_CHECK(
      input.size(1) == 32 || input.size(1) == 128 || input.size(1) == 1024,
      "output-only LayerNorm supports hidden dimensions 32, 128, and 1024");
  TORCH_CHECK(eps > 0.0, "LayerNorm epsilon must be positive");

  const c10::cuda::CUDAGuard guard(input.device());
  const int device = input.get_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  const int64_t rows = input.size(0);
  constexpr int threads = 256;
  const auto* input_half2 =
      reinterpret_cast<const half2*>(input.data_ptr<at::Half>());
  auto* output_half2 = reinterpret_cast<half2*>(output.data_ptr<at::Half>());
  if (input.size(1) == 32) {
    constexpr int warps_per_block = threads / 32;
    const int64_t blocks = (rows + warps_per_block - 1) / warps_per_block;
    output_only_layer_norm_warp_rows<32><<<blocks, threads, 0, stream>>>(
        input_half2, output_half2, rows, static_cast<float>(eps));
  } else if (input.size(1) == 128) {
    constexpr int warps_per_block = threads / 32;
    const int64_t blocks = (rows + warps_per_block - 1) / warps_per_block;
    output_only_layer_norm_warp_rows<128><<<blocks, threads, 0, stream>>>(
        input_half2, output_half2, rows, static_cast<float>(eps));
  } else {
    output_only_layer_norm_block_rows<1024><<<rows, threads, 0, stream>>>(
        input_half2, output_half2, rows, static_cast<float>(eps));
  }
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
