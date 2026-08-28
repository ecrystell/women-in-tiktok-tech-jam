#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cuda_fp16.h>

#include <cstdint>

namespace {

struct WelfordState {
  float mean;
  float m2;
  int count;
};

__device__ __forceinline__ WelfordState welford_combine(
    WelfordState left,
    WelfordState right) {
  if (right.count == 0) {
    return left;
  }
  if (left.count == 0) {
    return right;
  }
  const float delta = right.mean - left.mean;
  const int count = left.count + right.count;
  const float right_fraction =
      static_cast<float>(right.count) / static_cast<float>(count);
  left.mean += delta * right_fraction;
  left.m2 += right.m2 + delta * delta *
      (static_cast<float>(left.count) * right.count / count);
  left.count = count;
  return left;
}

__device__ __forceinline__ WelfordState warp_welford(WelfordState value) {
  constexpr unsigned mask = 0xffffffffu;
  const int lane = threadIdx.x & 31;
  for (int offset = 16; offset > 0; offset >>= 1) {
    WelfordState other{
        __shfl_down_sync(mask, value.mean, offset),
        __shfl_down_sync(mask, value.m2, offset),
        __shfl_down_sync(mask, value.count, offset)};
    if (lane < offset) {
      value = welford_combine(value, other);
    }
  }
  return value;
}

__global__ void layer_norm_identity_half2(
    const half2* input,
    half2* output,
    int64_t pairs_per_row,
    float eps) {
  const int64_t row = blockIdx.x;
  WelfordState local{0.0f, 0.0f, 0};
  for (int64_t pair = threadIdx.x; pair < pairs_per_row;
       pair += blockDim.x) {
    const half2 packed = input[row * pairs_per_row + pair];
    const float values[2] = {
        __half2float(__low2half(packed)),
        __half2float(__high2half(packed))};
#pragma unroll
    for (int index = 0; index < 2; ++index) {
      const WelfordState sample{values[index], 0.0f, 1};
      local = welford_combine(local, sample);
    }
  }

  local = warp_welford(local);
  __shared__ float warp_means[8];
  __shared__ float warp_m2[8];
  __shared__ int warp_counts[8];
  __shared__ float row_mean;
  __shared__ float row_rstd;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_means[warp] = local.mean;
    warp_m2[warp] = local.m2;
    warp_counts[warp] = local.count;
  }
  __syncthreads();

  if (warp == 0) {
    WelfordState block = lane < 8
        ? WelfordState{warp_means[lane], warp_m2[lane], warp_counts[lane]}
        : WelfordState{0.0f, 0.0f, 0};
    block = warp_welford(block);
    if (lane == 0) {
      row_mean = block.mean;
      row_rstd = rsqrtf(block.m2 / block.count + eps);
    }
  }
  __syncthreads();

  for (int64_t pair = threadIdx.x; pair < pairs_per_row;
       pair += blockDim.x) {
    const half2 packed = input[row * pairs_per_row + pair];
    const float low =
        (__half2float(__low2half(packed)) - row_mean) * row_rstd;
    const float high =
        (__half2float(__high2half(packed)) - row_mean) * row_rstd;
    output[row * pairs_per_row + pair] =
        __halves2half2(__float2half_rn(low), __float2half_rn(high));
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

}  // namespace

torch::Tensor layer_norm_identity_cuda(
    const torch::Tensor& input,
    double eps) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(input.scalar_type() == torch::kFloat16, "input must be float16");
  TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
  TORCH_CHECK(input.dim() == 3, "input must have shape [batch, sequence, model]");
  TORCH_CHECK(input.size(2) % 2 == 0, "model dimension must be even");
  const c10::cuda::CUDAGuard guard(input.device());
  const int device = input.get_device();
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  auto output = torch::empty_like(input);
  const int64_t rows = input.numel() / input.size(2);
  const int64_t pairs_per_row = input.size(2) / 2;
  constexpr int threads = 256;
  layer_norm_identity_half2<<<static_cast<int>(rows), threads, 0, stream>>>(
      reinterpret_cast<const half2*>(input.data_ptr<at::Half>()),
      reinterpret_cast<half2*>(output.data_ptr<at::Half>()),
      pairs_per_row,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
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
