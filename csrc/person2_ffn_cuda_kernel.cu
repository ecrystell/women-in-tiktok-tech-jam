#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cublasLt.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace {

void check_cublas(cublasStatus_t status, const char* operation) {
  TORCH_CHECK(
      status == CUBLAS_STATUS_SUCCESS,
      operation,
      " failed with cuBLAS status ",
      static_cast<int>(status));
}

struct PlanKey {
  int device;
  int64_t tokens;
  int64_t output_dim;
  int64_t hidden_dim;

  bool operator==(const PlanKey& other) const {
    return device == other.device && tokens == other.tokens &&
        output_dim == other.output_dim && hidden_dim == other.hidden_dim;
  }
};

struct PlanKeyHash {
  size_t operator()(const PlanKey& key) const {
    size_t result = std::hash<int>{}(key.device);
    result ^= std::hash<int64_t>{}(key.tokens) + 0x9e3779b9 + (result << 6) +
        (result >> 2);
    result ^= std::hash<int64_t>{}(key.output_dim) + 0x9e3779b9 +
        (result << 6) + (result >> 2);
    result ^= std::hash<int64_t>{}(key.hidden_dim) + 0x9e3779b9 +
        (result << 6) + (result >> 2);
    return result;
  }
};

struct MatmulPlan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatrixLayout_t hidden_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  std::vector<cublasLtMatmulAlgo_t> algorithms;
  std::vector<size_t> workspace_sizes;
};

std::unordered_map<PlanKey, std::unique_ptr<MatmulPlan>, PlanKeyHash>& plans() {
  static auto* cache = new std::unordered_map<
      PlanKey, std::unique_ptr<MatmulPlan>, PlanKeyHash>();
  return *cache;
}

std::mutex& plans_mutex() {
  static auto* mutex = new std::mutex();
  return *mutex;
}

MatmulPlan* create_plan(
    cublasLtHandle_t handle,
    const PlanKey& key,
    size_t maximum_workspace_size) {
  auto plan = std::make_unique<MatmulPlan>();
  check_cublas(
      cublasLtMatmulDescCreate(
          &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
      "cublasLtMatmulDescCreate");

  const cublasOperation_t transpose_weight = CUBLAS_OP_T;
  const cublasOperation_t transpose_hidden = CUBLAS_OP_N;
  const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  check_cublas(
      cublasLtMatmulDescSetAttribute(
          plan->operation,
          CUBLASLT_MATMUL_DESC_TRANSA,
          &transpose_weight,
          sizeof(transpose_weight)),
      "set TRANSA");
  check_cublas(
      cublasLtMatmulDescSetAttribute(
          plan->operation,
          CUBLASLT_MATMUL_DESC_TRANSB,
          &transpose_hidden,
          sizeof(transpose_hidden)),
      "set TRANSB");
  check_cublas(
      cublasLtMatmulDescSetAttribute(
          plan->operation,
          CUBLASLT_MATMUL_DESC_EPILOGUE,
          &epilogue,
          sizeof(epilogue)),
      "set EPILOGUE");

  check_cublas(
      cublasLtMatrixLayoutCreate(
          &plan->weight_layout,
          CUDA_R_16F,
          key.hidden_dim,
          key.output_dim,
          key.hidden_dim),
      "create weight layout");
  check_cublas(
      cublasLtMatrixLayoutCreate(
          &plan->hidden_layout,
          CUDA_R_16F,
          key.hidden_dim,
          key.tokens,
          key.hidden_dim),
      "create hidden layout");
  check_cublas(
      cublasLtMatrixLayoutCreate(
          &plan->output_layout,
          CUDA_R_16F,
          key.output_dim,
          key.tokens,
          key.output_dim),
      "create output layout");

  cublasLtMatmulPreference_t preference = nullptr;
  check_cublas(
      cublasLtMatmulPreferenceCreate(&preference),
      "cublasLtMatmulPreferenceCreate");
  check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
          &maximum_workspace_size,
          sizeof(maximum_workspace_size)),
      "set maximum workspace");

  constexpr int maximum_algorithms = 16;
  cublasLtMatmulHeuristicResult_t heuristics[maximum_algorithms]{};
  int returned = 0;
  check_cublas(
      cublasLtMatmulAlgoGetHeuristic(
          handle,
          plan->operation,
          plan->weight_layout,
          plan->hidden_layout,
          plan->output_layout,
          plan->output_layout,
          preference,
          maximum_algorithms,
          heuristics,
          &returned),
      "cublasLtMatmulAlgoGetHeuristic");
  cublasLtMatmulPreferenceDestroy(preference);
  TORCH_CHECK(returned > 0, "cuBLASLt returned no down-projection algorithm");
  for (int index = 0; index < returned; ++index) {
    if (heuristics[index].state == CUBLAS_STATUS_SUCCESS &&
        heuristics[index].workspaceSize <= maximum_workspace_size) {
      plan->algorithms.push_back(heuristics[index].algo);
      plan->workspace_sizes.push_back(heuristics[index].workspaceSize);
    }
  }
  TORCH_CHECK(!plan->algorithms.empty(), "no usable down-projection algorithm");

  MatmulPlan* result = plan.get();
  plans().emplace(key, std::move(plan));
  return result;
}

MatmulPlan* get_plan(
    cublasLtHandle_t handle,
    const PlanKey& key,
    size_t maximum_workspace_size) {
  std::lock_guard<std::mutex> lock(plans_mutex());
  auto found = plans().find(key);
  if (found != plans().end()) {
    return found->second.get();
  }
  return create_plan(handle, key, maximum_workspace_size);
}

__global__ void add_residual_and_zero_invalid_rows(
    half* output,
    const half* residual,
    const bool* valid_token_mask,
    int64_t elements,
    int64_t output_dim) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = valid_token_mask[index / output_dim]
        ? __hadd(output[index], residual[index])
        : __float2half(0.0f);
  }
}

__global__ void add_residual(
    half* output, const half* residual, int64_t elements) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __hadd(output[index], residual[index]);
  }
}

void validate_common(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& output) {
  TORCH_CHECK(hidden.is_cuda() && residual.is_cuda() && weight.is_cuda() &&
      bias.is_cuda() && output.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(hidden.scalar_type() == torch::kFloat16 &&
      residual.scalar_type() == torch::kFloat16 &&
      weight.scalar_type() == torch::kFloat16 &&
      bias.scalar_type() == torch::kFloat16 &&
      output.scalar_type() == torch::kFloat16, "all values must be float16");
  TORCH_CHECK(hidden.is_contiguous() && residual.is_contiguous() &&
      weight.is_contiguous() && bias.is_contiguous() && output.is_contiguous(),
      "all values must be contiguous");
  TORCH_CHECK(hidden.dim() == 2 && residual.dim() == 2 && weight.dim() == 2,
      "hidden, residual, and weight must be matrices");
  TORCH_CHECK(bias.dim() == 1, "bias must be a vector");
  TORCH_CHECK(weight.size(1) == hidden.size(1), "hidden dimension mismatch");
  TORCH_CHECK(residual.size(0) == hidden.size(0) &&
      residual.size(1) == weight.size(0), "residual shape mismatch");
  TORCH_CHECK(output.sizes() == residual.sizes(), "output shape mismatch");
  TORCH_CHECK(bias.numel() == weight.size(0), "bias shape mismatch");
}

void launch_down(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor* valid_token_mask,
    int64_t algorithm_index,
    torch::Tensor& output) {
  validate_common(hidden, residual, weight, bias, output);
  if (valid_token_mask != nullptr) {
    TORCH_CHECK(valid_token_mask->is_cuda() &&
        valid_token_mask->scalar_type() == torch::kBool &&
        valid_token_mask->is_contiguous(), "mask must be contiguous CUDA bool");
    TORCH_CHECK(valid_token_mask->numel() == hidden.size(0),
        "mask token count mismatch");
  }

  const c10::cuda::CUDAGuard guard(hidden.device());
  const int device = hidden.get_device();
  cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
  void* workspace = at::cuda::getCUDABlasLtWorkspace();
  const size_t maximum_workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
  const PlanKey key{device, hidden.size(0), weight.size(0), hidden.size(1)};
  MatmulPlan* plan = get_plan(handle, key, maximum_workspace_size);
  TORCH_CHECK(algorithm_index >= 0 &&
      algorithm_index < static_cast<int64_t>(plan->algorithms.size()),
      "cuBLASLt algorithm index is out of range");
  const void* bias_pointer = bias.data_ptr();
  check_cublas(
      cublasLtMatmulDescSetAttribute(
          plan->operation,
          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias_pointer,
          sizeof(bias_pointer)),
      "set bias pointer");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  check_cublas(
      cublasLtMatmul(
          handle,
          plan->operation,
          &alpha,
          weight.data_ptr(),
          plan->weight_layout,
          hidden.data_ptr(),
          plan->hidden_layout,
          &beta,
          output.data_ptr(),
          plan->output_layout,
          output.data_ptr(),
          plan->output_layout,
          &plan->algorithms[algorithm_index],
          workspace,
          plan->workspace_sizes[algorithm_index],
          stream),
      "cublasLtMatmul");

  const int64_t elements = output.numel();
  constexpr int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  if (valid_token_mask != nullptr) {
    add_residual_and_zero_invalid_rows<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        valid_token_mask->data_ptr<bool>(),
        elements,
        weight.size(0));
  } else {
    add_residual<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half*>(output.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        elements);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void down_residual_masked_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& valid_token_mask,
    torch::Tensor& output) {
  launch_down(hidden, residual, weight, bias, &valid_token_mask, 0, output);
}

void down_residual_unmasked_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output) {
  launch_down(hidden, residual, weight, bias, nullptr, 0, output);
}

int64_t down_algorithm_count_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output) {
  validate_common(hidden, residual, weight, bias, output);
  const c10::cuda::CUDAGuard guard(hidden.device());
  const int device = hidden.get_device();
  cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
  const PlanKey key{device, hidden.size(0), weight.size(0), hidden.size(1)};
  MatmulPlan* plan = get_plan(
      handle, key, at::cuda::getCUDABlasLtWorkspaceSize());
  return static_cast<int64_t>(plan->algorithms.size());
}

void down_residual_masked_algorithm_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& valid_token_mask,
    int64_t algorithm_index,
    torch::Tensor& output) {
  launch_down(
      hidden,
      residual,
      weight,
      bias,
      &valid_token_mask,
      algorithm_index,
      output);
}
