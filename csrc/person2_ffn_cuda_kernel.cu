#include <ATen/cuda/CUDAContextLight.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include <cublasLt.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

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
  cublasLtMatrixLayout_t residual_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  size_t workspace_size = 0;
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
    size_t maximum_workspace_size,
    const void* weight,
    const void* hidden,
    const void* residual,
    void* output,
    const void* bias,
    void* workspace,
    cudaStream_t stream) {
  auto plan = std::make_unique<MatmulPlan>();
  const cudaDataType_t data_type = CUDA_R_16F;
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
      cublasLtMatmulDescSetAttribute(
          plan->operation,
          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias,
          sizeof(bias)),
      "set tuning bias pointer");

  // Row-major [tokens, dim] tensors are viewed as transposed column-major
  // matrices. This keeps D column-major so the bias epilogue is supported.
  check_cublas(
      cublasLtMatrixLayoutCreate(
          &plan->weight_layout,
          data_type,
          key.hidden_dim,
          key.output_dim,
          key.hidden_dim),
      "create weight layout");
  check_cublas(
      cublasLtMatrixLayoutCreate(
          &plan->hidden_layout,
          data_type,
          key.hidden_dim,
          key.tokens,
          key.hidden_dim),
      "create hidden layout");
  check_cublas(
      cublasLtMatrixLayoutCreate(
          &plan->residual_layout,
          data_type,
          key.output_dim,
          key.tokens,
          key.output_dim),
      "create residual layout");
  check_cublas(
      cublasLtMatrixLayoutCreate(
          &plan->output_layout,
          data_type,
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
          plan->residual_layout,
          plan->output_layout,
          preference,
          maximum_algorithms,
          heuristics,
          &returned),
      "cublasLtMatmulAlgoGetHeuristic");
  cublasLtMatmulPreferenceDestroy(preference);
  TORCH_CHECK(returned > 0, "cuBLASLt returned no FFN output algorithm");

  const float alpha = 1.0f;
  const float beta = 0.0f;
  float best_ms = std::numeric_limits<float>::infinity();
  bool found_algorithm = false;
  for (int index = 0; index < returned; ++index) {
    const auto& heuristic = heuristics[index];
    if (heuristic.state != CUBLAS_STATUS_SUCCESS ||
        heuristic.workspaceSize > maximum_workspace_size) {
      continue;
    }
    cudaEvent_t start;
    cudaEvent_t end;
    C10_CUDA_CHECK(cudaEventCreate(&start));
    C10_CUDA_CHECK(cudaEventCreate(&end));
    C10_CUDA_CHECK(cudaEventRecord(start, stream));
    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    constexpr int tuning_repetitions = 5;
    for (int repetition = 0; repetition < tuning_repetitions; ++repetition) {
      status = cublasLtMatmul(
          handle,
          plan->operation,
          &alpha,
          weight,
          plan->weight_layout,
          hidden,
          plan->hidden_layout,
          &beta,
          residual,
          plan->residual_layout,
          output,
          plan->output_layout,
          &heuristic.algo,
          workspace,
          heuristic.workspaceSize,
          stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        break;
      }
    }
    C10_CUDA_CHECK(cudaEventRecord(end, stream));
    C10_CUDA_CHECK(cudaEventSynchronize(end));
    float elapsed_ms = 0.0f;
    C10_CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
    C10_CUDA_CHECK(cudaEventDestroy(start));
    C10_CUDA_CHECK(cudaEventDestroy(end));
    if (status == CUBLAS_STATUS_SUCCESS) {
      const float average_ms = elapsed_ms / tuning_repetitions;
      if (average_ms < best_ms) {
        best_ms = average_ms;
        plan->algorithm = heuristic.algo;
        plan->workspace_size = heuristic.workspaceSize;
        found_algorithm = true;
      }
    }
  }
  TORCH_CHECK(found_algorithm, "no cuBLASLt FFN output algorithm executed");

  MatmulPlan* result = plan.get();
  plans().emplace(key, std::move(plan));
  return result;
}

MatmulPlan* get_plan(
    cublasLtHandle_t handle,
    const PlanKey& key,
    size_t maximum_workspace_size,
    const void* weight,
    const void* hidden,
    const void* residual,
    void* output,
    const void* bias,
    void* workspace,
    cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(plans_mutex());
  auto found = plans().find(key);
  if (found != plans().end()) {
    return found->second.get();
  }
  return create_plan(
      handle,
      key,
      maximum_workspace_size,
      weight,
      hidden,
      residual,
      output,
      bias,
      workspace,
      stream);
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

}  // namespace

void ffn_out_residual_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& valid_token_mask,
    torch::Tensor& output) {
  TORCH_CHECK(hidden.is_cuda(), "hidden must be CUDA");
  TORCH_CHECK(residual.is_cuda(), "residual must be CUDA");
  TORCH_CHECK(weight.is_cuda(), "weight must be CUDA");
  TORCH_CHECK(bias.is_cuda(), "bias must be CUDA");
  TORCH_CHECK(valid_token_mask.is_cuda(), "mask must be CUDA");
  TORCH_CHECK(output.is_cuda(), "output must be CUDA");
  TORCH_CHECK(hidden.scalar_type() == torch::kFloat16, "hidden must be float16");
  TORCH_CHECK(residual.scalar_type() == torch::kFloat16, "residual must be float16");
  TORCH_CHECK(weight.scalar_type() == torch::kFloat16, "weight must be float16");
  TORCH_CHECK(bias.scalar_type() == torch::kFloat16, "bias must be float16");
  TORCH_CHECK(output.scalar_type() == torch::kFloat16, "output must be float16");
  TORCH_CHECK(valid_token_mask.scalar_type() == torch::kBool, "mask must be bool");
  TORCH_CHECK(hidden.is_contiguous(), "hidden must be contiguous");
  TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
  TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
  TORCH_CHECK(bias.is_contiguous(), "bias must be contiguous");
  TORCH_CHECK(valid_token_mask.is_contiguous(), "mask must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
  TORCH_CHECK(hidden.dim() == 2, "hidden must be [tokens, hidden_dim]");
  TORCH_CHECK(residual.dim() == 2, "residual must be [tokens, output_dim]");
  TORCH_CHECK(weight.dim() == 2, "weight must be [output_dim, hidden_dim]");
  TORCH_CHECK(bias.dim() == 1, "bias must be [output_dim]");

  const int64_t tokens = hidden.size(0);
  const int64_t hidden_dim = hidden.size(1);
  const int64_t output_dim = weight.size(0);
  TORCH_CHECK(weight.size(1) == hidden_dim, "weight hidden dimension mismatch");
  TORCH_CHECK(
      residual.size(0) == tokens && residual.size(1) == output_dim,
      "residual shape mismatch");
  TORCH_CHECK(output.sizes() == residual.sizes(), "output shape mismatch");
  TORCH_CHECK(bias.numel() == output_dim, "bias shape mismatch");
  TORCH_CHECK(valid_token_mask.numel() == tokens, "mask shape mismatch");

  const c10::cuda::CUDAGuard guard(hidden.device());
  const int device = hidden.get_device();
  cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();
  void* workspace = at::cuda::getCUDABlasLtWorkspace();
  const size_t maximum_workspace_size = at::cuda::getCUDABlasLtWorkspaceSize();
  const PlanKey key{device, tokens, output_dim, hidden_dim};
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();
  MatmulPlan* plan = get_plan(
      handle,
      key,
      maximum_workspace_size,
      weight.data_ptr(),
      hidden.data_ptr(),
      residual.data_ptr(),
      output.data_ptr(),
      bias.data_ptr(),
      workspace,
      stream);

  const void* bias_pointer = bias.data_ptr();
  check_cublas(
      cublasLtMatmulDescSetAttribute(
          plan->operation,
          CUBLASLT_MATMUL_DESC_BIAS_POINTER,
          &bias_pointer,
          sizeof(bias_pointer)),
      "set bias pointer");

  const float alpha = 1.0f;
  // Preserve baseline arithmetic order: the Linear+bias result is rounded to
  // fp16 before the residual is added by the following pointwise kernel.
  const float beta = 0.0f;
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
          residual.data_ptr(),
          plan->residual_layout,
          output.data_ptr(),
          plan->output_layout,
          &plan->algorithm,
          workspace,
          plan->workspace_size,
          stream),
      "cublasLtMatmul");

  const int64_t elements = output.numel();
  const int threads = 256;
  const int blocks = static_cast<int>((elements + threads - 1) / threads);
  add_residual_and_zero_invalid_rows<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<half*>(output.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
      valid_token_mask.data_ptr<bool>(),
      elements,
      output_dim);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
