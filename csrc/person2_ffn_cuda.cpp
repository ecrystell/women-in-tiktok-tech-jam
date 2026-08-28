#include <torch/extension.h>

void down_residual_masked_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& valid_token_mask,
    torch::Tensor& output);

void down_residual_unmasked_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output);

int64_t down_algorithm_count_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output);

void down_residual_masked_algorithm_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& valid_token_mask,
    int64_t algorithm_index,
    torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "down_residual_masked_out",
      &down_residual_masked_cuda,
      "cuBLASLt down projection plus strict residual add and row masking");
  module.def(
      "down_residual_unmasked_out",
      &down_residual_unmasked_cuda,
      "cuBLASLt down projection plus strict residual add");
  module.def(
      "down_algorithm_count",
      &down_algorithm_count_cuda,
      "number of heuristic cuBLASLt down-projection algorithms");
  module.def(
      "down_residual_masked_algorithm_out",
      &down_residual_masked_algorithm_cuda,
      "execute one cuBLASLt algorithm for untimed calibration");
}
