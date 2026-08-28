#include <torch/extension.h>

void residual_masked_cuda(
    const torch::Tensor& update,
    const torch::Tensor& residual,
    const torch::Tensor& valid_token_mask,
    torch::Tensor& output);

void residual_unmasked_cuda(
    const torch::Tensor& update,
    const torch::Tensor& residual,
    torch::Tensor& output);

torch::Tensor exact_gelu_down_masked(
    torch::Tensor preactivation,
    const torch::Tensor& weight_nt,
    const torch::Tensor& bias,
    const torch::Tensor& residual,
    const torch::Tensor& valid_token_mask) {
  at::gelu_(preactivation, "none");
  auto update = at::addmm(bias, preactivation, weight_nt);
  auto output = torch::empty_like(residual);
  residual_masked_cuda(update, residual, valid_token_mask, output);
  return output;
}

torch::Tensor exact_gelu_down_unmasked(
    torch::Tensor preactivation,
    const torch::Tensor& weight_nt,
    const torch::Tensor& bias,
    const torch::Tensor& residual) {
  at::gelu_(preactivation, "none");
  auto update = at::addmm(bias, preactivation, weight_nt);
  auto output = torch::empty_like(residual);
  residual_unmasked_cuda(update, residual, output);
  return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "residual_masked_out",
      &residual_masked_cuda,
      "FP16 residual addition plus invalid-row zeroing");
  module.def(
      "residual_unmasked_out",
      &residual_unmasked_cuda,
      "FP16 residual addition");
  module.def(
      "exact_gelu_down_masked",
      &exact_gelu_down_masked,
      "Exact GELU, down projection, residual add, and invalid-row zeroing");
  module.def(
      "exact_gelu_down_unmasked",
      &exact_gelu_down_unmasked,
      "Exact GELU, down projection, and residual add");
}
