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

void gelu_exact_inplace_cuda(torch::Tensor& hidden);

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
      "gelu_exact_inplace",
      &gelu_exact_inplace_cuda,
      "In-place exact FP16 GELU for an owned temporary");
}
