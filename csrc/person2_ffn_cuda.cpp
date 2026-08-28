#include <torch/extension.h>

void ffn_out_residual_cuda(
    const torch::Tensor& hidden,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& valid_token_mask,
    torch::Tensor& output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "ffn_out_residual_out",
      &ffn_out_residual_cuda,
      "cuBLASLt FFN output projection with residual, bias, and row masking");
}
