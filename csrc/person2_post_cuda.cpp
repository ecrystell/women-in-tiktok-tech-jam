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

void output_only_layer_norm_cuda(
    const torch::Tensor& input,
    torch::Tensor& output,
    double eps);

torch::Tensor output_only_layer_norm(
    const torch::Tensor& input,
    double eps) {
  auto output = torch::empty_like(input);
  output_only_layer_norm_cuda(input, output, eps);
  return output;
}

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

torch::Tensor identity_layer_norm_ffn_masked(
    const torch::Tensor& residual,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& down_weight_nt,
    const torch::Tensor& down_bias,
    const torch::Tensor& valid_token_mask,
    double eps) {
  const auto normalized = at::layer_norm(
      residual,
      {residual.size(-1)},
      c10::nullopt,
      c10::nullopt,
      eps,
      true);
  auto preactivation = at::linear(normalized, up_weight, up_bias);
  return exact_gelu_down_masked(
      preactivation,
      down_weight_nt,
      down_bias,
      residual,
      valid_token_mask);
}

torch::Tensor identity_layer_norm_ffn_unmasked(
    const torch::Tensor& residual,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& down_weight_nt,
    const torch::Tensor& down_bias,
    double eps) {
  const auto normalized = at::layer_norm(
      residual,
      {residual.size(-1)},
      c10::nullopt,
      c10::nullopt,
      eps,
      true);
  auto preactivation = at::linear(normalized, up_weight, up_bias);
  return exact_gelu_down_unmasked(
      preactivation, down_weight_nt, down_bias, residual);
}

torch::Tensor output_only_layer_norm_ffn_masked(
    const torch::Tensor& residual,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& down_weight_nt,
    const torch::Tensor& down_bias,
    const torch::Tensor& valid_token_mask,
    double eps) {
  auto normalized = output_only_layer_norm(residual, eps);
  auto preactivation = at::linear(normalized, up_weight, up_bias);
  return exact_gelu_down_masked(
      preactivation,
      down_weight_nt,
      down_bias,
      residual,
      valid_token_mask);
}

torch::Tensor output_only_layer_norm_ffn_unmasked(
    const torch::Tensor& residual,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& down_weight_nt,
    const torch::Tensor& down_bias,
    double eps) {
  auto normalized = output_only_layer_norm(residual, eps);
  auto preactivation = at::linear(normalized, up_weight, up_bias);
  return exact_gelu_down_unmasked(
      preactivation, down_weight_nt, down_bias, residual);
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
  module.def(
      "identity_layer_norm_ffn_masked",
      &identity_layer_norm_ffn_masked,
      "Identity LayerNorm, exact FFN, residual add, and invalid-row zeroing");
  module.def(
      "identity_layer_norm_ffn_unmasked",
      &identity_layer_norm_ffn_unmasked,
      "Identity LayerNorm, exact FFN, and residual add");
  module.def(
      "output_only_layer_norm",
      &output_only_layer_norm,
      "FP16 output-only LayerNorm with FP32 statistics");
  module.def(
      "output_only_layer_norm_ffn_masked",
      &output_only_layer_norm_ffn_masked,
      "Output-only LayerNorm, exact FFN, residual add, and invalid-row zeroing");
  module.def(
      "output_only_layer_norm_ffn_unmasked",
      &output_only_layer_norm_ffn_unmasked,
      "Output-only LayerNorm, exact FFN, and residual add");
}
