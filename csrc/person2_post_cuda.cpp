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

void layer_norm_correct_exact_gelu_cuda(
    torch::Tensor& raw_projection,
    const torch::Tensor& residual,
    const torch::Tensor& weight_row_sum,
    const torch::Tensor& bias,
    double eps);

torch::Tensor algebraic_layer_norm_up_gelu(
    const torch::Tensor& residual,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& up_weight_row_sum,
    double eps) {
  auto activated = at::linear(residual, up_weight, c10::nullopt);
  layer_norm_correct_exact_gelu_cuda(
      activated, residual, up_weight_row_sum, up_bias, eps);
  return activated;
}

torch::Tensor algebraic_layer_norm_ffn_masked(
    const torch::Tensor& residual,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& up_weight_row_sum,
    const torch::Tensor& down_weight_nt,
    const torch::Tensor& down_bias,
    const torch::Tensor& valid_token_mask,
    double eps) {
  auto activated = algebraic_layer_norm_up_gelu(
      residual, up_weight, up_bias, up_weight_row_sum, eps);
  auto update = at::addmm(down_bias, activated, down_weight_nt);
  auto output = torch::empty_like(residual);
  residual_masked_cuda(update, residual, valid_token_mask, output);
  return output;
}

torch::Tensor algebraic_layer_norm_ffn_unmasked(
    const torch::Tensor& residual,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& up_weight_row_sum,
    const torch::Tensor& down_weight_nt,
    const torch::Tensor& down_bias,
    double eps) {
  auto activated = algebraic_layer_norm_up_gelu(
      residual, up_weight, up_bias, up_weight_row_sum, eps);
  auto update = at::addmm(down_bias, activated, down_weight_nt);
  auto output = torch::empty_like(residual);
  residual_unmasked_cuda(update, residual, output);
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
      "algebraic_layer_norm_up_gelu",
      &algebraic_layer_norm_up_gelu,
      "Raw up GEMM with LayerNorm correction and exact GELU");
  module.def(
      "algebraic_layer_norm_ffn_masked",
      &algebraic_layer_norm_ffn_masked,
      "Algebraic LayerNorm FFN with masked residual postprocessing");
  module.def(
      "algebraic_layer_norm_ffn_unmasked",
      &algebraic_layer_norm_ffn_unmasked,
      "Algebraic LayerNorm FFN with unmasked residual postprocessing");
}
