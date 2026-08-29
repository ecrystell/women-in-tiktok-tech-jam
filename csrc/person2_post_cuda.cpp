#include <torch/extension.h>

#include <tuple>

void residual_masked_cuda(
    const torch::Tensor& update,
    const torch::Tensor& residual,
    const torch::Tensor& valid_token_mask,
    torch::Tensor& output);

void residual_unmasked_cuda(
    const torch::Tensor& update,
    const torch::Tensor& residual,
    torch::Tensor& output);

void attention_residual_layer_norm_cuda(
    const torch::Tensor& residual,
    const torch::Tensor& attention_update,
    double eps,
    torch::Tensor& attention_residual,
    torch::Tensor& normalized);

std::tuple<torch::Tensor, torch::Tensor> attention_residual_layer_norm(
    const torch::Tensor& residual,
    const torch::Tensor& attention_update,
    double eps) {
  auto attention_residual = torch::empty_like(residual);
  auto normalized = torch::empty_like(residual);
  attention_residual_layer_norm_cuda(
      residual, attention_update, eps, attention_residual, normalized);
  return std::make_tuple(attention_residual, normalized);
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

torch::Tensor attention_residual_identity_layer_norm_ffn_masked(
    const torch::Tensor& residual,
    const torch::Tensor& attention_update,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& down_weight_nt,
    const torch::Tensor& down_bias,
    const torch::Tensor& valid_token_mask,
    double eps) {
  auto fused = attention_residual_layer_norm(
      residual, attention_update, eps);
  auto preactivation = at::linear(std::get<1>(fused), up_weight, up_bias);
  return exact_gelu_down_masked(
      preactivation,
      down_weight_nt,
      down_bias,
      std::get<0>(fused),
      valid_token_mask);
}

torch::Tensor attention_residual_identity_layer_norm_ffn_unmasked(
    const torch::Tensor& residual,
    const torch::Tensor& attention_update,
    const torch::Tensor& up_weight,
    const torch::Tensor& up_bias,
    const torch::Tensor& down_weight_nt,
    const torch::Tensor& down_bias,
    double eps) {
  auto fused = attention_residual_layer_norm(
      residual, attention_update, eps);
  auto preactivation = at::linear(std::get<1>(fused), up_weight, up_bias);
  return exact_gelu_down_unmasked(
      preactivation,
      down_weight_nt,
      down_bias,
      std::get<0>(fused));
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
      "attention_residual_layer_norm",
      &attention_residual_layer_norm,
      "FP16 residual add plus FP32-statistics identity LayerNorm");
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
      "attention_residual_identity_layer_norm_ffn_masked",
      &attention_residual_identity_layer_norm_ffn_masked,
      "Attention residual, identity LayerNorm, exact FFN, residual add, and invalid-row zeroing");
  module.def(
      "attention_residual_identity_layer_norm_ffn_unmasked",
      &attention_residual_identity_layer_norm_ffn_unmasked,
      "Attention residual, identity LayerNorm, exact FFN, and residual add");
}
