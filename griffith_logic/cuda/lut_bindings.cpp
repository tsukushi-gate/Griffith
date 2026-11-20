#include <torch/extension.h>

// lut_kernels.cu で定義
torch::Tensor lut_forward(
    torch::Tensor x,        // (in_dim, batch) 転置済み
    torch::Tensor indices,   // (n_inputs, out_dim)
    torch::Tensor w,         // (out_dim, 2^n_inputs) softmax/sigmoid適用済み
    int n_inputs
);

torch::Tensor lut_backward_w(
    torch::Tensor x,         // (in_dim, batch)
    torch::Tensor indices,    // (n_inputs, out_dim)
    torch::Tensor grad_y,     // (out_dim, batch)
    int n_inputs
);

torch::Tensor lut_backward_x(
    torch::Tensor x,                      // (in_dim, batch)
    torch::Tensor indices,                 // (n_inputs, out_dim)
    torch::Tensor w,                       // (out_dim, 2^n_inputs)
    torch::Tensor grad_y,                  // (out_dim, batch)
    torch::Tensor rev_starts,              // (in_dim + 1,)
    torch::Tensor rev_flat_y,              // (total_connections,)
    torch::Tensor rev_flat_slot,           // (total_connections,)
    int n_inputs
);

// IWP (Input-Wise Parameterization) 用カーネル。lut_kernels.cu で定義
torch::Tensor iwp_forward(
    torch::Tensor x,        // (in_dim, batch) 転置済み
    torch::Tensor indices,   // (n_inputs, out_dim)
    torch::Tensor p,         // (out_dim, n_inputs) = sigmoid(weights)適用済み
    int n_inputs
);

torch::Tensor iwp_backward_w(
    torch::Tensor x,         // (in_dim, batch)
    torch::Tensor indices,    // (n_inputs, out_dim)
    torch::Tensor p,          // (out_dim, n_inputs)
    torch::Tensor grad_y,     // (out_dim, batch)
    int n_inputs
);

torch::Tensor iwp_backward_x(
    torch::Tensor x,                      // (in_dim, batch)
    torch::Tensor indices,                 // (n_inputs, out_dim)
    torch::Tensor p,                       // (out_dim, n_inputs)
    torch::Tensor grad_y,                  // (out_dim, batch)
    torch::Tensor rev_starts,              // (in_dim + 1,)
    torch::Tensor rev_flat_y,              // (total_connections,)
    torch::Tensor rev_flat_slot,           // (total_connections,)
    int n_inputs
);

// WARP (Walsh-Hadamard係数パラメータ化) 用カーネル。lut_kernels.cu で定義
torch::Tensor warp_forward(
    torch::Tensor x,                       // (in_dim, batch) 転置済み
    torch::Tensor indices,                  // (n_inputs, out_dim)
    torch::Tensor weights,                  // (out_dim, 2^n_inputs) = θ（度数キャップ済み）
    double inv_tau,
    c10::optional<torch::Tensor> noise,     // (out_dim, batch) or None
    int n_inputs
);

torch::Tensor warp_backward_w(
    torch::Tensor x,           // (in_dim, batch)
    torch::Tensor indices,      // (n_inputs, out_dim)
    torch::Tensor grad_logit,   // (out_dim, batch) = dL/dz（/τ未適用の生の勾配）
    int n_inputs
);

torch::Tensor warp_backward_x(
    torch::Tensor x,                      // (in_dim, batch)
    torch::Tensor indices,                 // (n_inputs, out_dim)
    torch::Tensor weights,                 // (out_dim, 2^n_inputs) = θ（/τ未適用）
    torch::Tensor grad_logit,              // (out_dim, batch)
    torch::Tensor rev_starts,              // (in_dim + 1,)
    torch::Tensor rev_flat_y,              // (total_connections,)
    torch::Tensor rev_flat_slot,           // (total_connections,)
    int n_inputs
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &lut_forward, "LUT forward (CUDA)");
    m.def("backward_w", &lut_backward_w, "LUT backward weights (CUDA)");
    m.def("backward_x", &lut_backward_x, "LUT backward input (CUDA)");
    m.def("iwp_forward", &iwp_forward, "LUT IWP forward (CUDA)");
    m.def("iwp_backward_w", &iwp_backward_w, "LUT IWP backward weights (CUDA)");
    m.def("iwp_backward_x", &iwp_backward_x, "LUT IWP backward input (CUDA)");
    // noiseは呼び出し側(functions.py)が常にTensorかNoneを明示的に渡すため、
    // デフォルト値は設定しない(pybindの引数順デフォルト制約を回避)
    m.def("warp_forward", &warp_forward, "LUT WARP forward (CUDA)");
    m.def("warp_backward_w", &warp_backward_w, "LUT WARP backward weights (CUDA)");
    m.def("warp_backward_x", &warp_backward_x, "LUT WARP backward input (CUDA)");
}
