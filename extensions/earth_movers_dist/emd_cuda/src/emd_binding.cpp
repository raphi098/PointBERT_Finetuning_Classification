#include <torch/extension.h>
#include <vector>

// Declare the CUDA implementations from the .cu file.
at::Tensor approxmatch_forward_cuda(const at::Tensor xyz1,
                                    const at::Tensor xyz2);

at::Tensor matchcost_forward_cuda(const at::Tensor xyz1,
                                  const at::Tensor xyz2,
                                  const at::Tensor match);

std::vector<at::Tensor> matchcost_backward_cuda(const at::Tensor grad_cost,
                                                const at::Tensor xyz1,
                                                const at::Tensor xyz2,
                                                const at::Tensor match);

// Expose clean C++ API to Python
at::Tensor approxmatch_forward(const at::Tensor xyz1, const at::Tensor xyz2) {
    return approxmatch_forward_cuda(xyz1, xyz2);
}

at::Tensor matchcost_forward(const at::Tensor xyz1,
                             const at::Tensor xyz2,
                             const at::Tensor match) {
    return matchcost_forward_cuda(xyz1, xyz2, match);
}

std::vector<at::Tensor> matchcost_backward(const at::Tensor grad_cost,
                                           const at::Tensor xyz1,
                                           const at::Tensor xyz2,
                                           const at::Tensor match) {
    return matchcost_backward_cuda(grad_cost, xyz1, xyz2, match);
}

// Register functions into Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("approxmatch_forward", &approxmatch_forward, "ApproxMatch forward (CUDA)");
    m.def("matchcost_forward", &matchcost_forward, "MatchCost forward (CUDA)");
    m.def("matchcost_backward", &matchcost_backward, "MatchCost backward (CUDA)");
}
