import os
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
from ._ext import forward as _chamfer_forward_func
from ._ext import backward as _chamfer_backward_func
__all__ = ["ChamferDistanceL1", "ChamferDistanceL2"]

# # --- Lazy‐load the C++ extension only when first used ---
# _chamfer_ext = None
# def _ensure_chamfer_ext():
#     global _chamfer_ext
#     if _chamfer_ext is None:
#         src_dir = os.path.dirname(__file__)
#         build_dir = os.path.join(src_dir, "_ext")
#         os.makedirs(build_dir, exist_ok=True)
#         _chamfer_ext = load(
#             name="chamfer",
#             sources=[os.path.join(src_dir, "chamfer_cuda.cpp"),
#                      os.path.join(src_dir, "chamfer.cu")],
#             build_directory=build_dir,
#             verbose=False,
#             extra_cuda_cflags=[
#                 "-DCUDA_HAS_FP16=1",
#                 "-D__CUDA_NO_HALF_OPERATORS__",
#                 "-D__CUDA_NO_HALF_CONVERSIONS__",
#                 "-D__CUDA_NO_HALF2_OPERATORS__",
#             ],
#             extra_cflags=["-O2"],
#             with_cuda=True,
#         )
#     return _chamfer_ext

# Wrap into Function + Modules
class ChamferFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        dist1, dist2, idx1, idx2 = _chamfer_forward_func(xyz1, xyz2)
        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2

    @staticmethod
    def backward(ctx, grad_dist1, grad_dist2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors
        grad_xyz1, grad_xyz2 = _chamfer_backward_func(
            xyz1, xyz2, idx1, idx2, grad_dist1, grad_dist2
        )
        return grad_xyz1, grad_xyz2

class ChamferDistanceL2(nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            nonzero1 = xyz1.sum(-1).ne(0)
            nonzero2 = xyz2.sum(-1).ne(0)
            xyz1 = xyz1[nonzero1].unsqueeze(0)
            xyz2 = xyz2[nonzero2].unsqueeze(0)
        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return dist1.mean() + dist2.mean()

class ChamferDistanceL1(nn.Module):
    def __init__(self, ignore_zeros=False):
        super().__init__()
        self.ignore_zeros = ignore_zeros

    def forward(self, xyz1, xyz2):
        batch_size = xyz1.size(0)
        if batch_size == 1 and self.ignore_zeros:
            nonzero1 = xyz1.sum(-1).ne(0)
            nonzero2 = xyz2.sum(-1).ne(0)
            xyz1 = xyz1[nonzero1].unsqueeze(0)
            xyz2 = xyz2[nonzero2].unsqueeze(0)
        dist1, dist2 = ChamferFunction.apply(xyz1, xyz2)
        return 0.5 * (dist1.sqrt().mean() + dist2.sqrt().mean())
