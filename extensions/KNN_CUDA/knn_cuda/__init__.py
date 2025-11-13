import os
import torch
import torch.nn as nn
from warnings import warn

__version__ = "0.2"

# Try to import the precompiled CUDA extension
from ._ext import knn as _knn_func

# User-facing KNN function
def knn(ref, query, k):
    d, i = _knn_func(ref, query, k)
    i -=1
    return d, i

# Convenience transpose helper
def _T(t, mode=False):
    return t.transpose(0, 1).contiguous() if mode else t

class KNN(nn.Module):
    def __init__(self, k, transpose_mode=False):
        super().__init__()
        self.k = k
        self._t = transpose_mode

    def forward(self, ref, query):
        assert ref.size(0) == query.size(0), \
            f"ref.shape={ref.shape} != query.shape={query.shape}"
        with torch.no_grad():
            D, I = [], []
            for r, q in zip(ref, query):
                r = _T(r, self._t)
                q = _T(q, self._t)
                d, i = knn(r.float(), q.float(), self.k)
                D.append(_T(d, self._t))
                I.append(_T(i, self._t))
            return torch.stack(D, dim=0), torch.stack(I, dim=0)
