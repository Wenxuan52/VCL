# from models.variational import DiagGaussianPosterior
# q = DiagGaussianPosterior.from_prior((2,2))
# print(q.sample((5,)).shape)

import torch
from models.mlp import make_var_mlp
m = make_var_mlp([10, 20, 30])
x = torch.randn(4, 10)
y1 = m(x, sample=False)
y2 = m(x, sample=False)
assert torch.allclose(y1, y2)
