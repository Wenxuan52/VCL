import torch
import torch.nn as nn
import torch.nn.functional as F

from models.variational import DiagGaussianPosterior


class VarLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, prior=None):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.use_bias = bool(bias)

        self.q_weight = DiagGaussianPosterior.from_prior(
            (self.out_features, self.in_features)
        )
        self.q_bias = (
            DiagGaussianPosterior.from_prior((self.out_features,)) if self.use_bias else None
        )

        self.weight_prior = None
        self.bias_prior = None

        if prior is not None:
            if not isinstance(prior, tuple) or len(prior) != 2:
                raise ValueError("prior must be None or tuple(weight_prior, bias_prior)")
            self.set_prior(prior[0], prior[1])

    def set_prior(self, weight_prior: DiagGaussianPosterior, bias_prior: DiagGaussianPosterior | None):
        self.weight_prior = weight_prior
        self.bias_prior = bias_prior

    def forward(self, x, sample: bool = True):
        weight = self.q_weight.sample() if sample else self.q_weight.mean()
        bias = None
        if self.q_bias is not None:
            bias = self.q_bias.sample() if sample else self.q_bias.mean()
        return F.linear(x, weight, bias)

    def kl(self):
        if self.weight_prior is None:
            return self.q_weight.mu.new_tensor(0.0)

        kl_val = self.q_weight.kl_to(self.weight_prior)
        if self.q_bias is not None and self.bias_prior is not None:
            kl_val = kl_val + self.q_bias.kl_to(self.bias_prior)
        return kl_val


if __name__ == "__main__":
    x = torch.randn(8, 3)
    layer = VarLinear(3, 4)

    y = layer(x, sample=True)
    print(f"shape ok: {tuple(y.shape)} == {(8, 4)}")

    y_det1 = layer(x, sample=False)
    y_det2 = layer(x, sample=False)
    print(f"sample=False deterministic allclose: {torch.allclose(y_det1, y_det2)}")

    y_sto1 = layer(x, sample=True)
    y_sto2 = layer(x, sample=True)
    print(f"sample=True stochastic not allclose: {not torch.allclose(y_sto1, y_sto2)}")

    w_prior = DiagGaussianPosterior.from_prior((4, 3))
    b_prior = DiagGaussianPosterior.from_prior((4,))
    layer.set_prior(w_prior, b_prior)
    kl_val = layer.kl()
    print(f"kl={kl_val.item():.8f}, finite={torch.isfinite(kl_val).item()}, nonneg={(kl_val >= 0).item()}")
