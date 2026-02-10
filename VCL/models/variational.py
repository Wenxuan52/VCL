import torch
from torch import nn


class DiagGaussianPosterior(nn.Module):
    def __init__(self, shape, init_mu=0.0, init_log_sigma=0.0, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        mu_init = torch.full(shape, float(init_mu), **factory_kwargs)
        log_sigma_init = torch.full(shape, float(init_log_sigma), **factory_kwargs)

        self.mu = nn.Parameter(mu_init)
        self.log_sigma = nn.Parameter(log_sigma_init)

    def std(self):
        # Clamp for numerical stability before exponentiation.
        return torch.exp(torch.clamp(self.log_sigma, min=-30.0, max=20.0))

    def sample(self, sample_shape=()):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        eps = torch.randn(sample_shape + tuple(self.mu.shape), device=self.mu.device, dtype=self.mu.dtype)
        return self.mu + self.std() * eps

    def mean(self):
        return self.mu

    def kl_to(self, other):
        if not isinstance(other, DiagGaussianPosterior):
            raise TypeError("other must be an instance of DiagGaussianPosterior")

        m1 = self.mu
        s1 = self.std()
        m0 = other.mu
        s0 = other.std()

        kl = torch.log(s0 / s1) + (s1.pow(2) + (m1 - m0).pow(2)) / (2.0 * s0.pow(2)) - 0.5
        return kl.sum()

    @classmethod
    def from_prior(cls, shape, prior_mu=0.0, prior_sigma=1.0, device=None, dtype=None):
        prior_sigma = float(prior_sigma)
        if prior_sigma <= 0:
            raise ValueError(f"prior_sigma must be > 0, got {prior_sigma}")
        return cls(
            shape=shape,
            init_mu=float(prior_mu),
            init_log_sigma=float(torch.log(torch.tensor(prior_sigma)).item()),
            device=device,
            dtype=dtype,
        )

    @classmethod
    def from_previous(cls, prev, log_sigma_init=-6.0):
        if not isinstance(prev, DiagGaussianPosterior):
            raise TypeError("prev must be an instance of DiagGaussianPosterior")

        new = cls(
            shape=tuple(prev.mu.shape),
            init_mu=0.0,
            init_log_sigma=float(log_sigma_init),
            device=prev.mu.device,
            dtype=prev.mu.dtype,
        )
        new.mu.data.copy_(prev.mu.data)
        return new


if __name__ == "__main__":
    q = DiagGaussianPosterior.from_prior((3, 4))
    kl_self = q.kl_to(q)
    print(f"KL(q || q) = {kl_self.item():.8f}")

    q2 = DiagGaussianPosterior.from_previous(q, log_sigma_init=-6)
    sample = q2.sample()
    print(f"sample shape correct: {tuple(sample.shape)} == {(3, 4)}")

    std = q2.std()
    print(f"std min={std.min().item():.8f}, max={std.max().item():.8f}")
    print(
        f"std finite={torch.isfinite(std).all().item()}, std nonnegative={(std >= 0).all().item()}"
    )
