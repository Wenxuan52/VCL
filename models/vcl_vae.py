import torch
import torch.nn as nn

from models.mlp import make_var_mlp


class TaskEncoder(nn.Module):
    def __init__(self, z_dim: int = 50, x_dim: int = 784, hidden_dim: int = 500):
        super().__init__()
        self.x_dim = x_dim
        self.z_dim = z_dim

        self.net = nn.Sequential(
            nn.Linear(x_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, z_dim)
        self.fc_logvar = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        if x.dim() == 4:
            x = x.view(x.size(0), -1)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"x must be [B,1,28,28] or [B,784], got shape={tuple(x.shape)}")

        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class HeadZtoH(nn.Module):
    def __init__(self, z_dim: int = 50, h_dim: int = 500):
        super().__init__()
        self.net = make_var_mlp([z_dim, 500, h_dim], activation="relu", bias=True)

    def forward(self, z, sample: bool = True):
        return self.net(z, sample=sample)

    def kl(self):
        return self.net.kl()


class SharedHtoX(nn.Module):
    def __init__(self, h_dim: int = 500, x_dim: int = 784):
        super().__init__()
        self.x_dim = x_dim
        self.net = make_var_mlp([h_dim, 500, x_dim], activation="relu", bias=True)

    def forward(self, h, sample: bool = True):
        # Return reconstruction logits (no sigmoid here).
        return self.net(h, sample=sample)

    def kl(self):
        return self.net.kl()


class MultiHeadVCLVAE(nn.Module):
    def __init__(self, z_dim: int = 50, h_dim: int = 500, x_dim: int = 784):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim
        self.x_dim = x_dim

        self.encoders = nn.ModuleDict()
        self.heads = nn.ModuleDict()
        self.shared = SharedHtoX(h_dim=h_dim, x_dim=x_dim)

    def ensure_task(self, task_id):
        key = str(task_id)

        # Keep newly created task modules on the same device/dtype as the shared module.
        ref_param = next(self.shared.parameters())
        target_device = ref_param.device
        target_dtype = ref_param.dtype

        if key not in self.encoders:
            self.encoders[key] = TaskEncoder(
                z_dim=self.z_dim, x_dim=self.x_dim, hidden_dim=500
            ).to(device=target_device, dtype=target_dtype)
        if key not in self.heads:
            self.heads[key] = HeadZtoH(z_dim=self.z_dim, h_dim=self.h_dim).to(
                device=target_device, dtype=target_dtype
            )

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, task_id, x, sample_theta: bool = True):
        self.ensure_task(task_id)
        key = str(task_id)

        mu, logvar = self.encoders[key](x)
        z = self.reparameterize(mu, logvar)

        h = self.heads[key](z, sample=sample_theta)
        recon_logits = self.shared(h, sample=sample_theta)

        # Per-sample KL(q(z|x) || p(z)) against standard normal prior.
        kl_z = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)

        return {
            "recon_logits": recon_logits,
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "kl_z": kl_z,
        }

    def decode(self, task_id, z, sample_theta: bool = False, reshape: bool = True):
        """
        Decode latent z to reconstruction logits.
        - reshape=False: returns logits with shape [B, 784]
        - reshape=True: returns logits reshaped to [B, 1, 28, 28]
        """
        self.ensure_task(task_id)
        key = str(task_id)

        h = self.heads[key](z, sample=sample_theta)
        logits = self.shared(h, sample=sample_theta)

        if reshape:
            return logits.view(logits.size(0), 1, 28, 28)
        return logits


if __name__ == "__main__":
    model = MultiHeadVCLVAE()
    task_id = 0

    x = torch.rand(8, 1, 28, 28)
    out = model.forward(task_id, x, sample_theta=False)
    print("forward:")
    print(f"  recon_logits shape: {tuple(out['recon_logits'].shape)}")
    print(f"  mu shape: {tuple(out['mu'].shape)}")
    print(f"  logvar shape: {tuple(out['logvar'].shape)}")
    print(f"  kl_z shape: {tuple(out['kl_z'].shape)}")

    z = torch.randn(8, 50)
    decoded = model.decode(task_id, z, sample_theta=False, reshape=True)
    print("decode:")
    print(f"  decoded shape: {tuple(decoded.shape)}")
