"""
Stage B diffusion process: standard DDPM (Ho et al. 2020) forward process +
epsilon-prediction training loss + DDIM (Song et al. 2020) sampler for fast
inference. Operates on latent vectors (dim=32 by default), not point clouds.
"""
import torch


class GaussianDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.T = T
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, T, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(self, z0, k, noise=None):
        """Forward process: sample z_k ~ q(z_k | z0)."""
        if noise is None:
            noise = torch.randn_like(z0)
        sqrt_ac = self.sqrt_alphas_cumprod[k].unsqueeze(-1)
        sqrt_omac = self.sqrt_one_minus_alphas_cumprod[k].unsqueeze(-1)
        return sqrt_ac * z0 + sqrt_omac * noise, noise

    def training_loss(self, model, z0, t_phys, dataset_id=None):
        """Standard epsilon-prediction MSE loss."""
        B = z0.shape[0]
        k = torch.randint(0, self.T, (B,), device=z0.device)
        z_noisy, noise = self.q_sample(z0, k)
        pred_noise = model(z_noisy, k, t_phys, dataset_id)
        return torch.nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddim_sample(self, model, shape, t_phys, dataset_id=None, num_steps=50, eta=0.0):
        """DDIM sampling: shape=(B, latent_dim). t_phys: (B,) physical times
        to condition on (held fixed throughout the reverse process)."""
        device = self.device
        z = torch.randn(shape, device=device)
        step_indices = torch.linspace(self.T - 1, 0, num_steps, device=device).long()

        for i in range(num_steps):
            k = step_indices[i].expand(shape[0])
            pred_noise = model(z, k, t_phys, dataset_id)

            alpha_bar_k = self.alphas_cumprod[k].unsqueeze(-1)
            if i < num_steps - 1:
                k_prev = step_indices[i + 1].expand(shape[0])
                alpha_bar_prev = self.alphas_cumprod[k_prev].unsqueeze(-1)
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_k)

            z0_pred = (z - torch.sqrt(1 - alpha_bar_k) * pred_noise) / torch.sqrt(alpha_bar_k)
            sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_k)) * torch.sqrt(1 - alpha_bar_k / alpha_bar_prev)
            dir_z = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma ** 2, min=0)) * pred_noise
            noise = torch.randn_like(z) if eta > 0 else 0.0
            z = torch.sqrt(alpha_bar_prev) * z0_pred + dir_z + sigma * noise

        return z
