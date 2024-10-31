import numpy as np
import torch
import torch.nn as nn

from midst_models.single_table_TabSyn.src.tabsyn.model.utils import EDMLoss

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

randn_like = torch.randn_like

SIGMA_MIN = 0.002
SIGMA_MAX = 80
rho = 7
S_churn = 1
S_min = 0
S_max = float("inf")
S_noise = 1


class Precond(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        sigma_min=0,  # Minimum supported noise level.
        sigma_max=float("inf"),  # Maximum supported noise level.
        sigma_data=0.5,  # Expected standard deviation of the training data.
    ):
        super().__init__()

        self.hid_dim = hid_dim
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.denoise_fn_F = denoise_fn

    def forward(self, x, sigma):
        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        dtype = torch.float32

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.denoise_fn_F((x_in).to(dtype), c_noise.flatten())

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


class Model(nn.Module):
    def __init__(
        self,
        denoise_fn,
        hid_dim,
        P_mean=-1.2,
        P_std=1.2,
        sigma_data=0.5,
        gamma=5,
        opts=None,
        pfgmpp=False,
    ):
        super().__init__()

        self.denoise_fn_D = Precond(denoise_fn, hid_dim)
        self.loss_fn = EDMLoss(
            P_mean, P_std, sigma_data, hid_dim=hid_dim, gamma=5, opts=None
        )

    def forward(self, x):
        loss = self.loss_fn(self.denoise_fn_D, x)
        return loss.mean(-1).mean()

    def sample(self, num_samples, dim, num_steps=50, device="cuda:0"):
        latents = torch.randn([num_samples, dim], device=device)

        step_indices = torch.arange(
            num_steps, dtype=torch.float32, device=latents.device
        )

        sigma_min = max(SIGMA_MIN, self.denoise_fn_D.sigma_min)
        sigma_max = min(SIGMA_MAX, self.denoise_fn_D.sigma_max)

        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat(
            [self.denoise_fn_D.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
        )

        x_next = latents.to(torch.float32) * t_steps[0]

        with torch.no_grad():
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                x_next = self.sample_step(
                    self.denoise_fn_D, num_steps, i, t_cur, t_next, x_next
                )

        return x_next

    def sample_step(self, num_steps, i, t_cur, t_next, x_next):
        x_cur = x_next
        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        )
        t_hat = self.denoise_fn_D.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat**2 - t_cur**2).sqrt() * S_noise * randn_like(x_cur)
        # Euler step.

        denoised = self.denoise_fn_D(x_hat, t_hat).to(torch.float32)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = self.denoise_fn_D(x_next, t_next).to(torch.float32)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_next
