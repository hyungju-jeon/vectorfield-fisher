from networkx import selfloop_edges
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.nn.functional import softplus
from vfim.visualize import plot_vector_field


# Small constant to prevent numerical instability
eps = 1e-6


class DynamicsWrapper:
    """Wrapper class for dynamics models that handles trajectory generation.

    Args:
        model: Underlying dynamics model.
    """

    def __init__(self, model):
        self.model = model

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    @torch.no_grad()
    def generate_trajectory(self, x0, n_steps, R):
        return self.model.sample_forward(x0, k=n_steps, var=R, return_trajectory=True)[
            0
        ]

    @torch.no_grad()
    def streamplot(self, **kwargs):
        return plot_vector_field(self.model, **kwargs)


class RNNDynamics(nn.Module):
    """RNN-based dynamics model that predicts state transitions.

    Args:
        dx (int): Dimension of the state space.
        dh (int, optional): Hidden dimension size. Defaults to 256.
        residual (bool, optional): Whether to use residual connections. Defaults to True.
        fixed_variance (bool, optional): Whether to use fixed variance. Defaults to True.
        device (str, optional): Device to run the model on. Defaults to "cpu".
    """

    def __init__(self, dx, dh=256, fixed_variance=True, device="cpu"):
        super().__init__()

        self.dx = dx
        self.fixed_variance = fixed_variance

        if fixed_variance:
            self.logvar = nn.Parameter(
                -2 * torch.randn(1, dx, device=device), requires_grad=True
            )
            d_out = dx
        else:
            d_out = 2 * dx

        self.prior = nn.Sequential(
            nn.Linear(dx, dh),
            nn.Tanh(),
            nn.Linear(dh, dh),
            nn.Tanh(),
            nn.Linear(dh, d_out),
        ).to(device)
        self.device = device

    @torch.no_grad()
    def compute_var(self):
        """Computes the variance of the state transition distribution.

        Returns:
            torch.Tensor: Variance of the state transition distribution.
        """
        return softplus(self.logvar) + eps

    def compute_param(self, x):
        """Computes mean and variance parameters of state transition distribution.

        Args:
            x (torch.Tensor): Input state tensor of shape (..., dx).

        Returns:
            tuple:
                mu (torch.Tensor): Mean of state transition distribution.
                var (torch.Tensor): Variance of state transition distribution.
        """
        out = self.prior(x)

        if self.fixed_variance:
            mu = out
            var = softplus(self.logvar) + eps
        else:
            mu, logvar = torch.split(out, [self.dx, self.dx], -1)
            var = softplus(logvar) + eps

        return mu, var

    def sample_forward(self, x, k=1, var=None, return_trajectory=False):
        """Generates samples from forward dynamics model.

        Args:
            x (torch.Tensor): Initial state tensor of shape (..., dx).
            k (int, optional): Number of forward steps. Defaults to 1.
            return_trajectory (bool, optional): If True, returns full trajectory. Defaults to False.

        Returns:
            If return_trajectory is False:
                tuple:
                    x_sample (torch.Tensor): Final sampled state.
                    mu (torch.Tensor): Final predicted mean.
                    var (torch.Tensor): Prediction variance.
            If return_trajectory is True:
                tuple:
                    trajectory (torch.Tensor): Full trajectory of sampled states.
                    means (torch.Tensor): Full trajectory of predicted means.
                    var (torch.Tensor): Prediction variance.
        """
        var = softplus(self.logvar) + eps

        x_samples, mus = [x], []
        for i in range(k):
            mus.append(self(x_samples[i]) + x_samples[i])
            x_samples.append(
                mus[i] + torch.sqrt(var) * torch.randn_like(mus[i], device=x.device)
            )

        if return_trajectory:
            return torch.cat(x_samples, dim=-2)[..., 1:, :], torch.cat(mus, dim=-2), var
        else:
            return x_samples[-1], mus[-1], var

    def _compute_log_prob(self, mu, var, x):
        """Computes log probability of observing state x given predicted distribution.

        Args:
            mu (torch.Tensor): Mean of predicted distribution.
            var (torch.Tensor): Variance of predicted distribution.
            x (torch.Tensor): Observed state.

        Returns:
            torch.Tensor: Log probability of observing x.
        """
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x), (-2, -1))
        return log_prob

    def forward(self, x_prev):
        """
        Predicts next state mean given current state.

        Args:
            x_prev (torch.Tensor): Current state tensor

        Returns:
            torch.Tensor: Predicted mean of next state
        """
        mu, _ = self.compute_param(x_prev)
        return mu
