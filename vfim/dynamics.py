import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.nn.functional import softplus


# Small constant to prevent numerical instability
eps = 1e-6


class DynamicsWrapper:
    """Wrapper class for dynamics models that handles trajectory generation.

    Args:
        model: Underlying dynamics model.
        dt (float): Time step size for trajectory generation.
    """

    def __init__(self, model, dt):
        self.dt = dt
        self.model = model

    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    @torch.no_grad()
    def generate_trajectory(self, x0, n_steps, R):
        # if x0 is a matrix of shape (n_samples, n_dim), generate n_samples trajectories
        if len(x0.shape) > 1:
            trajectories = []
            for x in x0:
                trajectories.append(self.generate_trajectory(x, n_steps, R))
            return torch.stack(trajectories)
        else:
            state = x0
            trajectory = [state]
            for _ in range(n_steps - 1):
                state = (
                    state
                    + (self.model(state) + np.sqrt(R) * torch.randn_like(state))
                    * self.dt
                )
                trajectory.append(state)
            return torch.stack(trajectory)


class RNNDynamics(nn.Module):
    """RNN-based dynamics model that predicts state transitions.

    Args:
        dx (int): Dimension of the state space.
        dh (int, optional): Hidden dimension size. Defaults to 256.
        residual (bool, optional): Whether to use residual connections. Defaults to True.
        fixed_variance (bool, optional): Whether to use fixed variance. Defaults to True.
        device (str, optional): Device to run the model on. Defaults to "cpu".
    """

    def __init__(self, dx, dh=256, residual=True, fixed_variance=True, device="cpu"):
        super().__init__()

        self.dx = dx
        self.residual = residual
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

        if self.residual:
            mu = mu + x

        return mu, var

    def sample_forward(self, x, k=1, return_trajectory=False):
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
            mus.append(self.compute_param(x_samples[i])[0])
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
