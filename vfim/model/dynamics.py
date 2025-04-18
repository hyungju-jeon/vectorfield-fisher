import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from torch.nn.functional import softplus
from vfim.utils.visualize import plot_vector_field


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


class LinearDynamics(nn.Module):
    def __init__(self, A, B, R, device="cpu"):
        super().__init__()
        self.A = nn.Parameter(A.to(device), requires_grad=True)
        self.B = nn.Parameter(B.to(device), requires_grad=True)
        self.device = device
        self.R = R

    def compute_param(self, x):
        return x @ self.A + self.B

    def sample_forward(self, x, input=None, k=1, var=None, return_trajectory=False):
        x_samples = [x]
        mus = []
        if var is None:
            var = self.R
        for i in range(k):
            if input is not None:
                mus.append(self(x_samples[i]) + x_samples[i] + input[i])
            else:
                mus.append(self(x_samples[i]) + x_samples[i])
            x_samples.append(
                mus[i] + torch.sqrt(var) * torch.randn_like(mus[i], device=x.device)
            )

        if return_trajectory:
            return torch.cat(x_samples, dim=-2)[..., 1:, :], torch.cat(mus, dim=-2), var
        else:
            return x_samples[-1], mus[-1], var

    def forward(self, x):
        return self.compute_param(x)


class RNNDynamics(nn.Module):
    """RNN-based dynamics model that predicts state transitions.

    Args:
        dx (int): Dimension of the state space.
        dh (int, optional): Hidden dimension size. Defaults to 256.
        residual (bool, optional): Whether to use residual connections. Defaults to True.
        fixed_variance (bool, optional): Whether to use fixed variance. Defaults to True.
        device (str, optional): Device to run the model on. Defaults to "cpu".
    """

    def __init__(self, dx, dh=128, fixed_variance=True, device="cpu"):
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
            nn.ReLU(),
            nn.Linear(dh, dh),
            nn.ReLU(),
            nn.Linear(dh, d_out),
        ).to(device)
        # Resnet
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

    def sample_forward(self, x, input=None, k=1, var=None, return_trajectory=False):
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
        if var is None:
            var = softplus(self.logvar) + eps

        x_samples, mus = [x], []
        for i in range(k):
            if input is not None:
                mus.append(self(x_samples[i]) + x_samples[i] + input[i])
            else:
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


class EnsembleRNN(nn.Module):
    """Ensemble of RNN-based dynamics models.

    Args:
        dx (int): Dimension of the state space
        n_models (int): Number of ensemble members
        dh (int, optional): Hidden dimension size. Defaults to 256
        device (str, optional): Device to run model on. Defaults to "cpu"
    """

    def __init__(self, dx, n_models=5, dh=256, device="cpu"):
        super().__init__()
        self.models = nn.ModuleList(
            [RNNDynamics(dx, dh=dh, device=device) for _ in range(n_models)]
        )
        self.n_models = n_models
        self.device = device

    def sample_forward(self, x, input=None, k=1, var=None, return_trajectory=False):
        """Generates samples using ensemble predictions.

        Each model in ensemble generates predictions independently.
        Final prediction is averaged across ensemble members.
        Variance includes both model uncertainty and prediction uncertainty.
        """
        all_samples = []
        all_means = []
        all_vars = []

        for model in self.models:
            samples, means, vars = model.sample_forward(
                x, input, k, var, return_trajectory
            )
            all_samples.append(samples)
            all_means.append(means)
            all_vars.append(vars)

        # Stack predictions from ensemble members
        samples = torch.stack(all_samples)  # (M, ..., T, D)
        means = torch.stack(all_means)  # (M, ..., T, D)
        vars = torch.stack(all_vars)  # (M, ..., D) or (M, D)

        # Combine predictions
        mean_prediction = means.mean(dim=0)
        # Total variance includes both model uncertainty and prediction uncertainty
        total_variance = vars.mean(dim=0) + means.var(dim=0)

        return samples, mean_prediction, total_variance

    def forward(self, x):
        """Forward pass averages predictions from all ensemble members."""
        predictions = []
        for model in self.models:
            predictions.append(model(x))
        return torch.stack(predictions)
