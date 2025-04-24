import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus

eps = 1e-6


class Encoder(nn.Module):
    """Encoder class using a simple MLP applied per time step.

    Args:
        dy (int): Dimension of the input features.
        dx (int): Dimension of the latent space.
        dh (int): Dimension of the hidden layer in the MLP.
        device (str, optional): Device to run the model on. Defaults to "cpu".

    Attributes:
        dh (int): Dimension of the hidden layer in the MLP.
        dx (int): Dimension of the latent space.
        mlp (nn.Sequential): Multi-layer perceptron for encoding each time step.
        device (str): Device to run the model on.
    """

    def __init__(self, dy, dx, dh, device="cpu"):
        super().__init__()

        self.dh = dh
        self.dx = dx

        # Define an MLP to process each time step independently
        self.mlp = nn.Sequential(
            nn.Linear(dy, dh),
            nn.ReLU(),
            nn.Linear(dh, 2 * dx),  # Output mu and logvar for each time step
        ).to(device)

        self.device = device

    def compute_param(self, x):
        """Computes the mean and variance of the latent distribution for each time step.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).

        Returns:
            tuple:
                mu (torch.Tensor): Mean of the latent distribution (Batch, Time, dx).
                var (torch.Tensor): Variance of the latent distribution (Batch, Time, dx).
        """
        B, T, _ = x.shape
        # Reshape input to process each time step independently: (B * T, dy)
        x_flat = x.view(B * T, -1)

        # Apply MLP
        out_flat = self.mlp(x_flat)  # Shape: (B * T, 2 * dx)

        # Reshape output back to (B, T, 2 * dx)
        out = out_flat.view(B, T, 2 * self.dx)

        # Split into mu and logvar
        mu, logvar = torch.split(out, [self.dx, self.dx], dim=-1)
        var = softplus(logvar) + eps
        return mu, var

    def sample(self, x, n_samples=1):
        """Samples from the latent distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).
            n_samples (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            tuple:
                samples (torch.Tensor): Sampled latent variables.
                mu (torch.Tensor): Mean of the latent distribution.
                var (torch.Tensor): Variance of the latent distribution.
        """
        mu, var = self.compute_param(x)
        # Ensure mu and var have the correct shape (B, T, dx) before sampling
        # The shape should already be correct from compute_param
        samples = mu + torch.sqrt(var) * torch.randn(
            [n_samples] + list(mu.shape), device=self.device
        )
        # If n_samples is 1, remove the sample dimension
        if n_samples == 1:
            samples = samples.squeeze(0)
        return samples, mu, var

    def forward(self, x, n_samples=1):
        """Computes samples, mean, variance, and log probability of the latent distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).
            n_samples (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            tuple:
                samples (torch.Tensor): Sampled latent variables.
                mu (torch.Tensor): Mean of the latent distribution.
                var (torch.Tensor): Variance of the latent distribution.
                log_prob (torch.Tensor): Log probability of the samples.
        """
        # compute parameters and sample
        samples, mu, var = self.sample(x, n_samples=n_samples)
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(samples), (-2, -1))
        return samples, mu, var, log_prob
