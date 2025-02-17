import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus

eps = 1e-6


class NormalDecoder(nn.Module):
    """Neural network module that decodes input features into normal distribution.

    Args:
        dx (int): Dimensionality of input features.
        dy (int): Dimensionality of output features.
        device (str, optional): Device to run computations on. Defaults to "cpu".

    Attributes:
        device (str): Device to run computations on.
        decoder (nn.Linear): Linear layer mapping input to output features.
        logvar (nn.Parameter): Learnable parameter for log variance.
    """

    def __init__(self, dx, dy, device="cpu", l2=0.0):
        """
        Initializes the NormalDecoder with input and output dimensions and the device to run on.
        Args:
          dx (int): The dimensionality of the input features.
          dy (int): The dimensionality of the output features.
          device (str, optional): The device to run the computations on. Default is "cpu".
        """
        super().__init__()
        self.device = device
        # Remove bias from linear layer
        self.decoder = nn.Linear(dx, dy, bias=False).to(device)
        self.logvar = nn.Parameter(
            0.01 * torch.randn(1, dy, device=device), requires_grad=True
        )
        self.l2 = l2
        self.normalize_weights()  # Initialize with normalized weights

    def normalize_weights(self):
        """Normalize decoder matrix C to have unit Frobenius norm."""
        with torch.no_grad():
            weight_norm = torch.norm(self.decoder.weight, dim=1)
            if all(weight_norm) > 0:
                self.decoder.weight.data = (
                    self.decoder.weight.data / weight_norm[:, None]
                )

    def compute_param(self, x):
        """Computes mean and variance of normal distribution given input features.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple:
                mu (torch.Tensor): Mean of normal distribution.
                var (torch.Tensor): Variance of normal distribution.
        """
        mu = self.decoder(x)  # This is now just matrix multiplication
        var = softplus(self.logvar) + eps
        return mu, var

    def get_regularization(self):
        """Computes regularization term for decoder weights."""
        weight_norm = torch.norm(self.decoder.weight)
        return self.l2 * (weight_norm - 1).pow(
            2
        )  # Penalty for deviating from unit norm

    def forward(self, samples, x):
        """Computes log probability of input data.

        Args:
            samples (torch.Tensor): Samples from latent space.
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Log probability of input data.
        """
        # Normalize weights before forward pass
        self.normalize_weights()
        # given samples, compute parameters of likelihood
        mu, var = self.compute_param(samples)

        # now compute log prob
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x), (-1, -2))

        # add regularization
        if self.l2 > 0:
            log_prob = log_prob - self.get_regularization()

        return log_prob


class PoissonDecoder(nn.Module):
    """Decoder module that outputs Poisson distribution parameters.

    Args:
        d_latent (int): Dimension of latent space.
        d_observation (int): Dimension of observation space.
        l2 (float, optional): L2 regularization parameter. Defaults to 0.0.
        device (str, optional): Device to run computations on. Defaults to "cpu".
    """

    def __init__(self, d_latent, d_observation, l2=0.0, device="cpu"):
        super().__init__()
        self.device = device
        # Remove bias from linear layer
        self.decoder = nn.Linear(d_latent, d_observation, bias=False).to(device)
        self.l2 = l2
        self.normalize_weights()  # Initialize with normalized weights

    def normalize_weights(self):
        """Normalize decoder matrix C to have unit Frobenius norm."""
        with torch.no_grad():
            weight_norm = torch.norm(self.decoder.weight)
            if weight_norm > 0:
                self.decoder.weight.data = self.decoder.weight.data / weight_norm

    def compute_param(self, x):
        """Compute rates = softplus(Cx) where C is normalized."""
        log_rates = self.decoder(x)  # This is now just matrix multiplication
        rates = softplus(log_rates)
        return rates

    def get_regularization(self):
        """Computes regularization term for decoder weights."""
        weight_norm = torch.norm(self.decoder.weight)
        return self.l2 * (weight_norm - 1).pow(
            2
        )  # Penalty for deviating from unit norm

    def forward(self, x_samples, y):
        # Normalize weights before forward pass
        self.normalize_weights()
        log_rates = self.decoder(x_samples)

        log_prob = -torch.nn.functional.poisson_nll_loss(
            log_rates, y, full=True, reduction="none"
        )

        log_prob = torch.sum(log_prob, (-1, -2))

        # add regularization
        if self.l2 > 0:
            log_prob = log_prob - self.get_regularization()

        return log_prob
