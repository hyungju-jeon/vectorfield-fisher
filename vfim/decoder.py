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

    def __init__(self, dx, dy, device="cpu"):
        """
        Initializes the NormalDecoder with input and output dimensions and the device to run on.
        Args:
          dx (int): The dimensionality of the input features.
          dy (int): The dimensionality of the output features.
          device (str, optional): The device to run the computations on. Default is "cpu".
        """
        super().__init__()
        self.device = device
        self.decoder = nn.Linear(dx, dy).to(device)
        self.logvar = nn.Parameter(
            0.01 * torch.randn(1, dy, device=device), requires_grad=True
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
        mu = self.decoder(x)
        var = softplus(self.logvar) + eps
        return mu, var

    def forward(self, samples, x):
        """Computes log probability of input data.

        Args:
            samples (torch.Tensor): Samples from latent space.
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Log probability of input data.
        """
        # given samples, compute parameters of likelihood
        mu, var = self.compute_param(samples)

        # now compute log prob
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x), (-1, -2))
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
        self.decoder = nn.Linear(d_latent, d_observation).to(device)

    def compute_param(self, x):
        log_rates = self.decoder(x)
        rates = softplus(log_rates)
        return rates

    def forward(self, x_samples, y):
        log_rates = self.decoder(x_samples)

        log_prob = -torch.nn.functional.poisson_nll_loss(
            log_rates, y, full=True, reduction="none"
        )

        return torch.sum(log_prob, (-1, -2))
