import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus

eps = 1e-6


class Encoder(nn.Module):
    """Encoder class for a variational autoencoder with a GRU-based architecture.

    Args:
        dy (int): Dimension of the input features.
        dx (int): Dimension of the latent space.
        dh (int): Dimension of the hidden state in the GRU.
        device (str, optional): Device to run the model on. Defaults to "cpu".

    Attributes:
        dh (int): Dimension of the hidden state in the GRU.
        dx (int): Dimension of the latent space.
        gru (nn.GRU): Gated Recurrent Unit layer for encoding the input sequence.
        readout (nn.Linear): Linear layer for producing the parameters of the latent distribution.
        device (str): Device to run the model on.
    """

    def __init__(self, dy, dx, dh, device="cpu"):
        super().__init__()

        self.dh = dh
        self.dx = dx

        # Define a bidirectional GRU layer
        self.gru = nn.GRU(
            input_size=dy, hidden_size=dh, bidirectional=True, batch_first=True
        ).to(device)

        # Define a linear layer to produce the parameters of the latent distribution
        self.readout = nn.Linear(2 * dh, 2 * dx).to(device)

        self.device = device

    def compute_param(self, x):
        """Computes the mean and variance of the latent distribution.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).

        Returns:
            tuple:
                mu (torch.Tensor): Mean of the latent distribution.
                var (torch.Tensor): Variance of the latent distribution.
        """
        h, _ = self.gru(x)

        h = h.view(x.shape[0], x.shape[1], 2, self.dh)
        h_cat = torch.cat(
            (h[:, :, 0], h[:, :, 1]), -1
        )  # TODO: can we achieve this with one view
        out = self.readout(h_cat)
        mu, logvar = torch.split(out, [self.dx, self.dx], -1)
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
        samples = mu + torch.sqrt(var) * torch.randn(
            [n_samples] + list(mu.shape), device=self.device
        )
        return samples.squeeze(0), mu, var

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
