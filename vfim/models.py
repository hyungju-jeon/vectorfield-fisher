import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.functional import softplus
from vfim.dynamics import RNNDynamics
from vfim.encoder import Encoder
from vfim.decoder import NormalDecoder


class SeqVae(nn.Module):
    """Sequential Variational Autoencoder (SeqVAE).

    Args:
        dynamics (nn.Module): The dynamics model used for sampling forward.
        state_encoder (nn.Module): The encoder model used to encode the input sequence.
        decoder (nn.Module): The decoder model used to reconstruct the input sequence.
        device (str, optional): The device to run the model on. Defaults to "cpu".
    """

    def __init__(self, dynamics, encoder, decoder, device="cpu"):
        super().__init__()

        self.dynamics = dynamics
        self.encoder = encoder
        self.decoder = decoder

        self.device = device

    @staticmethod
    def _kl_div(mu_q, var_q, mu_p, var_p):
        """Computes KL divergence between two multivariate normal distributions.

        Args:
            mu_q (torch.Tensor): Mean of the first distribution (q).
            var_q (torch.Tensor): Variance of the first distribution (q).
            mu_p (torch.Tensor): Mean of the second distribution (p).
            var_p (torch.Tensor): Variance of the second distribution (p).

        Returns:
            torch.Tensor: KL divergence between the two distributions.
        """
        kl_d = 0.5 * (
            torch.log(var_p / var_q)
            + ((mu_q - mu_p) ** 2) / var_p
            + (var_q / var_p)
            - 1
        )
        return torch.sum(kl_d, (-1, -2))

    def _compute_kld_x(self, mu_q, var_q, x_samples):
        """Computes KLD between the posterior and prior distribution.

        Args:
            mu_q (torch.Tensor): Mean of posterior distribution with shape (..., T, D).
            var_q (torch.Tensor): Variance of posterior distribution with shape (..., T, D).
            x_samples (torch.Tensor): Samples from distribution with shape (..., T+1, D).

        Returns:
            torch.Tensor: KL divergence between posterior and prior distributions.
        """

        _, mu_p_x, var_p_x = self.dynamics.sample_forward(x_samples[..., :-1, :])
        kl_d = self._kl_div(mu_q[..., 1:, :], var_q[..., 1:, :], mu_p_x, var_p_x)

        return kl_d

    def compute_elbo(self, y, n_samples=1, beta=1.0):
        """Computes Evidence Lower Bound (ELBO) for given data.

        Args:
            y (torch.Tensor): Observed data.
            n_samples (int, optional): Number of samples from variational posterior. Defaults to 1.
            beta (float, optional): Weight for KL divergence term. Defaults to 1.0.

        Returns:
            torch.Tensor: Negative ELBO value.
        """

        # samples from variational posterior
        x_samples, mu_q_x, var_q_x, log_q = self.encoder(y, n_samples=n_samples)

        kl_d_x = self._compute_kld_x(mu_q_x, var_q_x, x_samples)

        log_like = self.decoder(x_samples, y)

        elbo = torch.mean(log_like - beta * kl_d_x)

        return -elbo


if __name__ == "__main__":
    d_obs = 2
    d_latent = 2
    d_hidden = 16
    device = "cpu"

    encoder = Encoder(d_obs, d_latent, d_hidden, device=device)
    prior = RNNDynamics(d_latent, device=device)
    decoder = NormalDecoder(d_latent, d_obs, device=device)

    vae = SeqVae(prior, encoder, decoder, device=device)
