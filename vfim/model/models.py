import torch
import torch.nn as nn
from tqdm import tqdm
from vfim.model.dynamics import RNNDynamics, EnsembleRNN


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

    def _compute_kld_x(self, mu_q, var_q, x_samples, input):
        """Computes KLD between the posterior and prior distribution.

        Args:
            mu_q (torch.Tensor): Mean of posterior distribution with shape (..., T, D).
            var_q (torch.Tensor): Variance of posterior distribution with shape (..., T, D).
            x_samples (torch.Tensor): Samples from distribution with shape (..., T+1, D).

        Returns:
            torch.Tensor: KL divergence between posterior and prior distributions.
        """

        _, mu_p_x, var_p_x = self.dynamics.sample_forward(
            x_samples[..., :-1, :], input[..., :-1, :]
        )
        kl_d = self._kl_div(mu_q[..., 1:, :], var_q[..., 1:, :], mu_p_x, var_p_x)

        return kl_d

    def compute_elbo(self, y, input=None, n_samples=1, beta=1.0):
        """Computes Evidence Lower Bound (ELBO) for given data.

        Args:
            y (torch.Tensor): Observed data.
            n_samples (int, optional): Number of samples from variational posterior.
            beta (float, optional): Weight for KL divergence term.

        Returns:
            torch.Tensor: Negative ELBO value.
        """
        x_samples, mu_q_x, var_q_x, log_q = self.encoder(y, n_samples=n_samples)
        kl_d_x = self._compute_kld_x(mu_q_x, var_q_x, x_samples, input=input)
        log_like = self.decoder.compute_log_prob(x_samples, y)
        elbo = torch.mean(log_like - beta * kl_d_x)
        return -elbo

    def train_model(
        self, dataloader, lr, weight_decay, n_epochs, has_input=False, verbose=True
    ):
        param_list = list(self.parameters())
        opt = torch.optim.AdamW(params=param_list, lr=lr, weight_decay=weight_decay)
        training_losses = []
        for _ in tqdm(range(n_epochs), disable=not verbose):
            for batch in dataloader:
                opt.zero_grad()
                if has_input:
                    y = batch[..., :-2]
                    input = batch[..., -2:]
                    loss = self.compute_elbo(y.to(self.device), input.to(self.device))
                else:
                    loss = self.compute_elbo(batch.to(self.device))
                loss.backward()
                opt.step()
                with torch.no_grad():
                    training_losses.append(loss.item())


class EnsembleSeqVae(nn.Module):
    """Ensemble Sequential VAE that trains multiple dynamics models independently.

    Args:
        dynamics (EnsembleRNN): Ensemble of dynamics models.
        encoder (nn.Module): Encoder network.
        decoder (nn.Module): Decoder network.
        device (str, optional): Device to run on. Defaults to "cpu".
    """

    def __init__(self, dynamics, encoder, decoder, device="cpu"):
        super().__init__()
        self.dynamics = dynamics
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    @staticmethod
    def _kl_div(mu_q, var_q, mu_p, var_p):
        """Same as SeqVae._kl_div"""
        kl_d = 0.5 * (
            torch.log(var_p / var_q)
            + ((mu_q - mu_p) ** 2) / var_p
            + (var_q / var_p)
            - 1
        )
        return torch.sum(kl_d, (-1, -2))

    def _compute_kld_x(self, mu_q, var_q, x_samples, input, idx):
        """Computes KLD between the posterior and prior distribution.

        Args:
            mu_q (torch.Tensor): Mean of posterior distribution with shape (..., T, D).
            var_q (torch.Tensor): Variance of posterior distribution with shape (..., T, D).
            x_samples (torch.Tensor): Samples from distribution with shape (..., T+1, D).
            idx (int): Index of the dynamics model to use.

        Returns:
            torch.Tensor: KL divergence between posterior and prior distributions.
        """

        _, mu_p_x, var_p_x = self.dynamics.models[idx].sample_forward(
            x_samples[..., :-1, :], input[..., :-1, :]
        )
        kl_d = self._kl_div(mu_q[..., 1:, :], var_q[..., 1:, :], mu_p_x, var_p_x)

        return kl_d

    def compute_elbo(self, y, idx, input=None, n_samples=1, beta=1.0):
        """Computes Evidence Lower Bound (ELBO) for given data.

        Args:
            y (torch.Tensor): Observed data.
            idx (int): Index of the dynamics model to use.
            n_samples (int, optional): Number of samples from variational posterior.
            beta (float, optional): Weight for KL divergence term.

        Returns:
            torch.Tensor: Negative ELBO value.
        """
        x_samples, mu_q_x, var_q_x, log_q = self.encoder(y, n_samples=n_samples)
        kl_d_x = self._compute_kld_x(mu_q_x, var_q_x, x_samples, input, idx)
        log_like = self.decoder.compute_log_prob(x_samples, y)
        elbo = torch.mean(log_like - beta * kl_d_x)
        return -elbo

    def train_model(
        self, dataloader, lr, weight_decay, n_epochs, has_input=False, verbose=True
    ):
        for i in range(self.dynamics.n_models):
            param_list = (
                list(self.dynamics.models[i].parameters())
                + list(self.encoder.parameters())
                + list(self.decoder.parameters())
            )

            opt = torch.optim.AdamW(params=param_list, lr=lr, weight_decay=weight_decay)
            training_losses = []
            for _ in tqdm(range(n_epochs), disable=not verbose):
                for batch in dataloader:
                    opt.zero_grad()
                    if has_input:
                        y = batch[..., :-2]
                        input = batch[..., -2:]
                        loss = self.compute_elbo(
                            y.to(self.device), input=input.to(self.device), idx=i
                        )
                    else:
                        loss = self.compute_elbo(batch.to(self.device), i)
                    loss.backward()
                    opt.step()
                    with torch.no_grad():
                        training_losses.append(loss.item())


if __name__ == "__main__":
    d_obs = 2
    d_latent = 2
    d_hidden = 16
    device = "cpu"

    # Example with single dynamics
    encoder = Encoder(d_obs, d_latent, d_hidden, device=device)
    prior = RNNDynamics(d_latent, device=device)
    decoder = NormalDecoder(d_latent, d_obs, device=device)
    vae = SeqVae(prior, encoder, decoder, device=device)

    # Example with ensemble dynamics
    ensemble_prior = EnsembleRNN(d_latent, n_models=5, device=device)
    ensemble_vae = EnsembleSeqVae(ensemble_prior, encoder, decoder, device=device)
