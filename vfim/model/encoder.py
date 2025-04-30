import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.functional import softplus

eps = 1e-6


class StateEncoder(nn.Module):
    """Abstract base class for encoders"""

    def __init__(self, dy, dz, dh, device="cpu"):
        super().__init__()
        self.dz = dz
        self.dh = dh
        self.dy = dy
        self.network = None
        self.device = device

    def compute_param(self, z):
        """Computes the mean and variance of the latent distribution for each time step.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).

        Returns:
            tuple:
                mu (torch.Tensor): Mean of the latent distribution (Batch, Time, dz).
                var (torch.Tensor): Variance of the latent distribution (Batch, Time, dz).
        """
        B, T, _ = z.shape
        # Reshape input to process each time step independently: (B * T, dy)
        z_flat = z.view(B * T, -1)

        # Apply MLP
        param_flat = self.network(z_flat)  # Shape: (B * T, 2 * dz)

        # Reshape output back to (B, T, 2 * dz)
        params = param_flat.view(B, T, 2 * self.dz)

        # Split into mu and logvar
        mu, logvar = torch.split(params, [self.dz, self.dz], dim=-1)
        var = softplus(logvar) + eps
        return mu, var

    def sample(self, z, n_samples=1):
        """Samples from the latent distribution.

        Args:
            z (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).
            n_samples (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            tuple:
                samples (torch.Tensor): Sampled latent variables.
                mu (torch.Tensor): Mean of the latent distribution.
                var (torch.Tensor): Variance of the latent distribution.
        """
        mu, var = self.compute_param(z)

        samples = mu + torch.sqrt(var) * torch.randn(
            [n_samples] + list(mu.shape), device=self.device
        )
        # If n_samples is 1, remove the sample dimension
        if n_samples == 1:
            samples = samples.squeeze(0)
        return samples, mu, var

    def forward(self, z, n_samples=1):
        """Computes samples, mean, variance, and log probability of the latent distribution.

        Args:
            z (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).
            n_samples (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            tuple:
                samples (torch.Tensor): Sampled latent variables.
                mu (torch.Tensor): Mean of the latent distribution.
                var (torch.Tensor): Variance of the latent distribution.
                log_prob (torch.Tensor): Log probability of the samples.
        """
        # compute parameters and sample
        samples, mu, var = self.sample(z, n_samples=n_samples)
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(samples), (-2, -1))
        return samples, mu, var, log_prob


class LinearEncoder(StateEncoder):
    """Encoder class using a known linear transformation
    y = C * z + b -> z = inv(C^T * C) * C^T * (y - b)
    """

    def __init__(self, dy, dz, sigma_obs=None, C=None, b=None, device="cpu"):
        super().__init__(dy, dz, 0, device=device)
        self.network = nn.Sequential(
            nn.Linear(
                dy, dy, bias=True
            ),  # First layer with identity matrix and -b as bias
            nn.Linear(dy, dz, bias=False),  # Second layer with C_inv as weight
        )
        nn.init.eye_(self.network[0].weight)  # Set weights to identity matrix
        # If C or b is not provided, initialize them as nn.Parameters with gradients enabled
        if C is None:
            self.C = nn.Parameter(torch.randn(dy, dz), requires_grad=True)
        else:
            self.C = C.view(dy, dz)
        if b is None:
            self.b = nn.Parameter(torch.randn(dy), requires_grad=True)
        else:
            self.b = b.view(dy)
        self.C = self.C.to(device)
        self.b = self.b.to(device)
        # Update the network initialization based on the availability of C and b
        self.C_inv = torch.linalg.pinv(self.C)

        self.network[0].bias.data = -self.b  # Set bias to -b
        self.network[1].weight.data = self.C_inv  # Set second layer weight to C_inv
        if sigma_obs is None:
            self.sigma_obs = nn.Parameter(torch.randn(1), requires_grad=True)
        else:
            self.sigma_obs = sigma_obs.view(
                1
            )  # Ensure sigma_obs is a parameter of shape (1)

    def compute_param(self, y):
        """Computes the mean and variance of the latent distribution for each time step.

        Args:
            y (torch.Tensor): Input tensor of shape (Batch, Time, Dimension).

        Returns:
            tuple:
                mu (torch.Tensor): Mean of the latent distribution (Batch, Time, dz).
                var (torch.Tensor): Variance of the latent distribution (Batch, Time, dz).
        """
        B, T, _ = y.shape
        # Reshape input to process each time step independently: (B * T, dy)
        y_flat = y.view(B * T, -1)

        # Apply linear transformation using the defined network
        mu_flat = self.network(y_flat)
        # Reshape output back to (B, T, dz)
        mu = mu_flat.view(B, T, -1)
        # Compute variance from the known observation noise
        var = self.sigma_obs * torch.linalg.inv(torch.matmul(self.C.T, self.C)) + eps
        var = var.view(1, 1, -1).expand(B, T, -1)

        return mu, var


class MlpEncoder(StateEncoder):
    """Encoder class using a simple MLP applied per time step to infer latent variables from input features.

    Args:
        dy (int): Dimension of the input features.
        dz (int): Dimension of the latent space.
        dh (int): Dimension of the hidden layer in the MLP.
        device (str, optional): Device to run the model on. Defaults to "cpu".

    Attributes:
        dh (int): Dimension of the hidden layer in the MLP.
        dz (int): Dimension of the latent space.
        mlp (nn.Sequential): Multi-layer perceptron for encoding each time step.
        device (str): Device to run the model on.
    """

    def __init__(self, dy, dz, dh, device="cpu"):
        super().__init__(dy, dz, dh, device=device)

        # Define an MLP to process each time step independently
        self.network = nn.Sequential(
            nn.Linear(dy, dh),
            nn.ReLU(),
            nn.Linear(dh, 2 * dz),  # Output mu and logvar for each time step
        ).to(device)


class EmbeddingEncoder(nn.Module):
    """Encoder class using an embedding layer to infer latent variables from input features.

    Args:
        dy (int): Dimension of the input features.
        dz (int): Dimension of the latent space.
        dh (int): Dimension of the hidden layer in the MLP.
        device (str, optional): Device to run the model on. Defaults to "cpu".

    Attributes:
        dh (int): Dimension of the hidden layer in the MLP.
        dz (int): Dimension of the latent space.
        mlp (nn.Sequential): Multi-layer perceptron for encoding each time step.
        device (str): Device to run the model on.
    """

    def __init__(self, dy, dz, dh, device="cpu"):
        super().__init__()

        self.dh = dh
        self.dz = dz

        # Define an MLP to process each time step independently
        self.mlp = nn.Sequential(
            nn.Linear(dy, dh),
            nn.ReLU(),
            nn.Linear(dh, 2 * dz),  # Output mu and logvar for each time step
        ).to(device)

        self.device = device


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

    def __init__(self, dx, dy, device="cpu", l2=0.0, C=None, b=None):
        """
        Initializes the NormalDecoder with input and output dimensions and the device to run on.
        Args:
          dx (int): The dimensionality of the input features.
          dy (int): The dimensionality of the output features.
          device (str, optional): The device to run the computations on. Default is "cpu".
        """
        super().__init__()
        self.device = device
        if C is not None:
            self.linear = nn.Linear(dx, dy, bias=True).to(device).requires_grad_(False)
            self.linear.weight.data = C
            self.linear.bias.data = b
            # decoder ~ exp(Linear(x))
            self.decoder = nn.Sequential(
                self.linear,
                Exp(),  # Use Exp() instead of nn.Softplus()
            ).to(device)

        else:
            # Remove bias from linear layer
            self.decoder = nn.Linear(dx, dy, bias=True).to(device)
            self.normalize_weights()  # Initialize with normalized weights
        self.logvar = nn.Parameter(
            0.001 * torch.randn(1, dy, device=device), requires_grad=True
        )
        self.l2 = l2

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
        mu = self.decoder(x)  # This will apply Linear -> Exp if C is provided
        var = softplus(self.logvar) + eps
        return mu, var

    def get_regularization(self):
        """Computes regularization term for decoder weights."""
        weight_norm = torch.norm(self.decoder.weight)
        return self.l2 * (weight_norm - 1).pow(
            2
        )  # Penalty for deviating from unit norm

    def compute_log_prob(self, samples, x):
        """Computes log probability of input data.

        Args:
            samples (torch.Tensor): Samples from latent space.
            x (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Log probability of input data.
        """
        # Normalize weights before forward pass
        # self.normalize_weights()
        # given samples, compute parameters of likelihood
        mu, var = self.compute_param(samples)

        # now compute log prob
        log_prob = torch.sum(Normal(mu, torch.sqrt(var)).log_prob(x), (-1, -2))

        # add regularization
        # if self.l2 > 0:
        #     log_prob = log_prob - self.get_regularization()

        return log_prob

    def forward(self, samples):
        mu, _ = self.compute_param(samples)
        return mu
