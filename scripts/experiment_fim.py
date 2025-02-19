# %%
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from vfim.dynamics import DynamicsWrapper, RNNDynamics
from vfim.vector_field import VectorField
from vfim.encoder import Encoder
from vfim.decoder import NormalDecoder
from vfim.models import SeqVae, EnsembleSeqVae
from vfim.visualize import plot_vector_field


if __name__ == "__main__":
    # %% Step 0: Set up random seed and key
    torch_seed = 1
    torch.manual_seed(torch_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # %% Step 1: Define true dynamics and observation model
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=torch_seed, w_attractor=0.02)

    # Observation model parameters: y = exp(C z + b) + noise.
    n_neurons = 50
    C = torch.randn(n_neurons, 2)
    C = C / torch.norm(C, dim=1, keepdim=True)

    f_true = DynamicsWrapper(model=vf)

    # %% Step 2: Generate training trajectory (K x T x 2) and observation (K x T x D)
    K = 1000
    T = 200
    R = torch.tensor(1e-3)
    Q = torch.tensor(1e-3)
    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_star = f_true.generate_trajectory(x0, T, R)
    y_star = (x_star @ C.T) + torch.randn(K, T, n_neurons) * torch.sqrt(Q)

    # %% Step 3: Initialize VAE for 'true' f approximation
    d_obs = 50
    d_latent = 2
    d_hidden = 16

    encoder = Encoder(d_obs, d_latent, d_hidden, device=device)
    rnn_dynamics = RNNDynamics(d_latent, device=device)
    # Initialize decoder with pre-defined C matrix (known readout)
    decoder = NormalDecoder(d_latent, d_obs, device=device, l2=1.0, C=C)
    vae_star = SeqVae(rnn_dynamics, encoder, decoder, device=device)

    batch_size = 64
    n_epochs = 500
    lr = 1e-3
    weight_decay = 1e-4

    dataloader = DataLoader(y_star, batch_size=batch_size)
    vae_star.train_model(dataloader, lr, weight_decay, n_epochs)

    f_star = DynamicsWrapper(model=rnn_dynamics)

    # %% Step 4: Initialize VAE for f_hat with fewer training data
    K = 50
    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_train = f_true.generate_trajectory(x0, T, R)
    y_train = (x_train @ C.T) + torch.randn(K, T, n_neurons) * torch.sqrt(Q)

    encoder_hat = Encoder(d_obs, d_latent, d_hidden, device=device)
    rnn_dynamics_hat = RNNDynamics(d_latent, device=device)
    vae = SeqVae(rnn_dynamics_hat, encoder_hat, decoder, device=device)

    dataloader = DataLoader(y_train, batch_size=batch_size)
    vae_star.train_model(dataloader, lr, weight_decay, n_epochs)
    f_hat = DynamicsWrapper(model=rnn_dynamics_hat)

    # Step 5: Compute FIM and CRLB from two initial points

    # %% Step 4: Create Deep Ensemble Network to approximate the dynamics
    # n_ensemble = 5

    # dynamics_list = [RNNDynamics(d_latent, device=device) for _ in range(n_ensemble)]
    # # Initialize each dynamics differently
    # for dynamics in dynamics_list:
    #     dynamics.apply(
    #         lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None
    #     )
    # ensemble_vae = EnsembleSeqVae(dynamics_list, encoder, decoder, device=device)

    # Step 5: Update the inference model and evaluate the model accuracy
