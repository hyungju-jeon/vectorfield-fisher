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
from vfim.models import SeqVae

# from vfim.visualize import plot_vector_field


if __name__ == "__main__":
    # Step 0: Set up random seed and key
    torch_seed = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # Step 1: Define true dynamics and observation model
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=torch_seed, w_attractor=0.01)

    # Observation model parameters: y = exp(C z + b) + noise.
    n_neurons = 50
    C = torch.randn(n_neurons, 2)
    b = torch.randn(n_neurons)

    dt = 0.5
    f_true = DynamicsWrapper(dt=dt, model=vf)

    # Step 2: Generate training trajectory (K x T x 2) and observation (K x T x D)
    K = 1000
    T = 200
    R, Q = 0.005, 0.005
    x0 = torch.rand(K, 2) * 5 - 2.5
    x_train = f_true.generate_trajectory(x0, T, R)
    y_train = (x_train @ C.T) + torch.randn(K, T, n_neurons) * np.sqrt(Q)

    # vf.streamplot()
    # for i in range(K):
    #     plt.plot(x_train[i][:, 0], x_train[i][:, 1])
    # plt.show()

    # Step 3: Initialize the inference model and train
    d_obs = 50
    d_latent = 2
    d_hidden = 16

    encoder = Encoder(d_obs, d_latent, d_hidden, device=device)
    rnn_dynamics = RNNDynamics(d_latent, device=device)
    decoder = NormalDecoder(d_latent, d_obs, device=device)
    f_hat = DynamicsWrapper(dt=dt, model=rnn_dynamics)
    vae = SeqVae(rnn_dynamics, encoder, decoder, device=device)

    batch_size = 64
    n_epochs = 500
    lr = 1e-3
    weight_decay = 1e-4

    dataloader = DataLoader(y_train, batch_size=batch_size)
    # Set random seed for reproducibility
    torch.manual_seed(torch_seed)

    param_list = list(vae.parameters())
    opt = torch.optim.AdamW(params=param_list, lr=lr, weight_decay=weight_decay)
    training_losses = []
    for _ in tqdm(range(n_epochs)):
        for batch in dataloader:
            opt.zero_grad()
            loss = vae.compute_elbo(batch.to(device))
            loss.backward()
            opt.step()
            with torch.no_grad():
                training_losses.append(loss.item())

    # Plot the vector field of the latent dynamics
    # Step 4: Compute FIM and CRLB from two initial points

    # Step 5: Update the inference model and evaluate the model accuracy
