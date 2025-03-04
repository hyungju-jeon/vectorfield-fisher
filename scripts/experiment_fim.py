# %%
import torch
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader
from vfim.dynamics import DynamicsWrapper, RNNDynamics, LinearDynamics
from vfim.vector_field import VectorField
from vfim.encoder import Encoder
from vfim.decoder import NormalDecoder
from vfim.models import SeqVae, EnsembleSeqVae
from vfim.information import FisherMetrics
from vfim.visualize import plot_vector_field


def test_linear_dynamics():
    # test_RNN_dynamics()
    # #  Step 0: Set up random seed and key
    torch_seed = 1
    torch.manual_seed(torch_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # # Observation model parameters: y = exp(C z + b) + noise.
    n_neurons = 50
    d_latent = 2
    C = torch.randn(n_neurons, 2)
    C = C / torch.norm(C, dim=1, keepdim=True)
    A = torch.rand(d_latent, d_latent, device=device)
    A = A / torch.norm(A, dim=1, keepdim=True)
    A = A * 0.01

    # #  Step 2: Generate training trajectory (K x T x 2) and observation (K x T x D)
    R = torch.tensor(1e-3)
    Q = torch.tensor(1e-3)
    B = torch.rand(d_latent, device=device) * 0.1

    linear_dynamics = LinearDynamics(A, B, R, device=device)
    linear_dynamics.requires_grad_(False)
    f_true = DynamicsWrapper(model=linear_dynamics)

    # #  Step 3: Initialize VAE for 'true' f approximation
    d_obs = 50
    d_latent = 2
    decoder = NormalDecoder(d_latent, d_obs, device=device, l2=1.0, C=C)

    batch_size = 64
    n_epochs = 500
    lr = 1e-3
    weight_decay = 1e-4

    # dataloader = DataLoader(y_star, batch_size=batch_size)
    # vae_star.train_model(dataloader, lr, weight_decay, n_epochs)

    # f_star = DynamicsWrapper(model=linear_dynamics)

    # #  Step 4: Initialize VAE for f_hat with fewer training data
    K = 100
    T = 30
    d_hidden = 16
    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_train = f_true.generate_trajectory(x0, T, R)
    y_train = (x_train @ C.T) + torch.randn(K, T, n_neurons) * torch.sqrt(Q)

    encoder = Encoder(d_obs, d_latent, d_hidden, device=device)
    vae = SeqVae(linear_dynamics, encoder, decoder, device=device)

    dataloader = DataLoader(y_train, batch_size=batch_size)
    vae.train_model(dataloader, lr, weight_decay, n_epochs)
    # f_hat = DynamicsWrapper(model=linear_dynamics_hat)

    # #  Step 5: Compute FIM and CRLB from two initial points
    linear_dynamics_hat = LinearDynamics(
        A=A + torch.rand_like(A) * 1e-3,
        B=B + torch.rand_like(B) * 1e-2,
        R=R,
        device=device,
    )

    num_test = 5
    T_test = 10
    fisher = FisherMetrics(
        dynamics=linear_dynamics_hat,
        decoder=decoder,
        process_noise=R * torch.eye(d_latent),
        measurement_noise=Q * torch.eye(d_obs),
    )
    initial_states = torch.rand(num_test, d_latent) * 5 - 2.5
    initial_cov = torch.eye(d_latent, device=device)

    fims = fisher.compute_fim(initial_states, T_test, initial_cov, use_diag=False)
    CRLB = [torch.trace(torch.inverse(fim.cpu())) for fim in fims]

    # #  Step 6: Monte Carlo simulation to estimate CRLB
    n_mc = 500
    error_state = []
    x_mc = f_true.generate_trajectory(initial_states.unsqueeze(1), T_test, R)
    vae_mcs = []

    # # Create a single reusable network instance

    linear_mc = LinearDynamics(
        A=torch.rand_like(A), B=torch.rand_like(B), R=R, device=device
    )
    encoder_mc = Encoder(d_obs, d_latent, d_hidden, device=device)
    encoder_mc.load_state_dict(encoder.state_dict())
    encoder_mc.requires_grad_(False)
    vae_mc = SeqVae(linear_mc, encoder_mc, decoder, device=device)
    for i in range(num_test):
        errors = []
        for _ in range(n_mc):
            # Simply load parameters from rnn_dynamics_hat
            linear_mc.load_state_dict(linear_dynamics_hat.state_dict())

            y_mc = (x_mc[i] @ C.T) + torch.randn(1, T_test, n_neurons) * torch.sqrt(Q)
            dataloader = DataLoader(y_mc, batch_size=1)
            vae_mc.train_model(dataloader, 1e-4, weight_decay, 15)

            # compute mse for each of dynamics parameter
            with torch.no_grad():
                theta_star = torch.cat(
                    [p.flatten() for p in linear_dynamics.parameters()]
                )
                theta_mc = torch.cat([p.flatten() for p in linear_mc.parameters()])
                error = torch.norm(theta_star - theta_mc)
            errors.append(error)

        vae_mcs.append(copy.deepcopy(vae_mc))
        error_state.append(torch.tensor(errors).cpu())

    # #
    mean_error = torch.stack(error_state).mean(dim=1).cpu()
    std_error = torch.stack(error_state).std(dim=1).cpu()

    # scatter plot fim x error_state. Repeat fim n_mc times

    plt.errorbar(torch.arange(num_test).cpu(), mean_error, yerr=std_error)
    # fims_scatter = torch.tensor(fims).repeat(n_mc, 1).cpu()
    # plt.scatter(fims_scatter, error_state)
    print(CRLB)

    # #
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    f_true.streamplot(ax=ax[0, 0])
    for i in range(1, 6):
        plot_vector_field(vae_mcs[i - 1].dynamics, ax=ax[i // 3, i % 3])
        plt.plot(x_mc.cpu()[i - 1, :, 0], x_mc.cpu()[i - 1, :, 1])
        plt.title(f"mean error: {mean_error[i-1]:.4f}, Information: {CRLB[i-1]:.4f}")
    plt.show()


def test_RNN_dynamics():
    #  Step 0: Set up random seed and key
    torch_seed = 1
    torch.manual_seed(torch_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    #  Step 1: Define true dynamics and observation model
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=torch_seed, w_attractor=0.02)

    # Observation model parameters: y = exp(C z + b) + noise.
    n_neurons = 50
    C = torch.randn(n_neurons, 2)
    C = C / torch.norm(C, dim=1, keepdim=True)

    f_true = DynamicsWrapper(model=vf)

    #  Step 2: Generate training trajectory (K x T x 2) and observation (K x T x D)
    K = 1000
    T = 200
    R = torch.tensor(1e-3)
    Q = torch.tensor(1e-3)
    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_star = f_true.generate_trajectory(x0, T, R)
    y_star = (x_star @ C.T) + torch.randn(K, T, n_neurons) * torch.sqrt(Q)

    #  Step 3: Initialize VAE for 'true' f approximation
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

    #  Step 4: Initialize VAE for f_hat with fewer training data
    K = 20
    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_train = f_true.generate_trajectory(x0, T, R)
    y_train = (x_train @ C.T) + torch.randn(K, T, n_neurons) * torch.sqrt(Q)

    encoder_hat = Encoder(d_obs, d_latent, d_hidden, device=device)
    rnn_dynamics_hat = RNNDynamics(d_latent, device=device)
    vae = SeqVae(rnn_dynamics_hat, encoder_hat, decoder, device=device)

    dataloader = DataLoader(y_train, batch_size=batch_size)
    vae.train_model(dataloader, lr, weight_decay, n_epochs)
    f_hat = DynamicsWrapper(model=rnn_dynamics_hat)

    #  Step 5: Compute FIM and CRLB from two initial points
    num_test = 5
    T_test = 50
    fisher = FisherMetrics(
        dynamics=rnn_dynamics_hat,
        decoder=decoder,
        process_noise=R * torch.eye(d_latent),
        measurement_noise=Q * torch.eye(d_obs),
    )
    initial_states = torch.rand(num_test, d_latent) * 5 - 2.5
    initial_cov = torch.eye(d_latent, device=device)

    fims = fisher.compute_fim(initial_states, T_test, initial_cov, use_diag=True)
    fims = [torch.sum(1 / fim) for fim in fims]

    #  Step 6: Monte Carlo simulation to estimate CRLB
    n_mc = 100
    error_state = []
    x_mc = f_star.generate_trajectory(initial_states.unsqueeze(1), T_test, R)
    vae_mcs = []

    # Create a single reusable network instance
    rnn_mc = RNNDynamics(d_latent, device=device)
    encoder_mc = Encoder(d_obs, d_latent, d_hidden, device=device)
    encoder_mc.load_state_dict(encoder_hat.state_dict())
    encoder_mc.requires_grad_(False)
    vae_mc = SeqVae(rnn_mc, encoder_mc, decoder, device=device)
    for i in range(num_test):
        errors = []
        for _ in range(n_mc):
            # Simply load parameters from rnn_dynamics_hat
            rnn_mc.load_state_dict(rnn_dynamics_hat.state_dict())

            y_mc = (x_mc[i] @ C.T) + torch.randn(1, T_test, n_neurons) * torch.sqrt(Q)
            dataloader = DataLoader(y_mc, batch_size=1)
            vae_mc.train_model(dataloader, 1e-4, weight_decay, 15)

            # compute mse for each of dynamics parameter
            with torch.no_grad():
                theta_star = torch.cat([p.flatten() for p in rnn_dynamics.parameters()])
                theta_mc = torch.cat([p.flatten() for p in rnn_mc.parameters()])
                error = torch.norm(theta_star - theta_mc)
            errors.append(error)

        vae_mcs.append(copy.deepcopy(vae_mc))
        error_state.append(torch.tensor(errors).cpu())

    #
    mean_error = torch.stack(error_state).mean(dim=1).cpu()
    std_error = torch.stack(error_state).std(dim=1).cpu()

    # scatter plot fim x error_state. Repeat fim n_mc times

    plt.errorbar(torch.arange(num_test).cpu(), mean_error, yerr=std_error)
    # fims_scatter = torch.tensor(fims).repeat(n_mc, 1).cpu()
    # plt.scatter(fims_scatter, error_state)
    print(fims)

    #
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    f_true.streamplot(ax=ax[0, 0])
    for i in range(1, 6):
        plot_vector_field(vae_mcs[i - 1].dynamics, ax=ax[i // 3, i % 3])
        plt.plot(x_mc.cpu()[i - 1, :, 0], x_mc.cpu()[i - 1, :, 1])
        plt.title(f"mean error: {mean_error[i-1]:.4f}, Information: {fims[i-1]:.4f}")
    plt.show()


if __name__ == "__main__":
    #  Step 0: Set up random seed and key
    torch_seed = 1234
    torch.manual_seed(torch_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    #  Step 1: Define true dynamics and observation model
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=torch_seed, w_attractor=0.02)

    # Observation model parameters: y = exp(C z + b) + noise.
    n_neurons = 50
    C = torch.randn(n_neurons, 2)
    C = C / torch.norm(C, dim=1, keepdim=True)

    f_true = DynamicsWrapper(model=vf)

    #  Step 2: Generate training trajectory (K x T x 2) and observation (K x T x D)
    K = 1000
    T = 250
    R = torch.tensor(1e-4)
    Q = torch.tensor(1e-4)
    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_star = f_true.generate_trajectory(x0, T, R)
    y_star = (x_star @ C.T) + torch.randn(K, T, n_neurons) * torch.sqrt(Q)

    #  Step 3: Initialize VAE for 'true' f approximation
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

    #  Step 4: Initialize VAE for f_hat with fewer training data
    K = 50

    x_train = x_star[K:]
    y_train = y_star[K:]

    # encoder_hat = Encoder(d_obs, d_latent, d_hidden, device=device)
    encoder_hat = Encoder(d_obs, d_latent, d_hidden, device=device)
    encoder_hat.load_state_dict(encoder.state_dict())
    encoder_hat.requires_grad_(False)
    rnn_dynamics_hat = RNNDynamics(d_latent, device=device)
    vae = SeqVae(rnn_dynamics_hat, encoder_hat, decoder, device=device)

    dataloader = DataLoader(y_train, batch_size=batch_size)
    vae.train_model(dataloader, lr, weight_decay, n_epochs)
    f_hat = DynamicsWrapper(model=rnn_dynamics_hat)

    #  Step 5: Compute FIM and CRLB from two initial points
    num_test = 5
    T_test = 50
    fisher = FisherMetrics(
        dynamics=rnn_dynamics_hat,
        decoder=decoder,
        process_noise=R * torch.eye(d_latent),
        measurement_noise=Q * torch.eye(d_obs),
    )
    initial_states = torch.rand(num_test, d_latent) * 5 - 2.5
    initial_cov = torch.eye(d_latent, device=device)

    fims = fisher.compute_fim(initial_states, T_test, initial_cov, use_diag=True)
    CRLB = [(1 / fim) for fim in fims]

    #  Step 6: Monte Carlo simulation to estimate CRLB
    n_mc = 10
    error_state = []
    vae_mcs = []
    var_theta = []
    x_mcs = []

    # Create a single reusable network instance
    rnn_mc = RNNDynamics(d_latent, device=device)
    encoder_mc = Encoder(d_obs, d_latent, d_hidden, device=device)
    encoder_mc.load_state_dict(encoder.state_dict())
    encoder_mc.requires_grad_(False)
    vae_mc = SeqVae(rnn_mc, encoder_mc, decoder, device=device)
    for i in range(num_test):
        errors = []
        theta = []
        for j in range(n_mc):
            # Simply load parameters from rnn_dynamics_hat
            rnn_mc.load_state_dict(rnn_dynamics_hat.state_dict())

            x_mc = f_star.generate_trajectory(initial_states[i].unsqueeze(0), T_test, R)
            y_mc = (x_mc @ C.T) + torch.randn(1, T_test, n_neurons) * torch.sqrt(Q)

            dataloader = DataLoader(y_mc, batch_size=1)
            vae_mc.train_model(dataloader, 1e-4, weight_decay, 15)
            # compute mse for each of dynamics parameter
            with torch.no_grad():
                theta_star = torch.cat([p.flatten() for p in rnn_dynamics.parameters()])
                theta_mc = torch.cat([p.flatten() for p in rnn_mc.parameters()])
                error = torch.norm(theta_mc)
                theta.append(theta_mc - theta_star)
            errors.append(error)
        x_mcs.append(x_mc)
        vae_mcs.append(copy.deepcopy(vae_mc))
        var_theta.append(torch.stack(theta).std(dim=0))
        error_state.append(torch.tensor(errors).cpu())

    #
    mean_error = torch.stack(error_state).mean(dim=1).cpu()
    std_error = torch.stack(error_state).std(dim=1).cpu()
    plt.errorbar(torch.arange(num_test).cpu(), mean_error, yerr=std_error)
    print(CRLB)

    #
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    f_star.streamplot(ax=ax[0, 0])
    for i in range(5):
        plt.plot(x_mcs[i].cpu()[:, 0], x_mcs[i].cpu()[:, 1])
    for i in range(1, 6):
        plot_vector_field(vae_mcs[i - 1].dynamics, ax=ax[i // 3, i % 3])
        plt.plot(x_mcs[i - 1].cpu()[:, 0], x_mcs[i - 1].cpu()[:, 1])
        plt.title(f"mean error: {mean_error[i-1]:.4f}, Information: {CRLB[i-1]:.4f}")
    plt.show()
