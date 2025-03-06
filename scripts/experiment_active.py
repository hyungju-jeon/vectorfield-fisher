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


def calculate_r2_scores(x_true, x_pred, time_points):
    """Calculate R² scores for different prediction horizons"""
    r2_scores = []
    for t in time_points:
        mse = torch.mean((x_true[:, :t, :] - x_pred[:, :t, :]) ** 2)
        r2 = 1 - mse / torch.var(x_true[:, :t, :])
        r2_scores.append(r2)
    return r2_scores


def run_exp():
    initial_states = torch.rand(1, d_latent) * 5 - 2.5
    # %%Active learning comparison
    num_trials = 50
    trial_length = 5
    fisher_length = 5
    refine_epoch = 10
    lr_refine = 1e-4
    u_strength = 0.15

    u_x = torch.linspace(-u_strength, u_strength, 3, device=device)
    u_y = torch.linspace(-u_strength, u_strength, 3, device=device)
    U_X, U_Y = torch.meshgrid(u_x, u_y, indexing="xy")
    u_list = torch.stack([U_X.flatten(), U_Y.flatten()], dim=1)

    vae_naive = copy.deepcopy(vae)
    vae_random = copy.deepcopy(vae)
    vae_fisher = copy.deepcopy(vae)

    initial_states = torch.rand(1, d_latent) * 5 - 2.5
    # Naive learning
    x0 = initial_states.unsqueeze(0)
    x_naive = x0
    for trial_idx in range(num_trials):
        x_trial = f_star.generate_trajectory(x0, trial_length, R)
        y_trial = (x_trial @ C.T) + torch.randn(
            1, trial_length, n_neurons
        ) * torch.sqrt(Q)

        dataloader = DataLoader(y_trial, batch_size=1)
        vae_naive.train_model(dataloader, lr_refine, weight_decay, refine_epoch)
        x_naive = torch.cat([x_naive, x_trial], dim=1)
        x0 = x_trial[:, -1, :].unsqueeze(0)

    # Random learning
    x0 = initial_states.unsqueeze(0) + torch.randn(1, d_latent) * u_strength
    x_random = x0
    for trial_idx in range(num_trials):
        x_trial = f_star.generate_trajectory(x0, trial_length, R)
        y_trial = (x_trial @ C.T) + torch.randn(
            1, trial_length, n_neurons
        ) * torch.sqrt(Q)

        dataloader = DataLoader(y_trial, batch_size=1)
        vae_random.train_model(dataloader, lr_refine, weight_decay, refine_epoch)
        x_random = torch.cat([x_random, x_trial], dim=1)

        x0 = x_trial[:, -1, :].unsqueeze(0) + torch.randn(1, d_latent) * u_strength

    # Fisher learning
    x0 = initial_states
    x0 = torch.cat([x0 + u for u in u_list], dim=0)
    fisher = FisherMetrics(
        dynamics=vae_fisher.dynamics,
        decoder=decoder,
        process_noise=R * torch.eye(d_latent),
        measurement_noise=Q * torch.eye(d_obs),
    )
    initial_cov = 0 * torch.eye(d_latent, device=device)
    fims = fisher.compute_fim(x0, fisher_length, initial_cov, use_diag=True)
    fim = torch.tensor([torch.sum(f) for f in fims])
    max_idx = torch.argmax(fim)
    x0 = x0[max_idx].unsqueeze(0)
    x_fisher = x0.unsqueeze(0)

    for trial_idx in range(num_trials):
        x_trial = f_star.generate_trajectory(x0.unsqueeze(0), trial_length, R)
        y_trial = (x_trial @ C.T) + torch.randn(
            1, trial_length, n_neurons
        ) * torch.sqrt(Q)

        dataloader = DataLoader(y_trial, batch_size=1)
        vae_fisher.train_model(dataloader, lr_refine, weight_decay, refine_epoch)
        x_fisher = torch.cat([x_fisher, x_trial], dim=1)

        x0 = x_trial[:, -1, :]
        x0 = torch.cat([x0 + u for u in u_list], dim=0)
        fims = fisher.compute_fim(x0, fisher_length, initial_cov, use_diag=True)
        fim = torch.tensor([torch.sum(f) for f in fims])
        max_idx = torch.argmax(fim)
        x0 = x0[max_idx].unsqueeze(0)

    theta_star = torch.cat([p.flatten() for p in rnn_dynamics.parameters()])
    theta_naive = torch.cat([p.flatten() for p in vae_naive.dynamics.parameters()])
    theta_random = torch.cat([p.flatten() for p in vae_random.dynamics.parameters()])
    theta_fisher = torch.cat([p.flatten() for p in vae_fisher.dynamics.parameters()])

    mse_theta_naive = torch.mean((theta_star - theta_naive) ** 2)
    mse_theta_random = torch.mean((theta_star - theta_random) ** 2)
    mse_theta_fisher = torch.mean((theta_star - theta_fisher) ** 2)

    # Dynamics similarity check using prediction
    K, T = 200, 20
    x0 = (torch.rand(K, 2) * 5 - 2.5).unsqueeze(1)

    # Generate trajectories for all models
    models = {
        "star": vae_star,
        "naive": vae_naive,
        "random": vae_random,
        "fisher": vae_fisher,
    }
    trajectories = {
        name: model.dynamics.sample_forward(x0, T, R, return_trajectory=True)[0]
        for name, model in models.items()
    }

    # Calculate R² scores for different time horizons
    time_points = [5, 10, 15, T]
    r2_scores = {
        name: calculate_r2_scores(trajectories["star"], traj, time_points)
        for name, traj in trajectories.items()
        if name != "star"
    }

    return (
        mse_theta_naive.detach().cpu(),
        mse_theta_random.detach().cpu(),
        mse_theta_fisher.detach().cpu(),
        *[score.detach().cpu() for score in r2_scores["naive"]],
        *[score.detach().cpu() for score in r2_scores["random"]],
        *[score.detach().cpu() for score in r2_scores["fisher"]],
    )


if __name__ == "__main__":
    #  Step 0: Set up random seed and key
    torch_seed = 1234
    torch.manual_seed(torch_seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    #  Step 1: Define true dynamics and observation model
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=torch_seed, w_attractor=0.01)

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
    x_train = x_star[:K]
    y_train = y_star[:K]

    encoder_init = Encoder(d_obs, d_latent, d_hidden, device=device)
    encoder_init.load_state_dict(encoder.state_dict())
    encoder_init.requires_grad_(False)
    rnn_dynamics_hat = RNNDynamics(d_latent, device=device)
    vae = SeqVae(rnn_dynamics_hat, encoder_init, decoder, device=device)

    dataloader = DataLoader(y_train, batch_size=batch_size)
    vae.train_model(dataloader, lr, weight_decay, n_epochs)
    f_hat = DynamicsWrapper(model=rnn_dynamics_hat)
    # -----------------------------------------------------------------
    # %%  Step 5: Compare active learning strategies
    results = {
        "mse_theta": {"naive": [], "random": [], "fisher": []},
        "r2_scores": {"naive": [], "random": [], "fisher": []},
    }

    for _ in range(10):
        exp_results = run_exp()

        # Unpack MSE theta results
        results["mse_theta"]["naive"].append(exp_results[0])
        results["mse_theta"]["random"].append(exp_results[1])
        results["mse_theta"]["fisher"].append(exp_results[2])

        # Unpack R² scores (4 time horizons for each method)
        results["r2_scores"]["naive"].append(exp_results[3:7])
        results["r2_scores"]["random"].append(exp_results[7:11])
        results["r2_scores"]["fisher"].append(exp_results[11:15])

    # Convert to numpy arrays
    r2_arrays = {
        method: torch.stack(
            [torch.stack(x) for x in results["r2_scores"][method]]
        ).numpy()
        for method in ["naive", "random", "fisher"]
    }

    # Filter out negative R² scores
    for method in r2_arrays:
        r2_arrays[method] = r2_arrays[method][r2_arrays[method][:, -1] > 0]

    # Plot results
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    time_horizons = [5, 10, 15, 20]

    for method, style in zip(["naive", "random", "fisher"], ["b", "g", "r"]):
        mean = r2_arrays[method].mean(axis=0)
        std = r2_arrays[method].std(axis=0)

        ax.plot(time_horizons, mean, c=style, label=method.capitalize())
        ax.fill_between(time_horizons, mean - std, mean + std, alpha=0.2, color=style)

    ax.set_xlabel("Prediction Horizon")
    ax.set_ylabel("R²")
    ax.legend()
    plt.show()
