# %%
import torch
import matplotlib.pyplot as plt
import copy
import vfim.model as env
import vfim.agent as agent
import vfim.utils as utils
from tqdm import tqdm
from torch.utils.data import DataLoader


torch_seed = 111


def evaluate_models(vae_star, models, x0):
    """Evaluate models at different points"""
    latent_mse = {
        name: torch.norm(vae_star.dynamics(x0) - model.dynamics(x0))
        for name, model in models.items()
        if name != "star"
    }

    return latent_mse


def compute_parameter_mse(rnn_dynamics, models):
    """Compute MSE between true parameters and learned parameters"""
    theta_star = torch.cat([p.flatten() for p in rnn_dynamics.parameters()])
    mse_dict = {}

    for name, model in models.items():
        theta = torch.cat([p.flatten() for p in model.dynamics.parameters()])
        mse_dict[name] = torch.mean((theta_star - theta) ** 2)

    return mse_dict


def initialize_vectorfield(d_obs, d_latent, device="cpu"):
    #  Step 0: Set up random seed and key
    torch.manual_seed(torch_seed)
    torch.set_default_device(device)

    #  Step 1: Define true dynamics and observation model
    vf = utils.VectorField(model="multi", x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=torch_seed, w_attractor=0.05)

    # Observation model parameters: y = exp(C z + b) + noise.
    C = torch.randn(d_obs, d_latent)
    C = C / torch.norm(C, dim=1, keepdim=True)

    f_true = env.DynamicsWrapper(model=vf)

    return f_true, C


def approximate_vectorfield(f_true, C, device="cpu"):
    K = 1000
    T = 250
    R = torch.tensor(1e-4)
    Q = torch.tensor(1e-4)

    d_obs = C.shape[0]
    d_latent = C.shape[1]

    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_star = f_true.generate_trajectory(x0, T, R)
    y_star = (x_star @ C.T) + torch.randn(K, T, d_obs) * torch.sqrt(Q)

    #  Step 3: Initialize VAE for 'true' f approximation
    d_hidden = 16

    encoder = env.Encoder(d_obs, d_latent, d_hidden, device=device)
    rnn_dynamics = env.RNNDynamics(d_latent, dh=16, device=device)
    # Initialize decoder with pre-defined C matrix (known readout)
    decoder = env.NormalDecoder(d_latent, d_obs, device=device, l2=1.0, C=C)
    vae_star = env.SeqVae(rnn_dynamics, encoder, decoder, device=device)

    batch_size = 64
    n_epochs = 500
    lr = 1e-3
    weight_decay = 1e-4

    dataloader = DataLoader(y_star, batch_size=batch_size)
    vae_star.train_model(dataloader, lr, weight_decay, n_epochs)

    f_star = env.DynamicsWrapper(model=rnn_dynamics)

    return vae_star, f_star, x_star, y_star


def activelearning_exp():
    """Run active learning experiment"""
    d_latent = 2

    trial_params = {
        "num_trials": 50,
        "trial_length": 10,
        "fisher_length": 10,
        "refine_epoch": 10,
        "lr_refine": 1e-4,
        "u_strength": 0.25,
        "weight_decay": 1e-5,
    }

    initial_states = torch.rand(1, d_latent) * 5 - 2.5

    # Initialize models
    models = {
        "naive": copy.deepcopy(vae),
        "random": copy.deepcopy(vae),
        "fisher": copy.deepcopy(vae),
    }
    R = torch.tensor(1e-4)
    Q = torch.tensor(1e-4)
    n_neurons = 50

    # Initialize active learning for each strategy

    # Run trials for each strategy
    active_learner = agent.ActiveLearning(models["naive"], device, trial_params)
    x_naive = active_learner.run_naive_trial(initial_states, f_star, C, R, Q, n_neurons)
    print("Naive trial complete")

    active_learner = agent.ActiveLearning(models["random"], device, trial_params)
    x_random = active_learner.run_random_trial(
        initial_states, f_star, C, R, Q, n_neurons
    )
    print("Random trial complete")

    active_learner = agent.ActiveLearning(models["fisher"], device, trial_params)
    x_fisher = active_learner.run_fisher_trial(
        initial_states, f_star, C, R, Q, n_neurons, d_latent, decoder
    )
    print("Fisher trial complete")

    # Evaluate models
    K, T = 200, 20
    x0_eval = (torch.rand(K, 2) * 5 - 2.5).unsqueeze(1)
    r2_scores = evaluate_models(vae_star, models, x0_eval)
    mse_dict = compute_parameter_mse(vae_star.dynamics, models)

    f_star.streamplot()
    plt.plot(x_naive[0, :, 0].cpu(), x_naive[0, :, 1].cpu(), label="naive")
    plt.plot(x_random[0, :, 0].cpu(), x_random[0, :, 1].cpu(), label="random")
    plt.plot(x_fisher[0, :, 0].cpu(), x_fisher[0, :, 1].cpu(), label="fisher")
    plt.legend()
    plt.show()

    return mse_dict, r2_scores


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_obs = 50
    d_latent = 2

    f_true, C = initialize_vectorfield(d_obs, d_latent, device)
    vae_star, f_star, x_star, y_star = approximate_vectorfield(f_true, C, device)

    #  Train with a small subset of data
    K = 50
    x_train = x_star[:K]
    y_train = y_star[:K]

    encoder = copy.deepcopy(vae_star.encoder)
    encoder.requires_grad_(False)
    decoder = copy.copy(vae_star.decoder)
    decoder.requires_grad_(False)
    dynamics = env.RNNDynamics(d_latent, dh=16, device=device)
    vae = env.SeqVae(dynamics, encoder, decoder, device=device)

    batch_size = 64
    n_epochs = 500
    lr = 1e-3
    weight_decay = 1e-4

    dataloader = DataLoader(y_train, batch_size=batch_size)
    vae.train_model(dataloader, lr, weight_decay, n_epochs)
    f_hat = env.DynamicsWrapper(model=dynamics)
    # -----------------------------------------------------------------
    # %% Compare active learning strategies
    num_experiments = 10
    results = {
        "mse_theta": {"naive": [], "random": [], "fisher": []},
        "r2_scores": {"naive": [], "random": [], "fisher": []},
    }

    for _ in tqdm(range(10), desc="Running experiments"):
        exp_results = activelearning_exp()

        # Unpack MSE theta results
        results["mse_theta"]["naive"].append(exp_results[0]["naive"])
        results["mse_theta"]["random"].append(exp_results[0]["random"])
        results["mse_theta"]["fisher"].append(exp_results[0]["fisher"])

        # Unpack R² scores (4 time horizons for each method)
        results["r2_scores"]["naive"].append(exp_results[1]["naive"])
        results["r2_scores"]["random"].append(exp_results[1]["random"])
        results["r2_scores"]["fisher"].append(exp_results[1]["fisher"])

    # Convert to numpy arrays
    r2_arrays = {
        method: torch.stack(results["r2_scores"][method]).detach().cpu()
        for method in ["naive", "random", "fisher"]
    }

    # Plot results
    plt.style.use("ggplot")
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    models = list(r2_arrays.keys())
    means = [torch.mean(r2_arrays[m]) for m in r2_arrays]
    stds = [torch.std(r2_arrays[m]) for m in models]

    x = torch.arange(len(models)).cpu()

    # Plot individual trial data by connecting points with a dim line
    for trial in range(num_experiments):
        trial_values = [r2_arrays[m][trial] for m in models]
        plt.plot(x, trial_values, color="black", alpha=0.75, linewidth=0.25)

    # Plot the mean with error bars using errorbar.
    # The 'fmt' argument sets the marker style, here 'o' indicates circular markers.
    plt.errorbar(
        x,
        means,
        yerr=stds,
        fmt="o",
        capsize=5,
        markersize=8,
        color="red",
        linestyle="None",
    )

    # Customize plot
    plt.xticks(x, models)
    plt.xlabel("Models")
    plt.ylabel("R² Measurement")
    plt.title("R² Measurements with Error Bars and Individual Trials")

    plt.show()
