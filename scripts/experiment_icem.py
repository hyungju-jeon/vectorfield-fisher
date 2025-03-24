# %%
import torch
import matplotlib.pyplot as plt
import copy
import vfim.model as env
import vfim.agent as agent
import vfim.utils as utils
from tqdm import tqdm
from torch.utils.data import DataLoader

from vfim.controllers.simple_icem import SimpleICem


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
    ensemble_dynamics = env.EnsembleRNN(d_latent, n_models=5, device=device)
    ensemble_vae = env.EnsembleSeqVae(
        ensemble_dynamics, encoder, decoder, device=device
    )

    batch_size = 64
    n_epochs = 500
    lr = 1e-3
    weight_decay = 1e-4

    dataloader = DataLoader(y_train, batch_size=batch_size)
    vae.train_model(dataloader, lr, weight_decay, n_epochs)
    ensemble_vae.train_model(dataloader, lr, weight_decay, n_epochs)
    f_hat = env.DynamicsWrapper(model=dynamics)

    # %%
    def find_local_maxima_variance_vector_output(
        ensemble, dx, num_restarts=5, x_range=2.5, steps=100, lr=1e-2
    ):
        models = ensemble.dynamics.models
        device = next(models[0].parameters()).device
        local_maxima = []

        for _ in range(num_restarts):
            x = torch.randn(1, dx, requires_grad=True, device=device)
            optimizer = torch.optim.Adam([x], lr=lr)

            for _ in range(steps):
                optimizer.zero_grad()
                outputs = torch.stack([net(x) for net in models], dim=0).squeeze(
                    1
                )  # [N, 2]
                var = torch.var(outputs, dim=0)  # [2]
                loss = -var.sum()  # maximize total output variance
                loss.backward()
                optimizer.step()

                # Clamp x to be within the specified range
                x.data.clamp_(-x_range, x_range)

            local_maxima.append((x.detach().cpu(), var.sum().item()))

        idx = torch.argmax(torch.tensor([x[1] for x in local_maxima]))
        local_maxima = local_maxima[idx][0]

        return local_maxima

    def compute_ensemble_stats(ensemble, grid_x, grid_y):
        models = ensemble.dynamics.models
        device = next(models[0].parameters()).device
        xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")  # [H, W]
        grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        # grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(device)  # [H*W, 2]

        with torch.no_grad():
            preds = torch.stack(
                [net(grid).squeeze(-1) for net in models], dim=0
            )  # [N, H*W]
            var_map = preds.var(dim=0).sum(-1).reshape(len(grid_x), len(grid_y)).cpu()

        return var_map, xx.cpu(), yy.cpu()

    def plot_variance_map(var_map, xx, yy):
        plt.contourf(xx, yy, var_map, levels=100, cmap="plasma", alpha=0.3)
        plt.colorbar(label="Variance")

        plt.title("Ensemble Output Variance Map")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.tight_layout()

        # plt.show()

    grid_x = torch.linspace(-2.5, 2.5, 50)
    grid_y = torch.linspace(-2.5, 2.5, 50)

    var_map, std_map, xx, yy = compute_ensemble_stats(ensemble_vae, grid_x, grid_y)
    f_star.streamplot()
    plot_variance_map(var_map, xx, yy)
    local_max = find_local_maxima_variance_vector_output(ensemble_vae, d_latent)
    print(local_max)
    # -----------------------------------------------------------------
    # %% Compare active learning strategies
    num_experiments = 10
    results = {
        "mse_theta": {"naive": [], "random": [], "fisher": []},
        "r2_scores": {"naive": [], "random": [], "fisher": []},
    }

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
        "icem_params": {
            "horizon": 20,
            "num_iterations": 10,
            "population_size": 64,
            "num_elites": 10,
            "alpha": 0.1,
            "action_bounds": [-0.15, 0.15],
        },
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
    active_learner = agent.ActiveLearning(models["naive"], device, trial_params)
    x_naive = active_learner.run_naive_trial(initial_states, f_star, C, R, Q, n_neurons)
    print("Naive trial complete")

    # Add iCEM controller for trajectory optimization
    icem = SimpleICem(**trial_params["icem_params"])

    @torch.no_grad()
    def dynamics_fn(state, action):
        # Wrapper for your dynamics model
        return state + vae.dynamics(state) + action

    @torch.no_grad()
    def cost_fn(state, goal, action):
        # Define your cost function here
        return torch.sum((state - goal) ** 2, dim=-1)

    # Use iCEM for trajectory optimization
    goal = torch.tensor([[1.0, 2]])
    results = []
    states = [initial_states]
    for i in range(20):
        result = icem.optimize(
            states[i],
            dynamics_fn,
            cost_fn,
            goal,
            action_dim=d_latent,
            device=device,
        )
        state = states[i] + f_star(states[i]) + result
        states.append(state.detach())
        results.append(result.detach().unsqueeze(0))

    states = torch.cat(states)
    results = torch.cat(results)

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
