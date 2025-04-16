# %%
import torch
import matplotlib.pyplot as plt
import copy
from vfim.agent.information import FisherMetrics
import vfim.model as env
import vfim.agent as agent
import vfim.utils as utils
from tqdm import tqdm
from torch.utils.data import DataLoader

from vfim.controllers.simple_icem import SimpleICem
from vfim.utils.visualize import plot_vector_field


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
        ensemble, dx, init_x=None, num_restarts=10, x_range=2.5, steps=100, lr=1e-2
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

        # weight variance by distance to the init_x if provided
        if init_x is not None:
            local_maxima = [
                (x, var / torch.norm(x - init_x.cpu()).item() ** 0.5)
                for x, var in local_maxima
            ]
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
        # plt.colorbar(label="Variance")

        plt.title("Ensemble Output Variance Map")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.tight_layout()

        # plt.show()

    grid_x = torch.linspace(-2.5, 2.5, 50)
    grid_y = torch.linspace(-2.5, 2.5, 50)

    var_map, xx, yy = compute_ensemble_stats(ensemble_vae, grid_x, grid_y)
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
        "refine_epoch": 1,
        "lr_refine": 1e-3,
        "u_strength": 0.25,
        "weight_decay": 1e-5,
        "icem_params": {
            "horizon": 5,
            "num_iterations": 10,
            "population_size": 32,
            "num_elites": 5,
            "alpha": 0.1,
            "action_bounds": [-0.15, 0.15],
        },
    }

    # Initialize models
    models_fisher = {
        "fisher": copy.deepcopy(vae),
        "ensemble": copy.deepcopy(ensemble_vae),
    }

    models_goal = {
        "goal": copy.deepcopy(vae),
        "ensemble": copy.deepcopy(ensemble_vae),
    }

    R = torch.tensor(1e-4)
    Q = torch.tensor(1e-4)
    n_neurons = 50
    # initial_states = torch.rand(1, d_latent) * 5 - 2.5
    initial_states = torch.tensor([[0.0, 0.0]], device=device)
    goal_fisher = find_local_maxima_variance_vector_output(
        models_fisher["ensemble"], d_latent, init_x=initial_states
    ).to(device)
    goal_goal = find_local_maxima_variance_vector_output(
        models_fisher["ensemble"], d_latent, init_x=initial_states
    ).to(device)

    # Add iCEM controller for trajectory optimization
    icem = SimpleICem(**trial_params["icem_params"])

    @torch.no_grad()
    def dynamics_fn(state, action):
        # Wrapper for your dynamics model
        return state + models_fisher["fisher"].dynamics(state) + action

    # @torch.no_grad()
    def cost_f_fisher(state, goal, action=None):
        # Define your cost function here
        fim = fisher.compute_fim_trajectory(state)
        goal_cost = torch.sum((state - goal) ** 2, dim=-1).sum()
        fim_cost = torch.sum(fim) * 1e-7
        # print(f"Goal : {goal_cost}, \n FIM : {fim_cost}")
        return goal_cost - fim_cost

    # @torch.no_grad()
    def cost_fn_goal(state, goal, action=None):
        # Define your cost function here
        # fim = fisher.compute_fim_trajectory(state)
        return torch.sum((state - goal) ** 2, dim=-1).sum()

    model = "fisher"
    # Use iCEM for trajectory optimization
    inputs_fisher = []
    states_fisher = [initial_states]
    inputs_goal = []
    states_goal = [initial_states]
    for num_epoch in range(10):
        fisher = FisherMetrics(
            dynamics=models_fisher[model].dynamics,
            decoder=decoder,
            process_noise=R * torch.eye(d_latent),
            measurement_noise=Q * torch.eye(n_neurons),
        )
        for icem_run in tqdm(range(4)):
            print(
                f"Fisher Trajectory : Starting point {states_fisher[-1]}, Target {goal_fisher}"
            )
            result_fisher = icem.optimize(
                states_fisher[-1],
                dynamics_fn,
                cost_f_fisher if model == "fisher" else cost_fn_goal,
                goal_fisher,
                action_dim=d_latent,
                device=device,
            )
            print(
                f"Goal Trajectory : Starting point {states_goal[-1]}, Target {goal_goal}"
            )
            result_goal = icem.optimize(
                states_goal[-1],
                dynamics_fn,
                cost_fn_goal,
                goal_goal,
                action_dim=d_latent,
                device=device,
            )
            for i in range(trial_params["icem_params"]["horizon"]):
                states_fisher.append(
                    (
                        states_fisher[-1] + f_star(states_fisher[-1]) + result_fisher[i]
                    ).detach()
                )
                states_goal.append(
                    (
                        states_goal[-1] + f_star(states_goal[-1]) + result_goal[i]
                    ).detach()
                )
                inputs_fisher.append(result_fisher[i].detach().unsqueeze(0))
                inputs_goal.append(result_goal[i].detach().unsqueeze(0))

            x_fisher = torch.cat(states_fisher[-6:])
            input_fisher = torch.cat(
                inputs_fisher[-5:] + [torch.tensor([[0, 0]])]
            ).unsqueeze(0)
            y_fisher = (x_fisher @ C.T) + torch.randn(
                1, x_fisher.shape[0], n_neurons
            ) * torch.sqrt(Q)

            x_goal = torch.cat(states_goal[-6:])
            input_goal = torch.cat(
                inputs_goal[-5:] + [torch.tensor([[0, 0]])]
            ).unsqueeze(0)
            y_goal = (x_goal @ C.T) + torch.randn(
                1, x_goal.shape[0], n_neurons
            ) * torch.sqrt(Q)

            # Train model with both y and input
            models_fisher["fisher"].train_model(
                DataLoader(torch.cat([y_fisher, input_fisher], dim=-1), batch_size=1),
                trial_params["lr_refine"],
                trial_params["weight_decay"],
                trial_params["refine_epoch"],
                has_input=True,
                verbose=True,
            )

            # Train ensemble model
            models_fisher["ensemble"].train_model(
                DataLoader(torch.cat([y_fisher, input_fisher], dim=-1), batch_size=1),
                trial_params["lr_refine"],
                trial_params["weight_decay"],
                trial_params["refine_epoch"],
                has_input=True,
                verbose=False,
            )

            models_goal["goal"].train_model(
                DataLoader(torch.cat([y_goal, input_goal], dim=-1), batch_size=1),
                trial_params["lr_refine"],
                trial_params["weight_decay"],
                trial_params["refine_epoch"],
                has_input=True,
                verbose=True,
            )
            models_goal["ensemble"].train_model(
                DataLoader(torch.cat([y_goal, input_goal], dim=-1), batch_size=1),
                trial_params["lr_refine"],
                trial_params["weight_decay"],
                trial_params["refine_epoch"],
                has_input=True,
                verbose=False,
            )

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        plot_vector_field(models_fisher["fisher"].dynamics, ax=ax[0])
        plt.plot(
            torch.cat(states_fisher).cpu().numpy()[:, 0],
            torch.cat(states_fisher).cpu().numpy()[:, 1],
        )
        plt.scatter(
            goal_fisher.cpu().numpy()[0][0],
            goal_fisher.cpu().numpy()[0][1],
            color="red",
        )
        ax[0].set_title("Fisher Dynamics")
        plot_vector_field(models_goal["goal"].dynamics, ax=ax[1])
        plt.plot(
            torch.cat(states_goal).cpu().numpy()[:, 0],
            torch.cat(states_goal).cpu().numpy()[:, 1],
        )
        plt.scatter(
            goal_goal.cpu().numpy()[0][0], goal_goal.cpu().numpy()[0][1], color="red"
        )
        ax[1].set_title("Goal Dynamics")
        plt.show()

        goal_fisher = find_local_maxima_variance_vector_output(
            models_fisher["ensemble"], d_latent, init_x=states_fisher[-1]
        ).to(device)
        goal_goal = find_local_maxima_variance_vector_output(
            models_goal["ensemble"], d_latent, init_x=states_goal[-1]
        ).to(device)
        # goal_goal = goal_fisher

    var_map, xx, yy = compute_ensemble_stats(models_fisher["ensemble"], grid_x, grid_y)
    states_fisher = torch.cat(states_fisher, dim=0)
    states_goal = torch.cat(states_goal, dim=0)
    # plot_vector_field(models_fisher[model].dynamics)
    # plot_variance_map(var_map, xx, yy)
    # plt.plot(states_fisher.cpu().numpy()[:, 0], states_fisher.cpu().numpy()[:, 1])

# %%
# Create fisher information map by computing FIM on smapled point in the grid and intorpolate
grid_x = torch.linspace(-2.5, 2.5, 25)
grid_y = torch.linspace(-2.5, 2.5, 25)
xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")  # [H, W]
grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)
# grid = torch.stack([xx, yy], dim=-1).reshape(-1, 2).to(device)  # [H*W, 2]
fisher = FisherMetrics(
    dynamics=models_fisher["fisher"].dynamics,
    decoder=decoder,
    process_noise=R * torch.eye(d_latent),
    measurement_noise=Q * torch.eye(n_neurons),
)

fisher_map = [fisher.compute_fim_trajectory(x.unsqueeze(0)) for x in grid]
fisher_map = torch.stack(fisher_map, dim=0).reshape(len(grid_x), len(grid_y))


# [H, W]
# plot_vector_field(models_fisher["fisher"].dynamics)
plot_vector_field(models_fisher["fisher"].dynamics)
plt.contourf(
    xx.cpu(), yy.cpu(), fisher_map.cpu().T, levels=10, cmap="plasma", alpha=0.3
)
plt.colorbar(label="Fisher Information")
plt.title("Fisher Information Map")
plt.xlabel("x₁")
plt.ylabel("x₂")
plt.grid(True)
plt.tight_layout()

# %%
# 3 plot f_star, models["fisher"].dynamics, models["goal"].dynamics
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
plot_vector_field(f_star.model, ax=ax[0])
ax[0].set_title("True Dynamics")

# var_map, xx, yy = compute_ensemble_stats(
#     models_fisher["ensemble"], grid_x, grid_y
# )
plot_vector_field(models_fisher["fisher"].dynamics, ax=ax[1])
# plot_variance_map(var_map, xx, yy)
# plt.plot(states_fisher.cpu().numpy()[:, 0], states_fisher.cpu().numpy()[:, 1])
ax[1].set_title("Fisher Dynamics")

# var_map, xx, yy = compute_ensemble_stats(
#     models_goal["ensemble"], grid_x, grid_y
# )
plot_vector_field(vae.dynamics, ax=ax[2])
# plot_variance_map(var_map, xx, yy)
# plt.plot(states_goal.cpu().numpy()[:, 0], states_goal.cpu().numpy()[:, 1])
ax[2].set_title("Goal Dynamics")
plt.show()

# %%


def evaluate_models(vae_star, model, x0):
    """Evaluate models at different points"""
    latent_mse = torch.norm(vae_star.dynamics(x0) - model.dynamics(x0))

    return latent_mse


K = 1000
x0_eval = (torch.rand(K, 2) * 5 - 2.5).unsqueeze(1)
mse_fisher = evaluate_models(vae_star, models_fisher["fisher"], x0_eval)
mse_goal = evaluate_models(vae_star, models_goal["goal"], x0_eval)
mse_vae = evaluate_models(vae_star, vae, x0_eval)

print(f"VAE MSE: {mse_vae}")
print(f"Fisher MSE: {mse_fisher}")
print(f"Goal MSE: {mse_goal}")
