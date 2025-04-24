# %%
import matplotlib.animation as animation
import torch
import matplotlib.pyplot as plt
import copy
from vfim.agent.information import FisherMetrics
import vfim.model as env
import vfim.agent as agent
import vfim.utils as utils
from tqdm import tqdm
from torch.utils.data import DataLoader

from matplotlib import cm
from matplotlib.colors import Normalize
from vfim.controllers.simple_icem import SimpleICem
from vfim.utils.visualize import plot_vector_field
import os
import io
from PIL import Image


result_dir = "/home/hyungju/Desktop/vectorfield-fisher/results/experiment_meeting/"
os.makedirs(result_dir, exist_ok=True)

# %% Define Auxiliary Functions
torch_seed = 1111

device = "cuda" if torch.cuda.is_available() else "cpu"
grid_x = torch.linspace(-2.5, 2.5, 25)
grid_y = torch.linspace(-2.5, 2.5, 25)
xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")  # [H, W]

rbf_grid_x = torch.linspace(-2.5, 2.5, 10)
rbf_grid_y = torch.linspace(-2.5, 2.5, 10)
rbf_xx, rbf_yy = torch.meshgrid(rbf_grid_x, rbf_grid_y, indexing="ij")  # [H, W]
rbf_grid = torch.stack([rbf_xx.flatten(), rbf_yy.flatten()], dim=1)

Q = torch.tensor(1e-3)  # process noise
R = torch.tensor(1e-3)  # measurement noise


def initialize_vectorfield(d_obs, d_latent, device="cpu"):
    #  Step 0: Set up random seed and key
    torch.manual_seed(torch_seed)
    torch.set_default_device(device)

    #  Step 1: Define true dynamics and observation model
    vf = utils.VectorField(model="limitcycle", x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=torch_seed, w_attractor=0.05)

    # Observation model parameters: y = exp(C z + b) + noise.
    C = torch.randn(d_obs, d_latent)
    C = C / torch.norm(C, dim=1, keepdim=True)

    b = torch.randn(d_obs)
    b = b / torch.norm(b, dim=0, keepdim=True)

    f_true = env.DynamicsWrapper(model=vf)

    return f_true, C, b


def approximate_vectorfield(f_true, C, b, device="cpu"):
    K = 5000
    T = 100

    d_obs = C.shape[0]
    d_latent = C.shape[1]

    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_star = f_true.generate_trajectory(x0, T, Q)
    y_star = torch.exp(x_star @ C.T + b) + torch.randn(K, T, d_obs) * torch.sqrt(R)

    #  Step 3: Initialize VAE for 'true' f approximation
    d_hidden = 16

    encoder = env.Encoder(d_obs, d_latent, d_hidden, device=device)
    # dynamics = env.RNNDynamics(d_latent, dh=8, device=device)
    dynamics = env.RBFDynamics(rbf_grid, device=device)
    # Initialize decoder with pre-defined C matrix (known readout)
    decoder = env.NormalDecoder(d_latent, d_obs, device=device, l2=1.0, C=C, b=b)
    vae_star = env.SeqVae(dynamics, encoder, decoder, device=device)

    batch_size = 64
    n_epochs = 200
    lr = 1e-4
    weight_decay = 1e-4

    dataloader = DataLoader(y_star, batch_size=batch_size)
    vae_star.train_model(dataloader, lr, weight_decay, n_epochs, optimizer="AdamW")

    # param_list = list(vae_star.parameters())
    # opt = torch.optim.SGD(params=param_list, lr=lr)

    # for _ in tqdm(range(200)):
    #     for batch in dataloader:
    #         opt.zero_grad()

    #         x_samples, mu_q_x, var_q_x, log_q = vae_star.encoder(batch, n_samples=1)
    #         kl_d_x = vae_star._compute_kld_x(mu_q_x, var_q_x, x_samples, input=input)
    #         log_like = vae_star.decoder.compute_log_prob(x_samples, batch)
    #         elbo = torch.mean(log_like - 1 * kl_d_x)
    #         loss = elbo
    #         loss.backward()
    #         opt.step()
    #         with torch.no_grad():
    #             training_losses.append(loss.item())

    f_star = env.DynamicsWrapper(model=dynamics)

    return vae_star, f_star, x_star, y_star


def compute_uncertainty(ensemble, x):
    """Compute uncertainty of ensemble predictions."""
    models = ensemble.dynamics.models
    with torch.no_grad():
        preds = torch.stack([net(x) for net in models], dim=0)  # [N, B, 2]
        var = preds.var(dim=0)  # [B, 2]
        return var


def compute_uncertainty_map(ensemble, grid_x, grid_y, show_plot=False):
    xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")  # [H, W]
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    with torch.no_grad():
        var_map = (
            compute_uncertainty(ensemble, grid.to(device))
            .sum(-1)
            .reshape(len(grid_x), len(grid_y))
            .cpu()
        )

    if show_plot:
        plt.contourf(xx.cpu(), yy.cpu(), var_map.cpu().T, levels=10, cmap="plasma")
        plt.colorbar(label="Variance")
        plt.title("Uncertainty Map")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return torch.log(var_map), xx.cpu(), yy.cpu()


def find_local_maxima_uncertainty(
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


def compute_fisher_map(fisher, grid_x, grid_y, show_plot=False):
    """Create a Fisher information map by computing FIM on sampled points in the grid."""
    xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")  # [H, W]
    grid = torch.stack([xx.flatten(), yy.flatten()], dim=1)

    fisher_map = [
        fisher.compute_fim_point(x.unsqueeze(0).to(device), use_diag=True) for x in grid
    ]
    fisher_map = torch.stack(fisher_map, dim=0).reshape(len(grid_x), len(grid_y))

    if show_plot:
        plt.contourf(xx.cpu(), yy.cpu(), fisher_map.cpu().T, levels=10, cmap="plasma")
        plt.colorbar(label="Fisher Information")
        plt.title("Fisher Information Map")
        plt.xlabel("x₁")
        plt.ylabel("x₂")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return fisher_map, xx.cpu(), yy.cpu()


def plot_current_state(
    vae,
    ensemble_vae,
    f_star,
    fisher,
    fisher_map=None,
    var_map=None,
    x=None,
    map_alpha=0.3,
    title=None,
    show=False,
    var_range=(-10.0, -3),
    fisher_max=5,
):
    # Compute maps if not provided
    var_map, xx, yy = compute_uncertainty_map(ensemble_vae, grid_x, grid_y)
    if fisher_map is None:
        fisher_map, xx, yy = compute_fisher_map(
            fisher,
            grid_x,
            grid_y,
            show_plot=False,
        )

    # Create a 3-subplot figure
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))

    # Subplot 1: Vector field only
    plot_vector_field(vae.dynamics, ax=ax[0], cmax=0.5, show_cbar=True)
    ax[0].set_xlim(-2.5, 2.5)
    ax[0].set_ylim(-2.5, 2.5)
    ax[0].set_title("Vector Field")

    # Subplot 2: Fisher Information Map
    plot_vector_field(vae.dynamics, ax=ax[1], cmax=0.5)
    contour = ax[1].contourf(
        xx.cpu(),
        yy.cpu(),
        fisher_map.cpu().T,
        levels=10,
        cmap="plasma",
        alpha=map_alpha,
    )
    contour.set_clim(0, fisher_max)  # Set color axis limits
    plt.colorbar(
        ax=ax[1],
        label="Fisher Information",
        mappable=cm.ScalarMappable(
            norm=Normalize(vmin=0, vmax=fisher_max), cmap="plasma"
        ),
        alpha=map_alpha,
    )
    # Pick min(10, x.shape[0]) points to plot in bold
    num_bold = min(20, x.shape[0] // 10) if x is not None else 0

    if x is not None:
        ax[1].plot(
            x[:-num_bold, 0].cpu().numpy(),
            x[:-num_bold, 1].cpu().numpy(),
            color="red",
            alpha=0.5,
            lw=0.5,
        )
        ax[1].plot(
            x[-num_bold:, 0].cpu().numpy(),
            x[-num_bold:, 1].cpu().numpy(),
            color="red",
            alpha=0.7,
            marker=".",
            lw=1,
        )
    ax[1].set_xlim(-2.5, 2.5)
    ax[1].set_ylim(-2.5, 2.5)
    ax[1].set_title("Fisher Information Map")

    # Subplot 3: Uncertainty Map
    plot_vector_field(vae.dynamics, ax=ax[2], cmax=0.5)
    contour = ax[2].contourf(
        xx.cpu(),
        yy.cpu(),
        var_map.cpu().T,
        levels=10,
        cmap="plasma",
        alpha=map_alpha,
    )
    if var_range is not None:
        contour.set_clim(var_range[0], var_range[1])  # Set color axis limits
    plt.colorbar(
        ax=ax[2],
        label="Uncertainty",
        mappable=cm.ScalarMappable(
            norm=Normalize(vmin=var_range[0], vmax=var_range[1]), cmap="plasma"
        ),
        alpha=map_alpha,
    )
    if x is not None:
        ax[2].plot(
            x[:-num_bold, 0].cpu().numpy(),
            x[:-num_bold, 1].cpu().numpy(),
            color="red",
            alpha=0.5,
            lw=0.5,
        )
        ax[2].plot(
            x[-num_bold:, 0].cpu().numpy(),
            x[-num_bold:, 1].cpu().numpy(),
            color="red",
            alpha=0.7,
            marker=".",
            lw=1,
        )
    ax[2].set_xlim(-2.5, 2.5)
    ax[2].set_ylim(-2.5, 2.5)
    ax[2].set_title("Uncertainty Map (log)")

    if title is not None:
        fig.suptitle(title)

    if show:
        plt.show()

    return fig, ax, var_map, fisher_map


# %% Initialize vector field and approximate it with VAE
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_obs = 50
    d_latent = 2
    run_initialize = True

    true_dynamics_file = os.path.join(result_dir, "model_true.pt")
    if os.path.exists(true_dynamics_file) and not run_initialize:
        saved_models = torch.load(true_dynamics_file)
        for key, value in saved_models.items():
            globals()[key] = value
            if isinstance(value, torch.Tensor):
                globals()[key] = value.to(device)

        print("Loaded VAE from file.")
    else:
        f_true, C, b = initialize_vectorfield(d_obs, d_latent, device)
        vae_star, f_star, x_star, y_star = approximate_vectorfield(f_true, C, b, device)
        torch.save(
            {
                "f_true": f_true,
                "C": C,
                "b": b,
                "vae_star": vae_star,
            },
            true_dynamics_file,
        )

    encoder = copy.deepcopy(vae_star.encoder)
    encoder.requires_grad_(False)
    decoder = copy.copy(vae_star.decoder)
    decoder.requires_grad_(False)

    initial_model_file = os.path.join(result_dir, "model_initial.pt")
    if os.path.exists(initial_model_file) and not run_initialize:
        saved_models = torch.load(initial_model_file)
        for key, value in saved_models.items():
            globals()[key] = value
            if isinstance(value, torch.Tensor):
                globals()[key] = value.to(device)

        print("Loaded VAE and Ensemble VAE from file.")
    else:
        dynamics = env.RBFDynamics(rbf_grid, device=device)
        # dynamics = env.RNNDynamics(d_latent, dh=8, device=device)
        initial_vae = env.SeqVae(dynamics, encoder, decoder, device=device)
        # ensemble_dynamics = env.EnsembleRNN(d_latent, n_models=5, device=device)
        ensemble_dynamics = env.EnsembleRBF(rbf_grid, n_models=10, device=device)
        initial_ensemble = env.EnsembleSeqVae(
            ensemble_dynamics, encoder, decoder, device=device
        )
        torch.save(
            {"initial_vae": initial_vae, "initial_ensemble": initial_ensemble},
            initial_model_file,
        )
    # -----------------------------------------------------------------
    # %% Set Parameters
    # -----------------------------------------------------------------
    exp_params = {
        "num_steps": 10000,
        "trial_length": 20,
        "fisher_length": 10,
        "refine_epoch": 1,
        "lr_refine": 1e-4,
        "u_strength": 0.1,
        "weight_decay": 1e-5,
    }
    icem_params = {
        "horizon": 20,
        "num_iterations": 10,
        "num_traj": 32,
        "num_elites": 5,
        "alpha": 0.1,
        "action_bounds": [-exp_params["u_strength"], exp_params["u_strength"]],
    }

    # -----------------------------------------------------------------
    # %% Naive without map
    # -----------------------------------------------------------------
    run_train = False
    model = {
        "vae": copy.deepcopy(initial_vae).to(device),
        "ensemble": copy.deepcopy(initial_ensemble).to(device),
    }
    fisher = FisherMetrics(
        dynamics=model["vae"].dynamics,
        decoder=model["vae"].decoder,
        process_noise=Q * torch.eye(d_latent).to(device),
        measurement_noise=R * torch.eye(d_obs).to(device),
    )
    fisher_map = None
    if run_train:
        initial_state = torch.zeros((1, 1, d_latent), device=device)

        x_traj = [initial_state]
        u_traj = []

        @torch.no_grad()
        def dynamics_fn(state, action):
            new_state = (
                state
                + model["vae"].dynamics(state)
                + action
                + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        @torch.no_grad()
        def dynamics_true_fn(state, action):
            new_state = (
                state + f_true(state) + action + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        gif_frames = []  # Store frames for GIF
        training_losses = []
        for step in tqdm(range(exp_params["num_steps"])):
            if step > exp_params["trial_length"] + 1:
                # update model
                x_refine = torch.cat(x_traj[-(exp_params["trial_length"] + 1) :], dim=1)
                u_refine = torch.cat(
                    u_traj[-exp_params["trial_length"] :] + [torch.tensor([[0, 0]])]
                ).unsqueeze(0)
                y_refine = torch.exp(x_refine @ C.T + b) + torch.randn(
                    1, x_refine.shape[0], d_obs
                ) * torch.sqrt(R)

                loss = model["vae"].train_model(
                    DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                )
                training_losses.append(loss)

            u = (torch.randn(1, d_latent) * exp_params["u_strength"]).clamp_(
                -exp_params["u_strength"], exp_params["u_strength"]
            )
            new_state = dynamics_true_fn(x_traj[-1], u)

            x_traj.append(new_state)
            u_traj.append(u)
            if step % 1000 == 0:
                plot_vector_field(model["vae"].dynamics)
                plt.plot(
                    torch.cat(x_traj, dim=1).cpu().detach()[0, :, 0],
                    torch.cat(x_traj, dim=1).cpu().detach()[0, :, 1],
                    alpha=0.5,
                    lw=0.5,
                )
                plt.show()

            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        plot_vector_field(model["vae"].dynamics)
        plt.plot(
            torch.cat(x_traj, dim=1).cpu().detach()[0, :, 0],
            torch.cat(x_traj, dim=1).cpu().detach()[0, :, 1],
            alpha=0.5,
            lw=0.5,
        )
        plt.show()
    else:

        @torch.no_grad()
        def dynamics_fn(state, action):
            new_state = (
                state
                + model["vae"].dynamics(state)
                + action
                + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        @torch.no_grad()
        def dynamics_true_fn(state, action):
            new_state = (
                state + f_true(state) + action + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        gif_frames = []  # Store frames for GIF
        for step in tqdm(range(exp_params["num_steps"])):
            if step > exp_params["trial_length"] + 1:
                # update model
                x_refine = torch.cat(
                    x_traj[-(exp_params["trial_length"] + 1) + step : step], dim=1
                )
                u_refine = torch.cat(
                    u_traj[-exp_params["trial_length"] + step : step]
                    + [torch.tensor([[0, 0]])]
                ).unsqueeze(0)
                y_refine = torch.exp(x_refine @ C.T + b) + torch.randn(
                    1, x_refine.shape[0], d_obs
                ) * torch.sqrt(R)

                loss = model["vae"].train_model(
                    DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                )

            if step % 10 == 0 and step > 0:
                fig, ax, var_map, fisher_map = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    f_true,
                    fisher,
                    fisher_map=fisher_map,
                    x=torch.cat(x_traj[:step], dim=1).squeeze(0),
                    show=False,
                    fisher_max=1e-2,
                    var_range=0.05,
                )

                # Save the current figure to a buffer
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                gif_frames.append(Image.open(io.BytesIO(buf.read())))  # Copy image data
                buf.close()
                plt.close(fig)

        gif_frames[0].save(
            os.path.join(result_dir, f"random_trajectory.gif"),
            save_all=True,
            append_images=gif_frames[1:],
            fps=1000,
            loop=0,
        )

    # # Save the current figure to a buffer
    # buf = io.BytesIO()
    # fig.savefig(buf, format="png")
    # buf.seek(0)
    # gif_frames.append(Image.open(io.BytesIO(buf.read())))  # Copy image data
    # buf.close()
    # plt.close(fig)

    # gif_frames[0].save(
    #     os.path.join(result_dir, f"trajectory.gif"),
    #     save_all=True,
    #     append_images=gif_frames[1:],
    #     fps=1,
    #     loop=0,
    # )

    # -----------------------------------------------------------------
    # %% Run Experiment (Uncertainty only)
    # -----------------------------------------------------------------
    run_train = True
    # Add iCEM controller for trajectory optimization
    icem = SimpleICem(**icem_params)

    model = {
        "vae": copy.deepcopy(initial_vae).to(device),
        "ensemble": copy.deepcopy(initial_ensemble).to(device),
    }
    fisher_map = None
    if run_train:
        initial_state = torch.zeros((1, 1, d_latent), device=device)
        x_traj = [initial_state]
        u_traj = []

        fisher = FisherMetrics(
            dynamics=model["vae"].dynamics,
            decoder=model["vae"].decoder,
            process_noise=Q * torch.eye(d_latent).to(device),
            measurement_noise=R * torch.eye(d_obs).to(device),
        )

        def cost_fn(state, new_state, goal=None, action=None, temp=1e-4):
            # fim = fisher.compute_fim_point(new_state).sum(-1) * temp
            uncertainty = compute_uncertainty(model["ensemble"], new_state).sum(-1)
            control = (action**2).sum(-1)
            # print(f"Current control: {control}, Uncertainty: {uncertainty}")

            return -(uncertainty - control * 1).to(device)  # maximize FIM + uncertainty

        @torch.no_grad()
        def dynamics_fn(state, action):
            # Wrapper for your dynamics model
            new_state = (
                state
                + model["vae"].dynamics(state)
                + action
                + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        @torch.no_grad()
        def dynamics_true_fn(state, action):
            new_state = (
                state + f_true(state) + action + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        gif_frames = []  # Store frames for GIF

        for step in tqdm(range(exp_params["num_steps"])):
            # if step % exp_params["trial_length"] == 0 and step > 0:
            if step > exp_params["trial_length"] + 1:
                # update model
                x_refine = torch.cat(x_traj[-(exp_params["trial_length"] + 1) :], dim=1)
                u_refine = torch.cat(
                    u_traj[-exp_params["trial_length"] :]
                    + [torch.tensor([[0, 0]]).to(device)]
                ).unsqueeze(0)
                y_refine = torch.exp(x_refine @ C.T + b) + torch.randn(
                    1, x_refine.shape[0], d_obs, device=device
                ) * torch.sqrt(R)

                loss = model["vae"].train_model(
                    DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                )

                model["ensemble"].train_model(
                    DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                    perturbation=1e-3,
                )

            action_icem = icem.optimize(
                x_traj[-1],
                dynamics_fn,
                cost_fn,
                action_dim=d_latent,
                device=device,
            )
            new_state = dynamics_true_fn(x_traj[-1], action_icem)
            x_traj.append(new_state)
            u_traj.append(action_icem.unsqueeze(0))

            if step % 10 == 0 and step > 0:
                fig, ax, var_map, fisher_map = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    f_true,
                    fisher,
                    fisher_map=fisher_map,
                    x=torch.cat(x_traj, dim=1).squeeze(0),
                    show=False,
                    fisher_max=40,
                    var_range=0.05,
                )

                # Save the current figure to a buffer
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                gif_frames.append(Image.open(io.BytesIO(buf.read())))  # Copy image data
                buf.close()
                plt.close(fig)

        gif_frames[0].save(
            os.path.join(result_dir, f"uncertainty_perturb_trajectory.gif"),
            save_all=True,
            append_images=gif_frames[1:],
            fps=1000,
            loop=0,
        )
        # save x_traj and u_traj
        torch.save(
            {
                "x_traj": torch.cat(x_traj, dim=1).cpu(),
                "u_traj": torch.cat(u_traj, dim=0).cpu(),
            },
            os.path.join(result_dir, f"uncertainty_perturb_trajectory.pt"),
        )
    else:
        gif_frames = []  # Store frames for GIF

        fisher = FisherMetrics(
            dynamics=model["vae"].dynamics,
            decoder=model["vae"].decoder,
            process_noise=Q * torch.eye(d_latent).to(device),
            measurement_noise=R * torch.eye(d_obs).to(device),
        )

        for step in tqdm(range(exp_params["num_steps"])):
            if step > exp_params["trial_length"] + 1:
                # update model
                x_refine = torch.cat(
                    x_traj[-(exp_params["trial_length"] + 1) + step : step], dim=1
                )
                u_refine = torch.cat(
                    u_traj[-exp_params["trial_length"] + step : step]
                    + [torch.tensor([[0, 0]]).to(device)]
                ).unsqueeze(0)
                y_refine = torch.exp(x_refine @ C.T + b) + torch.randn(
                    1, x_refine.shape[0], d_obs, device=device
                ) * torch.sqrt(R)

                loss = model["vae"].train_model(
                    DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                )
                model["ensemble"].train_model(
                    DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                )
            if step % 10 == 0 and step > 0:
                fig, ax, var_map, fisher_map = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    f_true,
                    fisher,
                    fisher_map=fisher_map,
                    x=torch.cat(x_traj[:step], dim=1).squeeze(0),
                    show=False,
                    fisher_max=1e-2,
                    var_range=(-10, -3.5),
                )

                # Save the current figure to a buffer
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                buf.seek(0)
                gif_frames.append(Image.open(io.BytesIO(buf.read())))  # Copy image data
                buf.close()
                plt.close(fig)

        gif_frames[0].save(
            os.path.join(result_dir, f"uncertainty_perturb_trajectory.gif"),
            save_all=True,
            append_images=gif_frames[1:],
            fps=1000,
            loop=0,
        )

    # -----------------------------------------------------------------
    # %% Run Experiment (Fisher + Uncertainty)
    # -----------------------------------------------------------------

    # -----------------------------------------------------------------

    # Add iCEM controller for trajectory optimization
    icem = SimpleICem(**icem_params)

    model = {
        "vae": copy.deepcopy(initial_vae).to(device),
        "ensemble": copy.deepcopy(initial_ensemble).to(device),
    }

    initial_state = torch.zeros((1, 1, d_latent), device=device)
    x_traj = [initial_state]
    u_traj = []

    fisher = FisherMetrics(
        dynamics=model["vae"].dynamics,
        decoder=model["vae"].decoder,
        process_noise=Q * torch.eye(d_latent).to(device),
        measurement_noise=R * torch.eye(d_obs).to(device),
    )

    def cost_fn(state, new_state, goal=None, action=None, temp=1e-4):
        fim = fisher.compute_fim_point(new_state, use_diag=True) * 1e-8
        uncertainty = compute_uncertainty(model["ensemble"], new_state).sum(-1)
        # print(f"Current FIM: {fim}, Uncertainty: {uncertainty}")

        return -(uncertainty).to(device)  # maximize FIM + uncertainty

    @torch.no_grad()
    def dynamics_fn(state, action):
        # Wrapper for your dynamics model
        new_state = (
            state
            + model["vae"].dynamics(state)
            + action
            + torch.randn_like(state) * torch.sqrt(Q)
        )
        return new_state.clamp_(-2.5, 2.5)

    @torch.no_grad()
    def dynamics_true_fn(state, action):
        new_state = (
            state + f_true(state) + action + torch.randn_like(state) * torch.sqrt(Q)
        )
        return new_state.clamp_(-2.5, 2.5)

    gif_frames = []  # Store frames for GIF

    for step in tqdm(range(exp_params["num_steps"])):
        # if step % exp_params["trial_length"] == 0 and step > 0:
        if step > exp_params["trial_length"] + 1:
            # update model
            x_refine = torch.cat(x_traj[-(exp_params["trial_length"] + 1) :], dim=1)
            u_refine = torch.cat(
                u_traj[-exp_params["trial_length"] :]
                + [torch.tensor([[0, 0]]).to(device)]
            ).unsqueeze(0)
            y_refine = torch.exp(x_refine @ C.T + b) + torch.randn(
                1, x_refine.shape[0], d_obs, device=device
            ) * torch.sqrt(R)

            loss = model["vae"].train_model(
                DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                exp_params["lr_refine"],
                exp_params["weight_decay"],
                exp_params["refine_epoch"],
                has_input=True,
                verbose=False,
            )

            model["ensemble"].train_model(
                DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                exp_params["lr_refine"],
                exp_params["weight_decay"],
                exp_params["refine_epoch"],
                has_input=True,
                verbose=False,
            )

        action_icem = icem.optimize(
            x_traj[-1],
            dynamics_fn,
            cost_fn,
            action_dim=d_latent,
            device=device,
        )
        new_state = dynamics_true_fn(x_traj[-1], action_icem)
        x_traj.append(new_state)
        u_traj.append(action_icem.unsqueeze(0))

        if step % 100 == 0:
            fig, ax, var_map, fisher_map = plot_current_state(
                model["vae"],
                model["ensemble"],
                f_true,
                fisher,
                x=torch.cat(x_traj, dim=1).squeeze(0),
                show=True,
                fisher_max=0.05,
                var_range=0.05,
            )

    plot_vector_field(model["vae"].dynamics)
    plt.plot(
        torch.cat(x_traj, dim=1).cpu().detach()[0, :, 0],
        torch.cat(x_traj, dim=1).cpu().detach()[0, :, 1],
        alpha=0.5,
        lw=0.5,
    )
    plt.show()

    # -----------------------------------------------------------------
    # %% Run Experiment (Goal)
    # -----------------------------------------------------------------

    # Create a directory to save the figures
    result_dir = (
        "/home/hyungju/Desktop/vectorfield-fisher/results/experiment_meeting/rbf"
    )
    os.makedirs(result_dir, exist_ok=True)

    models = {
        "uncertainty": model_uncertainty,
    }

    initial_state = torch.tensor([[0.0, 0.0]], device=device)
    for model_name, model in models.items():
        x_traj = [initial_state]
        z_traj = [initial_state]
        u_traj = []
        i_traj = []

        fisher = FisherMetrics(
            dynamics=model["vae"].dynamics,
            decoder=model["vae"].decoder,
            process_noise=Q * torch.eye(d_latent),
            measurement_noise=R * torch.eye(d_obs),
        )

        def cost_fn(state, new_state, goal=None, action=None, temp=1e-7):
            fim = fisher.compute_fim_point(new_state).mean(-1) * temp
            target_dist = torch.norm(new_state - goal, dim=-1)
            # print(f"Current FIM: {fim}, Target dist: {target_dist}")

            return -(fim.to(device) - target_dist)  # maximize FIM + dist

        @torch.no_grad()
        def dynamics_fn(state, action):
            # Wrapper for your dynamics model
            new_state = (
                state
                + model["vae"].dynamics(state)
                + action
                + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        @torch.no_grad()
        def dynamics_true_fn(state, action):
            # Wrapper for your dynamics model
            new_state = (
                state
                + vae_star.dynamics(state)
                + action
                + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        fisher_map, _, _ = compute_fisher_map(
            fisher,
            grid_x,
            grid_y,
            show_plot=False,
        )
        var_map, _, _ = compute_uncertainty_map(
            model["ensemble"], grid_x, grid_y, show_plot=False
        )
        # plot distribution of var_map

        fig, ax, var_map, fisher_map = plot_current_state(
            model["vae"],
            model["ensemble"],
            f_true,
            fisher,
            fisher_map,
            var_map,
            show=True,
            fisher_max=0.05,
            var_range=0.5,
        )
        fig.savefig(os.path.join(result_dir, f"{model_name}_step_0.png"))

        gif_frames = []  # Store frames for GIF

        for step in range(exp_params["num_steps"]):
            if step % exp_params["trial_length"] == 0 and step > 0:
                # update model
                x_refine = torch.cat(
                    x_traj[-(exp_params["trial_length"] + 1) :]
                ).unsqueeze(0)
                u_refine = torch.cat(
                    u_traj[-exp_params["trial_length"] :] + [torch.tensor([[0, 0]])]
                ).unsqueeze(0)
                y_refine = (x_refine @ C.T + b) + torch.randn(
                    1, x_refine.shape[0], d_obs
                ) * torch.sqrt(R)

                model["vae"].train_model(
                    DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=True,
                )
                model["ensemble"].train_model(
                    DataLoader(torch.cat([y_refine, u_refine], dim=-1), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=True,
                )
                fisher_map, _, _ = compute_fisher_map(
                    fisher,
                    grid_x,
                    grid_y,
                    show_plot=False,
                )
                var_map, _, _ = compute_uncertainty_map(
                    model["ensemble"], grid_x, grid_y, show_plot=False
                )

            if model_name in ["fisher", "icem", "goal", "uncertainty"]:
                # update every icem horizon steps
                if step % icem_params["horizon"] == 0:
                    target_state = find_local_maxima_uncertainty(
                        model["ensemble"], d_latent
                    ).to(device)
                    print(f"Target state: {target_state}")
                    # update iCEM
                    action_icem = icem.optimize(
                        x_traj[-1],
                        dynamics_fn,
                        cost_fn,
                        goal=target_state,
                        action_dim=d_latent,
                        device=device,
                    )
                new_state = dynamics_true_fn(
                    x_traj[-1], action_icem[step % icem_params["horizon"]]
                )
                pred_state = dynamics_fn(
                    z_traj[-1], action_icem[step % icem_params["horizon"]]
                )
                x_traj.append(new_state)
                z_traj.append(pred_state)
                u_traj.append(action_icem[step % icem_params["horizon"]].unsqueeze(0))
                i_traj.append(torch.norm(pred_state - new_state))
            else:
                # update trajectory
                pass
            if step % exp_params["trial_length"] == 0:
                fig, ax, var_map, fisher_map = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    f_true,
                    fisher,
                    fisher_map,
                    var_map,
                    x=torch.cat(x_traj),
                    show=False,
                    fisher_max=0.05,
                    var_range=0.5,
                )
            else:
                fig, ax, var_map, fisher_map = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    f_true,
                    fisher,
                    fisher_map,
                    var_map,
                    x=torch.cat(x_traj),
                    fisher_max=0.05,
                    var_range=0.5,
                )

            # Save the current figure to a buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            gif_frames.append(Image.open(io.BytesIO(buf.read())))  # Copy image data
            buf.close()
            plt.close(fig)

        # Save the GIF directly
        gif_frames[0].save(
            os.path.join(result_dir, f"{model_name}_trajectory.gif"),
            save_all=True,
            append_images=gif_frames[1:],
            duration=exp_params["trial_length"] * 10,
            loop=0,
        )

    # -----------------------------------------------------------------
    # %% Run Experiment (Naive & Random)
    # -----------------------------------------------------------------

    # Create a directory to save the figures

    models = {
        "naive": model_naive,
        "random": model_random,
    }

    initial_state = torch.tensor([[0.0, 0.0]], device=device)
    for model_name, model in models.items():
        x_traj = [initial_state]
        z_traj = [initial_state]
        u_traj = []
        i_traj = []

        fisher = FisherMetrics(
            dynamics=model["vae"].dynamics,
            decoder=model["vae"].decoder,
            process_noise=Q * torch.eye(d_latent),
            measurement_noise=R * torch.eye(d_obs),
        )

        @torch.no_grad()
        def dynamics_fn(state, action):
            # Wrapper for your dynamics model
            new_state = (
                state
                + model["vae"].dynamics(state)
                + action
                + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        @torch.no_grad()
        def dynamics_true_fn(state, action):
            # Wrapper for your dynamics model
            new_state = (
                state
                + vae_star.dynamics(state)
                + action
                + torch.randn_like(state) * torch.sqrt(Q)
            )
            return new_state.clamp_(-2.5, 2.5)

        fisher_map, _, _ = compute_fisher_map(
            fisher,
            grid_x,
            grid_y,
            show_plot=False,
        )
        var_map, _, _ = compute_uncertainty_map(
            model["ensemble"], grid_x, grid_y, show_plot=False
        )
        # plot distribution of var_map

        fig, ax, var_map, fisher_map = plot_current_state(
            model["vae"],
            model["ensemble"],
            f_true,
            fisher,
            fisher_map,
            var_map,
            show=True,
            fisher_max=0.05,
            var_range=0.5,
        )
        fig.savefig(os.path.join(result_dir, f"{model_name}_step_0.png"))

        gif_frames = []  # Store frames for GIF

        for step in tqdm(range(exp_params["num_steps"])):
            # if step % exp_params["trial_length"] == 0 and step > 0:
            if step > exp_params["trial_length"] + 1:
                # update model
                x_refine = torch.cat(
                    x_traj[-(exp_params["trial_length"] + 1) :]
                ).unsqueeze(0)
                u_refine = torch.cat(
                    u_traj[-exp_params["trial_length"] :] + [torch.tensor([[0, 0]])]
                ).unsqueeze(0)
                y_refine = (x_refine @ C.T + b) + torch.randn(
                    1, x_refine.shape[0], d_obs
                ) * torch.sqrt(R)

                if model_name == "random":
                    model["vae"].train_model(
                        DataLoader(
                            torch.cat([y_refine, u_refine], dim=-1), batch_size=1
                        ),
                        exp_params["lr_refine"],
                        exp_params["weight_decay"],
                        exp_params["refine_epoch"],
                        has_input=True,
                        verbose=False,
                    )
                    model["ensemble"].train_model(
                        DataLoader(
                            torch.cat([y_refine, u_refine], dim=-1), batch_size=1
                        ),
                        exp_params["lr_refine"],
                        exp_params["weight_decay"],
                        exp_params["refine_epoch"],
                        has_input=True,
                        verbose=False,
                    )
                else:
                    model["vae"].train_model(
                        DataLoader(y_refine, batch_size=1),
                        exp_params["lr_refine"],
                        exp_params["weight_decay"],
                        exp_params["refine_epoch"],
                        has_input=False,
                        verbose=False,
                    )
                    model["ensemble"].train_model(
                        DataLoader(y_refine, batch_size=1),
                        exp_params["lr_refine"],
                        exp_params["weight_decay"],
                        exp_params["refine_epoch"],
                        has_input=False,
                        verbose=False,
                    )
                if step % exp_params["trial_length"] == 0 and step > 0:
                    fisher_map, _, _ = compute_fisher_map(
                        fisher,
                        grid_x,
                        grid_y,
                        show_plot=False,
                    )
                    var_map, _, _ = compute_uncertainty_map(
                        model["ensemble"], grid_x, grid_y, show_plot=False
                    )

            u = (
                (torch.randn(1, d_latent) * exp_params["u_strength"]).clamp_(
                    -exp_params["u_strength"], exp_params["u_strength"]
                )
                if model_name == "random"
                else torch.zeros(1, d_latent)
            )
            new_state = dynamics_true_fn(x_traj[-1], u)

            x_traj.append(new_state)
            u_traj.append(u)

            if step % 100 == 0:
                fig, ax, var_map, fisher_map = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    f_true,
                    fisher,
                    fisher_map,
                    var_map,
                    x=torch.cat(x_traj),
                    show=True,
                    fisher_max=0.05,
                    var_range=0.5,
                )
            else:
                fig, ax, var_map, fisher_map = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    f_true,
                    fisher,
                    fisher_map,
                    var_map,
                    x=torch.cat(x_traj),
                    fisher_max=0.05,
                    var_range=0.5,
                )

            # Save the current figure to a buffer
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            buf.seek(0)
            gif_frames.append(Image.open(io.BytesIO(buf.read())))  # Copy image data
            buf.close()
            plt.close(fig)

            if step % 1000 == 0 and step > 0:
                # Save the GIF directly
                gif_frames[0].save(
                    os.path.join(result_dir, f"{model_name}_trajectory_{step}.gif"),
                    save_all=True,
                    append_images=gif_frames[1:],
                    fps=1000,
                    loop=0,
                )

        # Save
        x_traj = torch.cat(x_traj)
        u_traj = torch.cat(u_traj)
        with h5py.File(os.path.join(result_dir, f"{model_name}.h5"), "w") as hf:
            hf.create_dataset("x_traj", data=x_traj.cpu().numpy())
            hf.create_dataset("u_traj", data=u_traj.cpu().numpy())

        # Save x_traj and y_traj as
        gif_frames[0].save(
            os.path.join(result_dir, f"{model_name}_trajectory_{step}.gif"),
            save_all=True,
            append_images=gif_frames[1:],
            fps=1000,
            loop=0,
        )
# %% Test if it learns
initial_vae = env.SeqVae(
    env.RBFDynamics(rbf_grid, device=device), encoder, decoder, device=device
)
x0 = torch.rand(1, 2) * 5 - 2.5
x0 = x0.unsqueeze(0)
exp_params["u_strength"] = 0.5


for k in range(10):
    T_test = 500
    u_random = torch.randn(1, T_test, 2) * exp_params["u_strength"]
    u_random.clamp_(-exp_params["u_strength"], exp_params["u_strength"])
    # u_random = torch.zeros_like(u_random)  # No control input
    x_random = f_true.generate_trajectory(x0, T_test, Q, input=u_random)
    y_random = ((x_random) @ C.T + b) + torch.randn(1, T_test, 50) * torch.sqrt(R)
    x0 = x_random[:, -1:]

    for i in tqdm(range(10, T_test)):
        initial_vae.train_model(
            DataLoader(
                torch.cat([y_random[:, i - 10 : i], u_random[:, i - 10 : i]], dim=-1),
                batch_size=1,
            ),
            1e-3,
            1e-5,
            1,
            has_input=True,
            verbose=False,
        )
    plot_vector_field(initial_vae.dynamics)
    plt.show()

# %% Visualize individual component of Fisher map
eps = 1e-6
# model = {
#     "vae": copy.deepcopy(initial_vae).to(device),
#     "ensemble": copy.deepcopy(initial_ensemble).to(device),
# }

grid_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device)
grid_fim = []
grid_lambda = []
fisher = FisherMetrics(
    dynamics=model["vae"].dynamics,
    decoder=model["vae"].decoder,
    process_noise=Q * torch.eye(d_latent).to(device),
    measurement_noise=R * torch.eye(d_obs).to(device),
)
fisher_2 = FisherMetrics(
    dynamics=initial_vae.dynamics,
    decoder=initial_vae.decoder,
    process_noise=Q * torch.eye(d_latent).to(device),
    measurement_noise=R * torch.eye(d_obs).to(device),
)
fisher = fisher

for grid_pt in grid_pts:
    # grid_pt = grid_pt.squeeze(0)  # Remove the first dimension
    df_dtheta = fisher.compute_jacobian_params(fisher.dynamics, grid_pt).detach()
    H = fisher.compute_jacobian_state(fisher.decoder, grid_pt).detach()
    sigma = H @ fisher.Q @ H.T + fisher.R
    J = H
    # sigma = fisher.Q
    # J = df_dtheta

    I = J.T @ torch.inverse(sigma) @ J
    # idx = torch.where(torch.diag(I) > 1e-6)[0]
    # I = I[idx, :]
    # I = I[:, idx]
    fim = torch.diag(torch.inverse(I))
    # fim = torch.reciprocal(torch.diag(I))

    fim = fim.sum()

    grid_fim.append(fim.cpu())

grid_fim = torch.stack(grid_fim, dim=0).reshape(len(grid_x), len(grid_y))


fig, axs = plt.subplots(1, 1, figsize=(5, 5))
# Plot grid_fim grid_lambda in contourf
contour_fim = axs.contourf(
    xx.cpu(),
    yy.cpu(),
    grid_fim.cpu().T,
    levels=10,
    cmap="plasma",
    alpha=0.5,
)

axs.set_xlim(-2.5, 2.5)
axs.set_ylim(-2.5, 2.5)
axs.set_title("Fisher Information Map")
