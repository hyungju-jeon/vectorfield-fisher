# %%
from sklearn import ensemble
import torch
import matplotlib.pyplot as plt
import copy
from vfim.agent.information import FisherMetrics
import vfim.model as env
import vfim.agent as agent
import vfim.utils as utils
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from matplotlib import cm
from matplotlib.colors import Normalize
from vfim.controllers.simple_icem import SimpleICem
from vfim.utils.visualize import plot_vector_field
import os
import imageio


result_dir = "/home/hyungju/Desktop/vectorfield-fisher/results/fisher/"
os.makedirs(result_dir, exist_ok=True)

# %% Define Auxiliary Functions
torch_seed = 1111
X_BOUND = 2.5

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
grid_x = torch.linspace(-X_BOUND, X_BOUND, 25)
grid_y = torch.linspace(-X_BOUND, X_BOUND, 25)
xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")  # [H, W]
grid_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1)

rbf_grid_x = torch.linspace(-X_BOUND, X_BOUND, 25)
rbf_grid_y = torch.linspace(-X_BOUND, X_BOUND, 25)
rbf_xx, rbf_yy = torch.meshgrid(rbf_grid_x, rbf_grid_y, indexing="ij")  # [H, W]
rbf_grid_pts = torch.stack([rbf_xx.flatten(), rbf_yy.flatten()], dim=1)

Q = torch.tensor(1e-3, device=device)  # process noise
R = torch.tensor(1e-3, device=device)  # measurement noise


def log_linear_obs(x, C, b, **kwargs):
    """Log-linear observation model."""
    return torch.exp(x @ C.T + b) + torch.randn(
        x.shape[0], x.shape[1], C.shape[0]
    ) * torch.sqrt(R)


def gaussian_obs(x, A, mu, sigma, **kwargs):
    """Gaussian observation model."""
    # C * (-(x-mu)^2/(2*sigma^2))
    # x : [B, T, d_latent]
    # A : [d_obs]
    # mu : [d_obs, d_latent]
    # sigma : [d_latent]
    XX = torch.einsum(
        "btlo, btlo -> btl",
        (x.unsqueeze(-2) - mu) * torch.reciprocal(sigma),
        (x.unsqueeze(-2) - mu),
    )
    return A * torch.exp(-XX) + torch.randn(
        x.shape[0], x.shape[1], A.shape[0]
    ) * torch.sqrt(R)


@torch.no_grad()
def dynamics_fn(state, action):
    new_state = (
        state
        + model["vae"].dynamics(state)
        + action
        + torch.randn_like(state) * torch.sqrt(Q)
    )
    return new_state.clamp_(-X_BOUND, X_BOUND)


@torch.no_grad()
def dynamics_true_fn(state, action):
    new_state = state + f_true(state) + action + torch.randn_like(state) * torch.sqrt(Q)
    return new_state.clamp_(-X_BOUND, X_BOUND)


def initialize_vectorfield(
    d_obs,
    d_latent,
    mean_rate=5,
    model="log-linear",
    dynamics="limitcycle",
    device="cpu",
):
    #  Step 0: Set up random seed and key
    torch.manual_seed(torch_seed)

    #  Step 1: Define true dynamics and observation model
    vf = utils.VectorField(model=dynamics, x_range=X_BOUND, n_grid=50)
    vf.generate_vector_field(random_seed=torch_seed, w_attractor=0.05)
    f_true = env.DynamicsWrapper(model=vf)

    if model == "log-linear":
        # Log-linear Observation model : y = exp(C z + b) + noise.
        C = torch.randn(d_obs, d_latent)
        C = C / torch.norm(C, dim=1, keepdim=True)  # Normalize rows of C
        C *= 0.5
        b = 1.0 * torch.randn(d_obs) - torch.log(torch.tensor(mean_rate))
        b += torch.log(torch.tensor(mean_rate) / torch.exp(grid_pts @ C.T + b).mean(0))

        model_params = {
            "model": "log-linear",
            "forward": log_linear_obs,
            "C": C,
            "b": b,
        }

        return f_true, model_params

    elif model == "gaussian":
        # Gaussian Observation model : y = C exp(-(x-u)^2/(2*sigma^2)) + noise
        A = torch.rand(d_obs) * 20
        mu = torch.rand(d_obs, d_latent) * (2 * X_BOUND) - X_BOUND
        sigma = torch.rand(d_obs, d_latent) * 0.5 + 2
        model_params = {
            "model": "gaussian",
            "forward": gaussian_obs,
            "A": A,
            "mu": mu,
            "sigma": sigma,
        }

        return f_true, model_params


def construct_vae_model(
    d_obs,
    d_latent,
    d_hidden,
    model_params,
    _encoder=None,
    _decoder=None,
    _dynamics=None,
    n_ensemble=1,
    device="cpu",
):
    """Construct VAE model with specified parameters."""
    # Initialize encoder and load state dict if provided
    encoder = env.MlpEncoder(d_obs, d_latent, d_hidden, device=device)
    if _encoder is not None:
        encoder.load_state_dict(_encoder.state_dict())
        if not all(p.requires_grad for p in _encoder.parameters()):
            for param in encoder.parameters():
                param.requires_grad = False

    # Initialize dynamics model and load state dict if provided
    if n_ensemble > 1:
        dynamics = env.EnsembleRBF(rbf_grid_pts, n_models=n_ensemble, device=device)
    else:
        dynamics = env.RBFDynamics(rbf_grid_pts, device=device)
    if _dynamics is not None:
        dynamics.load_state_dict(_dynamics.state_dict())

    # Initialize decoder and load state dict if provided
    if model_params["model"] == "log-linear":
        decoder = env.LogLinearNormalDecoder(
            dx=d_latent,
            dy=d_obs,
            C=model_params["C"],
            b=model_params["b"],
            device=device,
        )
    elif model_params["model"] == "gassusian":
        # TODO: Implement GaussianDecoder
        pass
    else:
        raise ValueError("Unknown observation model")
    if _decoder is not None:
        decoder.load_state_dict(_decoder.state_dict())
        if not all(p.requires_grad for p in _decoder.parameters()):
            for param in decoder.parameters():
                param.requires_grad = False

    if n_ensemble > 1:
        vae = env.EnsembleSeqVae(dynamics, encoder, decoder, device=device)
    else:
        vae = env.SeqVae(dynamics, encoder, decoder, device=device)

    return vae


def approximate_vectorfield(f_true, model_params, device="cpu"):
    K = 5000
    T = 100

    obs_model = model_params["forward"]

    x0 = torch.rand(K, 2) * 5 - 2.5
    x0 = x0.unsqueeze(1)
    x_star = f_true.generate_trajectory(x0, T, Q)
    y_star = obs_model(x_star, **model_params)

    d_obs = y_star.shape[-1]
    d_latent = x_star.shape[-1]
    #  Step 3: Initialize VAE for 'true' f approximation
    d_hidden = 16

    encoder = env.MlpEncoder(d_obs, d_latent, d_hidden, device=device)
    # dynamics = env.RNNDynamics(d_latent, dh=8, device=device)
    dynamics = env.RBFDynamics(rbf_grid_pts, device=device)
    # Initialize decoder with pre-defined C matrix (known readout)
    decoder = env.LogLinearNormalDecoder(
        dx=d_latent, dy=d_obs, C=model_params["C"], b=model_params["b"], device=device
    )
    vae_star = env.SeqVae(dynamics, encoder, decoder, device=device)

    batch_size = 64
    n_epochs = 200
    lr = 1e-4
    weight_decay = 1e-4

    dataloader = DataLoader(TensorDataset(y_star), batch_size=batch_size)
    vae_star.train_model(dataloader, lr, weight_decay, n_epochs, optimizer="AdamW")

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

    return var_map, xx.cpu(), yy.cpu()


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
        fisher.compute_crlb_point_rbf(x.unsqueeze(0).to(device)) for x in grid
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


@torch.no_grad()
def compute_delta_f(f, f_hat):
    grid_x = torch.linspace(-2.5, 2.5, 10)
    grid_y = torch.linspace(-2.5, 2.5, 10)
    xx, yy = torch.meshgrid(grid_x, grid_y, indexing="ij")  # [H, W]
    grid_pts = torch.stack([xx.flatten(), yy.flatten()], dim=1).to(device).unsqueeze(0)

    vf_diff = torch.norm(f(grid_pts) - f_hat(grid_pts)) / torch.norm(f(grid_pts))

    return vf_diff


def plot_current_state(
    model,
    ensemble_model,
    plot_list=["vectorfield", "fisher", "uncertainty", "difference"],
    delta_f=None,
    fisher=None,
    x=None,
    title=None,
    map_alpha=0.3,
    show=False,
    fisher_range=(-6, -4),
    var_range=(-10.0, -3),
):
    def plot_trajectory(x, ax):
        num_bold = min(20, x.shape[0] // 10)
        ax.plot(
            x[:-num_bold, 0].cpu().numpy(),
            x[:-num_bold, 1].cpu().numpy(),
            color="red",
            alpha=0.5,
            lw=0.5,
        )
        ax.plot(
            x[-num_bold:, 0].cpu().numpy(),
            x[-num_bold:, 1].cpu().numpy(),
            color="red",
            alpha=0.7,
            marker=".",
            lw=1,
        )

    num_plots = len(plot_list)
    # Depending on num_plots, make subplots
    if num_plots < 4:
        fig, axs = plt.subplots(1, num_plots, figsize=(num_plots * 7, 5))
    else:
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()
    idx = 0

    if "vectorfield" in plot_list:
        plot_vector_field(model.dynamics, ax=axs[idx], cmax=0.5, show_cbar=True)
        axs[idx].set_xlim(-2.5, 2.5)
        axs[idx].set_ylim(-2.5, 2.5)
        axs[idx].set_title("Vector Field")
        idx += 1

    if "difference" in plot_list:
        axs[idx].plot(
            torch.tensor(delta_f).cpu().numpy(),
            color="red",
        )
        axs[idx].set_title("norm(f-f_hat)")
        idx += 1

    if "fisher" in plot_list:
        fisher_map, _, _ = compute_fisher_map(
            fisher,
            grid_x,
            grid_y,
            show_plot=False,
        )
        plot_vector_field(model.dynamics, ax=axs[idx], cmax=0.5)
        contour = axs[idx].contourf(
            xx.cpu(),
            yy.cpu(),
            torch.log(fisher_map).cpu(),
            levels=10,
            cmap="plasma",
            alpha=map_alpha,
        )
        contour.set_clim(fisher_range[0], fisher_range[1])  # Set color axis limits
        plt.colorbar(
            ax=axs[idx],
            label="Fisher Information",
            mappable=cm.ScalarMappable(
                norm=Normalize(vmin=fisher_range[0], vmax=fisher_range[1]),
                cmap="plasma",
            ),
            alpha=map_alpha,
        )
        if x is not None:
            plot_trajectory(x, axs[idx])

        axs[idx].set_xlim(-2.5, 2.5)
        axs[idx].set_ylim(-2.5, 2.5)
        axs[idx].set_title("A-Optimal Tr(inv(FIM)) (log)")
        idx += 1

    if "uncertainty" in plot_list:
        var_map, _, _ = compute_uncertainty_map(ensemble_model, grid_x, grid_y)
        plot_vector_field(model.dynamics, ax=axs[idx], cmax=0.5)
        contour = axs[idx].contourf(
            xx.cpu(),
            yy.cpu(),
            torch.log(var_map).cpu(),
            levels=10,
            cmap="plasma",
            alpha=map_alpha,
        )
        if var_range is not None:
            contour.set_clim(var_range[0], var_range[1])  # Set color axis limits
        plt.colorbar(
            ax=axs[idx],
            label="Uncertainty",
            mappable=cm.ScalarMappable(
                norm=Normalize(vmin=var_range[0], vmax=var_range[1]), cmap="plasma"
            ),
            alpha=map_alpha,
        )
        if x is not None:
            plot_trajectory(x, axs[idx])

        axs[idx].set_xlim(-2.5, 2.5)
        axs[idx].set_ylim(-2.5, 2.5)
        axs[idx].set_title("Uncertainty Map (log)")
        idx += 1

    if title is not None:
        fig.suptitle(title)

    if show:
        plt.show()

    return fig, axs


# %% Initialize vector field and approximate it with VAE
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_obs = 50
    d_latent = 2
    run_initialize = False
    true_dynamics = "limitcycle"

    true_dynamics_file = os.path.join(result_dir, "model_true.pt")
    if os.path.exists(true_dynamics_file) and not run_initialize:
        saved_models = torch.load(true_dynamics_file)
        model_params = saved_models["model_params"]
        d_obs = model_params["C"].shape[0]
        d_latent = model_params["C"].shape[1]

        vae_star = construct_vae_model(
            d_obs,
            d_latent,
            d_hidden=16,
            model_params=model_params,
        )
        vae_star.load_state_dict(saved_models["vae_star"].state_dict())
        f_true = saved_models["f_true"]

        print("Loaded VAE from file.")
    else:
        f_true, model_params = initialize_vectorfield(
            50, 2, model="log-linear", dynamics=true_dynamics, device=device
        )
        vae_star, f_star, x_star, y_star = approximate_vectorfield(
            f_true, model_params, device
        )
        torch.save(
            {
                "model_params": model_params,
                "vae_star": vae_star,
                "f_true": f_true,
            },
            true_dynamics_file,
        )

    #  Initialize VAE and Ensemble VAE
    fixed_encoder = copy.deepcopy(vae_star.encoder)
    fixed_decoder = copy.deepcopy(vae_star.decoder)
    for param in fixed_encoder.parameters():
        param.requires_grad = False
    for param in fixed_decoder.parameters():
        param.requires_grad = False
    initial_vae = construct_vae_model(
        d_obs,
        d_latent,
        d_hidden=16,
        model_params=model_params,
        _encoder=fixed_encoder,
        _decoder=fixed_decoder,
        device=device,
    )
    initial_ensemble = construct_vae_model(
        d_obs,
        d_latent,
        d_hidden=16,
        model_params=model_params,
        _encoder=fixed_encoder,
        _decoder=fixed_decoder,
        n_ensemble=10,
        device=device,
    )

    # -----------------------------------------------------------------
    # %% Set Parameters
    # -----------------------------------------------------------------
    def get_param():
        exp_params = {
            "num_steps": 10000,
            "trial_length": 20,
            "refine_epoch": 1,
            "lr_refine": 1e-4,
            "u_strength": 0.2,
            "weight_decay": 1e-5,
            "result_dir": result_dir,
            "save_animation": True,
            "debug": True,
        }
        fisher_param = {
            "num_samples": 10,
        }
        icem_params = {
            "horizon": 20,
            "num_iterations": 10,
            "num_traj": 32,
            "num_elites": 5,
            "alpha": 0.1,
            "action_bounds": [-exp_params["u_strength"], exp_params["u_strength"]],
        }
        return exp_params, icem_params, fisher_param

    # -----------------------------------------------------------------
    # %% Run Experiment (Naive and Random)
    # -----------------------------------------------------------------
    debug = True
    model_list = ["naive", "random"]
    obs_model = model_params["forward"]

    for model_id in model_list:
        vae = construct_vae_model(
            d_obs,
            d_latent,
            d_hidden=16,
            _encoder=initial_vae.encoder,
            _decoder=initial_vae.decoder,
            _dynamics=initial_vae.dynamics,
            model_params=model_params,
            device=device,
        )
        ensemble = construct_vae_model(
            d_obs,
            d_latent,
            d_hidden=16,
            _encoder=initial_vae.encoder,
            _decoder=initial_vae.decoder,
            _dynamics=initial_ensemble.dynamics,
            model_params=model_params,
            n_ensemble=10,
            device=device,
        )
        model = {
            "vae": vae,
            "ensemble": ensemble,
        }
        fisher_map = None
        exp_params, icem_params, _ = get_param()
        exp_params["result_dir"] = os.path.join(
            exp_params["result_dir"], f"{true_dynamics}_{model_id}"
        )
        os.makedirs(exp_params["result_dir"], exist_ok=True)

        exp_params["u_strength"] = 0.2 if model_id == "random" else 0.0

        initial_state = torch.zeros((1, 1, d_latent), device=device)
        x_traj = [initial_state]
        u_traj = []
        training_losses = []
        delta_f = []

        for step in tqdm(range(exp_params["num_steps"])):
            step_slice = slice(-(exp_params["trial_length"]) + step, step)
            if step > exp_params["trial_length"] + 1:
                x_refine = torch.cat(
                    ([x_traj[int(step_slice.start)]] + x_traj[step_slice]), dim=1
                )
                u_refine = torch.cat(u_traj[step_slice], dim=1)
                y_refine = obs_model(x_refine, **model_params)

                model["vae"].train_model(
                    DataLoader(list(zip(y_refine, u_refine)), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                )

                model["ensemble"].train_model(
                    DataLoader(list(zip(y_refine, u_refine)), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                    perturbation=1e-3 if model_id == "uncertainty_perturb" else 0,
                )

            u = torch.randn(1, 1, d_latent, device=device) * exp_params["u_strength"]
            u = u.clamp(-exp_params["u_strength"], exp_params["u_strength"])
            new_state = dynamics_true_fn(x_traj[-1], u)

            x_traj.append(new_state)
            u_traj.append(u)
            delta_f.append(compute_delta_f(f_true, model["vae"].dynamics))

            if exp_params["save_animation"] and step % 10 == 0 and step > 0:
                fig, ax = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    plot_list=["vectorfield", "uncertainty", "difference"],
                    delta_f=delta_f,
                    x=torch.cat(x_traj[:step], dim=1).squeeze(0),
                    show=False,
                    fisher_range=(-10, -8),
                    var_range=(-5, 0),
                )

                # Save the current figure to exp_params["result_dir"] folder as png
                fname = os.path.join(exp_params["result_dir"], f"step_{step:04d}.png")
                fig.savefig(fname, dpi=150)
                plt.close(fig)

        if exp_params["save_animation"]:
            # Create mp4 video from png files using imageio
            png_files = [
                os.path.join(exp_params["result_dir"], f)
                for f in sorted(os.listdir(exp_params["result_dir"]))
                if f.endswith(".png")
            ]
            video_path = os.path.join(
                exp_params["result_dir"], f"{true_dynamics}_{model_id}_trajectory.mp4"
            )
            with imageio.get_writer(
                video_path, fps=60, codec="libx264", macro_block_size=1
            ) as writer:
                for png_file in png_files:
                    image = imageio.imread(png_file)
                    writer.append_data(image)

            # Remove png files
            for png_file in png_files:
                os.remove(png_file)

        torch.save(
            {
                "x_traj": torch.cat(x_traj, dim=1).cpu(),
                "u_traj": torch.cat(u_traj, dim=0).cpu(),
                "training_losses": training_losses,
                "delta_f": delta_f,
                "model": model,
            },
            os.path.join(exp_params["result_dir"], f"result.pt"),
        )

    # -----------------------------------------------------------------
    # %% Run Experiment (Uncertainty only)
    # -----------------------------------------------------------------
    debug = True
    model_list = ["uncertainty", "uncertainty_perturb"]
    obs_model = model_params["forward"]
    for model_id in model_list:
        exp_params, icem_params, _ = get_param()
        # Add iCEM controller for trajectory optimization
        icem = SimpleICem(**icem_params)
        vae = construct_vae_model(
            d_obs,
            d_latent,
            d_hidden=16,
            _encoder=initial_vae.encoder,
            _decoder=initial_vae.decoder,
            _dynamics=initial_vae.dynamics,
            model_params=model_params,
            device=device,
        )
        ensemble = construct_vae_model(
            d_obs,
            d_latent,
            d_hidden=16,
            _encoder=initial_vae.encoder,
            _decoder=initial_vae.decoder,
            _dynamics=initial_ensemble.dynamics,
            model_params=model_params,
            n_ensemble=10,
            device=device,
        )
        model = {
            "vae": vae,
            "ensemble": ensemble,
        }
        fisher_map = None
        exp_params["result_dir"] = os.path.join(
            exp_params["result_dir"], f"{true_dynamics}_{model_id}"
        )
        os.makedirs(exp_params["result_dir"], exist_ok=True)

        initial_state = torch.zeros((1, 1, d_latent), device=device)
        x_traj = [initial_state]
        u_traj = []
        training_losses = []
        delta_f = []

        def cost_fn(state, new_state, goal=None, action=None, temp=1e-4):
            uncertainty = compute_uncertainty(model["ensemble"], new_state).sum(-1)
            control = (action**2).sum(-1)
            # print(f"Current control: {control}, Uncertainty: {uncertainty}")

            return -(uncertainty - control * 0.5).to(device)

        gif_frames = []  # Store frames for GIF

        for step in tqdm(range(exp_params["num_steps"])):
            step_slice = slice(-(exp_params["trial_length"]) + step, step)
            if step > exp_params["trial_length"] + 1:
                x_refine = torch.cat(
                    ([x_traj[int(step_slice.start)]] + x_traj[step_slice]), dim=1
                )
                u_refine = torch.cat(u_traj[step_slice], dim=1)
                y_refine = obs_model(x_refine, **model_params)

                model["vae"].train_model(
                    DataLoader(list(zip(y_refine, u_refine)), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                )

                model["ensemble"].train_model(
                    DataLoader(list(zip(y_refine, u_refine)), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                    perturbation=1e-2 if model_id == "uncertainty_perturb" else 0,
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
            u_traj.append(action_icem.view_as(new_state))
            delta_f.append(compute_delta_f(f_true, model["vae"].dynamics))

            if exp_params["save_animation"] and step % 10 == 0 and step > 0:
                fig, ax = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    plot_list=["vectorfield", "uncertainty", "difference"],
                    delta_f=delta_f,
                    x=torch.cat(x_traj[:step], dim=1).squeeze(0),
                    show=False,
                    fisher_range=(-10, -8),
                    var_range=(-5, 0),
                )

                # Save the current figure to exp_params["result_dir"] folder as png
                fname = os.path.join(exp_params["result_dir"], f"step_{step:04d}.png")
                fig.savefig(fname, dpi=150)
                plt.close(fig)

        if exp_params["save_animation"]:
            # Create mp4 video from png files using imageio
            png_files = [
                os.path.join(exp_params["result_dir"], f)
                for f in sorted(os.listdir(exp_params["result_dir"]))
                if f.endswith(".png")
            ]
            video_path = os.path.join(
                exp_params["result_dir"], f"{model_id}_trajectory.mp4"
            )
            with imageio.get_writer(
                video_path, fps=60, codec="libx264", macro_block_size=1
            ) as writer:
                for png_file in png_files:
                    image = imageio.imread(png_file)
                    writer.append_data(image)

            # Remove png files
            for png_file in png_files:
                os.remove(png_file)

        torch.save(
            {
                "x_traj": torch.cat(x_traj, dim=1).cpu(),
                "u_traj": torch.cat(u_traj, dim=0).cpu(),
                "training_losses": training_losses,
                "delta_f": delta_f,
                "model": model,
            },
            os.path.join(exp_params["result_dir"], f"result.pt"),
        )
    # %% Run Experiment (Fisher)
    # -----------------------------------------------------------------
    debug = True
    model_list = ["fisher"]
    obs_model = model_params["forward"]
    for model_id in model_list:
        exp_params, icem_params, _ = get_param()
        # Add iCEM controller for trajectory optimization
        icem = SimpleICem(**icem_params)
        vae = construct_vae_model(
            d_obs,
            d_latent,
            d_hidden=16,
            _encoder=initial_vae.encoder,
            _decoder=initial_vae.decoder,
            _dynamics=initial_vae.dynamics,
            model_params=model_params,
            device=device,
        )
        ensemble = construct_vae_model(
            d_obs,
            d_latent,
            d_hidden=16,
            _encoder=initial_vae.encoder,
            _decoder=initial_vae.decoder,
            _dynamics=initial_ensemble.dynamics,
            model_params=model_params,
            n_ensemble=10,
            device=device,
        )
        model = {
            "vae": vae,
            "ensemble": ensemble,
        }
        fisher = FisherMetrics(
            dynamics=model["vae"].dynamics,
            decoder=model["vae"].decoder,
            process_noise=Q * torch.eye(d_latent).to(device),
            measurement_noise=R * torch.eye(d_obs).to(device),
        )
        fisher_map = None
        exp_params["result_dir"] = os.path.join(exp_params["result_dir"], model_id)
        os.makedirs(exp_params["result_dir"], exist_ok=True)

        initial_state = torch.zeros((1, 1, d_latent), device=device)
        x_traj = [initial_state]
        u_traj = []
        training_losses = []
        delta_f = []

        def cost_fn(state, new_state, goal=None, action=None):
            crlb = fisher.compute_crlb_point_rbf(state, action).sum(-1)
            uncertainty = compute_uncertainty(model["ensemble"], new_state).sum(-1)
            control = (action**2).sum(-1)
            # print(
            #     f"Current CRLB:{crlb.mean()}, Control: {control.mean()}, Uncertainty: {uncertainty.mean()}"
            # )

            return -(uncertainty - control * 0.5 - crlb * 10).to(device)

        gif_frames = []  # Store frames for GIF

        for step in tqdm(range(exp_params["num_steps"])):
            step_slice = slice(-(exp_params["trial_length"]) + step, step)
            if step > exp_params["trial_length"] + 1:
                x_refine = torch.cat(
                    ([x_traj[int(step_slice.start)]] + x_traj[step_slice]), dim=1
                )
                u_refine = torch.cat(u_traj[step_slice], dim=1)
                y_refine = obs_model(x_refine, **model_params)

                model["vae"].train_model(
                    DataLoader(list(zip(y_refine, u_refine)), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                )

                model["ensemble"].train_model(
                    DataLoader(list(zip(y_refine, u_refine)), batch_size=1),
                    exp_params["lr_refine"],
                    exp_params["weight_decay"],
                    exp_params["refine_epoch"],
                    has_input=True,
                    verbose=False,
                    perturbation=1e-2,
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
            u_traj.append(action_icem.view_as(new_state))
            delta_f.append(compute_delta_f(f_true, model["vae"].dynamics))

            if exp_params["save_animation"] and step % 10 == 0 and step > 0:
                fig, ax = plot_current_state(
                    model["vae"],
                    model["ensemble"],
                    plot_list=["vectorfield", "uncertainty", "difference", "fisher"],
                    delta_f=delta_f,
                    fisher=fisher,
                    x=torch.cat(x_traj[:step], dim=1).squeeze(0),
                    show=False,
                    fisher_range=(-10, -8),
                    var_range=(-5, 0),
                )

                # Save the current figure to exp_params["result_dir"] folder as png
                fname = os.path.join(exp_params["result_dir"], f"step_{step:04d}.png")
                fig.savefig(fname, dpi=150)
                plt.close(fig)

        if exp_params["save_animation"]:
            # Create mp4 video from png files using imageio
            png_files = [
                os.path.join(exp_params["result_dir"], f)
                for f in sorted(os.listdir(exp_params["result_dir"]))
                if f.endswith(".png")
            ]
            video_path = os.path.join(
                exp_params["result_dir"], f"{model_id}_trajectory.mp4"
            )
            with imageio.get_writer(
                video_path, fps=60, codec="libx264", macro_block_size=1
            ) as writer:
                for png_file in png_files:
                    image = imageio.imread(png_file)
                    writer.append_data(image)

            # Remove png files
            for png_file in png_files:
                os.remove(png_file)

        torch.save(
            {
                "x_traj": torch.cat(x_traj, dim=1).cpu(),
                "u_traj": torch.cat(u_traj, dim=0).cpu(),
                "training_losses": training_losses,
                "delta_f": delta_f,
                "model": model,
            },
            os.path.join(exp_params["result_dir"], f"result.pt"),
        )
