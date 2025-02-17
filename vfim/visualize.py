import numpy as np
import torch
import matplotlib.pyplot as plt


def compute_vector_field(dynamics, xmin, xmax, ymin, ymax, device="cpu"):
    """
    Produces a vector field for a given dynamical system
    :param queries: N by dx torch tensor of query points where each row is a query
    :param dynamics: function handle for dynamics
    """
    xt, yt = np.meshgrid(np.linspace(xmin, xmax, 100), np.linspace(ymin, ymax, 100))
    queries = np.stack([xt.ravel(), yt.ravel()]).T
    if hasattr(dynamics, "device"):
        queries = torch.from_numpy(queries).float().to(dynamics.device)
    else:
        queries = torch.from_numpy(queries).float().to(device)

    vel = torch.zeros(queries.shape, device=device)
    with torch.no_grad():
        for n in range(queries.shape[0]):
            vel[n, :] = (dynamics(queries[[n]]) - queries[[n]]).to("cpu")

    vel_x = vel[:, 0].reshape(xt.shape[0], xt.shape[1])
    vel_y = vel[:, 1].reshape(yt.shape[0], yt.shape[1])
    speed = torch.sqrt(vel_x**2 + vel_y**2)
    return xt, yt, vel_x, vel_y, speed


def plot_vector_field(vae, device):
    xt, yt, vel_x, vel_y, speed = compute_vector_field(
        vae.dynamics,
        xmin=-20,
        xmax=5,
        ymin=-20,
        ymax=10,
        device=device,
    )

    plt.figure(figsize=(10, 8))
    plt.streamplot(
        xt, yt, vel_x, vel_y, color=speed.numpy(), linewidth=1, cmap="viridis"
    )
    plt.colorbar(label="Speed")
    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title("Vector Field of Latent Dynamics")
    plt.show()
