# %%
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Decoder(nn.Module):

    def __init__(self, num_out):
        super(Decoder, self).__init__()
        self.decoder = (
            nn.Linear(1, num_out, bias=False).to(device).requires_grad_(False)
        )
        self.decoder.weight.data = torch.rand(1, num_out).T

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: shape (input_dim,)
        return self.decoder(z)


class Encoder(nn.Module):
    def __init__(self, h: Decoder, num_in):
        super(Encoder, self).__init__()
        self.encoder = nn.Linear(num_in, 1, bias=False).to(device).requires_grad_(False)
        self.encoder.weight.data = (1 / h.decoder.weight.data) / num_in

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: shape (input_dim,)
        return self.encoder(z)


class Dynamics(nn.Module):
    def __init__(self):
        super(Dynamics, self).__init__()
        self.dynamics = nn.Linear(1, 1, bias=False).to(device)
        self.dynamics.weight.data = torch.tensor([0.1])

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: shape (input_dim,)

        return self.dynamics(z)


def compute_jacobian_state(function, state, **kwargs):
    """Compute Jacobian of a function with respect to state.

    Args:
        state (torch.Tensor): Current state vector of shape (n_state,)

    Returns:
        torch.Tensor: Jacobian matrix of shape (n_state, n_state)
    """
    state = state.clone().detach().requires_grad_(True)
    J_f = torch.autograd.functional.jacobian(function, state, **kwargs)
    return J_f


def compute_jacobian_params(function, state, indices=None, detach=False, **kwargs):
    if detach:
        state = state.clone().detach().requires_grad_(True)
    f_val = function(state)
    out_dim = f_val.shape[0]
    params = list(function.parameters())
    if indices is not None:
        params = [params[i] for i in indices]

    J_list = []
    for i in range(out_dim):
        grads = torch.autograd.grad(
            f_val[i], params, retain_graph=True, allow_unused=True, **kwargs
        )
        grad_vec = torch.cat(
            [
                torch.zeros_like(p).view(-1) if g is None else g.view(-1)
                for p, g in zip(params, grads)
            ]
        )
        J_list.append(grad_vec.unsqueeze(0))
    return torch.cat(J_list, dim=0)


# %%
dynamics = Dynamics()
decoder = Decoder(5)
encoder = Encoder(decoder, 5)

P = torch.zeros(1, 1)
z = torch.tensor([[1.0]])
dP = torch.zeros(1, 1)
I = torch.eye(1)
dz_dtheta = torch.zeros(1, 1)
fim = 0

Q, R = 1e-4, 1e-4

for iter in range(10):
    df_dz = compute_jacobian_state(dynamics, z).detach().squeeze().unsqueeze(-1)
    B = (
        compute_jacobian_params(dynamics, z, detach=True)
        .detach()
        .squeeze()
        .unsqueeze(-1)
    )

    z_new = z + dynamics(z)
    A = I + df_dz  # A = I + ∂f/∂z
    P_new = (A @ P @ A.T + Q).detach()
    dz_dtheta = (A @ dz_dtheta + B).detach()

    dA_dtheta = torch.autograd.functional.jacobian(
        lambda z_in: compute_jacobian_params(dynamics, z_in, create_graph=True),
        z,
    )  # dA_dtheta = ∂^2f/∂z∂θ +  (∂^2f/∂z^2) ∂z/∂θ
    dA_dtheta = dA_dtheta.squeeze(-1).squeeze(-1)

    new_dP = (dA_dtheta @ P @ A.T + A @ dP @ A.T + A @ P @ dA_dtheta).detach()
    dP = new_dP.detach()
    P = P_new.detach()

    # Measurement step
    H = compute_jacobian_state(decoder, z).detach().squeeze().unsqueeze(-1)
    sigma = (H @ P @ H.T + R).detach()
    de_dtheta = (-H @ dz_dtheta).detach()
    dsigma_dtheta = H @ dP @ H.T
    z = z_new

    sigma_inv = torch.inverse(sigma)
    mean_term = de_dtheta.T @ sigma_inv @ de_dtheta
    trace_term = 0.5 * torch.trace(
        sigma_inv @ dsigma_dtheta @ sigma_inv @ dsigma_dtheta
    )

    fim += mean_term + trace_term
    print(
        f"Current z = {z}, fim = {fim}, df_dz = {df_dz}, A = {A}, B = {B}, dA_dtheta = {dA_dtheta}"
    )
