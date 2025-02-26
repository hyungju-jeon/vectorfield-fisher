import torch
import torch.nn as nn
import numpy as np


class FisherMetrics:
    """Compute Fisher Information Matrix and related metrics for dynamics models.

    Args:
        dynamics: Dynamics model implementing forward method
        observation_matrix: Matrix C for observation model of shape (n_obs, n_state)
        observation_bias: Bias vector b for observation model of shape (n_obs,)
        process_noise: Process noise covariance matrix Q of shape (n_state, n_state)
        measurement_noise: Measurement noise covariance R of shape (n_obs, n_obs)

    Attributes:
        dynamics: The dynamics model
        C: Observation matrix
        b: Observation bias
        Q: Process noise covariance
        R: Measurement noise covariance
        device: Device on which computations are performed
    """

    def __init__(
        self,
        dynamics,
        decoder,
        process_noise,
        measurement_noise,
        use_kfac=False,
    ):
        self.dynamics = dynamics
        self.decoder = decoder
        self.Q = process_noise
        self.R = measurement_noise
        self.use_kfac = use_kfac
        self.device = next(dynamics.parameters()).device

    def compute_jacobian_state(self, function, state):
        """Compute Jacobian of a function with respect to state.

        Args:
            state (torch.Tensor): Current state vector of shape (n_state,)

        Returns:
            torch.Tensor: Jacobian matrix of shape (n_state, n_state)
        """
        state = state.clone().detach().requires_grad_(True)
        J_f = torch.autograd.functional.jacobian(lambda z: function(z), state)
        return J_f

    def compute_jacobian_params(self, function, state):
        """Compute Jacobian of a function with respect to parameters.

        Args:
            state (torch.Tensor): Current state vector of shape (n_state,)

        Returns:
            torch.Tensor: Jacobian matrix of shape (n_state, n_params)
        """
        f_val = function(state)
        out_dim = f_val.shape[0]
        params = list(function.parameters())

        J_list = []
        for i in range(out_dim):
            grads = torch.autograd.grad(
                f_val[i], params, retain_graph=True, allow_unused=True
            )
            grad_vec = torch.cat(
                [
                    torch.zeros_like(p).view(-1) if g is None else g.view(-1)
                    for p, g in zip(params, grads)
                ]
            )
            J_list.append(grad_vec.unsqueeze(0))
        return torch.cat(J_list, dim=0)

    def compute_jacobian_params_kfac(self, state):
        pass

    def compute_H_derivative(self, state):
        """Compute derivative of H = d(decoder)/dz with respect to state.
        Args:
            state (torch.Tensor): State vector (n_state,)
        Returns:
            torch.Tensor: dH/dz of shape (n_state, n_obs, n_state), where the first index corresponds
                          to the derivative with respect to each element of state.
        """

        def H_fn(z):
            return self.compute_jacobian_state(self.decoder, z)

        dH_dz = torch.autograd.functional.jacobian(H_fn, state)
        # dH_dz shape: (n_state, n_obs, n_state)
        return dH_dz

    def compute_sigma(self, state, state_cov):
        """Compute innovation covariance matrix.

        Args:
            state (torch.Tensor): Current state vector of shape (n_state,)
            state_cov (torch.Tensor): State covariance matrix of shape (n_state, n_state)

        Returns:
            torch.Tensor: Innovation covariance matrix of shape (n_obs, n_obs)
        """
        H = self.compute_jacobian_state(self.decoder, state)
        Sigma = H @ state_cov @ H.T + self.R
        return Sigma

    def compute_fim(
        self,
        initial_state,
        trajectory_length,
        initial_cov,
        use_diag=False,
        use_kfac=False,
    ):
        """Compute recursive Fisher Information Matrix.

        Recursively computes FIM using the formula:
        [I(theta)]_{ij} ≈ ∑_{t=1}^{T} { (∂e/∂θ)_i^T Σ_t^{-1} (∂e/∂θ)_j
             - 1/2 * trace(Σ_t^{-1} (∂Σ_t/∂θ)_i Σ_t^{-1} (∂Σ_t/∂θ)_j) }

        Args:
            initial_state (torch.Tensor): Initial state z0 of shape (n_state,)
            trajectory_length (int): Number of time steps T
            initial_cov (torch.Tensor): Initial state covariance P0 of shape (n_state, n_state)
            use_diag (bool): If True, use diagonal approximation for FIM
            use_kfac (bool): If True, use K-FAC approximation for dynamics Jacobian.

        Returns:
            torch.Tensor: Fisher Information Matrix of shape (n_params, n_params)
        """
        d_latent = initial_state.shape[0]
        params = list(self.dynamics.parameters())
        n_params = sum(p.numel() for p in params)

        fim = torch.zeros(n_params, n_params, device=self.device)
        z = initial_state
        P = initial_cov
        dz_dtheta = torch.zeros(d_latent, n_params, device=self.device)  # Ψ = ∂z/∂θ
        # Initialize covariance sensitivity
        # ∂P/∂θ (n_state x n_state x n_params), assume P0 independent of θ.
        dP = torch.zeros(d_latent, d_latent, n_params, device=self.device)
        I = torch.eye(d_latent, device=self.device)

        for _ in range(trajectory_length):
            # Prediction step
            df_dz = self.compute_jacobian_state(self.dynamics, z)
            if use_kfac or self.use_kfac:
                B = self.compute_jacobian_params_kfac(z)  # B = ∂f/∂θ
            else:
                B = self.compute_jacobian_params(self.dynamics, z)

            # Update state and covariance
            z = z + self.dynamics(z)
            A = I + df_dz  # A = I + ∂f/∂z
            P_new = A @ P @ A.T + self.Q

            # Predictive Step
            # Propagate state sensitivity: Ψ_new = A Ψ + ∂f/∂θ.
            dz_dtheta = A @ dz_dtheta + B

            # Propagate covariance sensitivity
            dA_dtheta = torch.autograd.functional.jacobian(
                lambda z_in: self.compute_jacobian_params(self.dynamics, z_in), z
            )  # dA_dtheta = ∂^2f/∂z∂θ
            new_dP = torch.zeros_like(dP)
            for i in range(n_params):
                new_dP[:, :, i] = (
                    dA_dtheta[:, :, i] @ P @ A.T
                    + A @ dP[:, :, i] @ A.T
                    + A @ P @ (dA_dtheta[:, :, i]).T
                )
            dP = new_dP
            P = P_new

            # Measurement step
            H = self.compute_jacobian_state(self.decoder, z)
            sigma = H @ P @ H.T + self.R
            de_dtheta = -H @ dz_dtheta

            # Compute Sigma derivatives
            dH_dz = self.compute_H_derivative(z)
            # For each parameter i, compute Δ_i H = ∑_j dH_dz[j] * (dz_dtheta)[j,i].
            deltaH = torch.zeros(n_params, H.shape[0], H.shape[1], device=self.device)
            for i in range(n_params):
                for j in range(d_latent):
                    deltaH[i] += dH_dz[j] * dz_dtheta[j, i]
            # Now, for each parameter, compute Δ_i Σ = (Δ_i H) P Hᵀ + H (dP[:,:,i]) Hᵀ + H P (Δ_i H)ᵀ.
            dsigma_dtheta = torch.zeros(
                n_params, sigma.shape[0], sigma.shape[1], device=self.device
            )
            for i in range(n_params):
                dsigma_dtheta[i] = (
                    deltaH[i] @ P @ H.T + H @ dP[:, :, i] @ H.T + H @ P @ deltaH[i].T
                )

            # Update FIM
            sigma_inv = torch.inverse(sigma)
            mean_term = de_dtheta.T @ sigma_inv @ de_dtheta
            cov_term = torch.zeros(n_params, n_params, device=self.device)

            for i in range(n_params):
                for j in range(n_params):
                    trace_term = torch.trace(
                        sigma_inv @ dsigma_dtheta[i] @ sigma_inv @ dsigma_dtheta[j]
                    )
                    print(trace_term)
                    cov_term[i, j] = -0.5 * trace_term

            fim += mean_term + cov_term

        return fim

    def compute_crlb(self, fim):
        """Compute Cramér-Rao Lower Bound from FIM.

        Args:
            fim (torch.Tensor): Fisher Information Matrix of shape (n_params, n_params)

        Returns:
            torch.Tensor: CRLB matrix of shape (n_params, n_params)

        Raises:
            RuntimeError: If FIM is singular and cannot be inverted
        """
        try:
            return torch.inverse(fim)
        except RuntimeError:
            raise RuntimeError("FIM is singular and cannot be inverted")


# ----------------------------
# Main script
# ----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    n_state = 2
    Q = 0.01 * torch.eye(n_state)
    R = 0.1 * torch.eye(n_state)
    P0 = 0.1 * torch.eye(n_state)

    # Observation model parameters for h(z) = exp(Cz + b)
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    b = torch.tensor([0.0, 0.0], dtype=torch.float32)
    # Define decoder: here h(z) = exp(Cz+b)
    decoder_fn = lambda z: torch.exp(C @ z + b)

    # Define a dynamics model (MLP) matching your network architecture.
    class MLPModel(nn.Module):
        def __init__(self, dx, dh, d_out):
            super(MLPModel, self).__init__()
            self.fc1 = nn.Linear(dx, dh)
            self.tanh1 = nn.Tanh()
            self.fc2 = nn.Linear(dh, dh)
            self.tanh2 = nn.Tanh()
            self.fc3 = nn.Linear(dh, d_out)

        def forward(self, z):
            h1 = self.tanh1(self.fc1(z))
            h2 = self.tanh2(self.fc2(h1))
            out = self.fc3(h2)
            return out

    dynamics_model = MLPModel(dx=n_state, dh=16, d_out=n_state)

    # Initial state.
    z0 = torch.tensor([0.5, -0.5], dtype=torch.float32)

    T = 50  # trajectory length

    # Create a FisherMetrics object with extended covariance sensitivity propagation.
    fisher_metrics = FisherMetrics(dynamics_model, decoder_fn, Q, R, use_kfac=True)
    FIM_total = fisher_metrics.compute_fim(z0, T, P0, use_kfac=True)
    print("Recursive Fisher Information Matrix (FIM) with extended dΣ/dθ propagation:")
    print(FIM_total.detach().numpy())

    # Compute CRLB.
    try:
        CRLB = fisher_metrics.compute_crlb(FIM_total)
        print("\nCramér-Rao Lower Bound (CRLB):")
        print(CRLB.detach().numpy())
        crlb_variances = torch.diag(CRLB)
        print("\nCRLB Variances (diagonal elements):")
        print(crlb_variances.detach().numpy())
    except RuntimeError as e:
        print("FIM is singular and cannot be inverted. Details:", e)
