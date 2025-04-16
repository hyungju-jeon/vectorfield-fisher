from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

eps = 1e-6


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

    def compute_jacobian_state(self, function, state, **kwargs):
        """Compute Jacobian of a function with respect to state.

        Args:
            state (torch.Tensor): Current state vector of shape (n_state,)

        Returns:
            torch.Tensor: Jacobian matrix of shape (n_state, n_state)
        """
        state = state.clone().detach().requires_grad_(True)
        J_f = torch.autograd.functional.jacobian(function, state, **kwargs)
        return J_f

    def compute_jacobian_params(self, function, state, indices=None, **kwargs):
        """Compute Jacobian of a function with respect to parameters.

        Args:
            state (torch.Tensor): Current state vector of shape (n_state,)

        Returns:
            torch.Tensor: Jacobian matrix of shape (n_state, n_params)
        """
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
            return self.compute_jacobian_state(self.decoder, z, create_graph=True)

        dH_dz = torch.autograd.functional.jacobian(H_fn, state)
        return dH_dz.permute(1, 0, 2)

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

    def compute_dA_dtheta_finite_diff(self, z):
        d_latent = z.shape[0]
        n_params = sum(p.numel() for p in self.dynamics.parameters())

        dA_dtheta = torch.zeros(d_latent, n_params, d_latent, device=self.device)

        for i in range(d_latent):
            z_eps = z.clone().detach()
            z_eps[i] += eps
            f_plus = self.compute_jacobian_params(self.dynamics, z_eps)
            z_eps[i] -= 2 * eps
            f_minus = self.compute_jacobian_params(self.dynamics, z_eps)
            # Central difference approximation
            dA_dtheta[..., i] = (f_plus - f_minus) / (2 * eps)
        dA_dtheta = dA_dtheta.permute(1, 0, 2)

        return dA_dtheta

    def compute_fim(
        self,
        initial_state,
        trajectory_length,
        initial_cov,
        use_diag=False,
        use_kfac=False,
    ):
        # if initial_state is tensor array, run _compute_fim_point for each initial_state
        # else, run _compute_fim_point for initial_state
        if len(initial_state.shape) > 1:
            fim_list = []
            for state in initial_state:
                fim_list.append(
                    self._compute_fim_point(
                        state,
                        trajectory_length,
                        initial_cov,
                        use_diag=use_diag,
                        use_kfac=use_kfac,
                    )
                )
            return torch.stack(fim_list)
        else:
            return self._compute_fim_point(
                initial_state,
                trajectory_length,
                initial_cov,
                use_diag=use_diag,
                use_kfac=use_kfac,
            )

    def _compute_fim_point(
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

        fim = 0
        z_filter = initial_state
        P_filter = initial_cov
        dz_dtheta = torch.zeros(d_latent, n_params, device=self.device)  # Ψ = ∂z/∂θ
        # Initialize covariance sensitivity
        # ∂P/∂θ (n_state x n_state x n_params), assume P0 independent of θ.
        dP = torch.zeros(n_params, d_latent, d_latent, device=self.device)
        I = torch.eye(d_latent, device=self.device)
        indices = torch.arange(n_params)

        for _ in range(trajectory_length):
            # Prediction step
            df_dz = self.compute_jacobian_state(self.dynamics, z_filter).detach()
            if use_kfac or self.use_kfac:
                B = self.compute_jacobian_params_kfac(z_filter)  # B = ∂f/∂θ
            else:
                B = self.compute_jacobian_params(self.dynamics, z_filter).detach()

            # Update state and covariance
            z_filter = z_filter + self.dynamics(z_filter)
            A = (I + df_dz).detach()  # A = I + ∂f/∂z
            P_new = (A @ P_filter @ A.T + self.Q).detach()

            # Predictive Step
            # Propagate state sensitivity: Ψ_new = A Ψ + ∂f/∂θ.
            dz_dtheta = (A @ dz_dtheta + B).detach()

            # Propagate covariance sensitivity
            # dA_dtheta = torch.autograd.functional.jacobian(
            #     lambda z_in: self.compute_jacobian_params(
            #         self.dynamics, z_in, create_graph=True
            #     ),
            #     z,
            # )  # dA_dtheta = ∂^2f/∂z∂θ +  (∂^2f/∂z^2) ∂z/∂θ
            # dA_dtheta = dA_dtheta.permute(1, 0, 2)

            # Central difference approximation for dA_dtheta
            dA_dtheta = self.compute_dA_dtheta_finite_diff(z_filter).detach()

            new_dP = (
                dA_dtheta @ P_filter @ A.T
                + A @ dP @ A.T
                + A @ P_filter @ dA_dtheta.permute(0, 2, 1)
            ).detach()
            dP = new_dP.detach()
            P_filter = P_new.detach()

            # Measurement step
            H = self.compute_jacobian_state(self.decoder, z_filter).detach()
            sigma = (H @ P_filter @ H.T + self.R).detach()
            de_dtheta = (-H @ dz_dtheta).detach()

            # Compute Sigma derivatives
            # dH_dz = self.compute_H_derivative(z)
            # For each parameter i, compute Δ_i H = ∑_j dH_dz[j] * (dz_dtheta)[j,i].
            # deltaH = torch.zeros(n_params, H.shape[0], H.shape[1], device=self.device)
            # for i in range(n_params):
            #     for j in range(d_latent):
            #         deltaH[i] += dH_dz[j] * dz_dtheta[j, i]
            # Now, for each parameter, compute Δ_i Σ = (Δ_i H) P Hᵀ + H (dP[:,:,i]) Hᵀ + H P (Δ_i H)ᵀ.
            # dsigma_dtheta = torch.zeros(
            #     n_params, sigma.shape[0], sigma.shape[1], device=self.device
            # ).detach()
            # for i in indices:
            # dsigma_dtheta[i] = (
            #     deltaH[i] @ P @ H.T + H @ dP[i] @ H.T + H @ P @ deltaH[i].T
            # )
            dsigma_dtheta = H @ dP @ H.T
            dsigma_dtheta = dsigma_dtheta.detach()

            # Update FIM
            if use_diag:
                sigma_inv = torch.inverse(
                    sigma + eps * torch.eye(sigma.shape[0])
                ).detach()

                # mean_term = torch.trace((de_dtheta @ de_dtheta.T) @ sigma_inv)
                mean_term = torch.sum(
                    de_dtheta[:, indices] * (sigma_inv @ de_dtheta[:, indices]), dim=0
                )
                sigma_dsigma = sigma_inv @ dsigma_dtheta
                cov_term = torch.einsum("tij,tji->t", sigma_dsigma, sigma_dsigma)
                # for i, idx in enumerate(indices):
                #     sigma_dsigma = sigma_inv @ dsigma_dtheta[idx]
                #     mean_term[i] += 0.5 * torch.trace(sigma_dsigma @ sigma_dsigma)
                fim += (mean_term + cov_term[indices]).cpu()
            else:
                sigma_inv = torch.inverse(sigma + eps * torch.eye(sigma.shape[0]))
                mean_term = de_dtheta.T @ sigma_inv @ de_dtheta
                cov_term = torch.zeros(n_params, n_params, device=self.device)

                for i in range(n_params):
                    for j in range(n_params):
                        trace_term = torch.trace(
                            sigma_inv @ dsigma_dtheta[i] @ sigma_inv @ dsigma_dtheta[j]
                        )
                        cov_term[i, j] = 0.5 * trace_term

                fim += mean_term + cov_term

        if use_diag:
            return fim
        else:
            return torch.diag((fim))

    def compute_fim_trajectory(self, z, use_diag=True):
        """Compute Fisher Information Matrix for a trajectory.

        Args:
            z (torch.Tensor): Trajectory of states of shape (T, n_state)
            use_diag (bool): If True, use diagonal approximation for FIM

        Returns:
            torch.Tensor: Fisher Information Matrix of shape (n_params, n_params)
        """
        T = z.shape[0]
        d_latent = z.shape[1]
        params = list(self.dynamics.parameters())
        n_params = sum(p.numel() for p in params)
        I = torch.eye(d_latent, device=self.device)
        fim = 0
        dz_dtheta = torch.zeros(d_latent, n_params, device=self.device)  # Ψ = ∂z/∂θ

        z_t = z  # Trajectory of states (T, n_state)
        df_dz = torch.stack(
            [self.compute_jacobian_state(self.dynamics, z_i).detach() for z_i in z_t]
        )  # (T, n_state, n_state)
        B = torch.stack(
            [self.compute_jacobian_params(self.dynamics, z_i).detach() for z_i in z_t]
        )  # (T, n_state, n_params)

        # Update state and covariance
        A = (I + df_dz).detach()  # (T, n_state, n_state)

        # Predictive Step
        dz_dtheta = torch.cumsum(
            torch.einsum("tij,tjk->tik", A, dz_dtheta.unsqueeze(0)) + B, dim=0
        ).detach()  # (T, n_state, n_params)

        # Measurement step
        H = torch.stack(
            [self.compute_jacobian_state(self.decoder, z_i).detach() for z_i in z_t]
        )  # (T, n_obs, n_state)
        sigma = self.R.unsqueeze(0).expand(T, -1, -1).detach()  # (T, n_obs, n_obs)
        de_dtheta = (
            -torch.einsum("tij,tjk->tik", H, dz_dtheta)
        ).detach()  # (T, n_obs, n_params)

        # Update FIM
        if use_diag:
            sigma_inv = torch.diag_embed(
                torch.reciprocal(torch.diagonal(sigma, dim1=1, dim2=2))
            )  # (T, n_obs, n_obs)
            mean_term = torch.sum(
                de_dtheta * torch.einsum("tij,tjk->tik", sigma_inv, de_dtheta),
                dim=(1, 2),
            )  # (T,)
            fim = torch.sum(1.0 / mean_term)  # Scalar

        return fim.cpu()


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
