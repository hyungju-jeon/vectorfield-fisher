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
    ):
        self.dynamics = dynamics
        self.decoder = decoder
        self.Q = process_noise
        self.R = measurement_noise
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

        J_theta = []
        for i in range(out_dim):
            grads = torch.autograd.grad(
                f_val[i], params, retain_graph=True, allow_unused=True
            )
            grad_list = []
            for p, g in zip(params, grads):
                grad_list.append(
                    torch.zeros_like(p).view(-1) if g is None else g.view(-1)
                )
            J_theta.append(torch.cat(grad_list).unsqueeze(0))

        return torch.cat(J_theta, dim=0)

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

    def compute_sigma_derivative(self, state, state_cov):
        """Compute derivative of innovation covariance with respect to state.

        Args:
            state (torch.Tensor): Current state vector of shape (n_state,)
            state_cov (torch.Tensor): State covariance matrix of shape (n_state, n_state)

        Returns:
            torch.Tensor: Derivative tensor of shape (n_state, n_obs, n_obs)
        """

        def sigma_fn(z):
            sigma = self.compute_sigma(z, state_cov)
            return sigma

        return torch.autograd.functional.jacobian(
            lambda z: self.compute_sigma(z, state_cov), state
        )
        # return torch.autograd.functional.jacobian(sigma_fn, state)

    def compute_fim(
        self, initial_state, trajectory_length, initial_cov, compute_diag=False
    ):
        """Compute recursive Fisher Information Matrix.

        Recursively computes FIM using the formula:
        [I(theta)]_{ij} ≈ ∑_{t=1}^{T} { (∂e/∂θ)_i^T Σ_t^{-1} (∂e/∂θ)_j
             - 1/2 * trace(Σ_t^{-1} (∂Σ_t/∂θ)_i Σ_t^{-1} (∂Σ_t/∂θ)_j) }

        Args:
            initial_state (torch.Tensor): Initial state z0 of shape (n_state,)
            trajectory_length (int): Number of time steps T
            initial_cov (torch.Tensor): Initial state covariance P0 of shape (n_state, n_state)

        Returns:
            torch.Tensor: Fisher Information Matrix of shape (n_params, n_params)
        """
        d_latent = initial_state.shape[0]
        params = list(self.dynamics.parameters())
        num_params = sum(p.numel() for p in params)

        fim = torch.zeros(num_params, num_params, device=self.device)
        z = initial_state
        P = initial_cov
        dz_dtheta = torch.zeros(d_latent, num_params, device=self.device)
        I = torch.eye(d_latent, device=self.device)

        for _ in range(trajectory_length):
            # Prediction step
            df_dz = self.compute_jacobian_state(self.dynamics, z)
            # B = df_dtheta
            B = self.compute_jacobian_params(self.dynamics, z)

            # Update state and covariance
            z = z + self.dynamics(z)
            A = I + df_dz
            P = A @ P @ A.T + self.Q

            # Predictive Step
            dz_dtheta = A @ dz_dtheta + B

            # Measurement step
            H = self.compute_jacobian_state(self.decoder, z)
            sigma = H @ P @ H.T + self.R
            de_dtheta = -H @ dz_dtheta

            mask_idx = 66250

            # Compute Sigma derivatives
            dsigma_dz = self.compute_sigma_derivative(z, P)
            dsigma_dtheta = torch.stack(
                [
                    sum(dsigma_dz[..., j] * dz_dtheta[j, i] for j in range(d_latent))
                    for i in range(num_params)
                ]
            )

            # Update FIM
            sigma_inv = torch.inverse(sigma)
            mean_term = de_dtheta.T @ sigma_inv @ de_dtheta

            cov_term = torch.zeros_like(fim)
            for i in range(66500, num_params):
                for j in range(66500, num_params):
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
    # Set seeds for reproducibility.
    torch.manual_seed(0)
    np.random.seed(0)

    d = 2  # state dimension
    Q = 0.01 * torch.eye(d)  # process noise covariance
    R = 0.1 * torch.eye(d)  # observation noise covariance
    P0 = 0.1 * torch.eye(d)  # initial state covariance

    # Observation model parameters: y = exp(C z + b) + noise.
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([0.0, 0.0])

    # Initialize MLP (dynamics model); in practice, this would be pre-trained.
    mlp = MLP(input_dim=d, hidden_dim=16, output_dim=d)

    # Initial state z0.
    z0 = torch.tensor([0.5, -0.5])

    T = 50  # trajectory length

    # Compute the recursive Fisher Information Matrix.
    fisher_metrics = FisherMetrics(mlp, C, b, Q, R)
    FIM_total = fisher_metrics.compute_fim(z0, T, P0)
    print("Recursive Fisher Information Matrix (FIM):")
    print(FIM_total.detach().numpy())

    # ---- Compute CRLB ----
    # The Cramér-Rao Lower Bound is given by the inverse of the FIM.
    try:
        CRLB = fisher_metrics.compute_crlb(FIM_total)
        print("\nCramér-Rao Lower Bound (CRLB):")
        print(CRLB.detach().numpy())

        # Optionally, print the variance lower bounds for each parameter (diagonal of CRLB).
        crlb_variances = torch.diag(CRLB)
        print("\nCRLB Variances (diagonal elements):")
        print(crlb_variances.detach().numpy())
    except RuntimeError as e:
        print("FIM is singular and cannot be inverted. Details:", e)
