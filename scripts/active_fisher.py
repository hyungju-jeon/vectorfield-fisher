# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================
# 1. Dynamics Network (MLP)
# ============================
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: shape (input_dim,)
        z = F.relu(self.fc1(z))
        z = self.fc2(z)
        return z


# ============================
# 2. Ensemble of Dynamics Networks
# ============================
class DynamicsEnsemble:
    def __init__(self, num_models: int, input_dim=2, hidden_dim=16, output_dim=2):
        self.models = [
            MLP(input_dim, hidden_dim, output_dim) for _ in range(num_models)
        ]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model in self.models:
            model.to(self.device)
            model.eval()  # set in evaluation mode

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        For a given state (shape: (input_dim,)),
        returns a tensor of shape (num_models, output_dim) with predictions.
        """
        state = state.to(self.device)
        preds = []
        with torch.no_grad():
            for model in self.models:
                preds.append(model(state))
        return torch.stack(preds)  # shape: (num_models, output_dim)


# ============================
# 3. Epistemic Uncertainty Metric
# ============================
def compute_epistemic_uncertainty(
    state: torch.Tensor, ensemble: DynamicsEnsemble
) -> torch.Tensor:
    """
    Computes the scalar epistemic uncertainty at the given state as the variance
    (mean squared deviation) of the ensemble predictions.
    """
    preds = ensemble.predict(state)  # shape: (N, output_dim)
    mean_pred = torch.mean(preds, dim=0)
    # Mean squared error (averaged over dimensions) as the uncertainty
    variance = torch.mean((preds - mean_pred) ** 2)
    return variance


# ============================
# 4. Observation Model and SNR Computation
# ============================
def compute_predicted_observation(
    state: torch.Tensor, C: torch.Tensor, b: torch.Tensor
) -> torch.Tensor:
    """
    Computes the predicted observation using a log-linear model:
      h(z) = exp(C z + b)
    """
    return torch.exp(C @ state + b)


def compute_snr(
    state: torch.Tensor, C: torch.Tensor, b: torch.Tensor, R: torch.Tensor
) -> torch.Tensor:
    """
    Computes an SNR metric at the given state.
    Here, SNR = ||h(z)|| / sqrt(trace(R)).
    """
    h = compute_predicted_observation(state, C, b)
    signal_strength = torch.norm(h)
    noise_level = torch.sqrt(torch.trace(R))
    return signal_strength / noise_level


# ============================
# 5. Candidate Target Selection
# ============================
def select_target_state(
    candidate_states: np.ndarray, ensemble: DynamicsEnsemble
) -> (torch.Tensor, float):
    """
    Given an array of candidate states (shape: (num_candidates, state_dim)),
    select the state with the highest epistemic uncertainty (information gain).
    Returns the target state and its uncertainty value.
    """
    best_value = -np.inf
    best_state = None
    for s in candidate_states:
        state_tensor = torch.tensor(s, dtype=torch.float32)
        ep_unc = compute_epistemic_uncertainty(state_tensor, ensemble).item()
        if ep_unc > best_value:
            best_value = ep_unc
            best_state = state_tensor
    return best_state, best_value


# ============================
# 6. Path Planning from Current State to Target State
# ============================
def plan_path(
    current_state: torch.Tensor, target_state: torch.Tensor, num_steps: int = 10
) -> torch.Tensor:
    """
    Compute a simple straight-line path from the current_state to the target_state.
    Returns a tensor of shape (num_steps, state_dim) representing the path.
    """
    current = current_state.unsqueeze(0)
    target = target_state.unsqueeze(0)
    # Linear interpolation between current and target
    path = (
        torch.linspace(0, 1, steps=num_steps).unsqueeze(1) * (target - current)
        + current
    )
    return path.squeeze(1)  # shape: (num_steps, state_dim)


# ============================
# 7. Evaluate SNR Along a Path
# ============================
def evaluate_path_snr(
    path: torch.Tensor, C: torch.Tensor, b: torch.Tensor, R: torch.Tensor
) -> float:
    """
    Given a path (tensor of shape (num_steps, state_dim)), compute the average SNR along the path.
    """
    snr_values = []
    for state in path:
        snr_values.append(compute_snr(state, C, b, R).item())
    return np.mean(snr_values)


# ============================
# 8. Main Active Learning Function
# ============================
def active_learning_step(
    current_state: torch.Tensor,
    candidate_states: np.ndarray,
    ensemble: DynamicsEnsemble,
    C: torch.Tensor,
    b: torch.Tensor,
    R: torch.Tensor,
    num_path_steps: int = 10,
):
    """
    1. Choose a target state that maximizes epistemic uncertainty.
    2. Plan a straight-line path from the current state to that target.
    3. Evaluate the average SNR along that path.
    Returns:
      - target state
      - path (tensor of shape (num_path_steps, state_dim))
      - average SNR along the path
      - epistemic uncertainty at the target.
    """
    target_state, target_ep_unc = select_target_state(candidate_states, ensemble)
    path = plan_path(current_state, target_state, num_steps=num_path_steps)
    avg_snr = evaluate_path_snr(path, C, b, R)
    return target_state, path, avg_snr, target_ep_unc


# ============================
# 9. Example Usage in a Model-Based RL Setup
# ============================
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # Define state dimension and observation model parameters.
    d = 2
    C = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    b = torch.tensor([0.0, 0.0], dtype=torch.float32)
    R = 0.1 * torch.eye(d)  # observation noise covariance

    # Create an ensemble of dynamics models.
    ensemble_size = 5
    ensemble = DynamicsEnsemble(
        num_models=ensemble_size, input_dim=d, hidden_dim=16, output_dim=d
    )

    # Current state (e.g., from your current latent state estimate)
    current_state = torch.tensor([0.0, 0.0], dtype=torch.float32)

    # Generate candidate states (for example, a grid or random samples)
    num_candidates = 100
    candidate_states = np.random.uniform(low=-2, high=2, size=(num_candidates, d))

    # Active learning step: choose target and plan path.
    target_state, path, avg_snr, target_ep_unc = active_learning_step(
        current_state, candidate_states, ensemble, C, b, R, num_path_steps=10
    )

    print("Current State:", current_state.cpu().numpy())
    print("Selected Target State (max InfoGain):", target_state.cpu().numpy())
    print("Epistemic Uncertainty at Target:", target_ep_unc)
    print("Planned Path (states along the trajectory):")
    print(path.cpu().numpy())
    print("Average SNR along the path:", avg_snr)

    # In a model-based RL loop, you would now use the planned path to guide your agent
    # toward that target region, collect new observations, update your model,
    # and then iterate this active learning procedure.
