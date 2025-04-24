import numpy as np
import torch
import colorednoise
from tqdm import tqdm


class SimpleICem:
    def __init__(
        self,
        horizon,
        num_iterations=40,
        num_traj=64,
        num_elites=10,
        alpha=0.1,
        noise_beta=1.0,  # colored noise parameter
        fraction_elites_reused=0.3,  # fraction of elites to keep between iterations
        fraction_prev_elites=0.2,  # fraction of elites to keep between env steps
        factor_decrease_num=1.25,  # Added parameter
        action_bounds=None,  # Added parameter
        init_std=0.5,  # Added parameter
    ):
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.num_traj = num_traj
        self.num_elites = num_elites
        self.alpha = alpha
        self.noise_beta = noise_beta
        self.fraction_elites_reused = fraction_elites_reused
        self.fraction_prev_elites = fraction_prev_elites
        self.factor_decrease_num = factor_decrease_num
        self.action_bounds = action_bounds
        self.init_std = init_std

        self.actions_from_prev_elites = None
        self.prev_elite_states = None

        self.mean = None
        self.stf = None

    def get_init_mean(self, action_dim, device):
        if self.action_bounds is not None:
            return (
                torch.zeros(self.horizon, action_dim, device=device)
                + (self.action_bounds[1] + self.action_bounds[0]) / 2.0
            )
        return torch.zeros(self.horizon, action_dim, device=device)

    def get_init_std(self, action_dim, device):
        if self.action_bounds is not None:
            return (
                torch.ones(self.horizon, action_dim, device=device)
                * (self.action_bounds[1] - self.action_bounds[0])
                / 2.0
                * self.init_std
            )
        return self.init_std * torch.ones(self.horizon, action_dim, device=device)

    def sample_action_sequences(
        self, num_traj, action_dim, action_mean, action_std, device
    ):
        # Generate colored noise
        if self.noise_beta > 0:
            # Transpose for temporal correlations in last axis
            samples = torch.tensor(
                colorednoise.powerlaw_psd_gaussian(
                    self.noise_beta, size=(num_traj, action_dim, self.horizon)
                ),
                device=device,
                dtype=torch.float32,
            ).transpose(
                1, 2
            )  # [num_traj, horizon, action_dim]
        else:
            samples = torch.randn(num_traj, self.horizon, action_dim, device=device)

        actions = samples * action_std + action_mean

        # Clip actions if bounds are provided
        if self.action_bounds is not None:
            actions = torch.clamp(actions, self.action_bounds[0], self.action_bounds[1])
        return actions

    def optimize(
        self, initial_state, dynamics_fn, cost_fn, action_dim, goal=None, device="cpu"
    ):
        """
        Optimize trajectory using iterative Cross Entropy Method
        Args:
            initial_state: initial state tensor [batch, state_dim]
            dynamics_fn: function that takes (state, action) and returns next_state
            cost_fn: function that takes (state, action) and returns cost
            action_dim: dimension of action space
            device: torch device
        Returns:
            executed_action: first action of the optimal sequence
        """
        batch_size = initial_state.shape[0]
        initial_state = initial_state.squeeze(0)

        ### Shift initialization ###
        # Shift mean time-wise
        if self.mean is None:
            self.mean = self.get_init_mean(action_dim, device)
        else:
            self.mean[:-1] = self.mean[1:].clone()
        action_std = self.get_init_std(action_dim, device)

        best_cost = float("inf")
        best_first_action = None
        current_elite_actions = None

        current_population = self.num_traj
        for iteration in range(self.num_iterations):
            # Decay population size like original
            if iteration > 0:
                current_population = max(
                    self.num_elites * 2,
                    int(current_population / self.factor_decrease_num),
                )

            # Sample new action sequences
            actions = self.sample_action_sequences(
                current_population,
                action_dim,
                self.mean.unsqueeze(0),
                action_std.unsqueeze(0),
                device,
            )

            # Add elite actions from previous environment step if available
            if iteration == 0 and self.actions_from_prev_elites is not None:
                num_prev_elites = int(
                    self.actions_from_prev_elites.shape[0] * self.fraction_prev_elites
                )
                if num_prev_elites > 0:
                    shifted_prev_actions = self.actions_from_prev_elites[
                        :num_prev_elites, 1:
                    ]
                    # Sample new actions for the last timestep
                    new_last_actions = self.sample_action_sequences(
                        num_prev_elites,
                        action_dim,
                        self.mean,
                        action_std,
                        device,
                    )
                    shifted_prev_actions = torch.cat(
                        [shifted_prev_actions, new_last_actions[:, -1:]], dim=1
                    )
                    actions = torch.cat([actions, shifted_prev_actions], dim=0)

            # Forward simulation and cost calculation same as before
            simulated_paths = torch.zeros(
                actions.shape[0],
                self.horizon + 1,
                initial_state.shape[-1],
                device=device,
            )
            simulated_paths[:, 0] = initial_state.repeat(actions.shape[0], 1)
            total_costs = torch.zeros(actions.shape[0], device=device)

            for t in range(self.horizon):
                simulated_paths[:, t + 1] = dynamics_fn(
                    simulated_paths[:, t], actions[:, t]
                )
                total_costs += cost_fn(
                    simulated_paths[:, t],
                    simulated_paths[:, t + 1],
                    goal,
                    actions[:, t],
                )

            # Add elite actions from previous iteration if available
            if current_elite_actions is not None:
                num_elites_to_keep = int(self.num_elites * self.fraction_elites_reused)
                if num_elites_to_keep > 0:
                    actions = torch.cat(
                        [actions, current_elite_actions[:num_elites_to_keep]], dim=0
                    )
                    total_costs = torch.cat(
                        [total_costs, current_elite_costs[:num_elites_to_keep]], dim=0
                    )

            # Get elite samples
            elite_idxs = torch.topk(-total_costs, self.num_elites)[1]
            current_elite_actions = actions[elite_idxs]
            current_elite_costs = total_costs[elite_idxs]

            # Update mean and std
            new_mean = current_elite_actions.mean(dim=0)
            new_std = current_elite_actions.std(dim=0)

            self.mean = (1 - self.alpha) * new_mean + self.alpha * self.mean
            action_std = (1 - self.alpha) * new_std + self.alpha * action_std

            # Update best first action if we found better solution
            min_cost_idx = elite_idxs[0]
            if total_costs[min_cost_idx] < best_cost:
                best_cost = total_costs[min_cost_idx]
                best_first_action = actions[min_cost_idx, 0]

        # Shift mean time-wise like original iCEM
        # print(action_mean)
        # Keep last timestep mean unchanged

        # Store elite actions/states for next environment step
        self.actions_from_prev_elites = current_elite_actions

        return best_first_action
