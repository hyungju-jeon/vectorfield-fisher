import torch
from torch.utils.data import DataLoader
from vfim.agent.information import FisherMetrics


class ActiveLearning:
    def __init__(self, vae, device, trial_params):
        self.vae = vae
        self.device = device
        self.trial_params = trial_params
        self.u_list = self._create_control_inputs(trial_params["u_strength"])

    def _create_control_inputs(self, u_strength):
        u_x = torch.linspace(-u_strength, u_strength, 3, device=self.device)
        u_y = torch.linspace(-u_strength, u_strength, 3, device=self.device)
        U_X, U_Y = torch.meshgrid(u_x, u_y, indexing="xy")
        return torch.stack([U_X.flatten(), U_Y.flatten()], dim=1)

    def run_naive_trial(self, initial_state, f_star, C, R, Q, n_neurons):
        x0 = initial_state.unsqueeze(0)
        x_trajectory = x0

        for _ in range(self.trial_params["num_trials"]):
            x_trial = f_star.generate_trajectory(
                x0, self.trial_params["trial_length"], R
            )
            y_trial = (x_trial @ C.T) + torch.randn(
                1, self.trial_params["trial_length"], n_neurons
            ) * torch.sqrt(Q)

            dataloader = DataLoader(y_trial, batch_size=1)
            self.vae.train_model(
                dataloader,
                self.trial_params["lr_refine"],
                self.trial_params["weight_decay"],
                self.trial_params["refine_epoch"],
            )

            x_trajectory = torch.cat([x_trajectory, x_trial], dim=1)
            x0 = x_trial[:, -1, :].unsqueeze(0)

        return x_trajectory

    def run_random_trial(self, initial_state, f_star, C, R, Q, n_neurons):
        x0 = initial_state.unsqueeze(0)
        x_trajectory = (
            x0 + torch.rand(1, x0.shape[-1]) * self.trial_params["u_strength"]
        )

        for _ in range(self.trial_params["num_trials"]):
            x_trial = f_star.generate_trajectory(
                x0, self.trial_params["trial_length"], R
            )
            y_trial = (x_trial @ C.T) + torch.randn(
                1, self.trial_params["trial_length"], n_neurons
            ) * torch.sqrt(Q)

            dataloader = DataLoader(y_trial, batch_size=1)
            self.vae.train_model(
                dataloader,
                self.trial_params["lr_refine"],
                self.trial_params["weight_decay"],
                self.trial_params["refine_epoch"],
            )

            x_trajectory = torch.cat([x_trajectory, x_trial], dim=1)
            x0 = (
                x_trial[:, -1, :].unsqueeze(0)
                + torch.rand(1, x0.shape[-1]) * self.trial_params["u_strength"]
            )

        return x_trajectory

    def run_fisher_trial(
        self, initial_state, f_star, C, R, Q, n_neurons, d_latent, decoder
    ):
        x0 = initial_state
        x0 = torch.cat([x0 + u for u in self.u_list], dim=0)
        x_trajectory = None

        fisher = FisherMetrics(
            dynamics=self.vae.dynamics,
            decoder=decoder,
            process_noise=R * torch.eye(d_latent),
            measurement_noise=Q * torch.eye(n_neurons),
        )
        initial_cov = 0 * torch.eye(d_latent, device=self.device)

        for _ in range(self.trial_params["num_trials"]):
            # Compute FIM and select best action
            fims = fisher.compute_fim(
                x0, self.trial_params["fisher_length"], initial_cov, use_diag=False
            )
            fim = torch.tensor([torch.sum(f) for f in fims])
            max_idx = torch.argmax(fim)
            x0_selected = x0[max_idx].unsqueeze(0)

            # Generate trajectory and observations
            x_trial = f_star.generate_trajectory(
                x0_selected.unsqueeze(0), self.trial_params["trial_length"], R
            )
            y_trial = (x_trial @ C.T) + torch.randn(
                1, self.trial_params["trial_length"], n_neurons
            ) * torch.sqrt(Q)

            # Train model
            dataloader = DataLoader(y_trial, batch_size=1)
            self.vae.train_model(
                dataloader,
                self.trial_params["lr_refine"],
                self.trial_params["weight_decay"],
                self.trial_params["refine_epoch"],
            )

            # Update trajectories
            if x_trajectory is None:
                x_trajectory = x0_selected.unsqueeze(0)
            x_trajectory = torch.cat([x_trajectory, x_trial], dim=1)

            # Prepare next iteration
            x0 = x_trial[:, -1, :]
            x0 = torch.cat([x0 + u for u in self.u_list], dim=0)

        return x_trajectory
