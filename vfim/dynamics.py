import torch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from vfim.vector_field import VectorField


class DynamicsModel:
    def __init__(self, dt):
        self.dt = dt

    def predict_next_state(self, state, params):
        raise NotImplementedError("Subclasses should implement this method.")

    def generate_trajectory(self, x0, n_steps, params=None):
        # if x0 is a matrix of shape (n_samples, n_dim), generate n_samples trajectories
        if len(x0.shape) > 1:
            trajectories = []
            for x in x0:
                trajectories.append(self.generate_trajectory(x, n_steps, params))
            return torch.stack(trajectories)
        else:
            state = x0
            trajectory = [state]
            for _ in range(n_steps):
                state = self.predict_next_state(state, params)
                trajectory.append(state)
            return torch.stack(trajectory)


class VectorFieldDynamics(DynamicsModel):
    def __init__(self, dt, vf: VectorField):
        super().__init__(dt)
        self.vf = vf

    def predict_next_state(self, state, params=None):
        return self.vf(state) * self.dt + state


class RNNDynamics(DynamicsModel):
    def __init__(self, dt, params):
        super().__init__(dt)
        self.params = params  # (W, U, b) tuple

    def predict_next_state(self, state, params=None):
        if params is None:
            params = self.params
        W, U, b = params
        return torch.tanh(torch.mm(W, state.unsqueeze(-1)).squeeze() + b)


class GPDynamics(DynamicsModel):
    def __init__(self, dt, X_train, Y_train):
        super().__init__(dt)
        # Convert torch tensors to numpy for sklearn GP
        if torch.is_tensor(X_train):
            X_train = X_train.cpu().numpy()
        if torch.is_tensor(Y_train):
            Y_train = Y_train.cpu().numpy()

        kernel = C(1.0, (0.1, 10.0)) * RBF(
            length_scale=1.0, length_scale_bounds=(0.1, 10.0)
        )
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-2).fit(
            X_train, Y_train
        )

    def predict_next_state(self, state, params=None):
        # Handle torch tensor input/output
        if torch.is_tensor(state):
            device = state.device
            state_np = state.cpu().numpy()
            pred = self.gp.predict(state_np.reshape(1, -1)).reshape(-1)
            return torch.tensor(pred, device=device)
        return torch.tensor(self.gp.predict(state.reshape(1, -1)).reshape(-1))
