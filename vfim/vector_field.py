from typing import Tuple, Optional, Literal
import numpy as np  # kept only for matplotlib compatibility
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel

# Type aliases
ArrayType = torch.Tensor
ModelType = Literal["multi", "ring", "limitcycle", "single"]


class VectorField:
    """A class for generating and manipulating 2D vector fields.

    Args:
        model: Type of vector field model. Defaults to "multi".
        x_range: Range of coordinates (-x_range to x_range). Defaults to 2.
        n_grid: Number of grid points in each dimension. Defaults to 40.

    Attributes:
        x_range: Range of x and y coordinates (-x_range to x_range).
        n_grid: Number of grid points in each dimension.
        model: Type of vector field model to generate.
        X: X coordinates of the grid points.
        Y: Y coordinates of the grid points.
        xy: Combined XY coordinates.
        U: X components of the vector field.
        V: Y components of the vector field.
    """

    def __init__(
        self,
        model: ModelType = "multi",
        x_range: float = 2,
        n_grid: int = 40,
    ):
        """
        Initialize the VectorField instance.

        Args:
            model: Type of vector field model.
            x_range: Range of coordinates (-x_range to x_range).
            n_grid: Number of grid points in each dimension.
        """
        self.x_range = x_range
        self.n_grid = n_grid
        self.model = model
        self.X: Optional[ArrayType] = None
        self.Y: Optional[ArrayType] = None
        self.xy: Optional[ArrayType] = None
        self.U: Optional[ArrayType] = None
        self.V: Optional[ArrayType] = None

        self.create_grid(self.x_range, self.n_grid)

    def create_grid(self, x_range: float, n_grid: int) -> None:
        """Create a 2D grid for the vector field using PyTorch."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = torch.linspace(-x_range, x_range, n_grid, device=device)
        y = torch.linspace(-x_range, x_range, n_grid, device=device)
        X, Y = torch.meshgrid(x, y, indexing="xy")
        xy = torch.stack([X.flatten(), Y.flatten()], dim=1)

        self.X, self.Y, self.xy = X, Y, xy

    def generate_vector_field(self, **kwargs) -> None:
        """Generate vector field based on the selected model.

        Args:
            **kwargs: Additional parameters passed to the vector field generator.

        Raises:
            ValueError: If grid is not initialized.
        """
        if self.xy is None:
            raise ValueError("Grid not initialized. Call create_grid first.")

        self.U, self.V = generate_random_vector_field(self.model, self.xy, **kwargs)

    @torch.no_grad()
    def interpolate(self, x: ArrayType) -> ArrayType:
        """Interpolate vector field values at given points using PyTorch RBF interpolation.

        Args:
            x: Points to interpolate at, shape (trials, times, dimension).

        Returns:
            Array containing interpolated vector field values.

        Raises:
            ValueError: If vector field is not generated yet.
        """
        if any(v is None for v in [self.X, self.Y, self.U, self.V]):
            raise ValueError("Vector field not generated yet.")

        trials, times, _ = x.shape
        x = x.reshape(trials * times, 2)
        points = torch.stack([self.X.flatten(), self.Y.flatten()], dim=1)
        values = torch.stack([self.U.flatten(), self.V.flatten()], dim=1)

        # RBF interpolation
        dist = torch.cdist(x, points)
        weights = torch.exp(-dist / 0.1)  # RBF kernel with length scale 0.1
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        interpolated_values = torch.mm(weights, values)
        return interpolated_values.reshape(trials, times, 2)

    def __call__(self, x: ArrayType) -> ArrayType:
        """Callable interface for interpolation.

        Args:
            x: Points to interpolate at.

        Returns:
            Array containing interpolated vector field values.
        """
        return self.interpolate(x)

    def sample_forward(self, x, k, var, return_trajectory=True):
        x_samples, mus = [x], [x]
        for i in range(k):
            mus.append(self(mus[i]) + mus[i])
            x_samples.append(
                mus[i] + torch.sqrt(var) * torch.randn_like(mus[i], device=x.device)
            )
        if return_trajectory:
            return torch.cat(x_samples, dim=-2)[..., 1:, :], torch.cat(mus, dim=-2)
        else:
            return x_samples[-1], mus[-1]


def generate_random_vector_field(
    model: ModelType = "multi", xy: Optional[ArrayType] = None, **kwargs
) -> Tuple[ArrayType, ArrayType]:
    """Factory function for generating different types of vector fields.

    Args:
        model: Type of vector field to generate.
        xy: Grid points coordinates.
        **kwargs: Additional model-specific parameters.

    Returns:
        tuple: Contains:
            - U (ArrayType): X components of the vector field
            - V (ArrayType): Y components of the vector field

    Raises:
        ValueError: If xy coordinates are not provided or model is not supported.
    """
    if xy is None:
        raise ValueError("xy coordinates must be provided")

    if model == "multi":
        return multi_attractor(xy, **kwargs)
    else:
        raise ValueError(f"Unsupported model: {model}")


def multi_attractor(
    xy: ArrayType,
    norm: float = 0.1,
    random_seed: int = 49,
    length_scale: float = 0.5,
    **kwargs,
) -> Tuple[ArrayType, ArrayType]:
    """Generate a multi-attractor vector field using PyTorch.

    Args:
        xy: Grid points coordinates.
        norm: Normalization factor for perturbations. Defaults to 0.05.
        random_seed: Random seed for reproducibility. Defaults to 49.

    Returns:
        tuple: Contains:
            - U (ArrayType): X components of the vector field
            - V (ArrayType): Y components of the vector field
    """
    w_attractor = kwargs.get("w_attractor", 0)
    torch.manual_seed(random_seed)
    device = xy.device

    try:
        base_kernel = RBFKernel(ard_num_dims=2)
        base_kernel.lengthscale = length_scale
        kernel = ScaleKernel(base_kernel)
        kernel.outputscale = 0.5

        with torch.no_grad(), gpytorch.settings.fast_computations(True):
            kernel.eval()
            K = kernel(xy).evaluate()

        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        eigenvalues = eigenvalues.clamp(min=1e-4)
        eps = torch.randn(2, K.shape[0], device=device)
        samples = torch.matmul(eps * torch.sqrt(eigenvalues), eigenvectors.T)

    except Exception as e:
        print(f"Falling back to simple random field due to error: {str(e)}")
        grid_size = int(torch.sqrt(torch.tensor(xy.shape[0])))
        samples = torch.randn(2, xy.shape[0], device=device)

    # Reshape and normalize
    grid_size = int(torch.sqrt(torch.tensor(xy.shape[0])))
    U = samples[0].reshape(grid_size, grid_size)
    V = samples[1].reshape(grid_size, grid_size)

    magnitude = torch.hypot(U, V).clamp(min=1e-8)
    U = norm * U / magnitude
    V = norm * V / magnitude

    if w_attractor > 0:
        U_attract = -xy[:, 0] * torch.sqrt(torch.sum(xy**2, 1)) * w_attractor
        V_attract = -xy[:, 1] * torch.sqrt(torch.sum(xy**2, 1)) * w_attractor
        U += U_attract.reshape(grid_size, grid_size)
        V += V_attract.reshape(grid_size, grid_size)

    return U, V  # Return torch tensors directly


if __name__ == "__main__":
    # Example usage with smaller grid size
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=10)
    vf.streamplot()
