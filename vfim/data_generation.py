# %%
import random
from typing import Tuple, Optional, Literal
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import torch
import gpytorch
from gpytorch.kernels import RBFKernel, ScaleKernel

# Type aliases
ArrayType = np.ndarray
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
        """Create a 2D grid for the vector field.

        Args:
            x_range: Range of coordinates.
            n_grid: Number of grid points.
        """
        x = np.linspace(-x_range, x_range, n_grid)
        y = np.linspace(-x_range, x_range, n_grid)
        X, Y = np.meshgrid(x, y)
        xy = np.stack([X.ravel(), Y.ravel()], axis=1)

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

    def interpolate(self, x: ArrayType) -> ArrayType:
        """Interpolate vector field values at given points.

        Args:
            x: Points to interpolate at, shape (n_points, 2).

        Returns:
            Array containing interpolated vector field values.

        Raises:
            ValueError: If vector field is not generated yet.
        """
        if any(v is None for v in [self.X, self.Y, self.U, self.V]):
            raise ValueError("Vector field not generated yet.")

        x = x.reshape(1, -1)
        points = np.stack([self.X.ravel(), self.Y.ravel()], axis=1)
        values = np.stack([self.U.ravel(), self.V.ravel()], axis=1)
        return griddata(points, values, x, method="cubic")[0]

    def __call__(self, x: ArrayType) -> ArrayType:
        """Callable interface for interpolation.

        Args:
            x: Points to interpolate at.

        Returns:
            Array containing interpolated vector field values.
        """
        return self.interpolate(x)

    def streamplot(self, **kwargs) -> None:
        """Create a streamplot visualization of the vector field.

        Args:
            **kwargs: Additional arguments passed to plt.streamplot.
                fig: Optional matplotlib figure instance.

        Raises:
            ValueError: If vector field is not generated yet.
        """
        if any(v is None for v in [self.X, self.Y, self.U, self.V]):
            raise ValueError("Vector field not generated yet.")

        fig = kwargs.pop("fig", plt.figure(figsize=(10, 10)))
        ax = fig.add_subplot(111)

        # Create streamplot
        ax.streamplot(
            self.X,
            self.Y,
            self.U,
            self.V,
            density=2,
            linewidth=0.5,
            color="b",
            **kwargs,
        )

        # Configure plot
        ax.set_title("Streamline Plot of Vector Field")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.axis("off")
        ax.axis("equal")
        # plt.show()


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
    xy: ArrayType, norm: float = 0.05, random_seed: int = 49, length_scale: float = 0.5
) -> Tuple[ArrayType, ArrayType]:
    """Generate a multi-attractor vector field with random perturbations using GPyTorch.

    Args:
        xy: Grid points coordinates.
        norm: Normalization factor for perturbations. Defaults to 0.05.
        random_seed: Random seed for reproducibility. Defaults to 49.

    Returns:
        tuple: Contains:
            - U (ArrayType): X components of the vector field
            - V (ArrayType): Y components of the vector field
    """
    # Set random seeds
    torch.manual_seed(random_seed)
    rng = np.random.default_rng(random_seed)

    # Convert to PyTorch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xy_torch = torch.tensor(xy, dtype=torch.float32, device=device)

    try:
        # Initialize kernel with more stable parameters
        base_kernel = RBFKernel(ard_num_dims=2)
        base_kernel.lengthscale = length_scale
        kernel = ScaleKernel(base_kernel)
        kernel.outputscale = 0.5  # Reduced scale for stability

        # Compute kernel matrix efficiently using GPyTorch
        with torch.no_grad(), gpytorch.settings.fast_computations(True):
            kernel.eval()
            K = kernel(xy_torch).evaluate()

        # Use eigendecomposition directly
        eigenvalues, eigenvectors = torch.linalg.eigh(K)
        # Ensure eigenvalues are positive and well-conditioned
        eigenvalues = eigenvalues.clamp(min=1e-4)
        eps = torch.randn(2, K.shape[0], device=device)
        samples = torch.matmul(eps * torch.sqrt(eigenvalues), eigenvectors.T)

    except Exception as e:
        print(f"Falling back to simple random field due to error: {str(e)}")
        # Fallback to simple random field
        grid_size = int(np.sqrt(xy.shape[0]))
        samples = torch.randn(2, xy.shape[0], device=device)
        samples = torch.nn.functional.normalize(samples, dim=1) * norm

    # Reshape and normalize
    grid_size = int(np.sqrt(xy.shape[0]))
    U = samples[0].reshape(grid_size, grid_size)
    V = samples[1].reshape(grid_size, grid_size)

    # Add small epsilon to avoid division by zero
    magnitude = torch.hypot(U, V).clamp(min=1e-8)
    U = norm * U / magnitude
    V = norm * V / magnitude

    return U.cpu().numpy(), V.cpu().numpy()


if __name__ == "__main__":
    # Example usage with smaller grid size
    vf = VectorField(x_range=2.5, n_grid=50)
    vf.generate_vector_field(random_seed=10)
    vf.streamplot()
