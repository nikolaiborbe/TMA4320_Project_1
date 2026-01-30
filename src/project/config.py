"""Configuration loader for PINN project."""

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import yaml
import numpy as np


@dataclass
class Config:
    """Configuration for the PINN simulation."""

    data: dict

    # Domain
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    t_min: float
    t_max: float

    # Physics
    alpha: float
    k: float
    h: float
    T_outside: float

    # Source
    source_locations: jnp.ndarray
    source_sizes: jnp.ndarray
    source_strength: float

    # Grid
    nx: int
    ny: int
    nt: int

    # Sensors
    sensor_rate: float
    sensor_noise: float
    sensor_locations: jnp.ndarray

    # Training
    layer_sizes: list
    learning_rate: float
    num_epochs: int
    seed: int
    lambda_physics: float
    lambda_ic: float
    lambda_bc: float
    lambda_data: float
    num_collocation: int
    num_ic: int
    num_bc: int

    def is_source(self, x, y):
        """Check if point(s) are inside any heat source."""
        # source_locations: (S, 2), source_sizes: (S,)
        cx = self.source_locations[:, 0]  # (S,)
        cy = self.source_locations[:, 1]  # (S,)
        sizes = self.source_sizes  # (S,)

        # Broadcast x, y against source centers
        # x, y can be scalars or arrays of any shape
        dx = jnp.abs(x - cx[:, None, None])  # (S, ...) broadcasts with x
        dy = jnp.abs(y - cy[:, None, None])  # (S, ...) broadcasts with y

        inside = (dx <= sizes[:, None, None]) & (dy <= sizes[:, None, None])
        return jnp.any(inside, axis=0)  # same shape as x, y

    def heat_source(self, x, y, t):
        """Heat source term at point (x, y, t)."""
        return jnp.where(self.is_source(x, y), self.source_strength, 0.0)

    def set_sensor_pos(self, sensor_index: int, x_pos: float, y_pos: float):
        """Set the new position of one of the sensors"""
        self.sensor_locations = self.sensor_locations.at[sensor_index].set(
            jnp.array([x_pos, y_pos])
        )

    def solve_heat_equation(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Solve the 2D heat equation using implicit Euler.

        Returns:
            x: x-coordinates (nx,)
            y: y-coordinates (ny,)
            t: time points (nt,)
            T: temperature solution (nt, nx, ny)
        """
        # Create grids
        x = np.linspace(self.x_min, self.x_max, self.nx)
        y = np.linspace(self.y_min, self.y_max, self.ny)
        t = np.linspace(self.t_min, self.t_max, self.nt)

        dx, dy = x[1] - x[0], y[1] - y[0]
        dt = t[1] - t[0]

        X, Y = np.meshgrid(x, y, indexing="ij")

        #######################################################################
        # Oppgave 3.2: Start
        #######################################################################

        # Placeholder initialization — replace this with your implementation
        T: np.ndarray = np.zeros((self.nt, self.nx, self.ny))
        T_0 = np.zeros((self.nx, self.ny))

        first = True
        for n in range(len(t) - 1):
            if first:
                T_0.fill(self.T_outside)
            else:
                T_0 = T[n]

            b_0 = self._build_rhs(T_0, X, Y, dx, dy, dt, t[n + 1])
            A = self._build_matrix(dx, dy, dt)

            T_1 = np.linalg.solve(A, b_0)
            T[n + 1] = T_1.reshape(self.nx, self.ny)

            first = False

        #######################################################################
        # Oppgave 3.2: Slutt
        #######################################################################
        self.data = dict(x=x, y=y, t=t, T=T)
        return x, y, t, T

    def _build_matrix(self, dx: float, dy: float, dt: float) -> np.ndarray:
        """Build the implicit Euler system matrix."""
        n = self.nx * self.ny
        A = np.zeros((n, n))

        rx = self.alpha * dt / dx**2
        ry = self.alpha * dt / dy**2

        def idx(i, j):
            return i * self.ny + j

        I, J = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="ij")

        # Boundary masks
        left = I == 0
        right = I == self.nx - 1
        bottom = J == 0
        top = J == self.ny - 1

        # Diagonal entries
        diag = np.full((self.nx, self.ny), 1 + 2 * rx + 2 * ry)
        diag[left | right] -= rx
        diag[bottom | top] -= ry
        diag[left | right] += rx * self.h * dx / self.k
        diag[bottom | top] += ry * self.h * dy / self.k

        p = idx(I, J)
        A[p, p] = diag

        # Off-diagonals
        mask = ~left
        A[idx(I[mask], J[mask]), idx(I[mask] - 1, J[mask])] = -rx

        mask = ~right
        A[idx(I[mask], J[mask]), idx(I[mask] + 1, J[mask])] = -rx

        mask = ~bottom
        A[idx(I[mask], J[mask]), idx(I[mask], J[mask] - 1)] = -ry

        mask = ~top
        A[idx(I[mask], J[mask]), idx(I[mask], J[mask] + 1)] = -ry

        return A

    def _build_rhs(
        self,
        T_curr,
        X: np.ndarray,
        Y: np.ndarray,
        dx: float,
        dy: float,
        dt: float,
        t_next: int,
    ) -> np.ndarray:
        """Build right-hand side for implicit system.

        Args:
            T_curr: np.ndarray, shape(50, 25)

        Returns:
            np.ndarray, shape(1250)

        """
        rhs = T_curr.copy()

        # Heat source
        q = np.array(self.heat_source(X, Y, t_next))
        rhs += dt * q

        # Robin BC contributions
        rx = self.alpha * dt / dx**2
        ry = self.alpha * dt / dy**2
        bc_term = self.T_outside

        rhs[0, :] += rx * (self.h * dx / self.k) * bc_term
        rhs[-1, :] += rx * (self.h * dx / self.k) * bc_term
        rhs[:, 0] += ry * (self.h * dy / self.k) * bc_term
        rhs[:, -1] += ry * (self.h * dy / self.k) * bc_term

        return rhs.flatten()


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load configuration from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    return Config(
        # Data
        data = {},
        # Domain
        x_min=data["domain"]["x_min"],
        x_max=data["domain"]["x_max"],
        y_min=data["domain"]["y_min"],
        y_max=data["domain"]["y_max"],
        t_min=data["time"]["t_min"],
        t_max=data["time"]["t_max"],
        # Physics
        alpha=data["physics"]["alpha"],
        k=data["physics"]["k"],
        h=data["physics"]["h"],
        T_outside=data["physics"]["T_outside"],
        # Source
        source_locations=jnp.asarray(
            data["source"]["locations"],
        ),
        source_sizes=jnp.asarray(
            data["source"]["sizes"],
        ),
        source_strength=data["source"]["strength"],
        # Grid
        nx=data["grid"]["nx"],
        ny=data["grid"]["ny"],
        nt=data["grid"]["nt"],
        # Sensors
        sensor_rate=data["sensors"]["measure_rate"],
        sensor_noise=data["sensors"]["noise_std"],
        sensor_locations=jnp.asarray(
            data["sensors"]["locations"],
        ),  # shape (n_sensors, 2)
        # Training
        layer_sizes=data["training"]["layer_sizes"],
        learning_rate=data["training"]["learning_rate"],
        num_epochs=data["training"]["num_epochs"],
        seed=data["training"]["seed"],
        lambda_physics=data["training"]["lambda_physics"],
        lambda_ic=data["training"]["lambda_ic"],
        lambda_bc=data["training"]["lambda_bc"],
        lambda_data=data["training"]["lambda_data"],
        num_collocation=data["training"]["num_collocation"],
        num_ic=data["training"]["num_ic"],
        num_bc=data["training"]["num_bc"],
    )
