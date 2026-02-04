import numpy as np
from project import (Config)
from viz import create_animation, plot_snapshots
from project import (
    load_config,
    solve_heat_equation,
)

def _build_matrix(cfg: Config, dx: float, dy: float, dt: float) -> np.ndarray:
    """Build the implicit Euler system matrix."""
    n = cfg.nx * cfg.ny
    A = np.zeros((n, n))

    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2

    def idx(i, j):
        return i * cfg.ny + j

    I, J = np.meshgrid(np.arange(cfg.nx), np.arange(cfg.ny), indexing="ij")

    # Boundary masks
    left = I == 0
    right = I == cfg.nx - 1
    bottom = J == 0
    top = J == cfg.ny - 1

    # Diagonal entries
    diag = np.full((cfg.nx, cfg.ny), 1 + 2 * rx + 2 * ry)
    diag[left | right] -= rx
    diag[bottom | top] -= ry
    diag[left | right] += rx * cfg.h * dx / cfg.k
    diag[bottom | top] += ry * cfg.h * dy / cfg.k

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


def _build_rhs_save_data(
    cfg: Config,
    T_curr,
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    t_next: int,
) -> tuple:
    """Build right-hand side for implicit system.

    Args:
        T_curr: np.ndarray, shape(50, 25)

    Returns:
        np.ndarray, shape(1250)

    """
    rhs = T_curr.copy()

    # Heat source
    q = np.array(cfg.heat_source(X, Y, t_next))

    # Grid points, needed for addded code
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)

    # Find the average room temperature, excluding the heat source
    T_outside_heatsource = list()
    for i in range(cfg.nx):
        row = list()
        for j in range(cfg.ny):
            if q[i][j] > 0:                    # q is zero where there is no heat source
                pass
            else:
                row.append(rhs[i][j])
        T_outside_heatsource.extend(row)
    T_room_mean = np.mean(np.array(T_outside_heatsource))

    # Find the mean sensor reading
    T_readings = list()
    for loc in cfg.sensor_locations:
        if np.any(np.all(cfg.source_locations==loc, axis = 1)):          # Exclude temperature reading at source locations
            pass
        else:
            sens_x, sens_y  = loc[0], loc[1]
            i = np.argmin(np.abs(x - sens_x))                  # Find the nearest grid point
            j = np.argmin(np.abs(y - sens_y))
            sens_reading = rhs[i][j] + np.random.normal(0, cfg.sensor_noise)
            T_readings.append(sens_reading)
    T_sens_mean = np.mean(np.array(T_readings))
    sens_error = np.abs(T_room_mean - T_sens_mean)      # Keep track of sensor accuracy

    # Temperature regulation
    energy_usage = 0
    if T_sens_mean < 1:
        rhs += dt * q
        energy_usage += cfg.source_strength * dt        # Keep track of energy consumption (unit: degrees)

    # Robin BC contributions
    rx = cfg.alpha * dt / dx**2
    ry = cfg.alpha * dt / dy**2
    bc_term = cfg.T_outside

    rhs[0, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[-1, :] += rx * (cfg.h * dx / cfg.k) * bc_term
    rhs[:, 0] += ry * (cfg.h * dy / cfg.k) * bc_term
    rhs[:, -1] += ry * (cfg.h * dy / cfg.k) * bc_term

    return rhs.flatten(), sens_error, energy_usage

def solve_heat_equation_save_data(
    cfg: Config,
) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,float,float]:
    """Solve the 2D heat equation using implicit Euler.

    Args:
        cfg: Configuration object

    Returns:
        x: x-coordinates (nx,)
        y: y-coordinates (ny,)
        t: time points (nt,)
        T: temperature solution (nt, nx, ny)
    """
    # Create grids
    x = np.linspace(cfg.x_min, cfg.x_max, cfg.nx)
    y = np.linspace(cfg.y_min, cfg.y_max, cfg.ny)
    t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)

    dx, dy = x[1] - x[0], y[1] - y[0]
    dt = t[1] - t[0]

    X, Y = np.meshgrid(x, y, indexing="ij")

    #######################################################################
    # Oppgave 3.2: Start
    #######################################################################

    # Placeholder initialization — replace this with your implementation
    T: np.ndarray = np.zeros((cfg.nt, cfg.nx, cfg.ny))
    T_0 = np.zeros((cfg.nx, cfg.ny))

    # Misc data
    energy_usage = np.zeros(len(t) - 1)
    sens_errors = np.zeros(len(t) - 1)

    first = True
    for n in range(len(t) - 1):
        if first: T_0.fill(cfg.T_outside)
        else: T_0 = T[n]

        b_0, error, energy = _build_rhs_save_data(cfg, T_0, X, Y, dx, dy, dt, t[n+1])
        A = _build_matrix(cfg, dx, dy, dt)

        T_1 = np.linalg.solve(A, b_0)
        T[n + 1] = T_1.reshape(cfg.nx, cfg.ny)

        first = False

        energy_usage[n] = energy
        sens_errors[n] = error

    Power_eff = np.sum(energy_usage)/(len(t)-1)          # Effective power [degrees/hour]
    Error_eff = np.sum(sens_errors)/(len(t)-1)           # Avg error in sensor reading

    #######################################################################
    # Oppgave 3.2: Slutt
    #######################################################################

    return x, y, t, T, Power_eff, Error_eff

def main():
    cfg = load_config("config.yaml")

    print("Solving heat equation with FDM...")
    x, y, t, T_fdm, Power_eff, Error = solve_heat_equation_save_data(cfg)

    print("\nGenerating FDM visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_fdm,
        save_path="output/fdm/fdm_snapshots.png",
    )
    create_animation(
        x, y, t, T_fdm, title="FDM", save_path="output/fdm/fdm_animation.gif"
    )

    print(f'Effective power consumption: {Power_eff:.2f} degrees/hour\nSensor error: {Error:.2f}')


if __name__ == "__main__":
    main()
