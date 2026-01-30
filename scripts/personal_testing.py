import numpy as np
import jax.numpy as jnp
import platform
import matplotlib
import matplotlib.pyplot as plt

if platform.system() == "Linux":
    matplotlib.use("QtAgg")

from project import solve_heat_equation, generate_training_data, load_config, Config


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def set_sensor_pos(config: Config, sensor_index: int, x_pos: float, y_pos: float):
    config.sensor_locations = config.sensor_locations.at[sensor_index].set(jnp.array([x_pos,y_pos]))
    return config

def load_config_1():
    config: Config = load_config()
    # Replace all sensors with just one at (0, 0)
    config = set_sensor_pos(config, 0, 0, 1)

    return config


def testing():
    # configs = [load_config_1(), load_config_2(), load_config_3()]
    configs = [load_config_1()]
    for config in configs:
        plot_config(config)


def plot_config(config: Config):
    solve_heat_equation(config)
    x, y, t, T_fdm, sensor_data = generate_training_data(config)

    # Group data by sensor location using a dict
    sensors: dict[tuple[float, float], dict[str, list[float]]] = {}
    for x_i, y_i, t_i, T_i in sensor_data:
        key = (float(x_i), float(y_i))
        if key not in sensors:
            sensors[key] = {"times": [], "temps": []}
        sensors[key]["times"].append(t_i)
        sensors[key]["temps"].append(T_i)

    # Plot each sensor
    for (x_pos, y_pos), data in sensors.items():
        plt.plot(
            data["times"], data["temps"], label=f"Sensor ({x_pos:.1f}, {y_pos:.1f})"
        )

    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    testing()
