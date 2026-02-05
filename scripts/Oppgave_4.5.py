import matplotlib.pyplot as plt
import numpy as np
from project import (
    load_config,
    solve_heat_equation,
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)

cfg = load_config("config.yaml")

t = np.linspace(cfg.t_min, cfg.t_max, cfg.nt)

"""
print("Solving heat equation with FDM...")
x, y, t, T_fdm = solve_heat_equation(cfg)

print("generate training data...")
x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

print("Train NN...")
nn_params, losses = train_nn(sensor_data, cfg)

print("Predict on gridd...")
T_pred = predict_grid(nn_params, x, y, t, cfg)
"""

# NN vs FDM
error = T_pred - T_fdm
mse = np.mean(error**2)
print(mse)

# Plott
plt.plot(T_pred, t)
plt.plot(T_fdm, t)
plt.xlabel("Temperature")
plt.ylabel("time t")
plt.show()
