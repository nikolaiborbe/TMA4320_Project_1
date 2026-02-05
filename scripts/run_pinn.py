"""Script for training and plotting the PINN model."""

import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_pinn,
)

os.makedirs("output/Pinn", exist_ok=True)

def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################

    print("generate training data...")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    print("Train Pinn...")
    pinn_params, losses = train_pinn(sensor_data, cfg)

    print("Predict on gridd...")
    T_pred = predict_grid(pinn_params["nn"], x, y, t, cfg)

    print("\nGenerating Pinn visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/Pinn/Pinn_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="Pinn", save_path="output/Pinn/Pinn_animation.gif"
    )

   
    epochs = np.arange(len(next(iter(losses.values()))))

    # Plott losses
    for key, value in losses.items():
        plt.plot(epochs, value, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.yscale("log")  # ofte nyttig
    plt.legend()
    plt.grid(True)
    plt.savefig("output/Pinn/losses.png", dpi=200)
    plt.close()

    """
    # Plott parameters
    pinn_params = {
        "nn": init_nn_params(cfg, key=nn_key),
        "log_alpha": np.log(jax.random.normal(scalars_key, (1, ))), # np.log defaults to np.ln()
        "log_power": np.log(20),
        "log_k": np.log(0.09),
        "log_h": np.log(0.17)
    }
    """
   
    

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
