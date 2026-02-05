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


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 5.4: Start
    #######################################################################

    print("generate training data...")
        x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    print("Train NN...")
    pin_params, losses = train_pinn(cfg, sensor_data)

    print("Predict on gridd...")
    T_pred = predict_grid(cfg, nn_params, x, y, t)

    print("\nGenerating NN visualizations...")
    plot_snapshots(
        x,
        y,
        t,
        T_pred,
        save_path="output/NN/NN_snapshots.png",
    )
    create_animation(
        x, y, t, T_pred, title="NN", save_path="output/NN/NN_animation.gif"
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

    # Plott parameters

    

    #######################################################################
    # Oppgave 5.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
