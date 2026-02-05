"""Script for training and plotting the NN model.

I scripts/run_nn.py , skriv et liknende script som scripts/run_fdm.py for å trene og
evaluere NN. Plott hvordan tapene utvikler seg i løpet av treningen, og visualiser prediksjonene
fra det ferdig trente nettverket.
Tips:
• Benytt generate_training_data(), train_nn() , og den ferdigskrevne hjelpefunksjonen
predict_grid() for å generere data, trene nettverket, og gjøre prediksjoner på
hele det diskretiserte domenet.


"""

import os

import matplotlib.pyplot as plt
import numpy as np
from viz import create_animation, plot_snapshots

from project import (
    generate_training_data,
    load_config,
    predict_grid,
    train_nn,
)


def main():
    cfg = load_config("config.yaml")

    #######################################################################
    # Oppgave 4.4: Start
    #######################################################################

    print("Generating training data...")
    x, y, t, T_fdm, sensor_data = generate_training_data(cfg)

    #print("Training NN...")
    nn_params, losses = train_nn(sensor_data, cfg)

    print("Predicting on grid...")
    T_pred = predict_grid(nn_params, x, y, t, cfg)

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
    plt.savefig("output/NN/losses.png", dpi=200)
    plt.close()

    #######################################################################
    # Oppgave 4.4: Slutt
    #######################################################################


if __name__ == "__main__":
    main()
