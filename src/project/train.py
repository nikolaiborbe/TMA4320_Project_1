"""Training routines for NN and PINN models."""

import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

from .config import Config
from .loss import bc_loss, data_loss, ic_loss, physics_loss
from .model import init_nn_params, init_pinn_params
from .optim import adam_step, init_adam
from .sampling import sample_bc, sample_ic, sample_interior


def train_nn(
    sensor_data: jnp.ndarray, cfg: Config
) -> tuple[list[tuple[jnp.ndarray, jnp.ndarray]], dict]:
    """Train a standard neural network on sensor data only.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        params: Trained network parameters
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    nn_params = init_nn_params(cfg)
    adam_state = init_adam(nn_params)

    losses = {"total": [], "data": [], "ic": []}  # Fill with loss histories

    #######################################################################
    # Oppgave 4.3: Start
    #######################################################################

    def objective_fn(ic_loss, data_loss):
        return cfg.lambda_ic*ic_loss + cfg.lambda_data*data_loss

    for i in tqdm(range(cfg.num_epochs), desc='Training NN'):
        ic_epoch, key = sample_ic(key, cfg)

        ic_loss_epoch = ic_loss(nn_params, ic_epoch, cfg)
        data_loss_epoch = data_loss(nn_params, sensor_data, cfg)
        obj_val, obj_grad = jax.value_and_grad(objective_fn)(ic_loss_epoch, data_loss_epoch)

        # Update the nn_params and losses dictionary
        holder = adam_step(nn_params, obj_grad, adam_state, lr=cfg.learning_rate)
        nn_params: list[tuple[jnp.ndarray, jnp.ndarray]] = holder[0]
        adam_state = holder[1]
        losses["data"].append(data_loss_epoch)
        losses["ic"].append(ic_loss_epoch)
        losses["total"].append(obj_val)

    
        
    #######################################################################
    # Oppgave 4.3: Slutt
    #######################################################################

    return nn_params, {k: jnp.array(v) for k, v in losses.items()}


def train_pinn(sensor_data: jnp.ndarray, cfg: Config) -> tuple[dict, dict]:
    """Train a physics-informed neural network.

    Args:
        sensor_data: Sensor measurements [x, y, t, T]
        cfg: Configuration

    Returns:
        pinn_params: Trained parameters (nn weights + alpha)
        losses: Dictionary of loss histories
    """
    key = jax.random.key(cfg.seed)
    pinn_params = init_pinn_params(cfg)
    opt_state = init_adam(pinn_params)

    losses = {"total": [], "data": [], "physics": [], "ic": [], "bc": []}

    #######################################################################
    # Oppgave 5.3: Start
    #######################################################################

    # Update the nn_params and losses dictionary

    #######################################################################
    # Oppgave 5.3: Slutt
    #######################################################################

    return pinn_params, {k: jnp.array(v) for k, v in losses.items()}
