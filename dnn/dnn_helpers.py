import os
import time
from typing import List, Optional
from tensorflow import keras
from tensorflow.keras.callbacks import Callback


def get_run_logdir() -> str:
    """
    Creates a unique log directory for each training run using timestamps.

    Returns:
        str: Path to the log directory
    """
    try:
        root_logdir = os.path.join(os.curdir, "logs")
        os.makedirs(root_logdir, exist_ok=True)
        run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        return os.path.join(root_logdir, run_id)
    except Exception as e:
        raise RuntimeError(f"Failed to create log directory: {str(e)}")


def get_early_stopping(
    monitor: str = 'val_loss',
    patience: int = 10,
    restore_best_weights: bool = True
) -> keras.callbacks.EarlyStopping:
    """
    Creates an EarlyStopping callback to prevent overfitting.

    Args:
        monitor (str): Metric to monitor for early stopping
        patience (int): Number of epochs to wait before stopping if no
            improvement
        restore_best_weights (bool): Whether to restore model weights from best
            epoch

    Returns:
        keras.callbacks.EarlyStopping: Configured early stopping callback
    """
    return keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=restore_best_weights
    )


def get_tensorboard_cb() -> keras.callbacks.TensorBoard:
    """
    Creates a TensorBoard callback for logging training metrics.

    Returns:
        keras.callbacks.TensorBoard: Configured TensorBoard callback
    """
    return keras.callbacks.TensorBoard(get_run_logdir())


def save_checkpoint(
    model_name: str,
    monitor: str = 'val_loss',
    save_best_only: bool = True,
    save_weights_only: bool = True
) -> keras.callbacks.ModelCheckpoint:
    """
    Creates a ModelCheckpoint callback to save model weights.

    Args:
        model_name (str): Name of the model for checkpoint file
        monitor (str): Metric to monitor for saving checkpoints
        save_best_only (bool): Whether to save only the best model
        save_weights_only (bool): Whether to save only model weights

    Returns:
        keras.callbacks.ModelCheckpoint: Configured checkpoint callback
    """
    try:
        os.makedirs("./data", exist_ok=True)
        return keras.callbacks.ModelCheckpoint(
            filepath=f"./data/checkpoint-{model_name}.weights.h5",
            save_best_only=save_best_only,
            save_weights_only=save_weights_only,
            monitor=monitor,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create checkpoint directory: {str(e)}")


def get_exponential_decay_scheduler(
    initial_learning_rate: float = 0.001,
    decay_steps: int = 1000,
    decay_rate: float = 0.9,
    staircase: bool = True
) -> keras.optimizers.schedules.ExponentialDecay:
    """
    Creates an exponential learning rate decay schedule.

    Args:
        initial_learning_rate (float): Initial learning rate
        decay_steps (int): Number of steps to decay over
        decay_rate (float): Rate of decay
        staircase (bool): Whether to use discrete steps

    Returns:
        keras.optimizers.schedules.ExponentialDecay: Configured learning rate
            schedule
    """
    return keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=staircase
    )


def get_adam_optimizer(
    initial_learning_rate: float = 0.001
) -> keras.optimizers.Adam:
    """
    Creates an Adam optimizer with a learning rate schedule.

    Args:
        learning_rate (float): Initial learning rate

    Returns:
        keras.optimizers.Adam: Configured Adam optimizer
    """
    return keras.optimizers.Adam(
        learning_rate=get_exponential_decay_scheduler(
            initial_learning_rate=initial_learning_rate)
    )


def get_all_callbacks(
    model_name: str,
    early_stopping_params: Optional[dict] = None,
    checkpoint_params: Optional[dict] = None,


) -> List[Callback]:
    """
    Combines all training callbacks into a single list.

    Args:
        model_name (str): Name of the model for checkpointing
        early_stopping_params (dict, optional): Parameters for early stopping
        checkpoint_params (dict, optional): Parameters for model checkpointing
        lr_scheduler_params (dict, optional): Parameters for learning rate
            scheduling

    Returns:
        List[Callback]: List of configured callbacks
    """
    return [get_early_stopping(**(early_stopping_params or {})),
            get_tensorboard_cb(),
            save_checkpoint(model_name, **(checkpoint_params or {}))]
