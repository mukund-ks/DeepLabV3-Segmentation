import os
import click
import tensorflow as tf
from model import createModel
from utils import splitData, shuffling, tmp_cleanup
from train import tf_dataset
from metrics import calc_loss, dice_coef, iou
from custom_callbacks import IoUThresholdCallback
from keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)
from keras.optimizers import Adam
from keras.metrics import Recall, Precision, Accuracy


@click.command()
@click.option(
    "--n-models",
    type=int,
    default=4,
    help="No. of Models in Ensemble. Defaults to 4.",
)
@click.option(
    "--epochs",
    type=int,
    default=60,
    help="Epochs to train each model for. Defaults to 60.",
)
@click.option(
    "--batches",
    type=int,
    default=8,
    help="Batch size for data. Defaults to 8.",
)
@click.option(
    "--initial-lr",
    type=float,
    default=1e-4,
    help="Initial Learning Rate for training. Defaults to 1e-4.",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=0.7,
    help="Starting threshold for IoU. Defaults to 0.7.",
)
@click.option(
    "--max-threshold",
    type=float,
    default=0.95,
    help="Maximum threshold for IoU. Defaults to 0.95.",
)
@click.option(
    "--scale-factor",
    type=float,
    default=0.05,
    help="Scaling factor for IoU threshold. Defaults to 0.05.",
)
def ens_trainer(
    n_models: int,
    epochs: int,
    batches: int,
    initial_lr: float,
    iou_threshold: float,
    max_threshold: float,
    scale_factor: float,
) -> None:
    """Ensemble Trainer script for DeepLabV3+ Model with ResNet50 backbone. The script trains a specified number of models, loads previous model's weights into subsequent models and keeps track of poor performing examples in the validation dataset (through IoU Threshold). Learning Rate and IoU Threshold are adjusted for each model during loop.

    Args:
        n_models (int): No. of Models in Ensemble
        epochs (int): Epochs to train each model for
        batches (int): Batch size for data
        initial_lr (float): Initial Learning Rate for training
        iou_threshold (float): Starting threshold for IoU
        max_threshold (float): Maximum threshold for IoU
        scale_factor (float): Scaling factor for IoU threshold

    Raises:
        OSError: In case Data Path (./tmp/data) does not exist.
    """
    os.makedirs("tmp/weights", exist_ok=True)
    os.makedirs("tmp/logs", exist_ok=True)

    data_path = os.path.join("tmp", "data")

    if not os.path.exists(data_path):
        raise OSError(f"Data Dir: {data_path} does not exist.")

    for model_idx in range(1, n_models + 1):
        tf.keras.backend.clear_session()
        weights_path = os.path.join("tmp", "weights", f"model_{model_idx-1}.h5")
        weights_save_path = os.path.join("tmp", "weights", f"model_{model_idx}.h5")
        log_path = os.path.join("tmp", "logs", f"model_{model_idx}_epoch_log.csv")

        x_train, y_train, x_val, y_val = splitData(data_path)
        x_train, y_train = shuffling(x_train, y_train)

        train_dataset = tf_dataset(x_train, y_train, batch=batches)
        val_dataset = tf_dataset(x_val, y_val, batch=batches)

        click.secho(f"Train Size: {len(x_train)}\nValidation Size: {len(x_val)}\n", fg="green")

        model = createModel("ResNet50")

        if os.path.isfile(weights_path):
            click.secho(
                f"Weights of Model_{model_idx-1} found!\nLoading weights in Model_{model_idx} 👍\n",
                fg="green",
            )
            model.load_weights(weights_path, by_name=True)

        loss_fn = calc_loss(model=model)

        model.compile(
            loss=loss_fn,
            optimizer=Adam(initial_lr / (2 ** (model_idx - 1))),
            metrics=[dice_coef, iou, Recall(), Precision(), Accuracy()],
        )

        click.secho(
            f"Threshold for Model_{model_idx}: {iou_threshold}\nLR for Model_{model_idx}: {initial_lr / (2 ** (model_idx - 1))}\n",
            fg="blue",
        )

        callbacks = [
            ModelCheckpoint(
                weights_save_path, verbose=1, save_best_only=True, save_weights_only=True
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1),
            CSVLogger(log_path),
            TensorBoard(),
            IoUThresholdCallback(
                model=model,
                model_idx=model_idx,
                x_val=x_val,
                y_val=y_val,
                threshold=iou_threshold,
            ),
        ]

        click.secho(f"Training Model_{model_idx}....\n", fg="blue")
        model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)

        iou_threshold = min(iou_threshold + scale_factor, max_threshold)

    click.secho("Ensemble Training Done!", fg="green")
    click.secho("\nRunning cleanup!\n", fg="blue")
    tmp_cleanup()
    return


if __name__ == "__main__":
    ens_trainer()
