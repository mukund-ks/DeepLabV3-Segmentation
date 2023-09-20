import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import numpy as np
import cv2
import tensorflow as tf
from keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)
from keras.optimizers import Adam
from keras.metrics import Recall, Precision, Accuracy
from model import createModel
from metrics import dice_loss, dice_coef, iou
from utils import createDir, loadData, shuffling
from typing import Any

H = 256
W = 256
LR = 2e-4

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def read_image(path: Any) -> Any:
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32)
    x = x / 255.0  # normalizing and standardizing image with Imagenet specifications
    x -= MEAN
    x /= STD
    return x


def read_mask(path: Any) -> Any:
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    x = x / 255.0  # normalizing mask
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x: Any, y: Any) -> Any:
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X: Any, Y: Any, batch: int = 2) -> Any:
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


def trainer(
    stop_early: bool,
    batches: int,
    epochs: int,
    modelType: str,
) -> None:
    """Function to train the model.

    Args:
        stop_early (bool): Opt-in to use Early-Stopping during training.
        batches (int): No. of batches.
        epochs (int): No. of epochs.
        modelType (str): Choice of backbone. ResNet50 or ResNet101.
    """
    np.random.seed(42)
    tf.random.set_seed(42)

    createDir("output")
    files_dir = "output"

    model_path = os.path.join(files_dir, "model.h5")
    csv_path = os.path.join(files_dir, "Epoch_Log.csv")

    train_path = os.path.join("./augmented_data_ews", "Train")
    val_path = os.path.join("./augmented_data_ews", "Test")

    x_train, y_train = loadData(train_path)
    x_train, y_train = shuffling(x_train, y_train)
    x_val, y_val = loadData(val_path)

    print(f"Train:\nImages: {len(x_train)}\tMasks: {len(y_train)}")
    print(f"Validation:\nImages: {len(x_val)}\tMasks: {len(y_val)}")

    train_dataset = tf_dataset(x_train, y_train, batch=batches)
    val_dataset = tf_dataset(x_val, y_val, batch=batches)

    model = createModel(shape=(H, W, 3), modelType=modelType)

    loss_fn = dice_loss(model=model)

    model.compile(
        loss=loss_fn,
        optimizer=Adam(LR),
        metrics=[dice_coef, iou, Recall(), Precision(), Accuracy()],
    )

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(log_dir="logs"),
    ]

    if stop_early:
        callbacks.append(
            EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=False),
        )

    print(f"\nUsing {modelType} as Encoder{' with Early Stopping.' if stop_early else '.'}\n")

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
    return


if __name__ == "__main__":
    trainer(batches=4, epochs=80, modelType="ResNet50", stop_early=False)
