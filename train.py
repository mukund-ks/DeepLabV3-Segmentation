import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
from keras.metrics import Recall, Precision
from model import createModel
from metrics import dice_loss, dice_coef, iou
from utils import createDir, loadData, shuffling

H = 256
W = 256
LR = 1e-4


def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0  # normalizing image
    x = x.astype(np.float32)
    return x


def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x / 255.0  # normalizing the mask
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x


def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y


def tf_dataset(X, Y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


def trainer(
    augmentation: bool,
    batches: int = 4,
    epochs: int = 25,
    modelType: str = "ResNet101",
    data_dir: str = "",
):
    np.random.seed(42)
    tf.random.set_seed(42)

    createDir("output")
    files_dir = "output"

    model_path = os.path.join(files_dir, "model.h5")
    csv_path = os.path.join(files_dir, "epoch_logs.csv")

    if augmentation:
        train_path = os.path.join("./augmented_data", "Train")
        val_path = os.path.join("./augmented_data", "Test")
    else:
        train_path = os.path.join(data_dir, "Train")
        val_path = os.path.join(data_dir, "Test")

    x_train, y_train = loadData(train_path)
    x_train, y_train = shuffling(x_train, y_train)
    x_val, y_val = loadData(val_path)

    print(f"Train:\nImages: {len(x_train)}\tMasks: {len(y_train)}")
    print(f"Validation:\nImages: {len(x_val)}\tMasks: {len(y_val)}")

    train_dataset = tf_dataset(x_train, y_train, batch=batches)
    val_dataset = tf_dataset(x_val, y_val, batch=batches)

    model = createModel((H, W, 3), modelType=modelType)
    model.compile(
        loss=dice_loss, optimizer=Adam(LR), metrics=[dice_coef, iou, Recall(), Precision()]
    )

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        TensorBoard(),
        EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=False),
    ]

    model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks)
    return


if __name__ == "__main__":
    trainer(batches=4, epochs=25, modelType="ResNet101", augmentation=True, data_dir="./data")
