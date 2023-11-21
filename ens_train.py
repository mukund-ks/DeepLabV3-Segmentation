import os
import tensorflow as tf
from model import createModel
from utils import splitData, shuffling
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

os.makedirs("tmp/weights", exist_ok=True)
os.makedirs("tmp/data/Image", exist_ok=True)
os.makedirs("tmp/data/Mask", exist_ok=True)
os.makedirs("tmp/logs", exist_ok=True)

BATCH = 8
N_MODELS = 4
EPOCH = 50

threshold = 0.7
max_threshold = 0.95
scale_factor = 0.05

initial_lr = 1e-4

data_path = os.path.join("tmp", "data")

for model_idx in range(1, N_MODELS + 1):
    tf.keras.backend.clear_session()
    weights_path = os.path.join("tmp", "weights", f"model_{model_idx-1}.h5")
    weights_save_path = os.path.join("tmp", "weights", f"model_{model_idx}.h5")
    log_path = os.path.join("tmp", "logs", f"model_{model_idx}_epoch_log.csv")

    x_train, y_train, x_val, y_val = splitData(data_path)
    x_train, y_train = shuffling(x_train, y_train)

    train_dataset = tf_dataset(x_train, y_train, batch=BATCH)
    val_dataset = tf_dataset(x_val, y_val, batch=BATCH)

    print(f"Train Size: {len(x_train)}")
    print(f"Validation Size: {len(x_val)}")

    model = createModel("ResNet50")

    if os.path.isfile(weights_path):
        print(f"Weights of Model_{model_idx-1} found!\nLoading weights in Model_{model_idx}")
        model.load_weights(weights_path, by_name=True)

    loss_fn = calc_loss(model=model)

    model.compile(
        loss=loss_fn,
        optimizer=Adam(initial_lr / (2 ** (model_idx - 1))),
        metrics=[dice_coef, iou, Recall(), Precision(), Accuracy()],
    )

    threshold = min(threshold + scale_factor, max_threshold)

    print(f"Threshold for Model_{model_idx}: {threshold}")

    callbacks = [
        ModelCheckpoint(weights_save_path, verbose=1, save_best_only=True, save_weights_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1),
        CSVLogger(log_path),
        TensorBoard(),
        IoUThresholdCallback(
            model=model, model_idx=model_idx, x_val=x_val, y_val=y_val, threshold=threshold
        ),
    ]

    print(f"Training Model_{model_idx}....")
    model.fit(train_dataset, epochs=EPOCH, validation_data=val_dataset, callbacks=callbacks)

print("Ensemble Training Done!")
