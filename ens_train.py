from model import createModel
from utils import splitData, shuffling
from train import tf_dataset
from metrics import calc_loss, dice_coef, iou
from keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
)
from keras.optimizers import Adam
from keras.metrics import Recall, Precision, Accuracy
import os

#TODO: Add IoU Threshold Callback

BATCH = 8
N_ENS = 4
LR = 1e-4
EPOCH = 50
modelType = "ResNet50"

os.makedirs("tmp/weights", exist_ok=True)
os.makedirs("tmp/data/Image", exist_ok=True)
os.makedirs("tmp/data/Mask", exist_ok=True)
os.makedirs("tmp/logs", exist_ok=True)

for i in range(N_ENS):
    weights_path = os.path.join("tmp", "weights", f"model_{i-1}.h5")
    data_path = os.path.join("tmp", "data")
    weights_save_path = os.path.join("tmp", "weights", f"model_{i}.h5")
    log_path = os.path.join("tmp", "logs", f"model_{i}_epoch_log.csv")

    x_train, y_train, x_val, y_val = splitData(data_path)
    x_train, y_train = shuffling(x_train, y_train)

    train_dataset = tf_dataset(x_train, y_train, batch=BATCH)
    val_dataset = tf_dataset(x_val, y_val, batch=BATCH)

    model = createModel(modelType)

    if os.path.exists(weights_path):
        model.load_weights(weights_path, by_name=True)

    loss_fn = calc_loss(model=model)

    model.compile(
        loss=loss_fn,
        optimizer=Adam(LR),
        metrics=[dice_coef, iou, Recall(), Precision(), Accuracy()],
    )

    callbacks = [
        ModelCheckpoint(weights_save_path, verbose=1, save_best_only=True, save_weights_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=5, min_lr=1e-8, verbose=1),
        CSVLogger(log_path),
        TensorBoard(),
    ]
    
    model.fit(train_dataset, epochs=EPOCH, validation_data=val_dataset, callbacks=callbacks)
