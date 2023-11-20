import os
import numpy as np
import cv2
import tensorflow as tf
from model import createModel
from utils import splitData, shuffling
from train import tf_dataset
from metrics import calc_loss, dice_coef, iou, eval_iou
from keras.callbacks import (
    ModelCheckpoint,
    CSVLogger,
    ReduceLROnPlateau,
    EarlyStopping,
    TensorBoard,
    Callback,
)
from keras.optimizers import Adam
from keras.metrics import Recall, Precision, Accuracy

BATCH = 8
N_MODELS = 4
LR = 1e-4
EPOCH = 5
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

modelType = "ResNet50"

os.makedirs("tmp/weights", exist_ok=True)
os.makedirs("tmp/data/Image", exist_ok=True)
os.makedirs("tmp/data/Mask", exist_ok=True)
os.makedirs("tmp/logs", exist_ok=True)

data_path = os.path.join("tmp", "data")


class IoUThresholdCallback(Callback):
    def __init__(self, model, model_n, threshold):
        super().__init__()
        self.model = model
        self.model_n = model_n
        self.threshold = threshold

    def on_train_end(self, logs=None):
        low_iou_indices = []

        for idx, (x, y) in enumerate(zip(x_val, y_val)):
            image = cv2.imread(x, cv2.IMREAD_COLOR).astype(np.float32)
            image_resized = cv2.resize(image, dsize=(256, 256, 3))
            image_resized = image_resized / 255.0
            image_resized -= MEAN
            image_resized /= STD
            image_resized = np.expand_dims(image_resized, axis=0)

            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask_resized = cv2.resize(mask, dsize=(256, 256, 1))
            mask_flatten = mask_resized.flatten()

            y_pred = model.predict(image_resized)[0]
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)
            y_pred = y_pred.flatten()

            iou = eval_iou(y_true=mask_flatten, y_pred=y_pred)

            if iou < self.threshold:
                low_iou_indices.append(idx)
                img_path = os.path.join("tmp", "data", "Image", f"tmp_img_{idx}_{self.model_n}.png")
                mask_path = os.path.join(
                    "tmp", "data", "Mask", f"tmp_img_{idx}_mask_{self.model_n}.png"
                )
                cv2.imwrite(img_path, image)
                cv2.imwrite(mask_path, mask_resized)

        print(f"Examples with IoU < {self.threshold} on last epoch: {len(low_iou_indices)}")
        return


for i in range(N_MODELS):
    tf.keras.backend.clear_session()
    weights_path = os.path.join("tmp", "weights", f"model_{i-1}.h5")
    weights_save_path = os.path.join("tmp", "weights", f"model_{i}.h5")
    log_path = os.path.join("tmp", "logs", f"model_{i}_epoch_log.csv")

    x_train, y_train, x_val, y_val = splitData(data_path)
    x_train, y_train = shuffling(x_train, y_train)

    train_dataset = tf_dataset(x_train, y_train, batch=BATCH)
    val_dataset = tf_dataset(x_val, y_val, batch=BATCH)

    print(f"Train Size: {len(x_train)}")
    print(f"Validation Size: {len(x_val)}")

    model = createModel(modelType)

    if os.path.exists(weights_path):
        print(f"Weights of Model_{i-1} found!\nLoading weights in Model_{i}")
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
        IoUThresholdCallback(model=model, model_n=i, threshold=0.7),
    ]

    print(f"Training Model_{i}....")
    model.fit(train_dataset, epochs=EPOCH, validation_data=val_dataset, callbacks=callbacks)

print("Ensemble Training Done!")
