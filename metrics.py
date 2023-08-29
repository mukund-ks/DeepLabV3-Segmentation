import numpy as np
import tensorflow as tf
from keras.layers import Flatten

SMOOTH = 1e-15


def iou(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + SMOOTH) / (union + SMOOTH)
        x = x.astype(np.float32)
        return x

    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


def dice_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + SMOOTH) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + SMOOTH)


def dice_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 1.0 - dice_coef(y_true, y_pred)
