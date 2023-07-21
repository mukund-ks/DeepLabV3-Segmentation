import numpy as np
import tensorflow as tf
from keras.layers import Flatten


def iou(y_pred, y_true):
    def f(y_true, y_pred):
        intersection = (y_true * y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f, [y_true, y_pred], tf.float32)


SMOOTH = 1e-15
def dice_coef(y_true, y_pred):
    y_true = Flatten()(y_true)
    y_pred = Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + SMOOTH) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + SMOOTH)


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_pred, y_true)
