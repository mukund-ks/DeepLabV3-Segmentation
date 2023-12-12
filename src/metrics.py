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


def calc_loss(model: tf.keras.Model) -> float:
    def dice_loss(y_true: np.ndarray, y_pred: np.ndarray, l2_weight: float = 1e-8):
        dice = dice_coef(y_true, y_pred)
        l2_loss = tf.add_n([l2_weight * tf.nn.l2_loss(var) for var in model.trainable_weights])
        return 1.0 - dice + l2_loss

    return dice_loss


# Evaluation Metrics


def eval_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) for binary masks.

    Parameters:
    y_true (numpy.ndarray): Ground truth binary mask (0 to 255).
    y_pred (numpy.ndarray): Predicted binary mask (0 or 1).

    Returns:
    float: IoU score.
    """
    y_true[y_true > 0] = 1  # normalize binary mask
    y_true = y_true.astype(np.float32)

    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection

    x = (intersection + SMOOTH) / (union + SMOOTH)
    x = x.astype(np.float32)
    return x


def eval_dice_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Dice Coefficient for binary masks.

    Parameters:
    y_true (numpy.ndarray): Ground truth binary mask (0 to 255).
    y_pred (numpy.ndarray): Predicted binary mask (0 or 1).

    Returns:
    float: Dice Coefficient score.
    """
    y_true[y_true > 0] = 1  # normalize binary mask
    y_true = y_true.astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)

    if union == 0:
        return 1.0  # Handle the case where both masks are empty

    dice = (2.0 * intersection + SMOOTH) / (union + SMOOTH)
    return dice


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Accuracy score for binary classification.

    Parameters:
    y_true (numpy.ndarray): Ground truth binary labels (0 to 255).
    y_pred (numpy.ndarray): Predicted binary labels (0 or 1).

    Returns:
    float: Accuracy score.
    """
    y_true[y_true > 0] = 1  # normalize binary mask
    y_true = y_true.astype(np.float32)

    correct = np.sum(y_true == y_pred)
    total = y_true.size
    accuracy = correct / total

    return accuracy


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Precision score for binary classification.

    Parameters:
    y_true (numpy.ndarray): Ground truth binary labels (0 to 255).
    y_pred (numpy.ndarray): Predicted binary labels (0 or 1).

    Returns:
    float: Precision score.
    """
    y_true[y_true > 0] = 1  # normalize binary mask
    y_true = y_true.astype(np.float32)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))

    if TP + FP == 0:
        return 1.0  # case where both TP and FP are zero

    precision = TP / (TP + FP)

    return precision


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Recall (Sensitivity) score for binary classification.

    Parameters:
    y_true (numpy.ndarray): Ground truth binary labels (0 to 255).
    y_pred (numpy.ndarray): Predicted binary labels (0 or 1).

    Returns:
    float: Recall (Sensitivity) score.
    """
    y_true[y_true > 0] = 1  # normalize binary mask
    y_true = y_true.astype(np.float32)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    if TP + FN == 0:
        return 1.0  # case where TP and FN are zero

    recall = TP / (TP + FN)
    return recall


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the F1 Score for binary classification.

    Parameters:
    y_true (numpy.ndarray): Ground truth binary labels (0 to 255).
    y_pred (numpy.ndarray): Predicted binary labels (0 or 1).

    Returns:
    float: F1 Score.
    """
    y_true[y_true > 0] = 1  # normalize binary mask
    y_true = y_true.astype(np.float32)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0  # Handle the case where both precision and recall are zero

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1
