import os
import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from typing import Any, Union

H = 256
W = 256


def createDir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    return


def loadData(path: str) -> list[str]:
    x = sorted(glob(os.path.join(path, "Image", "*png")))
    y = sorted(glob(os.path.join(path, "Mask", "*png")))
    if len(x) == 0 or len(y) == 0:
        raise OSError("No Input in provided Directory.")
    return x, y


def shuffling(x: Any, y: Any) -> tuple:
    x, y = shuffle(x, y, random_state=42)
    return x, y


def splitData(path: str) -> list[str]:
    X = sorted(glob(os.path.join(path, "Image", "*.png")))
    Y = sorted(glob(os.path.join(path, "Mask", "*.png")))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test


def createDirs(paths: tuple[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    return


def getMaskLen(y_pred: np.ndarray) -> Union[float, int]:
    """Determines a bounding box around the predicted mask and returns diagonal, horizontal and vertical length of the box in pixels

    Args:
        y_pred (numpy.ndarray): Binary mask from model prediction

    Returns:
        float | int: diagonal, horizontal & vertical length
    """
    binary_mask = np.array(y_pred, dtype=np.uint8)

    # Finding Contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Getting the Largest Contour
    try:
        largest_contour = max(contours, key=cv2.contourArea)
    except Exception as _:
        return 0.0, 0.0, 0.0

    # Getting the co-ordinates for smalles possible Bounding Box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Calculating line lengths
    diagonal_length = round(np.sqrt(w**2 + h**2), 2)
    horizontal_length = round(w, 2)
    vertical_length = round(h, 2)

    return diagonal_length, horizontal_length, vertical_length


def saveResults(img: Any, mask: Any, y_pred: Any, save_img_path: str) -> None:
    line = np.ones((H, 10, 3)) * 128

    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    # mask = mask * 255

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)

    masked_img = img * y_pred
    y_pred = y_pred * 255

    imgs = np.concatenate([img, line, mask, line, y_pred, line, masked_img], axis=1)
    cv2.imwrite(save_img_path, imgs)

    return
