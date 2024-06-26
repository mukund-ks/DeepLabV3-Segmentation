import os
from glob import glob
from typing import Union

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

H = 256
W = 256


def createDir(path: str) -> None:
    """Helper function to create a directory.

    Args:
        path (str): Path to create.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return


def loadData(path: str) -> list[str]:
    """Helper function to load data in lists from a directory.

    Args:
        path (str): Path to Data Directory.

    Raises:
        OSError: In case the Data Directory provided does not exist.

    Returns:
        list[str]: List of Images and Masks.
    """
    x = sorted(glob(os.path.join(path, "Image", "*png")))
    y = sorted(glob(os.path.join(path, "Mask", "*png")))
    if len(x) == 0 or len(y) == 0:
        raise OSError("No Input in provided Directory.")
    return x, y


def shuffling(x: list[str], y: list[str]) -> list[str]:
    """Helper function to shuffle training set.

    Args:
        x (list[str]): Images
        y (list[str]): Masks

    Returns:
        list[str]: Shuffled list of images and masks.
    """
    x, y = shuffle(x, y, random_state=42)
    return x, y


def splitData(path: str) -> list[str]:
    """Helper function to split data into training and validation sets.

    Args:
        path (str): Path to Data Directory.

    Returns:
        list[str]: Paths to images and masks.
    """
    if not os.path.exists(os.path.join(path, "Image")) or not os.path.exists(
        os.path.join(path, "Mask")
    ):
        raise OSError(f"Incomplete Data Directory: {path}")

    X = sorted(glob(os.path.join(path, "Image", "*.png")))
    Y = sorted(glob(os.path.join(path, "Mask", "*.png")))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    return x_train, y_train, x_test, y_test


def createDirs(paths: tuple[str]) -> None:
    """Helper function to create a set of directories.

    Args:
        paths (tuple[str]): Tuple containing paths.
    """
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


def saveResults(img: np.ndarray, mask: np.ndarray, y_pred: np.ndarray, save_img_path: str) -> None:
    """Helper function to save evaluation results as a single png, with provided image, ground truth, predicted mask and segmented output, from left to right.

    Args:
        img (np.ndarray): Input Image
        mask (np.ndarray): Ground Truth
        y_pred (np.ndarray): Predicted Mask
        save_img_path (str): Directory to save the results in.
    """
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


def tmp_cleanup() -> None:
    """Utility function for ens_train.py to delete temporary images and masks."""
    for dir in os.listdir("./tmp/data"):
        if not os.path.isdir(os.path.join("./tmp/data", dir)):
            continue
        for filename in os.listdir(os.path.join("./tmp/data", dir)):
            if not (filename.startswith("tmp") and filename.endswith(".png")):
                continue
            file_path = os.path.join("./tmp/data", dir, filename)
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        print(f"Deleted temporary files in ./tmp/data/{dir}\n")
    return
