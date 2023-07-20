import os
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from typing import Any


def createDir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
    return


def loadData(path: str) -> list[str]:
    x = sorted(glob(os.path.join(path, "Image", "*png")))
    y = sorted(glob(os.path.join(path, "Mask", "*png")))
    return x, y


def shuffling(x, y):
    x, y = shuffle(x, y, random_state=42)
    return x, y


def splitData(path: str) -> list[str]:
    X = sorted(glob(os.path.join(path, "images", "*.png")))
    Y = sorted(glob(os.path.join(path, "masks", "*.png")))

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    return x_train, y_train, x_test, y_test


def createDirs(paths: tuple[str]) -> None:
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)
    return
