import os

import cv2
import numpy as np
from albumentations import (
    ChannelShuffle,
    HorizontalFlip,
    RandomBrightnessContrast,
    RandomSizedCrop,
    Rotate,
    VerticalFlip,
)
from tqdm import tqdm
from utils import createDirs, loadData, splitData


def augment_data(images: list, masks: list, save_path: str, augment: bool) -> None:
    """Function to apply augmentions and/or resize images to specific dimensions.

    Args:
        images (list): List containing paths to images.
        masks (list): List containing paths to masks.
        save_path (str): Path to save the new set.
        augment (bool): Opt-in to apply augmentations to the dataset.
    """
    H = 256
    W = 256

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = os.path.split(x)[1].split(".")[0]

        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        if augment:
            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x1 = augmented["image"]
            y1 = augmented["mask"]

            aug = ChannelShuffle(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented["image"]
            y2 = augmented["mask"]

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented["image"]
            y3 = augmented["mask"]

            aug = RandomBrightnessContrast(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented["image"]
            y4 = augmented["mask"]

            aug = Rotate(limit=45, p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented["image"]
            y5 = augmented["mask"]

            aug = RandomSizedCrop(min_max_height=(256, 256), width=256, height=256, p=1.0)
            augmented = aug(image=x, mask=y)
            x6 = augmented["image"]
            y6 = augmented["mask"]

            X = [x, x1, x2, x3, x4, x5, x6]
            Y = [y, y1, y2, y3, y4, y5, y6]

        else:
            X = [x]
            Y = [y]

        index = 0
        for i, m in zip(X, Y):
            i = cv2.resize(i, (H, W))
            m = cv2.resize(m, (H, W))

            tmp_image_name = f"{name}_{index}.png"
            tmp_mask_name = f"{name}_{index}.png"

            image_path = os.path.join(save_path, "Image", tmp_image_name)
            mask_path = os.path.join(save_path, "Mask", tmp_mask_name)

            if not cv2.imwrite(rf"{image_path}", i):
                print("Could not save image.")
                break
            if not cv2.imwrite(rf"{mask_path}", m):
                print("Could not save mask")
                break

            index += 1
    return


def processData(data_dir: str, augmentation: bool, split_data: bool) -> None:
    """Function to prepare dataset for training.

    Args:
        data_dir (str): Path to Data Directory.
        augmentation (bool): Opt-in to apply augmentations to provided data.
        split_data (bool): Opt-in to split data into training & validation set.

    Raises:
        OSError: In case the provided Data Directory does not exist.
    """
    np.random.seed(42)

    if not os.path.exists(data_dir):
        raise OSError("Directory does not exist.", data_dir)

    if split_data:
        x_train, y_train, x_test, y_test = splitData(data_dir)
    else:
        train_path = os.path.join(data_dir, "Train")
        x_train, y_train = loadData(path=train_path)

        test_path = os.path.join(data_dir, "Test")
        x_test, y_test = loadData(path=test_path)

    print(f"Train\t: {len(x_train)} - {len(y_train)}")
    print(f"Test\t: {len(x_test)} - {len(y_test)}")

    createDirs(
        (
            "./new_data/Train/Image/",
            "./new_data/Train/Mask/",
            "./new_data/Test/Image/",
            "./new_data/Test/Mask/",
        )
    )

    print("Creating New Training Set: ")
    augment_data(x_train, y_train, "./new_data/Train/", augment=augmentation)
    print("Creating New Testing Set: ")
    augment_data(x_test, y_test, "./new_data/Test/", augment=False)

    return


if __name__ == "__main__":
    processData(data_dir="./data_pss", augmentation=False, split_data=True)
