import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import traceback
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from keras.models import load_model
from metrics import (
    iou as model_iou,
    dice_loss,
    dice_coef,
    eval_iou,
    eval_dice_coef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from utils import loadData, createDir, getMaskLen, saveResults

H = 256
W = 256


def evaluator(eval_dir: str) -> None:
    """Function to evaluate the trained model.

    Args:
        eval_dir (str): Path to evaluation directory.

    Raises:
        OSError: In case the provided Evaluation Path does not exist.
    """
    np.random.seed(42)
    tf.random.set_seed(42)

    if not os.path.exists(eval_dir):
        raise OSError("Path does not exist.", eval_dir)

    createDir("eval_results")

    with CustomObjectScope({"iou": model_iou, "dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = load_model("./output/model.h5")

    try:
        x_test, y_test = loadData(eval_dir)
    except Exception as _:
        traceback.print_exc()
        exit("No input available.")

    print(f"Test:\nImages: {len(x_test)}\tMasks: {len(y_test)}")

    SCORE = []
    for x, y in tqdm(zip(x_test, y_test), total=len(x_test)):
        name = os.path.split(x)[1].split(".")[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        diagonal_len, horizontal_len, vertical_len = getMaskLen(y_pred)

        save_img_path = f"./eval_results/{name}.png"

        saveResults(image, mask, y_pred, save_img_path)

        mask = mask.flatten()
        y_pred = y_pred.flatten()

        acc_scr = accuracy_score(y_pred=y_pred, y_true=mask)
        f1_scr = f1_score(y_pred=y_pred, y_true=mask)
        recall_val = recall_score(y_pred=y_pred, y_true=mask)
        precison_val = precision_score(y_pred=y_pred, y_true=mask)
        iou = eval_iou(y_pred=y_pred, y_true=mask)
        dice = eval_dice_coef(y_pred=y_pred, y_true=mask)
        SCORE.append(
            [
                name,
                acc_scr,
                f1_scr,
                recall_val,
                precison_val,
                iou,
                dice,
                diagonal_len,
                horizontal_len,
                vertical_len,
            ]
        )

    score = [s[1:6] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1-Score: {score[1]:0.5f}")
    print(f"Jaccard-Score: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precison: {score[4]:0.5f}")

    df = pd.DataFrame(
        SCORE,
        columns=[
            "Image",
            "Accuracy",
            "F1",
            "Jaccard",
            "Recall",
            "Precison",
            "Diagonal Length (px)",
            "Horizontal Length (px)",
            "Vertical Length (px)",
        ],
    )
    df.to_csv("./eval_results/Evaluation_Score.csv")

    return


if __name__ == "__main__":
    evaluator(eval_dir="./eval_data")
