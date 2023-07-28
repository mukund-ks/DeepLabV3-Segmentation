import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from typing import Any
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef, iou
from utils import loadData, createDir

H = 256
W = 256


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


def evaluator(eval_dir: str) -> None:
    np.random.seed(42)
    tf.random.set_seed(42)

    if not os.path.exists(eval_dir):
        raise OSError("Path does not exist.", eval_dir)

    createDir("eval_results")

    with CustomObjectScope({"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = load_model("./output/model.h5")

    x_test, y_test = loadData(eval_dir)
    print(f"Test:\nImages: {len(x_test)}\tMasks: {len(y_test)}")

    SCORE = []
    for x, y in tqdm(zip(x_test, y_test), total=len(x_test)):
        name = x.split("\\")[-1].split(".")[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)

        y_pred = model.predict(x)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred > 0.5
        y_pred = y_pred.astype(np.int32)

        save_img_path = f"./eval_results/{name}.png"

        saveResults(image, mask, y_pred, save_img_path)

        mask = mask.flatten()
        y_pred = y_pred.flatten()

        acc_scr = accuracy_score(mask, y_pred)
        f1_scr = f1_score(mask, y_pred, labels=[0, 1], average="micro")
        jac_scr = jaccard_score(mask, y_pred, labels=[0, 1], average="micro")
        recall_val = recall_score(mask, y_pred, labels=[0, 1], average="micro")
        precison_val = precision_score(mask, y_pred, labels=[0, 1], average="micro")
        SCORE.append([name, acc_scr, f1_scr, jac_scr, recall_val, precison_val])

    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1-Score: {score[1]:0.5f}")
    print(f"Jaccard-Score: {score[2]:0.5f}")
    print(f"Recall: {score[3]:0.5f}")
    print(f"Precison: {score[4]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "Accuracy", "F1", "Jaccard", "Recall", "Precison"])
    df.to_csv("./eval_results/Evaluation_Score.csv")

    return


if __name__ == "__main__":
    evaluator(eval_dir="./eval_cvppp")
