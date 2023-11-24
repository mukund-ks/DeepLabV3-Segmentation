import os
import cv2
import click
import numpy as np
import pandas as pd
from tqdm import tqdm
from model_cloud import createModel
from utils import loadData
from metrics import (
    eval_iou,
    eval_dice_coef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from utils import saveResults, getMaskLen

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


@click.command()
@click.option(
    "--data-dir",
    type=str,
    prompt="Evaluation Data Directory",
    required=True,
    help="Directory containing data for evaluation",
)
@click.option(
    "--weights-dir", type=str, default="./tmp/weights", help="Directory containing data for evaluation"
)
def ens_eval(data_dir: str, weights_dir: str) -> None:
    """Evaluation script to test models trained in ensemble mode. Find results in 'ens_results' directory.

    Args:
        data_dir (str): Directory containing data for evaluation
        weights_dir (str): Directory containing ensemble model weights.

    Raises:
        OSError: In the event that weights are not found.
    """
    os.makedirs("ens_results", exist_ok=True)

    all_files = os.listdir(weights_dir)

    model_files = [file for file in all_files if file.startswith("model_") and file.endswith(".h5")]
    
    if not len(model_files):
        raise OSError(f"No weights in weights directory, {weights_dir}")

    x_test, y_test = loadData(data_dir)

    preds = []
    for m in model_files:
        model = createModel("ResNet50")
        model.load_weights(os.path.join(weights_dir, m))

        model_name = m.split(".")[0]
        model_preds = []

        click.secho(f"\nEval Images: {len(x_test)}\nEval Masks: {len(y_test)}\n", fg="blue")

        click.secho(f"Testing {model_name}...\n", fg="green")
        for x in tqdm(x_test, total=len(x_test)):
            image = cv2.imread(x, cv2.IMREAD_COLOR)
            image_resized = cv2.resize(image, dsize=(256, 256))
            x = image_resized / 255.0
            x -= MEAN
            x /= STD
            x = np.expand_dims(x, axis=0)

            y_pred = model.predict(x)[0]
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)

            model_preds.append(y_pred)

        preds.append(model_preds)

    preds = np.array(preds)
    avg_result = np.mean(preds, axis=0)
    avg_result_binary = (avg_result > 0).astype(np.int32)

    SCORE = []
    for x, y, pred in zip(x_test, y_test, avg_result_binary):
        name = os.path.split(x)[1].split(".")[0]

        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image_resized = cv2.resize(image, dsize=(256, 256))

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask_resized = cv2.resize(mask, dsize=(256, 256))

        diagonal_len, horizontal_len, vertical_len = getMaskLen(pred)

        save_img_path = f"./ens_results/{name}.png"

        saveResults(image_resized, mask_resized, pred, save_img_path)

        mask_resized = mask_resized.flatten()
        pred = pred.flatten()

        acc_scr = accuracy_score(y_pred=pred, y_true=mask_resized)
        f1_scr = f1_score(y_pred=pred, y_true=mask_resized)
        recall_val = recall_score(y_pred=pred, y_true=mask_resized)
        precison_val = precision_score(y_pred=pred, y_true=mask_resized)
        iou = eval_iou(y_pred=pred, y_true=mask_resized)
        dice = eval_dice_coef(y_pred=pred, y_true=mask_resized)
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

    score = [s[1:7] for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Accuracy: {score[0]:0.5f}")
    print(f"F1-Score: {score[1]:0.5f}")
    print(f"Recall: {score[2]:0.5f}")
    print(f"Precison: {score[3]:0.5f}")
    print(f"IoU: {score[4]:0.5f}")
    print(f"Dice Coefficient: {score[5]:0.5f}")

    df = pd.DataFrame(
        SCORE,
        columns=[
            "Image",
            "Accuracy",
            "F1",
            "Recall",
            "Precison",
            "IoU",
            "Dice Coeff",
            "Diagonal Length (px)",
            "Horizontal Length (px)",
            "Vertical Length (px)",
        ],
    )
    df.to_csv("./ens_results/Evaluation_Score.csv")
    return

if __name__=="__main__":
    ens_eval()