import cv2
import numpy as np
import os
from keras.callbacks import Callback
from metrics import eval_iou

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class IoUThresholdCallback(Callback):
    """Custom Callback to calculate IoU on validation dataset on final epoch and save poor performing examples.
    """
    def __init__(self, model, model_idx, x_val, y_val, threshold):
        super().__init__()
        self.model = model
        self.model_idx = model_idx
        self.x_val = x_val
        self.y_val = y_val
        self.threshold = threshold

    def on_train_end(self, logs=None):
        low_iou_indices = []

        for idx, (x, y) in enumerate(zip(self.x_val, self.y_val)):
            image = cv2.imread(x, cv2.IMREAD_COLOR).astype(np.float32)
            image_resized = cv2.resize(image, dsize=(256, 256))
            image_resized = image_resized / 255.0
            image_resized -= MEAN
            image_resized /= STD
            image_resized = np.expand_dims(image_resized, axis=0)

            mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            mask_resized = cv2.resize(mask, dsize=(256, 256))
            mask_flatten = mask_resized.flatten()

            y_pred = self.model.predict(image_resized)[0]
            y_pred = np.squeeze(y_pred, axis=-1)
            y_pred = y_pred > 0.5
            y_pred = y_pred.astype(np.int32)
            y_pred = y_pred.flatten()

            iou = eval_iou(y_true=mask_flatten, y_pred=y_pred)

            if iou < self.threshold:
                low_iou_indices.append(idx)
                for i in range(5):
                    save_idx = f"{idx}_{self.model_idx}_{i}"
                    img_path = os.path.join("tmp", "data", "Image", f"tmp_img_{save_idx}.png")
                    mask_path = os.path.join("tmp", "data", "Mask", f"tmp_img_{save_idx}_mask.png")
                    cv2.imwrite(img_path, image)
                    cv2.imwrite(mask_path, mask_resized)

        print(f"Examples with IoU < {self.threshold} on last epoch: {len(low_iou_indices)}")
        return
