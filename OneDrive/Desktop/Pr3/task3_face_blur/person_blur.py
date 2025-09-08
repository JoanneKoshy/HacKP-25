# person_blur.py
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO segmentation model (fast)
model = YOLO("yolov8n-seg.pt")  # auto-downloads if needed


def blur_persons(image_rgb, blur_strength: int = 35, double_blur: bool = True):
    """
    Detect persons in an image and blur only those regions.
    :param image_rgb: input image (numpy array in RGB format)
    :param blur_strength: Gaussian blur kernel size (higher = stronger blur)
    :param double_blur: if True, applies Gaussian blur twice for stronger anonymization
    :return: blurred image (RGB numpy array)
    """
    h, w = image_rgb.shape[:2]

    # Run YOLO segmentation
    results = model.predict(image_rgb, conf=0.25, iou=0.45, verbose=False)

    # Take first result (single image)
    res = results[0]

    # Copy of original image
    output = image_rgb.copy()

    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()  # (N, H, W)
        classes = res.boxes.cls.cpu().numpy().astype(int)

        for i, cls in enumerate(classes):
            label = res.names[int(cls)]
            if label.lower() == "person":
                mask = masks[i] > 0.5  # boolean mask
                # Ensure mask matches image size
                if mask.shape != (h, w):
                    mask = cv2.resize(mask.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)

                # Blur the full image first
                k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
                blurred = cv2.GaussianBlur(output, (k, k), 0)

                # Apply double blur if desired
                if double_blur:
                    blurred = cv2.GaussianBlur(blurred, (k, k), 0)

                # Apply blur only on person mask
                mask3 = np.stack([mask] * 3, axis=-1)
                output = np.where(mask3, blurred, output)

    return output
