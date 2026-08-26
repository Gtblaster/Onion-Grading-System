import cv2
import numpy as np
from typing import List, Tuple


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Normalizes lighting across an image using CLAHE on the L channel in LAB color space.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input for lighting normalization.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def get_onion_color_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    Extracts binary mask corresponding to ALL ONION VARIETIES across all lighting conditions
    (Webcams, Phone Cameras, Room Light, Studio Light, Web Images):
    1. Red / Pink / Purple Onions: Hue [0-28] or [135-180], Saturation >= 10, Value >= 20
    2. Yellow / Golden / Brown Onions: Hue [10-50], Saturation >= 10, Value >= 20
    3. White / Cream / Ivory Onions: Saturation <= 65, Value >= 45
    """
    # Red / Pink / Purple
    lower_red1 = np.array([0, 10, 20], dtype=np.uint8)
    upper_red1 = np.array([28, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([135, 10, 20], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Yellow / Golden / Brown
    lower_yellow = np.array([10, 10, 20], dtype=np.uint8)
    upper_yellow = np.array([50, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # White / Cream / Ivory (S <= 65, V >= 45)
    lower_white = np.array([0, 0, 45], dtype=np.uint8)
    upper_white = np.array([180, 65, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    onion_mask = cv2.bitwise_or(mask_red, mask_yellow)
    onion_mask = cv2.bitwise_or(onion_mask, mask_white)

    return onion_mask


def subtract_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs robust background subtraction for natural produce photos, webcam feeds, and studio white/light backgrounds.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input for background subtraction.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    is_white_background = np.mean(border_pixels) > 200

    if is_white_background:
        _, mask = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY_INV)
        hsv_mask = get_onion_color_mask(hsv)
        mask = cv2.bitwise_and(mask, hsv_mask)
    else:
        mask = get_onion_color_mask(hsv)

    # Fallback to Otsu thresholding if color mask yields less than 2% foreground
    if np.count_nonzero(mask) < (gray.size * 0.02):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(border_pixels) > 128:
            mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    foreground = cv2.bitwise_and(image, image, mask=mask)
    return foreground, mask


if __name__ == "__main__":
    print("Testing Webcam & Broad Variety HSV Segmentation...")
    # Low saturation webcam dim onion image test
    webcam_dim = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(webcam_dim, (200, 200), 80, (40, 50, 110), -1)

    fg, mask = subtract_background(webcam_dim)
    print("Isolated dim webcam onion pixel count:", np.count_nonzero(mask))
    assert np.count_nonzero(mask) > 1000, "Dim webcam onion segmentation failed!"
    print("Webcam & Broad Variety HSV Segmentation verified successfully.")
