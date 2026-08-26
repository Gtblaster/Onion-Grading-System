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
    Extracts binary mask corresponding to ALL ONION VARIETIES (Red, Yellow, Brown, White):
    1. Red / Pink / Purple Onions: Hue [0-22] or [145-180], Saturation >= 25
    2. Yellow / Golden / Brown Onions: Hue [12-45], Saturation >= 25
    3. White Onions: Low Saturation (S <= 45), Bright Value (V >= 80)
    """
    # Red / Pink / Purple
    lower_red1 = np.array([0, 25, 35], dtype=np.uint8)
    upper_red1 = np.array([22, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([145, 25, 35], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Yellow / Golden / Brown
    lower_yellow = np.array([12, 25, 35], dtype=np.uint8)
    upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # White / Cream (S <= 45, V >= 80)
    lower_white = np.array([0, 0, 80], dtype=np.uint8)
    upper_white = np.array([180, 45, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    onion_mask = cv2.bitwise_or(mask_red, mask_yellow)
    onion_mask = cv2.bitwise_or(onion_mask, mask_white)

    return onion_mask


def subtract_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs robust background subtraction for both natural produce photos and studio white/light backgrounds.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input for background subtraction.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Check if image has a studio white background (edges are pure white/light)
    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    is_white_background = np.mean(border_pixels) > 210

    if is_white_background:
        # For studio white backgrounds: Otsu thresholding isolates foreground object
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        # Combine with HSV color bounds
        hsv_mask = get_onion_color_mask(hsv)
        mask = cv2.bitwise_and(mask, hsv_mask)
    else:
        mask = get_onion_color_mask(hsv)

    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    foreground = cv2.bitwise_and(image, image, mask=mask)
    return foreground, mask


if __name__ == "__main__":
    print("Testing Studio White Background & Yellow Onion Segmentation...")
    # Synthetic studio image: White background (255) with Yellow Onion (BGR: 30, 100, 210)
    studio_img = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.circle(studio_img, (200, 200), 80, (30, 100, 210), -1)

    fg, mask = subtract_background(studio_img)
    print("Isolated yellow onion pixel count:", np.count_nonzero(mask))
    assert np.count_nonzero(mask) > 1000, "Studio yellow onion segmentation failed!"
    print("Studio White Background Subtraction verified successfully.")
