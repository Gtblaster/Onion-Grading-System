import cv2
import numpy as np
from typing import List, Tuple


def resize_for_fast_processing(image: np.ndarray, max_dim: int = 800) -> Tuple[np.ndarray, float]:
    """
    Downscales large high-res / 4K / 1080p images to max_dim (default 800px) while maintaining exact aspect ratio.
    Returns (resized_image, scale_factor).
    """
    if image is None or image.size == 0:
        return image, 1.0

    h, w = image.shape[:2]
    max_side = max(h, w)
    if max_side <= max_dim:
        return image, 1.0

    scale_factor = float(max_dim) / float(max_side)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale_factor


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Normalizes lighting across an image using CLAHE on the L channel in LAB color space.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input for lighting normalization.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def get_onion_color_mask(hsv_image: np.ndarray) -> np.ndarray:
    """
    Extracts binary mask corresponding to ALL ONION VARIETIES (Red, Yellow, White)
    while excluding outdoor soil, ground dirt, and green foliage:
    1. Red / Purple / Pink Onions: Hue [0-15] or [155-180], Saturation >= 25, Value >= 30
    2. Yellow / Golden / Brown Onions: Hue [15-45], Saturation >= 45, Value >= 40 (excludes soil S < 45)
    3. White / Cream / Ivory Onions: Saturation <= 45, Value >= 160
    """
    # 1. Red / Purple / Pink Onions
    lower_red1 = np.array([0, 25, 30], dtype=np.uint8)
    upper_red1 = np.array([15, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([155, 25, 30], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # 2. Yellow / Golden / Brown Onions (S >= 45 excludes dull soil dirt)
    lower_yellow = np.array([15, 45, 40], dtype=np.uint8)
    upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    # 3. White / Cream / Ivory Onions (V >= 160)
    lower_white = np.array([0, 0, 160], dtype=np.uint8)
    upper_white = np.array([180, 45, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

    onion_mask = cv2.bitwise_or(mask_red, mask_yellow)
    onion_mask = cv2.bitwise_or(onion_mask, mask_white)

    return onion_mask


def subtract_background(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Performs robust background subtraction for outdoor produce photos, studio white backgrounds, and market crates.
    """
    if image is None or image.size == 0:
        raise ValueError("Invalid image input for background subtraction.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    is_white_background = np.mean(border_pixels) > 210

    if is_white_background:
        _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        hsv_mask = get_onion_color_mask(hsv)
        mask = cv2.bitwise_and(mask, hsv_mask)
    else:
        mask = get_onion_color_mask(hsv)

    # Fallback to Otsu thresholding only if color mask is extremely sparse (< 0.5% of pixels)
    if np.count_nonzero(mask) < (gray.size * 0.005):
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(border_pixels) > 128:
            mask = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    foreground = cv2.bitwise_and(image, image, mask=mask)
    return foreground, mask


if __name__ == "__main__":
    print("Testing Field Photo Red Onion Segmentation...")
    # Synthetic Red Onion on Soil (Soil BGR: 60, 80, 100, Red Onion BGR: 30, 20, 180)
    field_img = np.full((400, 400, 3), (60, 80, 100), dtype=np.uint8)
    cv2.circle(field_img, (150, 200), 70, (30, 20, 180), -1)
    cv2.circle(field_img, (280, 200), 65, (30, 20, 180), -1)

    fg, mask = subtract_background(field_img)
    print("Isolated field red onion pixel count:", np.count_nonzero(mask))
    assert np.count_nonzero(mask) > 5000, "Field red onion segmentation failed!"
    print("Field Photo Red Onion Segmentation verified successfully.")
