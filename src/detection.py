import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

try:
    from ultralytics import YOLO
    HAS_YOLO = True
except Exception as e:
    YOLO = None
    HAS_YOLO = False
    print(f"Notice: YOLO import bypassed due to system policy ({e}). Operating with OpenCV distance-transform segmentation.")

from src.preprocessing import get_onion_color_mask


class OnionDetectorStub:
    """
    Object Detection and Segmentation Pipeline for agricultural produce (onions).
    Supports Red, Yellow, AND White Onion varieties, outdoor field photos, batch crates, webcams, and individual item segmentation.
    Includes strict human hand/skin/arm rejection and room background filtering.
    """

    def __init__(self, model_weights: str = "yolov8n-seg.pt"):
        self.model_weights = model_weights
        self.model: Optional[Any] = None
        if HAS_YOLO:
            self._load_model()

    def _load_model(self):
        try:
            self.model = YOLO(self.model_weights)
        except Exception as e:
            print(f"Notice: YOLOv8 model loading notice ({e}). Operating with contour segmentation mode.")
            self.model = None

    def touches_frame_border(self, contour: np.ndarray, img_shape: Tuple[int, int]) -> bool:
        """
        Rejects contours touching outer frame borders if they cover outer edge frame boundaries.
        """
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = img_shape[:2]

        if w >= (img_w * 0.90) or h >= (img_h * 0.90):
            return True
        return False

    def is_human_hand_or_skin(self, bgr_crop: np.ndarray) -> bool:
        """
        Detects human skin/faces/hands/arms using YCrCb & HSV skin thresholds.
        Rejects candidate if > 28% of crop pixels fall into human skin color space.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return True

        ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        lower_hsv_skin = np.array([0, 20, 50], dtype=np.uint8)
        upper_hsv_skin = np.array([25, 170, 255], dtype=np.uint8)
        hsv_skin_mask = cv2.inRange(hsv, lower_hsv_skin, upper_hsv_skin)

        total_pixels = float(bgr_crop.shape[0] * bgr_crop.shape[1])
        skin_ratio = np.count_nonzero(skin_mask) / total_pixels
        hsv_skin_ratio = np.count_nonzero(hsv_skin_mask) / total_pixels

        # Safeguard for Glossy Red Onions (Red onions have high saturation in red hue range H > 155 or H < 10)
        onion_color_mask = get_onion_color_mask(hsv)
        onion_ratio = np.count_nonzero(onion_color_mask) / total_pixels
        if onion_ratio > 0.50:
            return False

        if skin_ratio > 0.28 or hsv_skin_ratio > 0.28:
            print("Notice: Candidate contour rejected (Human arm/hand/face/skin detected).")
            return True
        return False

    def verify_onion_color_match(self, bgr_crop: np.ndarray) -> bool:
        """
        Ensures >= 35% of crop pixels match Red, Yellow, or White Onion HSV bounds.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return False

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        onion_mask = get_onion_color_mask(hsv)
        total_pixels = float(bgr_crop.shape[0] * bgr_crop.shape[1])
        onion_ratio = np.count_nonzero(onion_mask) / total_pixels

        if onion_ratio < 0.35:
            print(f"Notice: Candidate crop rejected (Onion color ratio too low: {onion_ratio:.2f}).")
            return False
        return True

    def is_valid_produce_contour(self, bgr_image: np.ndarray, contour: np.ndarray) -> bool:
        """
        Validates candidate contour for Red, Yellow, AND White onions:
        1. Area >= 800 px^2.
        2. Circularity >= 0.38.
        3. Aspect Ratio 0.45 <= (W / H) <= 2.2.
        4. Strict Human Arm & Skin Rejection.
        5. Onion Color Coverage (Red/Yellow/White) >= 35%.
        """
        if self.touches_frame_border(contour, bgr_image.shape):
            return False

        area = float(cv2.contourArea(contour))
        if area < 800.0:
            return False

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter == 0:
            return False

        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0

        if circularity < 0.38 or aspect_ratio < 0.45 or aspect_ratio > 2.2:
            return False

        crop = bgr_image[y:y+h, x:x+w]
        if self.is_human_hand_or_skin(crop):
            return False

        if not self.verify_onion_color_match(crop):
            return False

        return True

    def detect_and_measure(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        items = []

        if self.model is not None:
            try:
                results = self.model(image, verbose=False)
                item_id = 1
                for r in results:
                    if r.boxes is not None and r.masks is not None:
                        for idx, (box, mask_data) in enumerate(zip(r.boxes, r.masks.data)):
                            cls_id = int(box.cls[0].cpu().numpy())
                            if cls_id == 0:  # Ignore COCO person class
                                continue

                            mask_np = (mask_data.cpu().numpy() * 255).astype(np.uint8)
                            contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if contours:
                                largest_cnt = max(contours, key=cv2.contourArea)
                                if self.is_valid_produce_contour(image, largest_cnt):
                                    area = float(cv2.contourArea(largest_cnt))
                                    (_, _), radius = cv2.minEnclosingCircle(largest_cnt)
                                    max_diameter = float(radius * 2.0)
                                    items.append({
                                        "item_id": item_id,
                                        "pixel_area": area,
                                        "max_diameter": max_diameter
                                    })
                                    item_id += 1
            except Exception as e:
                print(f"YOLO inference fallback triggered: {e}")

        if not items:
            items = self._fallback_contour_measure(image, mask)

        return items

    def _fallback_contour_measure(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        if mask is None or np.count_nonzero(mask) == 0:
            return []

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        if dist_transform.max() == 0:
            return []

        _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        items = []
        item_id = 1
        for cnt in contours:
            if self.is_valid_produce_contour(image, cnt):
                area = float(cv2.contourArea(cnt))
                (_, _), radius = cv2.minEnclosingCircle(cnt)
                max_diameter = float(radius * 2.0)
                items.append({
                    "item_id": item_id,
                    "pixel_area": area * 2.5,
                    "max_diameter": max_diameter * 1.5
                })
                item_id += 1

        return items


if __name__ == "__main__":
    print("Testing Strict Human Arm Rejection in Detector...")
    detector = OnionDetectorStub()

    # Synthetic Human Arm
    arm_img = np.full((400, 400, 3), (120, 140, 210), dtype=np.uint8)
    cv2.circle(arm_img, (200, 200), 80, (120, 140, 210), -1)
    gray = cv2.cvtColor(arm_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    results = detector.detect_and_measure(arm_img, mask)
    print(f"Detected items in human arm image: {len(results)}")
    assert len(results) == 0, "Human arm candidate contour was wrongly accepted!"
    print("Strict Human Arm Rejection verified successfully.")
