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
    Supports Red, Yellow, AND White Onion varieties, batch crate images, and individual item segmentation.
    Includes OpenCV distance-transform fallback when YOLO/PyTorch DLLs are restricted by OS policies.
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

    def touches_frame_border(self, contour: np.ndarray, img_shape: Tuple[int, int], border_margin: int = 5) -> bool:
        """
        Only rejects contours touching the border if they are extreme outer frame edges and not part of a batch crate.
        """
        x, y, w, h = cv2.boundingRect(contour)
        img_h, img_w = img_shape[:2]

        if w >= (img_w * 0.95) or h >= (img_h * 0.95):
            return True
        return False

    def is_human_hand_or_skin(self, bgr_crop: np.ndarray) -> bool:
        """
        Detects human skin/faces/hands using YCrCb & HSV skin thresholds.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return True

        ycrcb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        lower_hsv_skin = np.array([0, 20, 60], dtype=np.uint8)
        upper_hsv_skin = np.array([25, 170, 255], dtype=np.uint8)
        hsv_skin_mask = cv2.inRange(hsv, lower_hsv_skin, upper_hsv_skin)

        total_pixels = float(bgr_crop.shape[0] * bgr_crop.shape[1])
        skin_ratio = np.count_nonzero(skin_mask) / total_pixels
        hsv_skin_ratio = np.count_nonzero(hsv_skin_mask) / total_pixels

        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        if np.mean(gray) > 180 and np.std(gray) < 40:
            return False

        if skin_ratio > 0.50 and hsv_skin_ratio > 0.50:
            print("Notice: Candidate contour rejected (Human hand/face/skin detected).")
            return True
        return False

    def verify_onion_color_match(self, bgr_crop: np.ndarray) -> bool:
        """
        Ensures >= 40% of crop pixels match Red, Yellow, or White Onion HSV bounds.
        """
        if bgr_crop is None or bgr_crop.size == 0:
            return False

        hsv = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2HSV)
        onion_mask = get_onion_color_mask(hsv)
        total_pixels = float(bgr_crop.shape[0] * bgr_crop.shape[1])
        onion_ratio = np.count_nonzero(onion_mask) / total_pixels

        if onion_ratio < 0.40:
            print(f"Notice: Candidate crop rejected (Onion color ratio too low: {onion_ratio:.2f}).")
            return False
        return True

    def is_valid_produce_contour(self, bgr_image: np.ndarray, contour: np.ndarray) -> bool:
        """
        Validates candidate contour for Red, Yellow, AND White onions:
        1. Area >= 1000 px^2.
        2. Circularity >= 0.38.
        3. Aspect Ratio 0.45 <= (W / H) <= 2.1.
        4. Human Skin Rejection.
        5. Onion Color Coverage (Red/Yellow/White) >= 40%.
        """
        if self.touches_frame_border(contour, bgr_image.shape):
            return False

        area = float(cv2.contourArea(contour))
        if area < 1000.0:
            return False

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter == 0:
            return False

        circularity = (4.0 * np.pi * area) / (perimeter ** 2)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / float(h) if h > 0 else 0.0

        if circularity < 0.38 or aspect_ratio < 0.45 or aspect_ratio > 2.1:
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
                            if cls_id == 0:
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
        if mask is None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
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
    print("Testing Updated Onion Detector with Safe Policy Fallback...")
    detector = OnionDetectorStub()

    white_img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(white_img, (200, 200), 70, (230, 240, 240), -1)

    gray = cv2.cvtColor(white_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    results = detector.detect_and_measure(white_img, mask)
    print(f"Detected valid white onion items: {len(results)}")
    assert len(results) >= 1, "White onion detection failed!"
    print("Safe Policy Fallback verified successfully.")
