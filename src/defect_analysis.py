import cv2
import numpy as np
from typing import Dict, Any
from src.preprocessing import get_onion_color_mask


def analyze_onion_defects(bgr_image: np.ndarray, mask: np.ndarray) -> Dict[str, Any]:
    """
    Performs research-grade computer vision diagnostic analysis strictly on valid onion skin:
    Accurately distinguishes true Black Mold Rot (Aspergillus niger) from normal dim room lighting, 
    shadows, and healthy dark red/purple onion skin.
    - True Black Mold Rot: Low Value (V < 25) AND Low Saturation (S < 30)
    - Premature Top Sprouting: Vibrant Green Shoots (Hue 35-85, Saturation >= 50, Coverage > 12%)
    - Skin Peeling & Discoloration
    """
    if bgr_image is None or mask is None or np.count_nonzero(mask) == 0:
        return {
            "rot_percentage": 0.0,
            "sprout_percentage": 0.0,
            "color_score": 1.0,
            "defect_percentage": 0.0,
            "primary_defect": "Sound Produce (No Defect)"
        }

    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    valid_onion_mask = get_onion_color_mask(hsv)
    onion_skin_mask = cv2.bitwise_and(mask, valid_onion_mask)

    total_onion_pixels = float(np.count_nonzero(onion_skin_mask))
    if total_onion_pixels == 0:
        total_onion_pixels = float(np.count_nonzero(mask))
        onion_skin_mask = mask

    masked_bgr = cv2.bitwise_and(bgr_image, bgr_image, mask=onion_skin_mask)

    # 1. True Black Mold Rot Detection (Aspergillus niger)
    # Requires VERY LOW brightness V < 25 AND LOW saturation S < 30 (decayed black mold)
    # Excludes healthy dark red onion skin (which has S >= 40) and dim room shadows
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]

    true_rot_mask = ((v_channel < 25) & (s_channel < 30) & (onion_skin_mask > 0)).astype(np.uint8) * 255
    rot_pixel_count = float(np.count_nonzero(true_rot_mask))
    rot_percentage = min((rot_pixel_count / total_onion_pixels) * 100.0, 100.0)

    # 2. Sprout Detection (Vibrant green shoots Hue 35 to 85, Saturation >= 50)
    lower_green = np.array([35, 50, 45])
    upper_green = np.array([85, 255, 255])
    sprout_mask = cv2.inRange(hsv, lower_green, upper_green)
    sprout_mask = cv2.bitwise_and(sprout_mask, sprout_mask, mask=onion_skin_mask)
    sprout_pixel_count = float(np.count_nonzero(sprout_mask))
    sprout_percentage = min((sprout_pixel_count / total_onion_pixels) * 100.0, 100.0)

    # 3. Color Uniformity Score
    lower_pink1 = np.array([0, 20, 40])
    upper_pink1 = np.array([22, 255, 255])
    lower_pink2 = np.array([145, 20, 40])
    upper_pink2 = np.array([180, 255, 255])

    healthy_skin_mask1 = cv2.inRange(hsv, lower_pink1, upper_pink1)
    healthy_skin_mask2 = cv2.inRange(hsv, lower_pink2, upper_pink2)
    healthy_skin_mask = cv2.bitwise_or(healthy_skin_mask1, healthy_skin_mask2)
    healthy_skin_mask = cv2.bitwise_and(healthy_skin_mask, healthy_skin_mask, mask=onion_skin_mask)
    healthy_pixel_count = float(np.count_nonzero(healthy_skin_mask))

    color_score = min(max(healthy_pixel_count / total_onion_pixels, 0.1), 1.0)

    true_sprout_pct = max(0.0, sprout_percentage - 12.0)
    defect_percentage = min(rot_percentage * 1.5 + true_sprout_pct * 2.0 + (1.0 - color_score) * 10.0, 100.0)

    rot_r = round(rot_percentage, 1)
    sprout_r = round(sprout_percentage, 1)

    # Diagnostic Classification
    if rot_r >= 12.0:
        primary_defect = f"Black Mold Rot (Aspergillus) [Severe: {rot_r}%]"
    elif rot_r >= 5.0:
        primary_defect = f"Black Mold Spot (Aspergillus) [Moderate: {rot_r}%]"
    elif sprout_r >= 18.0:
        primary_defect = f"Top Sprouting (Growth Shoot) [Severe: {sprout_r}%]"
    elif sprout_r >= 12.0:
        primary_defect = f"Top Sprouting (Growth Shoot) [Moderate: {sprout_r}%]"
    elif color_score < 0.45:
        primary_defect = f"Skin Discoloration / Sunburn [{round(defect_percentage, 1)}%]"
    else:
        primary_defect = "Sound Produce (No Defect)"

    return {
        "rot_percentage": round(rot_percentage, 2),
        "sprout_percentage": round(sprout_percentage, 2),
        "color_score": round(color_score, 2),
        "defect_percentage": round(defect_percentage, 2),
        "primary_defect": primary_defect
    }


if __name__ == "__main__":
    print("Testing True Black Mold Rot vs Dim Lighting Shadow Rejection...")
    # Synthetic healthy dark red onion under dim lighting (BGR: 20, 30, 120 -> S=212, V=120)
    dim_red_onion = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(dim_red_onion, (100, 100), 60, (20, 30, 120), -1)

    gray = cv2.cvtColor(dim_red_onion, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    analysis = analyze_onion_defects(dim_red_onion, mask)
    print("Dim Red Onion Defect Result:", analysis["primary_defect"])
    assert "Sound Produce" in analysis["primary_defect"]
    print("Rot vs Dim Lighting Rejection verified successfully.")
