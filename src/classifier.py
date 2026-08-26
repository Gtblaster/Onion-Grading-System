import numpy as np
from typing import Dict, Any


class OnionAGMARKClassifier:
    """
    AGMARK (Indian Agricultural Produce Grading & Marking Standards) Compliant Classifier.
    Evaluates produce metrics against official Government AGMARK standards with DYNAMIC REAL-TIME DAILY MANDI PRICES:
    - Grade A: >= 55 mm (Large / Export Grade) -> 100% of Today's Live Modal Mandi Price
    - Grade B: 40 - 55 mm (Medium / Wholesale Grade) -> 65% of Today's Live Modal Mandi Price
    - Small Grade: 30 - 40 mm (Sambar / Pickle Size) -> 45% of Today's Live Modal Mandi Price
    - Reject: < 30 mm OR Severe Rot (>= 12%) -> 0.0 (Unfit for Market / No Commercial Price)
    """

    def __init__(self, pixels_per_mm: float = 2.5):
        self.pixels_per_mm = pixels_per_mm

    def classify_onion(self, pixel_area: float, max_diameter_px: float, defect_analysis: Dict[str, Any], today_mandi_modal_price: float = 40.0) -> Dict[str, Any]:
        diameter_mm = float(max_diameter_px / self.pixels_per_mm)
        area_mm2 = float(pixel_area / (self.pixels_per_mm ** 2))
        area_cm2 = float(area_mm2 / 100.0)

        rot_pct = defect_analysis.get("rot_percentage", 0.0)
        sprout_pct = defect_analysis.get("sprout_percentage", 0.0)
        color_score = defect_analysis.get("color_score", 0.8)
        defect_pct = defect_analysis.get("defect_percentage", 0.0)
        primary_defect = defect_analysis.get("primary_defect", "Sound Produce (No Defect)")

        # AGMARK Standard Alignment & Dynamic Price Calculation
        if rot_pct >= 12.0 or sprout_pct >= 8.0 or defect_pct >= 30.0 or diameter_mm < 30.0:
            grade = "Reject"
            grade_description = "Unfit for retail sale (Severe rot/sprout or undersized <30mm)"
            mandi_price_per_kg = 0.0  # Rejected produce has no commercial price
            quality_status = "Substandard / Reject"
        elif diameter_mm >= 55.0 and defect_pct < 15.0 and rot_pct < 4.0:
            grade = "Grade A"
            grade_description = "AGMARK Grade A (Large >=55mm, High Quality Export Grade)"
            mandi_price_per_kg = round(today_mandi_modal_price * 1.0, 1)
            quality_status = "Premium Export Quality"
        elif diameter_mm >= 40.0 and rot_pct < 8.0:
            grade = "Grade B"
            grade_description = "AGMARK Grade B (Medium 40-55mm, Standard Commercial Grade)"
            mandi_price_per_kg = round(today_mandi_modal_price * 0.65, 1)
            quality_status = "Standard Domestic Quality"
        else:
            grade = "Small Grade"
            grade_description = "AGMARK Small Grade (30-40mm, Sambar / Pickle Size)"
            mandi_price_per_kg = round(today_mandi_modal_price * 0.45, 1)
            quality_status = "Small Commercial Grade"

        return {
            "grade": grade,
            "quality_status": quality_status,
            "agmark_standard": grade_description,
            "metrics": {
                "diameter_mm": round(diameter_mm, 1),
                "area_cm2": round(area_cm2, 2),
                "pixel_area": round(pixel_area, 0),
                "max_diameter_px": round(max_diameter_px, 1)
            },
            "defects": {
                "color_score": color_score,
                "rot_percentage": rot_pct,
                "sprout_percentage": sprout_pct,
                "defect_percentage": defect_pct,
                "primary_defect": primary_defect
            },
            "market_estimate": {
                "estimated_price_inr_per_kg": mandi_price_per_kg,
                "currency": "INR (₹)"
            }
        }


if __name__ == "__main__":
    print("Testing AGMARK Classifier with 0.0 Price for Rejects...")
    classifier = OnionAGMARKClassifier(pixels_per_mm=2.5)

    res_r = classifier.classify_onion(2000.0, 40.0, {"rot_percentage": 15.0, "sprout_percentage": 0.0, "color_score": 0.3, "defect_percentage": 40.0, "primary_defect": "Severe Rot"}, today_mandi_modal_price=42.0)
    assert res_r["market_estimate"]["estimated_price_inr_per_kg"] == 0.0
    print("Zero price for rejected produce verified successfully.")
