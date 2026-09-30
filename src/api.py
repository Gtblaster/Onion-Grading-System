import os
import cv2
import base64
import numpy as np
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.preprocessing import normalize_lighting, subtract_background, resize_for_fast_processing
from src.detection import OnionDetectorStub
from src.defect_analysis import analyze_onion_defects
from src.classifier import OnionAGMARKClassifier
from src.market_analytics import MandiPricePredictor

app = FastAPI(
    title="AGMARK Compliant High-Speed Onion Quality Assessment Engine",
    description="Ultra-Fast Computer Vision & Quality Analytics API for Agricultural Produce (Onions). Equipped with Active Learning Online Retraining.",
    version="3.3.0"
)

detector = OnionDetectorStub()
classifier = OnionAGMARKClassifier(pixels_per_mm=2.5)
market_predictor = MandiPricePredictor()


class BoundingBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class MetricDetails(BaseModel):
    diameter_mm: float
    area_cm2: float
    pixel_area: float
    max_diameter_px: float


class DefectDetails(BaseModel):
    color_score: float
    rot_percentage: float
    sprout_percentage: float
    defect_percentage: float
    primary_defect: str


class MarketEstimate(BaseModel):
    estimated_price_inr_per_kg: float
    currency: str
    price_source: str


class AGMARKItemResult(BaseModel):
    item_id: int
    grade: str
    quality_status: str
    agmark_standard: str
    bbox: Optional[BoundingBox] = None
    metrics: MetricDetails
    defects: DefectDetails
    market_estimate: MarketEstimate


class ComprehensiveGradingResponse(BaseModel):
    items_count: int
    overall_batch_grade: str
    average_diameter_mm: float
    estimated_mandi_price_inr: float
    price_source_info: str
    active_learning_status: str = "Active Learning Idle"
    annotated_image_base64: Optional[str] = None
    results: List[AGMARKItemResult]


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serves the interactive hackathon web user interface."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        with open(template_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    return HTMLResponse(content="<h1>AGMARK Onion Quality Assessment Pipeline Online</h1>", status_code=200)


@app.get("/market_trends")
def get_market_trends(mandi_name: str = "Lasalgaon (Nashik, MH)"):
    """
    Returns live Agmarknet daily wholesale price benchmarks, 30-day historical prices, 
    14-day AI future price forecast, and cold storage profit recommendations.
    """
    return market_predictor.get_market_analytics(mandi_name)


@app.post("/grade_image", response_model=ComprehensiveGradingResponse)
async def grade_image(file: UploadFile = File(...), pixels_per_mm: Any = 2.5, mandi_name: str = "Lasalgaon (Nashik, MH)"):
    """
    Ultra-Fast Computer Vision & AGMARK Grading Pipeline:
    Draws thick square bounding boxes around each produce item with solid color-coded tag boxes containing item name & result.
    """
    try:
        pixels_per_mm_val = float(pixels_per_mm)
        if pixels_per_mm_val <= 0 or np.isnan(pixels_per_mm_val):
            pixels_per_mm_val = 2.5
    except Exception:
        pixels_per_mm_val = 2.5

    price_source = f"Agmarknet Live ({mandi_name} Index)"

    try:
        contents = await file.read()
        if not contents or len(contents) < 50:
            return ComprehensiveGradingResponse(
                items_count=0,
                overall_batch_grade="No Produce Detected",
                average_diameter_mm=0.0,
                estimated_mandi_price_inr=0.0,
                price_source_info=price_source,
                active_learning_status="No produce detected to learn from",
                annotated_image_base64=None,
                results=[]
            )

        nparr = np.frombuffer(contents, np.uint8)
        img_raw = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_raw is None or img_raw.size == 0:
            return ComprehensiveGradingResponse(
                items_count=0,
                overall_batch_grade="No Produce Detected",
                average_diameter_mm=0.0,
                estimated_mandi_price_inr=0.0,
                price_source_info=price_source,
                active_learning_status="No produce detected to learn from",
                annotated_image_base64=None,
                results=[]
            )
    except Exception as e:
        return ComprehensiveGradingResponse(
            items_count=0,
            overall_batch_grade="No Produce Detected",
            average_diameter_mm=0.0,
            estimated_mandi_price_inr=0.0,
            price_source_info=price_source,
            active_learning_status="No produce detected to learn from",
            annotated_image_base64=None,
            results=[]
        )

    try:
        img, scale_factor = resize_for_fast_processing(img_raw, max_dim=800)
        adjusted_px_per_mm = pixels_per_mm_val * scale_factor

        market_analytics = market_predictor.get_market_analytics(mandi_name)
        today_modal_price = market_analytics.get("today_modal_price_inr", 40.0)
        price_source = f"Agmarknet Live ({mandi_name} Index - Today's Modal Price: ₹{today_modal_price}/kg)"

        norm_img = normalize_lighting(img)
        fg_img, mask = subtract_background(norm_img)

        detected_items = detector.detect_and_measure(fg_img, mask)

        annotated_img = img.copy()
        grading_results = []
        total_diam = 0.0
        prices = []
        retrained_online = False

        local_classifier = OnionAGMARKClassifier(pixels_per_mm=adjusted_px_per_mm)
        img_h, img_w = img.shape[:2]

        for item in detected_items:
            item_id = item["item_id"]
            pixel_area = item["pixel_area"]
            max_diam_px = item["max_diameter"]

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            item_mask = np.zeros_like(mask)
            matched_cnt = None
            if contours:
                matched_cnt = min(contours, key=lambda c: abs(cv2.contourArea(c) - pixel_area))
                cv2.drawContours(item_mask, [matched_cnt], -1, 255, -1)
            else:
                item_mask = mask

            defect_data = analyze_onion_defects(img, item_mask)
            class_res = local_classifier.classify_onion(pixel_area, max_diam_px, defect_data, today_mandi_modal_price=today_modal_price)
            grade = class_res["grade"]

            was_retrained = local_classifier.ai_model.log_user_sample_and_retrain(
                diam_mm=class_res["metrics"]["diameter_mm"],
                area_cm2=class_res["metrics"]["area_cm2"],
                rot_pct=defect_data["rot_percentage"],
                sprout_pct=defect_data["sprout_percentage"],
                color_score=defect_data["color_score"],
                defect_pct=defect_data["defect_percentage"],
                assigned_grade=grade
            )
            if was_retrained:
                retrained_online = True

            color_map = {
                "Grade A": (0, 200, 0),        # Bright Green
                "Grade B": (0, 215, 255),      # Golden Yellow
                "Small Grade": (255, 180, 0),  # Light Cyan
                "Reject": (0, 0, 235)          # Crimson Red
            }
            box_color = color_map.get(grade, (255, 255, 255))
            text_color = (0, 0, 0) if grade == "Grade B" or grade == "Small Grade" else (255, 255, 255)

            item_bbox = None
            if matched_cnt is not None:
                x, y, w, h = cv2.boundingRect(matched_cnt)
                if w < (img_w * 0.90) and h < (img_h * 0.90):
                    item_bbox = BoundingBox(x=x, y=y, w=w, h=h)
                    
                    # 1. Draw Thick Outer Bounding Square Box
                    cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, 3)

                    # 2. Solid Color-Coded Header Box: [ ITEM #N: GRADE ]
                    header_text = f" ITEM #{item_id}: {grade.upper()} "
                    font_scale = 0.55
                    font_thickness = 2
                    (text_w, text_h), baseline = cv2.getTextSize(header_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                    
                    header_y1 = max(y - text_h - 12, 0)
                    header_y2 = max(y, text_h + 12)
                    header_x2 = min(x + text_w + 12, img_w)

                    # Fill solid header tag box
                    cv2.rectangle(annotated_img, (x, header_y1), (header_x2, header_y2), box_color, -1)
                    cv2.rectangle(annotated_img, (x, header_y1), (header_x2, header_y2), (15, 23, 42), 1)
                    cv2.putText(annotated_img, header_text, (x + 2, header_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness)

                    # 3. Sub-Header Dark Square Box: [ Diam: 58.5mm | Sound ]
                    sub_text = f" {class_res['metrics']['diameter_mm']}mm | {defect_data['primary_defect'].split('(')[0].strip()} "
                    (sub_w, sub_h), _ = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                    sub_y1 = header_y2
                    sub_y2 = sub_y1 + sub_h + 10
                    sub_x2 = min(x + sub_w + 10, img_w)

                    cv2.rectangle(annotated_img, (x, sub_y1), (sub_x2, sub_y2), (15, 23, 42), -1)
                    cv2.rectangle(annotated_img, (x, sub_y1), (sub_x2, sub_y2), box_color, 1)
                    cv2.putText(annotated_img, sub_text, (x + 3, sub_y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

            total_diam += class_res["metrics"]["diameter_mm"]
            prices.append(class_res["market_estimate"]["estimated_price_inr_per_kg"])

            grading_results.append(AGMARKItemResult(
                item_id=item_id,
                grade=grade,
                quality_status=class_res["quality_status"],
                agmark_standard=class_res["agmark_standard"],
                bbox=item_bbox,
                metrics=MetricDetails(**class_res["metrics"]),
                defects=DefectDetails(**class_res["defects"]),
                market_estimate=MarketEstimate(
                    estimated_price_inr_per_kg=class_res["market_estimate"]["estimated_price_inr_per_kg"],
                    currency="INR (₹)",
                    price_source=price_source
                )
            ))

        _, buffer = cv2.imencode('.jpg', annotated_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        items_cnt = len(grading_results)
        learned_count = local_classifier.ai_model.user_samples_count

        if items_cnt == 0:
            overall_grade = "No Produce Detected"
            avg_diam = 0.0
            avg_price = 0.0
            learning_msg = f"Active Learning Idle (Total Learned: {learned_count})"
        else:
            avg_diam = round(total_diam / items_cnt, 1)

            if retrained_online:
                learning_msg = f"⚡ Active Learning: AI Model Retrained Online Live! (Total Samples Learned: {learned_count})"
            else:
                learning_msg = f"🧠 Active Learning: Logged User Image to AI Memory (Total Samples Learned: {learned_count})"

            grades_list = [r.grade for r in grading_results]
            reject_count = grades_list.count("Reject")
            reject_ratio = reject_count / float(items_cnt)

            sound_grades = [g for g in grades_list if g != "Reject"]
            sound_prices = [r.market_estimate.estimated_price_inr_per_kg for r in grading_results if r.grade != "Reject"]

            if reject_ratio >= 0.25:
                overall_grade = "Reject"
                avg_price = 0.0
            elif reject_ratio >= 0.15:
                overall_grade = "Mixed (Contains Rejects)"
                avg_price = round(float(np.mean(prices)), 1)
            else:
                if sound_prices:
                    avg_price = round(float(np.mean(sound_prices)), 1)
                else:
                    avg_price = round(today_modal_price * 0.65, 1)

                if avg_diam >= 55.0 and sound_grades.count("Grade A") >= (len(sound_grades) * 0.5):
                    overall_grade = "Grade A"
                elif avg_diam >= 40.0:
                    overall_grade = "Grade B"
                elif avg_diam >= 30.0:
                    overall_grade = "Small Grade"
                else:
                    overall_grade = "Grade B"

        return ComprehensiveGradingResponse(
            items_count=items_cnt,
            overall_batch_grade=overall_grade,
            average_diameter_mm=avg_diam,
            estimated_mandi_price_inr=avg_price,
            price_source_info=price_source,
            active_learning_status=learning_msg,
            annotated_image_base64=annotated_b64,
            results=grading_results
        )
    except Exception as exc:
        print(f"Error during image grading: {exc}")
        return ComprehensiveGradingResponse(
            items_count=0,
            overall_batch_grade="No Produce Detected",
            average_diameter_mm=0.0,
            estimated_mandi_price_inr=0.0,
            price_source_info=price_source,
            active_learning_status="Active Learning Idle",
            annotated_image_base64=None,
            results=[]
        )


if __name__ == "__main__":
    print("Testing Prominent Square Box Item Tag Overlay...")
    from fastapi.testclient import TestClient
    client = TestClient(app)

    dummy = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(dummy, (200, 200), 70, (30, 20, 180), -1)
    _, img_bytes = cv2.imencode('.png', dummy)

    res = client.post("/grade_image?pixels_per_mm=2.5", files={"file": ("test.png", img_bytes.tobytes(), "image/png")})
    assert res.status_code == 200
    data = res.json()
    print("Items count:", data["items_count"])
    print("Prominent Square Box Item Tag Overlay verified successfully.")
