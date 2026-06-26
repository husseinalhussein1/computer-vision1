"""
error_analysis.py - Stage 6: Failure Case Analysis
Runs the trained model on test images and identifies at least 5 real
failure cases without modifying the model.

Failure types examined:
  1. False Positive  — helmet predicted as no-helmet (or vice versa)
  2. False Negative  — missed detection (ground-truth box with no matching prediction)
  3. Low-confidence  — correct class but very low confidence (distant person)
  4. Occlusion       — partially hidden head, detection missed
  5. Lighting        — bright or dark scene causing misclassification
"""

import sys
import cv2
import yaml
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ─── Configuration ────────────────────────────────────────────────────────────

BEST_WEIGHTS  = "runs/train/helmet_detector/weights/best.pt"
DATA_YAML     = "dataset/data.yaml"
TEST_IMG_DIR  = Path("dataset/test/images")
TEST_LBL_DIR  = Path("dataset/test/labels")
OUTPUT_DIR    = Path("outputs/error_analysis")
CLASS_NAMES   = ["helmet", "no-helmet"]
CONF_THRESH   = 0.25
IOU_THRESH    = 0.45
MAX_FAILURES  = 8   # Collect up to 8 failure cases; display at least 5


# ─── IoU Helper ───────────────────────────────────────────────────────────────

def iou(boxA: list, boxB: list) -> float:
    """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
    xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


# ─── Label Parser ─────────────────────────────────────────────────────────────

def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> list:
    """
    Parse a YOLO label file and return absolute [cls, x1, y1, x2, y2] boxes.
    """
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)
            boxes.append([cls_id, x1, y1, x2, y2])
    return boxes


# ─── Failure Classifier ───────────────────────────────────────────────────────

def classify_failure(gt_box: list, pred_boxes: list, img_h: int, img_w: int) -> dict | None:
    """
    Given one ground-truth box, decide if there is a failure and what kind.
    Returns a dict describing the failure, or None if detection was correct.
    """
    gt_cls          = gt_box[0]
    gt_coords       = gt_box[1:]
    gt_name         = CLASS_NAMES[gt_cls] if gt_cls < len(CLASS_NAMES) else "unknown"
    gt_area         = (gt_coords[2] - gt_coords[0]) * (gt_coords[3] - gt_coords[1])
    total_area      = img_w * img_h

    best_iou    = 0.0
    best_pred   = None

    for pred in pred_boxes:
        pred_cls    = int(pred.cls[0])
        pred_coords = list(map(int, pred.xyxy[0].tolist()))
        score       = iou(gt_coords, pred_coords)
        if score > best_iou:
            best_iou  = score
            best_pred = pred

    # ── No matching prediction → False Negative ──────────────────
    if best_iou < 0.3 or best_pred is None:
        # Determine likely reason
        if gt_area / total_area < 0.005:
            reason = "Person is very small / far from camera (distant person)."
        elif gt_coords[0] < 5 or gt_coords[1] < 5 or gt_coords[2] > img_w - 5 or gt_coords[3] > img_h - 5:
            reason = "Person is at image boundary; likely partially cropped (occlusion)."
        else:
            reason = "Detection missed entirely — may be due to occlusion or unusual pose."

        return {
            "type":     "False Negative (Missed Detection)",
            "expected": gt_name,
            "actual":   "No detection",
            "reason":   reason,
            "gt_box":   gt_coords,
            "pred_box": None,
        }

    # ── Prediction exists — check class match ────────────────────
    pred_cls_id   = int(best_pred.cls[0])
    pred_cls_name = CLASS_NAMES[pred_cls_id] if pred_cls_id < len(CLASS_NAMES) else "unknown"
    pred_conf     = float(best_pred.conf[0])

    if pred_cls_name != gt_name:
        # Misclassification
        if gt_name == "no-helmet" and pred_cls_name == "helmet":
            reason = "Worker without helmet classified as wearing one — possible partial occlusion or similar head shape."
        elif gt_name == "helmet" and pred_cls_name == "no-helmet":
            reason = "Helmet misclassified — may be unusual helmet color, angle, or heavy shadow."
        else:
            reason = "Class mismatch due to ambiguous appearance."

        return {
            "type":     "False Positive (Wrong Class)",
            "expected": gt_name,
            "actual":   pred_cls_name,
            "reason":   reason,
            "gt_box":   gt_coords,
            "pred_box": list(map(int, best_pred.xyxy[0].tolist())),
        }

    # ── Correct class but very low confidence ────────────────────
    if pred_conf < 0.35:
        return {
            "type":     "Low Confidence Detection",
            "expected": gt_name,
            "actual":   f"{pred_cls_name} (conf={pred_conf:.2f})",
            "reason":   "Model is uncertain — likely due to poor lighting, motion blur, or small person size.",
            "gt_box":   gt_coords,
            "pred_box": list(map(int, best_pred.xyxy[0].tolist())),
        }

    return None   # Correct detection; not a failure


# ─── Visualization ────────────────────────────────────────────────────────────

def draw_failure(image: np.ndarray, failure: dict) -> np.ndarray:
    """Annotate an image with GT box (green) and predicted box (red/blue)."""
    out = image.copy()

    # Ground-truth box in green
    if failure["gt_box"]:
        x1, y1, x2, y2 = failure["gt_box"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(out, f"GT: {failure['expected']}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

    # Predicted box in red
    if failure["pred_box"]:
        px1, py1, px2, py2 = failure["pred_box"]
        cv2.rectangle(out, (px1, py1), (px2, py2), (0, 0, 220), 2)
        cv2.putText(out, f"Pred: {failure['actual']}", (px1, py2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)

    # Failure type label at top
    label = f"[{failure['type']}]"
    cv2.putText(out, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

    return out


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_error_analysis(weights_path: str = BEST_WEIGHTS) -> None:
    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights.resolve()}")
        sys.exit(1)

    if not TEST_IMG_DIR.exists():
        print(f"[ERROR] Test images directory not found: {TEST_IMG_DIR.resolve()}")
        sys.exit(1)

    print("=" * 60)
    print("   Safety Helmet Detector - Error Analysis")
    print("=" * 60)

    model     = YOLO(weights_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(TEST_IMG_DIR.glob("*.*"))
    failures    = []

    for img_path in image_files:
        if len(failures) >= MAX_FAILURES:
            break

        image = cv2.imread(str(img_path))
        if image is None:
            continue

        img_h, img_w = image.shape[:2]
        label_path   = TEST_LBL_DIR / (img_path.stem + ".txt")
        gt_boxes     = load_ground_truth(label_path, img_w, img_h)

        if not gt_boxes:
            continue

        results   = model.predict(source=str(img_path), conf=CONF_THRESH,
                                  iou=IOU_THRESH, verbose=False)
        pred_boxes = results[0].boxes

        for gt_box in gt_boxes:
            if len(failures) >= MAX_FAILURES:
                break
            failure = classify_failure(gt_box, pred_boxes, img_h, img_w)
            if failure is not None:
                failure["image_path"] = str(img_path)
                failures.append(failure)

    if not failures:
        print("\n[INFO] No failures detected in the sampled test images.")
        print("       The model performed correctly on all sampled cases.")
        return

    # Print and save each failure
    print(f"\nFound {len(failures)} failure case(s). Displaying details:\n")

    for i, failure in enumerate(failures, start=1):
        print(f"  Case {i}: {failure['type']}")
        print(f"    Image    : {failure['image_path']}")
        print(f"    Expected : {failure['expected']}")
        print(f"    Actual   : {failure['actual']}")
        print(f"    Reason   : {failure['reason']}")
        print()

        image      = cv2.imread(failure["image_path"])
        annotated  = draw_failure(image, failure)
        out_path   = OUTPUT_DIR / f"failure_{i:02d}_{Path(failure['image_path']).name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"    Saved to : {out_path}")
        print()

    print("=" * 60)
    print(f"  Error analysis complete. {len(failures)} failure images saved to:")
    print(f"  {OUTPUT_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze failure cases of the Safety Helmet Detector")
    parser.add_argument("--weights", type=str, default=BEST_WEIGHTS, help="Path to model weights")
    args = parser.parse_args()

    run_error_analysis(weights_path=args.weights)
