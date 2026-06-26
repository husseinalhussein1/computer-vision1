"""
infer_image.py - Stage 5: Image Inference
Runs the trained model on a new image, draws bounding boxes with
class names and confidence scores, then saves the result to outputs/.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ─── Configuration ────────────────────────────────────────────────────────────

BEST_WEIGHTS = "runs/train/helmet_detector/weights/best.pt"
OUTPUT_DIR   = Path("outputs")
CLASS_NAMES  = ["helmet", "no-helmet"]

# Color per class: helmet=green, no-helmet=red
CLASS_COLORS = {
    "helmet":    (0, 200, 0),
    "no-helmet": (0, 0, 220),
}


# ─── Drawing Helper ───────────────────────────────────────────────────────────

def draw_detections(image: np.ndarray, boxes, class_names: list) -> np.ndarray:
    """
    Draw bounding boxes, class names, and confidence scores on the image.

    Args:
        image:       BGR image (numpy array).
        boxes:       Ultralytics Boxes object from model results.
        class_names: List of class name strings.

    Returns:
        Annotated BGR image.
    """
    annotated = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        cls_name   = class_names[cls_id] if cls_id < len(class_names) else "unknown"
        color      = CLASS_COLORS.get(cls_name, (255, 255, 0))
        label      = f"{cls_name}  {confidence:.2f}"

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)

        # Label text
        cv2.putText(
            annotated, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2,
        )

    return annotated


# ─── Inference Function ───────────────────────────────────────────────────────

def run_inference(image_path: str, weights_path: str = BEST_WEIGHTS, conf: float = 0.25) -> str:
    """
    Run detection on a single image and save the annotated result.

    Args:
        image_path:   Path to the input image file.
        weights_path: Path to trained model weights (.pt).
        conf:         Confidence threshold for detections.

    Returns:
        Path to the saved output image.
    """
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path.resolve()}")
        sys.exit(1)

    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights.resolve()}")
        print("        Train the model first by running train.py")
        sys.exit(1)

    print(f"[INFO] Loading model from {weights_path}")
    model = YOLO(weights_path)

    print(f"[INFO] Running inference on: {image_path}")
    results = model.predict(source=str(img_path), conf=conf, verbose=False)

    # Read original image
    image = cv2.imread(str(img_path))
    if image is None:
        print(f"[ERROR] Could not read image: {img_path}")
        sys.exit(1)

    # Draw detections
    annotated = draw_detections(image, results[0].boxes, CLASS_NAMES)

    # Save output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_filename = f"detected_{img_path.name}"
    out_path     = OUTPUT_DIR / out_filename
    cv2.imwrite(str(out_path), annotated)

    # Print detection summary
    print(f"\n[INFO] Detections found: {len(results[0].boxes)}")
    for box in results[0].boxes:
        cls_id     = int(box.cls[0])
        confidence = float(box.conf[0])
        cls_name   = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
        print(f"       {cls_name:<12} conf={confidence:.2f}")

    print(f"\n[OK]  Annotated image saved to: {out_path}")

    # Display image (non-blocking; press any key to close)
    cv2.imshow("Safety Helmet Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return str(out_path)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Safety Helmet Detector on a single image")
    parser.add_argument("image",       type=str,            help="Path to the input image")
    parser.add_argument("--weights",   type=str, default=BEST_WEIGHTS, help="Path to model weights")
    parser.add_argument("--conf",      type=float, default=0.25,       help="Confidence threshold")
    args = parser.parse_args()

    run_inference(image_path=args.image, weights_path=args.weights, conf=args.conf)
