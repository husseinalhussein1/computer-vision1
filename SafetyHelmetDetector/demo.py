"""
demo.py - Stage 8: Demo Script
Runs the model on a set of new worksite images, draws bounding boxes,
applies the warning rule, and saves all results to outputs/demo/.
"""

import sys
import cv2
from pathlib import Path
from ultralytics import YOLO

from warning import check_and_warn, overlay_warning
from infer_image import draw_detections


# ─── Configuration ────────────────────────────────────────────────────────────

BEST_WEIGHTS = "runs/train/helmet_detector/weights/best.pt"
DEMO_OUTPUT  = Path("outputs/demo")
CLASS_NAMES  = ["helmet", "no-helmet"]
CONF_THRESH  = 0.25


# ─── Demo Runner ──────────────────────────────────────────────────────────────

def run_demo(image_paths: list[str], weights_path: str = BEST_WEIGHTS) -> None:
    """
    Run detection and warning on each image and save results to outputs/demo/.

    Args:
        image_paths:  List of paths to new worksite images.
        weights_path: Path to trained model weights (.pt).
    """
    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights.resolve()}")
        print("        Run train.py first.")
        sys.exit(1)

    DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("   Safety Helmet Detector - Demo")
    print("=" * 60)
    print(f"  Model   : {weights_path}")
    print(f"  Images  : {len(image_paths)}")
    print(f"  Output  : {DEMO_OUTPUT.resolve()}")
    print("=" * 60 + "\n")

    model = YOLO(weights_path)

    for idx, img_path_str in enumerate(image_paths, start=1):
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[WARNING] Image not found, skipping: {img_path}")
            continue

        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[WARNING] Could not read image, skipping: {img_path}")
            continue

        print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")

        # Run detection
        results    = model.predict(source=str(img_path), conf=CONF_THRESH, verbose=False)
        pred_boxes = results[0].boxes

        # Draw bounding boxes, class names, and confidence scores
        annotated = draw_detections(image, pred_boxes, CLASS_NAMES)

        # Apply warning rule and overlay text if no-helmet detected
        warning_triggered = check_and_warn(pred_boxes)
        if warning_triggered:
            annotated = overlay_warning(annotated)

        # Print per-image detection summary
        n_helmet    = sum(1 for b in pred_boxes if CLASS_NAMES[int(b.cls[0])] == "helmet")
        n_no_helmet = sum(1 for b in pred_boxes if CLASS_NAMES[int(b.cls[0])] == "no-helmet")
        print(f"         Detections: {n_helmet} helmet | {n_no_helmet} no-helmet")
        if warning_triggered:
            print("         [!] WARNING triggered")

        # Save result
        out_path = DEMO_OUTPUT / f"demo_{idx:03d}_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        print(f"         Saved: {out_path}\n")

    print("=" * 60)
    print(f"  Demo complete. Results saved to: {DEMO_OUTPUT.resolve()}")
    print("=" * 60)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Safety Helmet Detector - Demo on new worksite images"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more image paths (e.g. demo_imgs/*.jpg)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=BEST_WEIGHTS,
        help="Path to trained model weights (.pt)",
    )
    args = parser.parse_args()

    run_demo(image_paths=args.images, weights_path=args.weights)
