"""
evaluate.py - Stage 4: Model Evaluation
Evaluates the trained model on the test split and reports mAP@50
and per-class accuracy (precision used as class-level accuracy proxy).
"""

import sys
from pathlib import Path
from ultralytics import YOLO


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_YAML    = "dataset/data.yaml"
BEST_WEIGHTS = "runs/train/helmet_detector/weights/best.pt"
CLASS_NAMES  = ["helmet", "no-helmet"]


# ─── Evaluation Function ──────────────────────────────────────────────────────

def evaluate_model(weights_path: str = BEST_WEIGHTS) -> dict:
    """
    Run validation on the test split and return key metrics.

    Args:
        weights_path: Path to trained model weights (.pt file).

    Returns:
        Dictionary with mAP@50 and per-class accuracy values.
    """
    weights = Path(weights_path)
    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights.resolve()}")
        print("        Train the model first by running train.py")
        sys.exit(1)

    data_yaml = Path(DATA_YAML)
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found at: {data_yaml.resolve()}")
        sys.exit(1)

    print("=" * 60)
    print("   Safety Helmet Detector - Model Evaluation")
    print("=" * 60)
    print(f"  Weights : {weights_path}")
    print(f"  Dataset : {DATA_YAML}")
    print(f"  Split   : test")
    print("=" * 60 + "\n")

    model = YOLO(weights_path)

    # Validate on the test split
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        verbose=True,
        plots=True,
    )

    # ── Extract mAP@50 ───────────────────────────────────────────
    map50 = float(metrics.box.map50)   # mean Average Precision @ IoU 0.50

    # ── Per-class precision as accuracy proxy ────────────────────
    # metrics.box.ap_class_index → tensor of class indices with results
    # metrics.box.p → precision per class (same order as ap_class_index)
    class_precision = {}
    ap_class_index  = metrics.box.ap_class_index.tolist()
    precisions      = metrics.box.p.tolist()

    for idx, cls_id in enumerate(ap_class_index):
        if cls_id < len(CLASS_NAMES):
            cls_name = CLASS_NAMES[cls_id]
            class_precision[cls_name] = precisions[idx]

    # Fill missing classes with 0
    for name in CLASS_NAMES:
        class_precision.setdefault(name, 0.0)

    # ── Print organized report ───────────────────────────────────
    print("\n" + "=" * 60)
    print("   Evaluation Report (Test Split)")
    print("=" * 60)
    print(f"  mAP@50                : {map50 * 100:.2f}%")
    print(f"  Accuracy (helmet)     : {class_precision['helmet'] * 100:.2f}%")
    print(f"  Accuracy (no-helmet)  : {class_precision['no-helmet'] * 100:.2f}%")
    print("=" * 60)

    return {
        "map50":              map50,
        "accuracy_helmet":    class_precision["helmet"],
        "accuracy_no_helmet": class_precision["no-helmet"],
    }


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate Safety Helmet Detector")
    parser.add_argument(
        "--weights",
        type=str,
        default=BEST_WEIGHTS,
        help="Path to model weights (.pt)",
    )
    args = parser.parse_args()

    evaluate_model(weights_path=args.weights)
