"""
train.py - Stage 3: Model Training
Trains a YOLOv8n model on the Safety Helmet Detection dataset.
All results are saved under runs/.
"""

import sys
from pathlib import Path
from ultralytics import YOLO


# ─── Configuration ────────────────────────────────────────────────────────────

DATA_YAML   = "dataset/data.yaml"
MODEL_NAME  = "yolov8n.pt"       # Nano model — fast and suitable for this task
RUNS_DIR    = "runs"
PROJECT_DIR = "runs/train"


# ─── Training Function ────────────────────────────────────────────────────────

def train_model(epochs: int = 50, batch: int = -1, imgsz: int = 640) -> str:
    """
    Train YOLOv8n on the helmet dataset.

    Args:
        epochs: Number of training epochs (default 50).
        batch:  Batch size. -1 = auto-select based on available GPU/CPU memory.
        imgsz:  Input image size (default 640).

    Returns:
        Path to the best saved weights file.
    """
    data_path = Path(DATA_YAML)
    if not data_path.exists():
        print(f"[ERROR] data.yaml not found at: {data_path.resolve()}")
        print("        Run data_check.py first and make sure the dataset is in place.")
        sys.exit(1)

    print("=" * 60)
    print("   Safety Helmet Detector - Model Training")
    print("=" * 60)
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Epochs  : {epochs}")
    print(f"  Batch   : {'auto' if batch == -1 else batch}")
    print(f"  Img size: {imgsz}")
    print(f"  Data    : {DATA_YAML}")
    print("=" * 60 + "\n")

    # Load pre-trained YOLOv8n weights
    model = YOLO(MODEL_NAME)

    # Start training
    results = model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        optimizer="auto",
        project=PROJECT_DIR,
        name="helmet_detector",
        exist_ok=True,         # Overwrite previous run with same name
        verbose=True,          # Print per-epoch progress
        save=True,             # Save best and last checkpoints
        plots=True,            # Save training plots (loss curves, etc.)
        patience=15,           # Early stopping if no improvement for 15 epochs
    )

    best_weights = Path(PROJECT_DIR) / "helmet_detector" / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("  Training Complete")
    print(f"  Best weights saved at: {best_weights}")
    print("=" * 60)

    return str(best_weights)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Safety Helmet Detector (YOLOv8n)")
    parser.add_argument("--epochs", type=int, default=50,  help="Number of training epochs")
    parser.add_argument("--batch",  type=int, default=-1,  help="Batch size (-1 = auto)")
    parser.add_argument("--imgsz",  type=int, default=640, help="Image size for training")
    args = parser.parse_args()

    train_model(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz)
