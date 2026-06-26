"""
data_check.py - Stage 2: Dataset Verification
Reads the dataset, validates structure and class labels, and prints an organized report.
"""

import os
import sys
import yaml
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

DATASET_DIR = Path("dataset")
DATA_YAML   = DATASET_DIR / "data.yaml"
SPLITS      = ["train", "valid", "test"]
VALID_CLASSES = {"helmet", "no-helmet"}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_yaml(path: Path) -> dict:
    if not path.exists():
        print(f"[ERROR] data.yaml not found at: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def count_images(images_dir: Path) -> int:
    """Return the number of image files in a directory."""
    if not images_dir.exists():
        return 0
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sum(1 for f in images_dir.iterdir() if f.suffix.lower() in extensions)


def read_labels(labels_dir: Path, class_names: list) -> dict:
    """
    Parse all YOLO .txt label files and return per-class sample counts.
    Returns a dict: {class_name: count}
    """
    counts = {name: 0 for name in class_names}
    if not labels_dir.exists():
        return counts
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if not parts:
                    continue
                class_id = int(parts[0])
                if 0 <= class_id < len(class_names):
                    counts[class_names[class_id]] += 1
    return counts


def validate_classes(class_names: list) -> None:
    """Ensure the dataset contains only the allowed classes. Exit on mismatch."""
    found = set(class_names)
    extra = found - VALID_CLASSES
    missing = VALID_CLASSES - found

    if extra:
        print(f"\n[ERROR] Unexpected classes found in dataset: {extra}")
        print(f"        Allowed classes are only: {VALID_CLASSES}")
        sys.exit(1)

    if missing:
        print(f"\n[WARNING] Some expected classes are missing from dataset: {missing}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("   Safety Helmet Detector - Dataset Verification Report")
    print("=" * 60)

    # ── Load data.yaml ──────────────────────────────────────────
    config      = load_yaml(DATA_YAML)
    class_names = config.get("names", [])
    nc          = config.get("nc", 0)

    print(f"\n[INFO] Dataset config : {DATA_YAML}")
    print(f"[INFO] Declared classes: {nc} → {class_names}")

    # ── Validate class names ─────────────────────────────────────
    validate_classes(class_names)
    print("[OK]   Class names are valid: helmet, no-helmet")

    # ── Check each split ─────────────────────────────────────────
    split_stats   = {}
    missing_dirs  = []

    for split in SPLITS:
        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        if not images_dir.exists():
            missing_dirs.append(str(images_dir))
            split_stats[split] = {"images": 0, "labels": {}}
            continue

        n_images  = count_images(images_dir)
        lbl_counts = read_labels(labels_dir, class_names)
        split_stats[split] = {"images": n_images, "labels": lbl_counts}

    if missing_dirs:
        print(f"\n[WARNING] Missing directories detected:")
        for d in missing_dirs:
            print(f"          {d}")

    # ── Print organized report ───────────────────────────────────
    print("\n" + "-" * 60)
    print("  Split Statistics")
    print("-" * 60)

    total_images = 0
    total_labels = {name: 0 for name in class_names}

    for split in SPLITS:
        stats  = split_stats[split]
        n_img  = stats["images"]
        lbls   = stats["labels"]
        total_images += n_img

        print(f"\n  [{split.upper()}]")
        print(f"    Images : {n_img}")
        for cls in class_names:
            cnt = lbls.get(cls, 0)
            total_labels[cls] += cnt
            print(f"    {cls:<12}: {cnt} samples")

    print("\n" + "-" * 60)
    print("  Overall Totals")
    print("-" * 60)
    print(f"  Total images  : {total_images}")
    for cls in class_names:
        print(f"  {cls:<14}: {total_labels[cls]} samples")

    print("\n" + "=" * 60)
    print("  Dataset verification complete. No critical errors found.")
    print("=" * 60)


if __name__ == "__main__":
    main()
