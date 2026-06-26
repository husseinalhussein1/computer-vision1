"""
warning.py - Stage 7: Warning Rule
Checks detection results for persons without a safety helmet
and prints/overlays a warning message.
"""

import cv2
import numpy as np


# ─── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES     = ["helmet", "no-helmet"]
WARNING_TEXT    = "WARNING: Person detected without safety helmet."
WARNING_COLOR   = (0, 0, 255)   # Red in BGR


# ─── Warning Functions ────────────────────────────────────────────────────────

def check_and_warn(boxes) -> bool:
    """
    Inspect detection boxes and print a warning if any no-helmet person is found.

    Args:
        boxes: Ultralytics Boxes object from model prediction results.

    Returns:
        True if at least one no-helmet detection was found, else False.
    """
    no_helmet_found = False

    for box in boxes:
        cls_id   = int(box.cls[0])
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown"
        if cls_name == "no-helmet":
            no_helmet_found = True
            break

    if no_helmet_found:
        print("\n" + "!" * 50)
        print(WARNING_TEXT)
        print("!" * 50 + "\n")

    return no_helmet_found


def overlay_warning(image: np.ndarray) -> np.ndarray:
    """
    Write the warning message in red at the top of the image.

    Args:
        image: BGR image (numpy array).

    Returns:
        Image with the warning text overlaid.
    """
    annotated = image.copy()

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.75
    thickness  = 2

    (tw, th), baseline = cv2.getTextSize(WARNING_TEXT, font, font_scale, thickness)

    # Dark background strip behind the text
    cv2.rectangle(annotated, (0, 0), (tw + 16, th + baseline + 12), (0, 0, 0), -1)

    # Warning text in red
    cv2.putText(
        annotated,
        WARNING_TEXT,
        (8, th + 6),
        font,
        font_scale,
        WARNING_COLOR,
        thickness,
        cv2.LINE_AA,
    )

    return annotated
