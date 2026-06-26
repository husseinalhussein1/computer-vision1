"""
app.py - Streamlit User Interface
Two-tab interface:
  Tab 1 — Training  : configure and launch model training, view results.
  Tab 2 — Testing   : upload an image, run detection, view output and warning.
"""

import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image

from train import train_model
from evaluate import evaluate_model
from infer_image import draw_detections
from warning import check_and_warn, overlay_warning
from ultralytics import YOLO


# ─── Constants ────────────────────────────────────────────────────────────────

BEST_WEIGHTS = "runs/train/helmet_detector/weights/best.pt"
OUTPUT_DIR   = Path("outputs")
CLASS_NAMES  = ["helmet", "no-helmet"]


# ─── Page Setup ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Safety Helmet Detector",
    page_icon="🪖",
    layout="wide",
)

st.title("Safety Helmet Detector")
st.caption("YOLOv8n — Object Detection for Workplace Safety")


# ─── Tab Layout ───────────────────────────────────────────────────────────────

tab_train, tab_test = st.tabs(["Training", "Testing"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Training
# ══════════════════════════════════════════════════════════════════════════════

with tab_train:
    st.header("Model Training")
    st.write("Configure the training parameters and start training YOLOv8n on the helmet dataset.")

    col1, col2 = st.columns(2)

    with col1:
        epochs = st.slider(
            label="Number of Epochs",
            min_value=10,
            max_value=150,
            value=50,
            step=5,
            help="More epochs generally improve accuracy but take longer.",
        )

    with col2:
        batch_options = {
            "Auto (-1)": -1,
            "8":          8,
            "16":        16,
            "32":        32,
        }
        batch_label = st.selectbox(
            label="Batch Size",
            options=list(batch_options.keys()),
            index=0,
            help="Auto lets YOLO select the best batch size for your hardware.",
        )
        batch = batch_options[batch_label]

    st.markdown("---")

    if st.button("Start Training", type="primary", use_container_width=True):
        with st.spinner(f"Training for {epochs} epochs — this may take several minutes..."):
            try:
                train_model(epochs=epochs, batch=batch)
                st.success("Training completed successfully!")
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.stop()

        # ── Show evaluation results after training ───────────────
        weights_path = Path(BEST_WEIGHTS)
        if weights_path.exists():
            st.subheader("Evaluation Results (Test Split)")
            with st.spinner("Evaluating model on test split..."):
                try:
                    metrics = evaluate_model(weights_path=BEST_WEIGHTS)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("mAP@50",              f"{metrics['map50'] * 100:.1f}%")
                    m2.metric("Accuracy — helmet",   f"{metrics['accuracy_helmet'] * 100:.1f}%")
                    m3.metric("Accuracy — no-helmet",f"{metrics['accuracy_no_helmet'] * 100:.1f}%")
                except Exception as e:
                    st.warning(f"Could not run evaluation: {e}")
        else:
            st.warning("Best weights not found after training. Check the runs/ directory.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Testing
# ══════════════════════════════════════════════════════════════════════════════

with tab_test:
    st.header("Image Testing")
    st.write("Upload a worksite image to detect whether workers are wearing safety helmets.")

    uploaded_file = st.file_uploader(
        label="Upload an image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
    )

    if uploaded_file is not None:
        weights_path = Path(BEST_WEIGHTS)
        if not weights_path.exists():
            st.error(
                "No trained model found. "
                "Please go to the **Training** tab and train the model first."
            )
            st.stop()

        # Decode uploaded image to numpy array (BGR for OpenCV)
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image_bgr is None:
            st.error("Could not decode the uploaded image. Please try a different file.")
            st.stop()

        col_orig, col_result = st.columns(2)

        with col_orig:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)

        with st.spinner("Running detection..."):
            model   = YOLO(str(weights_path))
            results = model.predict(source=image_bgr, conf=0.25, verbose=False)
            boxes   = results[0].boxes

            annotated         = draw_detections(image_bgr, boxes, CLASS_NAMES)
            warning_triggered = check_and_warn(boxes)

            if warning_triggered:
                annotated = overlay_warning(annotated)

        with col_result:
            st.subheader("Detection Result")
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

        # ── Save output ──────────────────────────────────────────
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"app_result_{uploaded_file.name}"
        cv2.imwrite(str(out_path), annotated)

        # ── Detection summary ────────────────────────────────────
        st.markdown("---")
        n_helmet    = sum(1 for b in boxes if CLASS_NAMES[int(b.cls[0])] == "helmet")
        n_no_helmet = sum(1 for b in boxes if CLASS_NAMES[int(b.cls[0])] == "no-helmet")

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Detections",  len(boxes))
        c2.metric("With Helmet",        n_helmet)
        c3.metric("Without Helmet",     n_no_helmet)

        # ── Warning banner ───────────────────────────────────────
        if warning_triggered:
            st.error(
                "WARNING: Person detected without safety helmet.",
                icon="⚠️",
            )
        else:
            st.success("All detected persons are wearing safety helmets.", icon="✅")

        st.caption(f"Result saved to: {out_path}")
