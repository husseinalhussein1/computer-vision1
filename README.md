# Safety Helmet Detector

A YOLOv8n-based object detection system that identifies workers wearing or not wearing safety helmets in workplace images and triggers a safety warning when a violation is detected.

---

## Dataset

**Selected Dataset:** Safety Helmet Detection — Roboflow Universe

| Property        | Details                                      |
|-----------------|----------------------------------------------|
| Source          | Roboflow Universe                            |
| Format          | YOLOv8 (YOLO annotation format)              |
| Classes         | `helmet`, `no-helmet`                        |
| Total Images    | 3,000+ (varies by version)                   |
| Splits          | train / valid / test                         |
| Scenes          | Construction sites, factories, work zones    |
| Annotation      | Bounding boxes around each person            |

**Why this dataset?**
- Two clean classes exactly matching project requirements (helmet / no-helmet).
- Bounding boxes represent persons, not just helmets — matches the detection goal.
- Diverse scenes: real construction sites, multiple lighting conditions, crowd scenarios.
- High-quality annotations and sufficient volume for reliable YOLOv8n training.

### How to Download the Dataset

1. Go to [Roboflow Universe](https://universe.roboflow.com) and search for **"Safety Helmet Detection"**.
2. Choose a dataset with classes `helmet` and `no-helmet`.
3. Export in **YOLOv8** format.
4. Unzip into the `dataset/` folder so the structure is:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml        ← already provided in this repo
```

---

## Project Structure

```
SafetyHelmetDetector/
│
├── dataset/              ← Place your downloaded dataset here
│   └── data.yaml         ← Dataset configuration (classes, paths)
│
├── models/               ← (Optional) Copy best.pt here for archiving
├── outputs/              ← Detection results and demo outputs
│   └── demo/             ← Demo script outputs
├── runs/                 ← YOLO training runs (auto-generated)
├── utils/                ← Shared utilities (future use)
│
├── data_check.py         ← Stage 2: Dataset verification
├── train.py              ← Stage 3: Model training
├── evaluate.py           ← Stage 4: Model evaluation
├── infer_image.py        ← Stage 5: Single image inference
├── error_analysis.py     ← Stage 6: Failure case analysis
├── warning.py            ← Stage 7: Warning rule
├── demo.py               ← Stage 8: Batch demo on new images
├── app.py                ← Streamlit UI (Training + Testing tabs)
│
├── requirements.txt      ← Python dependencies
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

Run each stage in order:

### 1. Verify Dataset
```bash
python data_check.py
```
Validates splits, class names, and prints a statistics report.

### 2. Train the Model
```bash
python train.py --epochs 50 --batch -1
```
Saves best and last weights to `runs/train/helmet_detector/weights/`.

### 3. Evaluate on Test Split
```bash
python evaluate.py
```
Reports mAP@50 and per-class accuracy for `helmet` and `no-helmet`.

### 4. Run Inference on a Single Image
```bash
python infer_image.py path/to/image.jpg
```
Saves annotated result to `outputs/`.

### 5. Analyze Failure Cases
```bash
python error_analysis.py
```
Finds and visualizes at least 5 real detection failures from the test split.

### 6. Run Demo on New Images
```bash
python demo.py img1.jpg img2.jpg img3.jpg
```
Processes each image, applies warning rule, and saves results to `outputs/demo/`.

### 7. Launch Streamlit UI
```bash
streamlit run app.py
```
Opens a browser interface with Training and Testing tabs.

---

## Model

| Setting    | Value       |
|------------|-------------|
| Base model | YOLOv8n     |
| Epochs     | 50          |
| Image size | 640 × 640   |
| Optimizer  | Auto        |
| Batch size | Auto (-1)   |

---

## Warning Rule

If any person classified as `no-helmet` is detected in an image, the system:

1. Prints to console:
   ```
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   WARNING: Person detected without safety helmet.
   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
   ```
2. Overlays the same warning message in red on the output image.

---

## Ethics Statement

This system is designed to assist safety officers in monitoring workplace compliance with helmet regulations. It is intended as a decision-support tool, not a replacement for human judgment. Workers' privacy must be respected, and footage should be handled in accordance with applicable privacy laws and workplace policies. Like all machine learning models, this detector may produce false positives or false negatives, particularly in challenging lighting, occlusion, or crowd conditions. The final safety decision must always rest with a qualified safety officer — never with the model alone.

---

## Output Colors

| Color  | Meaning           |
|--------|-------------------|
| Green  | Helmet detected   |
| Red    | No helmet detected|
