# Vietnamese Sidewalk Encroachment Detection

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Streamlit](https://img.shields.io/badge/Interface-Streamlit-FF4B4B)
![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED)
![Computer Vision](https://img.shields.io/badge/Task-Computer%20Vision-green)

An integrated deep learning system for detecting sidewalk encroachment objects, segmenting road and sidewalk regions, and assessing encroachment severity in Vietnamese street images.

---

## Table of Contents

- [Project Overview](#project-overview)
- [System Pipeline](#system-pipeline)
- [Main Features](#main-features)
- [Dataset](#dataset)
- [Models](#models)
- [Detected Classes](#detected-classes)
- [Severity Assessment](#severity-assessment)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Run with Streamlit](#run-with-streamlit)
- [Run with Docker](#run-with-docker)
- [Configuration](#configuration)
- [Experiments](#experiments)
- [Output Files](#output-files)
- [Limitations](#limitations)
- [Repository](#repository)

---

## Project Overview

Sidewalk encroachment is a common urban problem in Vietnamese cities, where pedestrian areas are frequently occupied by motorbikes, cars, vendor carts, tables, chairs, advertising boards, umbrellas, bins, and other roadside objects.

This project develops an integrated computer vision pipeline that combines:

1. Road and sidewalk segmentation.
2. Sidewalk encroachment object detection.
3. Spatial validation between detected objects and sidewalk regions.
4. Duplicate detection suppression.
5. Encroachment scoring.
6. Severity classification.
7. A Streamlit interface for image-based analysis.

The current system focuses on static Vietnamese street images.

---

## System Pipeline

```text
Input Street Image
        │
        ▼
Road–Sidewalk Segmentation
        │
        ▼
Encroachment Object Detection
        │
        ▼
Spatial Validation
        │
        ├── Sidewalk overlap
        ├── Lower contact region
        ├── Bottom-center support
        └── Road–sidewalk contact comparison
        │
        ▼
Duplicate Detection Suppression
        │
        ▼
Object Score + Sidewalk Coverage Score
        │
        ▼
Encroachment Severity
```

The segmentation model identifies road and sidewalk regions, while the object detection model identifies potential sidewalk obstacles.

Detected objects are not automatically considered encroachments. Each object must pass additional spatial validation rules before contributing to the final score.

---

## Main Features

- Upload and analyze street images.
- Randomly select images from `sample_images/`.
- Segment road and sidewalk regions.
- Detect seven sidewalk encroachment object classes.
- Validate object positions using sidewalk overlap and contact information.
- Compare sidewalk and road contact for vehicles.
- Suppress duplicate detections of the same object.
- Calculate object-based and sidewalk-coverage scores.
- Classify encroachment into four severity levels.
- Display accepted and ignored detections.
- Export annotated result images.
- Export scoring and detection results as CSV files.
- Configure thresholds and weights through YAML files.
- Run locally with Streamlit.
- Package and deploy the application with Docker.

---

## Dataset

The project uses a multi-source collection of Vietnamese street and sidewalk images obtained from:

- Self-collected Vietnamese street images.
- Public sidewalk and urban-scene datasets.
- Public object-detection datasets.
- Additional publicly available Internet images.

The complete collection contains **3,765 images**.

Different subsets of the collection are used for the two main computer vision tasks:

| Task | Number of images |
|---|---:|
| Road–sidewalk segmentation | 2,734 |
| Sidewalk encroachment object detection | 2,766 |

The segmentation and object detection datasets partially overlap but use different annotation formats.

The complete training datasets, images, and labels are not included in this repository because of their size and data distribution restrictions.

Detailed information about data sources, dataset composition, annotations, and train–validation–test splits is available in:

```text
datasets/README.md
```

---

## Models

The application requires two trained model files:

| Model file | Purpose |
|---|---|
| `best_object_detection.pt` | Detect sidewalk encroachment objects |
| `best_road_sidewalk_seg.pt` | Segment road and sidewalk regions |

Expected structure:

```text
models/
├── README.md
├── best_object_detection.pt
└── best_road_sidewalk_seg.pt
```

The trained model weights are not included in this repository.

Before running the application, place both `.pt` files inside the `models/` directory.

Additional model information is available in:

```text
models/README.md
```

---

## Detected Classes

### Object Detection Classes

| Class | Description |
|---|---|
| `ad_board` | Advertising boards and signboards |
| `bin` | Trash bins |
| `car` | Cars occupying or overlapping sidewalks |
| `motorbike` | Motorbikes occupying or overlapping sidewalks |
| `table_chair` | Tables and chairs |
| `umbrella` | Umbrellas associated with roadside activities |
| `vendor_cart` | Street vending carts |

### Segmentation Classes

| Class | Description |
|---|---|
| `road` | Road surface |
| `sidewalk` | Pedestrian sidewalk region |

---

## Severity Assessment

The final encroachment score combines two components:

```text
Encroachment Score = Object Score + Sidewalk Coverage Score
```

### Object Score

The contribution of each accepted object depends on:

- Detection confidence.
- Class-specific weight.
- Sidewalk overlap ratio.
- Spatial validation result.

### Sidewalk Coverage Score

The coverage component measures how much of the detected sidewalk area is occupied by accepted obstacle regions.

### Severity Levels

| Score range | Severity |
|---|---|
| `< 0.42` | No encroachment |
| `0.42 – 1.50` | Light |
| `1.51 – 3.00` | Moderate |
| `> 3.00` | Heavy |

The system also returns:

- Number of accepted obstacle regions.
- Raw object score.
- Final object score.
- Sidewalk coverage ratio.
- Sidewalk coverage score.
- Final severity level.

---

## Project Structure

```text
vietnamese-sidewalk-encroachment-detection/
├── app.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── .gitignore
├── .dockerignore
│
├── configs/
│   ├── contact_thresholds.yaml
│   ├── duplicate_detection.yaml
│   ├── scoring.yaml
│   └── sidewalk_postprocess.yaml
│
├── datasets/
│   └── README.md
│
├── models/
│   └── README.md
│
├── experiments/
│   ├── experiments.ipynb
│   └── input/
│       └── experiment_cases.csv
│
├── sample_images/
├── outputs/
└── temp_uploads/
```

The following files and directories are intentionally excluded from GitHub:

```text
models/*.pt
dataset images and labels
experiment images
sample images
generated outputs
temporary uploaded files
```

---

## Installation

### Requirements

- Python 3.10
- pip
- Git
- Docker Desktop, when using Docker

### 1. Clone the Repository

```bash
git clone https://github.com/quocdata0602/vietnamese-sidewalk-encroachment-detection.git
cd vietnamese-sidewalk-encroachment-detection
```

### 2. Create a Virtual Environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
python -m venv .venv
.venv\Scripts\activate
```

#### Linux or macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add Trained Models

Place the trained model files at:

```text
models/best_object_detection.pt
models/best_road_sidewalk_seg.pt
```

---

## Run with Streamlit

Start the application:

```bash
streamlit run app.py
```

Open the following address in a web browser:

```text
http://localhost:8501
```

The Streamlit interface provides:

- Image upload.
- Random sample-image selection.
- Detection confidence configuration.
- Segmentation confidence configuration.
- Sidewalk overlap threshold configuration.
- Class-specific confidence thresholds.
- Class-specific scoring weights.
- Duplicate suppression settings.
- Sidewalk mask post-processing options.
- Accepted and ignored detection inspection.
- Result image and CSV download.

---

## Run with Docker

### Requirements

- Docker Desktop.
- Docker Compose.

### 1. Build and Start the Application

```bash
docker compose up --build
```

### 2. Open the Application

```text
http://localhost:8501
```

### 3. Stop the Application

```bash
docker compose down
```

### 4. Rebuild after Code or Dependency Changes

```bash
docker compose down
docker compose up --build
```

The Docker container loads the model weights from the `models/` directory.

Ensure that the following files exist before starting the container:

```text
models/best_object_detection.pt
models/best_road_sidewalk_seg.pt
```

---

## Configuration

Pipeline parameters are stored in YAML files under the `configs/` directory.

### `configs/contact_thresholds.yaml`

Contains:

- Lower bounding-box contact-band ratios.
- Minimum sidewalk-contact thresholds for each object class.

### `configs/duplicate_detection.yaml`

Contains:

- Classes requiring duplicate suppression.
- Class-specific IoU thresholds.
- Bounding-box containment threshold.
- Normalized center-distance threshold.

### `configs/scoring.yaml`

Contains:

- Class-specific confidence thresholds.
- Class-specific scoring weights.

### `configs/sidewalk_postprocess.yaml`

Contains:

- Minimum connected-component ratio.
- Morphological closing kernel ratio.
- Maximum convex-hull expansion ratio.

These files allow pipeline parameters to be modified without changing the main application code.

---

## Experiments

The experimental notebook is located at:

```text
experiments/experiments.ipynb
```

The notebook contains case studies for evaluating:

- Sidewalk overlap thresholds.
- Bottom-contact validation.
- Road-context validation.
- Duplicate detection suppression.
- Class-specific scoring weights.
- Complete-pipeline severity assessment.

Experiment metadata is stored in:

```text
experiments/input/experiment_cases.csv
```

Experiment images and generated experiment outputs are not included in the public repository.

---

## Output Files

After processing an image, the application generates files under:

```text
outputs/
```

Example output structure:

```text
outputs/
├── pipeline_<image_name>.jpg
├── <image_name>_encroachment_score_results.csv
├── <image_name>_obstacle_predictions_used.csv
├── <image_name>_obstacle_predictions_ignored.csv
└── <image_name>_all_predictions.csv
```

| File | Description |
|---|---|
| `pipeline_<image_name>.jpg` | Annotated result with sidewalk mask, bounding boxes, score, and severity |
| `encroachment_score_results.csv` | Overall scoring and coverage summary |
| `obstacle_predictions_used.csv` | Objects accepted for score calculation |
| `obstacle_predictions_ignored.csv` | Objects rejected by confidence or spatial rules |
| `all_predictions.csv` | Complete detection output |

Generated output files are excluded from GitHub.

---

## Limitations

- The current system supports static images only.
- Video and real-time camera processing are outside the current scope.
- Performance depends on sidewalk visibility and segmentation quality.
- Small, blurred, or heavily occluded objects may be missed.
- Complex scenes may produce inaccurate road–sidewalk boundaries.
- Buildings, storefronts, or sky regions may occasionally be misclassified by the segmentation model.
- Vehicle classification depends strongly on lower-contact and road-context information.
- The trained model weights and complete datasets are not publicly included in this repository.

---

## Repository

```text
https://github.com/quocdata0602/vietnamese-sidewalk-encroachment-detection
```

This repository contains the application source code, configuration files, Docker setup, experiment notebook, and project documentation.