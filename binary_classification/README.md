# Binary Classification for Sidewalk Encroachment

This folder contains the binary image-classification baseline used in the paper **“A Vietnamese Street-Image Dataset and Integrated Deep Learning System for Sidewalk Encroachment Detection and Severity Assessment.”**

## Purpose

This experiment was created to provide a simple image-level baseline for the sidewalk encroachment problem.

Given a street image, the model predicts whether sidewalk encroachment is present:

- **Encroachment**
- **Non-encroachment**

The binary classifier is used to show the difference between a simple classification approach and the complete integrated pipeline proposed in the study.

The binary model can determine whether an image contains signs of sidewalk encroachment, but it cannot:

- Identify which objects cause the encroachment.
- Locate objects with bounding boxes.
- Segment road or sidewalk regions.
- Measure the occupied sidewalk area.
- Calculate an encroachment score.
- Classify severity as Clear, Light, Moderate, or Heavy.

Therefore, this experiment serves only as a baseline and is not used as the final sidewalk encroachment assessment system.

## Folder Structure

```text
binary_classification/
├── train_binary_classification.ipynb
└── README.md
```

The dataset, trained weights, checkpoints, and evaluation outputs are stored in Google Drive and are not included in this repository.

## Dataset

The binary-classification dataset contains **2,734 images**:

| Class | Images |
|---|---:|
| Encroachment | 1,735 |
| Non-encroachment | 999 |
| **Total** | **2,734** |

The dataset is divided using a stratified 70/15/15 split:

| Split | Images |
|---|---:|
| Training | 1,913 |
| Validation | 410 |
| Test | 411 |
| **Total** | **2,734** |

## Models

The notebook compares three ImageNet-pretrained convolutional neural networks:

- MobileNetV2
- EfficientNetB0
- ResNet50V2

All images are resized to **224 × 224 pixels**.

Each pretrained backbone is connected to a task-specific classification head containing:

- Global Average Pooling
- Dropout
- Sigmoid output layer

Training uses:

- ImageNet-pretrained weights
- Binary cross-entropy loss
- Adam optimizer
- Training-only data augmentation
- Validation-AUC checkpointing
- Early stopping

## Training Environment

The notebook is designed to run on:

- Google Colab
- Google Drive
- Python 3
- TensorFlow / Keras

Google Drive is used to store:

- Input datasets
- Dataset split information
- Training logs
- Evaluation results
- Model checkpoints
- Final trained models

## How to Run

1. Open `train_binary_classification.ipynb` in Google Colab.

2. Enable a GPU runtime:

```text
Runtime → Change runtime type → T4 GPU
```

3. Mount Google Drive when prompted by the notebook.

4. Update the dataset and output paths in the configuration cells so that they match the folders in Google Drive.

5. Run all cells from top to bottom.

The notebook will:

- Read the binary image dataset from Google Drive.
- Create stratified train, validation, and test splits.
- Train the three transfer-learning models.
- Evaluate all models on the test set.
- Generate comparison metrics and confusion matrices.
- Save results and trained models back to Google Drive.

## Results

| Model | Accuracy | Precision | Recall | F1-score | AUC |
|---|---:|---:|---:|---:|---:|
| MobileNetV2 | 0.886 | 0.877 | 0.954 | 0.914 | 0.957 |
| **EfficientNetB0** | **0.908** | **0.897** | **0.966** | **0.930** | 0.960 |
| ResNet50V2 | 0.903 | 0.890 | 0.966 | 0.926 | **0.963** |

**EfficientNetB0** is selected as the best binary-classification model because it achieves the highest accuracy and F1-score.

ResNet50V2 achieves the highest AUC, while EfficientNetB0 provides the best overall balance for binary encroachment recognition.

## Generated Outputs

The notebook generates and saves files such as:

```text
classification_results.csv
image_level_split.csv
model_comparison_metrics.png
confusion_matrices.png
summary.json
training logs
model checkpoints
best_model.keras
best_model.pt
```

These generated files are stored in Google Drive and are not included in this GitHub repository.

## Scope

This binary classifier is included for experimental comparison only.

The final system uses object detection, road–sidewalk segmentation, spatial validation, duplicate suppression, sidewalk coverage estimation, and rule-based severity assessment to provide a more detailed and explainable result.