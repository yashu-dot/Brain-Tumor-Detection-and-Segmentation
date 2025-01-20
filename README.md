# Brain Tumor Detection and Segmentation Using Deep Learning

This repository contains the implementation of a two-step approach for brain tumor detection and segmentation using deep learning. The project employs a Convolutional Neural Network (CNN) for classification and the U-Net architecture for segmentation, achieving high accuracy in processing MRI brain scans.

## Project Overview

Detecting and segmenting brain tumors is a critical task in medical imaging. This project addresses the limitations of manual analysis by introducing a robust automated system:
- **Classification:** A CNN classifies MRI scans as tumorous or non-tumorous with **97% accuracy**.
- **Segmentation:** If a tumor is detected, the U-Net architecture segments the tumor boundaries with a **93% accuracy**.

## Key Features

- **Two-step Approach:** Combines tumor detection and segmentation for enhanced diagnostic precision.
- **High Performance:** Optimized CNN and U-Net architectures for accurate classification and segmentation.
- **Real-world Applications:**
  - Clinical diagnosis and treatment planning.
  - Remote diagnostic systems for underserved regions.
  - Facilitates research by processing large-scale MRI datasets.

## Dataset

The project utilizes the [Br35H: Brain Tumor Detection 2020 dataset](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection):
- **Classification Task:**
  - 3,060 MRI images (1,500 tumorous, 1,500 non-tumorous).
- **Segmentation Task:**
  - Annotated images for tumor regions with augmentation applied to enhance dataset size.

### Preprocessing
- Images resized to 256x256 pixels.
- Normalized pixel values for faster convergence.
- Data augmentation techniques:
  - Horizontal flips, rotations, and vertical flips.

## Model Architectures

### Classification Model (CNN)
- **Layers:**
  - 3 convolutional layers with ReLU activation.
  - MaxPooling and dropout layers for feature extraction and overfitting prevention.
  - Fully connected layers for binary classification.
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy

### Segmentation Model (U-Net)
- **Architecture:**
  - Encoder-decoder structure with skip connections.
  - Convolutional layers with ReLU activation.
  - UpSampling layers to reconstruct segmented output.
- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Additional Techniques:**
  - Learning rate scheduler.
  - Early stopping.

## Results

| Task           | Accuracy |
|----------------|----------|
| Classification | 97%      |
| Segmentation   | 93%      |

## Usage

### Prerequisites
- Python 3.8+
- Required libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib

### Steps to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-repo-name/brain-tumor-detection.git
   cd brain-tumor-detection
