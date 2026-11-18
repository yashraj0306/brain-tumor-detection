---
title: Brain Tumor Detection
emoji: 🧠
colorFrom: red
colorTo: purple
sdk: docker
app_port: 7860
---

# 🧠 Brain Tumor Detection

An AI-powered MRI brain tumor detection system that performs simultaneous tumor segmentation and classification using a multi-task deep learning architecture.

## Live Demo

[brain-tumor-detection.hf.space](https://jatinredhu-brain-tumor-detection.hf.space)

## Model Architecture

- **Segmentation:** Attention UNet++ with EfficientNetB3 encoder
- **Classification:** 4-class tumor classification (Glioma, Meningioma, Pituitary, No Tumor)
- **Attention:** ScSE (Spatial and Channel Squeeze & Excitation) blocks
- **Framework:** PyTorch + segmentation_models_pytorch

## Performance

| Metric                  | Score |
| ----------------------- | ----- |
| Dice Score              | 0.890 |
| Classification Accuracy | 99%   |
| Macro F1                | 0.99  |

## Stack

- **Backend:** FastAPI + PyTorch
- **Frontend:** Vanilla HTML/CSS/JS
- **Deployment:** Hugging Face Spaces (Docker)
- **Model Weights:** Hugging Face Hub

## Usage

Upload an MRI brain scan image and the model will return:

- Tumor segmentation overlay
- Tumor type classification
- Confidence scores for each class

## Author

Made by [Jatin Redhu](https://github.com/Jatinredhu)
