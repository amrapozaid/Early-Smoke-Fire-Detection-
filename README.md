Early-Smoke-Fire-Detection

An intelligent vision-based fire and smoke detection framework that integrates a Vision Transformer (ViT) with the YOLOv8 detection architecture.

========================================
------------------------------------------------------------
Project Overview
------------------------------------------------------------

This project implements a stable hybrid detection model that combines:

- Vision Transformer (ViT) global context features
- YOLOv8 (CNN backbone + FPN)
- Residual gated fusion mechanism

The model is designed for Fire/Smoke detection datasets.

------------------------------------------------------------
Project Structure
------------------------------------------------------------

├── ViT_YOLOv8_model.py        # Hybrid ViT+YOLOv8 model
├── train.py                   # Fully training script
├── load_dataset.py            # YOLO TXT dataset loader
├── data.yaml                  # Dataset configuration
├── requirements.txt
├── README.txt


------------------------------------------------------------
Installation
------------------------------------------------------------

1) Create virtual environment:

    conda create -n yolo_gpu python=3.10 -y
    conda activate yolo_gpu

2) Install dependencies:

    pip install -r requirements.txt

3) Verify GPU:

    python -c "import torch; print(torch.cuda.is_available())"

------------------------------------------------------------
Training
------------------------------------------------------------

Run:

    python train.py

------------------------------------------------------------
Outputs
------------------------------------------------------------

After training you will find:

best.pt        → best model by mAP50-95
last.pt        → final epoch
history.csv    → full training log
curves_*.png   → training plots

------------------------------------------------------------
Model Architecture
------------------------------------------------------------


------------------------------------------------------------
Datasets
------------------------------------------------------------
For access to our custom optimized Fire/Smoke dataset, please email:

aaabozezd@ju.edu.sa

You can also train/test the model using the following public Fire and Smoke datasets:

1-Kaggle – DataCluster Labs Fire and Smoke Dataset
https://www.kaggle.com/datasets/dataclusterlabs/fire-and-smoke-dataset
https://universe.roboflow.com/datacluster-labs-agryi/fire-and-smoke-dataset-umfsp

2-Roboflow Universe – Fire and Smoke Detection (Hiwia) v2
https://universe.roboflow.com/middle-east-tech-university/fire-and-smoke-detection-hiwia/dataset/2

------------------------------------------------------------
Trained Model Weights
------------------------------------------------------------
The trained model weights for the ViT +YOLOv8  Hybrid Fire/Smoke Detection Model will be made publicly available upon official publication of our research paper. 

------------------------------------------------------------
Citation
------------------------------------------------------------
Amr Abozeid and Rayan Alanazi, An Intelligent Approach for Early Smoke/Fire Detection 
Using Vision Sensors in Smart Cities, Scientific Reports (under review).

------------------------------------------------------------
End of README
------------------------------------------------------------
