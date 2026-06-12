# Automated Defects Identification System in Printed Circuit Boards Using CNN

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange?logo=tensorflow)
![Models](https://img.shields.io/badge/Models-CNN%20%7C%20ConvNeXt%20%7C%20Inception-green)
![Deployment](https://img.shields.io/badge/Deployment-Roboflow-purple)
![Target](https://img.shields.io/badge/Target-Manufacturing%20Plants-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

> An automated deep learning system for detecting and classifying defects in Printed Circuit Boards (PCBs), designed to support quality control in manufacturing environments and reduce production losses.

---

## 📌 Overview

Manual inspection of PCBs on production lines is slow, error-prone, and costly. This project implements an **Automated Defect Identification System** using Convolutional Neural Networks (CNNs) and modern deep learning architectures to detect PCB defects in real time.

The system compares multiple CNN-based architectures — including **ConvNeXt** and **Inception** — and deploys the best-performing model via **Roboflow** as a cloud-based inference interface, making it accessible directly from a web browser or API without requiring local GPU hardware.

---

## 🎯 Key Goals

- Automate PCB quality inspection to reduce human error
- Minimise production defects and associated costs in manufacturing plants
- Provide a cloud-deployable inference pipeline accessible via Roboflow
- Compare the performance of multiple CNN architectures on PCB defect data

---

## 🔍 Defect Categories

The system is trained to detect common PCB manufacturing defects, including:

| Defect Class     | Description                                      |
|------------------|--------------------------------------------------|
| Missing Hole     | Absence of a required drill hole                 |
| Mouse Bite       | Irregular notch along the PCB edge               |
| Open Circuit     | Broken or incomplete conductive trace            |
| Short            | Unintended connection between conductors         |
| Spur             | Unwanted copper protrusion from a trace          |
| Spurious Copper  | Residual copper remaining after etching          |

---

## 🏗️ Models & Architecture

Three deep learning approaches are implemented and compared:

| Notebook | Architecture | Description |
|----------|-------------|-------------|
| `Automated_Defects_Identification...ipynb` | Custom CNN | Primary detection model, exported as `.pt` |
| `ConvNeXt (1).ipynb` | ConvNeXt | Modern CNN variant with improved feature hierarchy |
| `inception (1).ipynb` | InceptionV3 | Multi-scale feature extraction via Inception modules |

All models are trained on the **DeepPCB** dataset and evaluated on classification accuracy, precision, recall, and F1-score.

---

## 📁 Repository Structure

```
├── Automated_Defects_Identification_in_Printed_Circuit_Boards_using_Deep_learning_Techniques_ (updated).ipynb
│   └── Main training pipeline — exports pcbdetection.pt
├── ConvNeXt (1).ipynb
│   └── ConvNeXt architecture experiments
├── inception (1).ipynb
│   └── InceptionV3 architecture experiments
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites

```bash
Python >= 3.8
torch / torchvision
tensorflow / keras
opencv-python
pillow
inference-sdk        # Roboflow inference
```

### Install Dependencies

```bash
pip install torch torchvision tensorflow keras opencv-python pillow inference-sdk
```

---

## 🚀 Running the Project

Follow these steps to train and deploy the model:

### Step 1 — Train the Model
Open and run the main notebook on **Google Colab** (GPU recommended):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Denuwanweerakkody/Automated-Defects-Identification-System-in-Printed-Circuit-Boards-Using-CNN-/blob/main/Automated_Defects_Identification_in_Printed_Circuit_Boards_using_Deep_learning_Techniques_%20(updated).ipynb)

### Step 2 — Export the Model
After training, export and download the model weights:
```
pcbdetection.pt
```

### Step 3 — Set Up Local Environment
Create a new project folder, open a terminal at that path, and install the required packages:
```bash
pip install inference-sdk pillow opencv-python
```

### Step 4 — Deploy via Roboflow
1. Create a free account at [Roboflow](https://roboflow.com)
2. Upload your trained model (`pcbdetection.pt`) to a new Roboflow project
3. Use the generated API key and model endpoint for inference:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="YOUR_API_KEY"
)

result = client.infer("your_pcb_image.jpg", model_id="pcbdetection/1")
print(result)
```

### Step 5 — Cloud-Based Interface
Access the Roboflow-hosted model via the web dashboard or REST API — no local GPU required.

---

## 📦 Dataset

This project uses the **DeepPCB** benchmark dataset.

- **Classes**: 6 PCB defect types
- **Source**: [DeepPCB GitHub Repository](https://github.com/tangsanli5201/DeepPCB)

> Download the dataset and place it under `data/` before running the notebooks.

---

## 📊 Results

Model comparison across CNN architectures (update with your actual values):

| Architecture | Accuracy (%) | Precision | Recall | F1-Score |
|-------------|-------------|-----------|--------|----------|
| Custom CNN  | —           | —         | —      | —        |
| ConvNeXt    | —           | —         | —      | —        |
| InceptionV3 | —           | —         | —      | —        |

> Fill in with your experimental results from the notebooks.

---

## 💡 Use Case

This system is designed for **manufacturing plant integration**, where:

- PCB images are captured by cameras on production lines
- The model runs inference in real time via the Roboflow API
- Defective boards are flagged automatically for rejection or rework
- Production teams receive defect statistics to monitor and improve quality

---

## 👤 Author

**Denuwan Weerakkody** — [@Denuwanweerakkody](https://github.com/Denuwanweerakkody)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

## 🔗 Related Project

Also check out the advanced few-shot version of this work:
👉 [FSODM-Siamese: Few-Shot PCB Defect Detection](https://github.com/Denuwanweerakkody/A-Few-Shot-Siamese-Network-with-Adaptive-Metric-Learning-for-Printed-Circuit-Board-Defect-Detection)
