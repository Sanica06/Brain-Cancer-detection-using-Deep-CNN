# 🧠 Brain Cancer Detection using Deep CNN

A deep learning-based project for detecting brain cancer from MRI images using Convolutional Neural Networks (CNNs).  
This project demonstrates the power of Deep Learning for early cancer diagnosis — built with TensorFlow/Keras.

---

##  Table of Contents
1. [Project Overview](#-project-overview)  
2. [Features](#-features)  
3. [Tech Stack](#-tech-stack)  
4. [Setup & Running Locally](#-setup--running-locally)  
5. [Dataset](#-dataset)  
6. [Training](#-training)  
7. [Usage](#-usage)  
8. [Model Performance](#-model-performance)  
9. [Screenshots](#-screenshots)  
10. [Contributing](#-contributing)  
11. [License & Credits](#-license--credits)

---

## 🧠 Project Overview

The goal of this project is to train a Convolutional Neural Network (CNN) to detect the presence of brain tumors in MRI images.  
The model learns to classify MRI scans into *Tumor Present* or *No Tumor*.  

---

## ✅ Features
- Deep CNN architecture for high-accuracy classification  
- Image preprocessing and augmentation  
- Training history visualization (accuracy/loss curves)  
- Model evaluation on test set  
- Portable trained model (.h5) for deployment

---

## 🛠️ Tech Stack
- Python 3.x  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib / Seaborn  
- Scikit-learn

---

## ⚙️ Setup & Running Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Sanica06/Brain-Cancer-detection-using-Deep-CNN.git
cd Brain-Cancer-detection-using-Deep-CNN
```

### 2️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## 📂 Dataset

- Dataset used: Brain MRI Images for Brain Tumor Detection  
- Source: [Kaggle Brain Tumor Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

Place the dataset in:
```txt
/dataset/
```

Expected structure:
```
dataset/
├── yes/     # Tumor present
└── no/      # No tumor
```

---

## 🎯 Training

Run the training script:
```bash
python train.py
```

After training, the model will be saved as:
```txt
model/brain_tumor_cnn_model.h5
```

---

## 📝 Usage

Run the prediction on new MRI images:
```bash
python predict.py --image path/to/image.jpg
```

---
