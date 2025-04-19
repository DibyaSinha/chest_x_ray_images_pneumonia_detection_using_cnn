# 🩺 Pneumonia Detection Using CNN on Chest X-Ray Images

This project uses a **Convolutional Neural Network (CNN)** to detect **pneumonia** from **chest X-ray images**. The model is trained using TensorFlow and Keras, and performs binary classification: identifying whether an X-ray is `NORMAL` or indicates `PNEUMONIA`.

## 📁 Dataset

The dataset used is the **Chest X-Ray Images (Pneumonia)** dataset available on [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

It is divided into three folders:
- `train/`
- `val/` (validation)
- `test/`

Each folder contains:
- `NORMAL/` – images of healthy lungs
- `PNEUMONIA/` – images with pneumonia

## 🧠 Project Workflow

### 1. 📊 Data Exploration & Visualization
- Count and display the number of images in each class.
- Display sample X-ray images from both categories to understand visual differences.

### 2. ⚙️ Data Preprocessing
- Use `ImageDataGenerator` for:
  - Normalizing pixel values
  - Real-time data augmentation (rotation, zoom, flip)

### 3. 🏗️ CNN Model Architecture
- 3 convolutional layers with ReLU activation and MaxPooling
- Flatten layer followed by Dense + Dropout
- Final Dense layer with sigmoid activation for binary classification

### 4. 🧪 Model Training & Evaluation
- Train the model using the training set
- Validate performance on the validation set
- Evaluate final accuracy and loss on the test set
- Plot training vs validation accuracy and loss

## 🚀 How to Run

### 🔧 Requirements
- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn

### 📦 Install Dependencies
```bash
pip install tensorflow numpy matplotlib seaborn
