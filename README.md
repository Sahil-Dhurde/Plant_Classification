# 🌿 Plant Leaf Disease Classification — CNN Project

A complete end-to-end deep learning pipeline for classifying plant leaf diseases using the **PlantVillage dataset** from Kaggle. Supports both a **custom CNN** trained from scratch and a **MobileNetV2 transfer learning** model with fine-tuning.

---

## 📁 Project Structure

```
Plant_classification/
│
├── main.py                  ← Single entry point (interactive menu + CLI)
├── config.py                ← All hyperparameters and paths
├── download_dataset.py      ← Kaggle dataset downloader
├── data_preprocessing.py    ← tf.data pipeline + augmentation
├── model.py                 ← Custom CNN & MobileNetV2 architectures
├── train.py                 ← Training script (both models)
├── evaluate.py              ← Confusion matrix, classification report, charts
├── predict.py               ← Predict: single image / folder / webcam
├── visualize.py             ← Class distribution, augmentation, prediction grid
├── requirements.txt         ← Python dependencies
│
├── dataset/                 ← Downloaded & extracted PlantVillage images
├── models/                  ← Saved .keras model files + class_names.json
└── results/                 ← Training plots, confusion matrix, reports
```

---

## 🗂️ Dataset

| Detail | Value |
|---|---|
| **Name** | PlantVillage |
| **Kaggle ID** | `abdallahalidev/plantvillage-dataset` |
| **Classes** | 38 (healthy + diseased leaves of 14 plant species) |
| **Images** | ~54,000 color images |
| **Image types** | Color, Grayscale, Segmented |

> We use the **color** subset for best CNN performance.

---

## ⚙️ Setup

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Kaggle API Key
1. Go to [https://www.kaggle.com/settings](https://www.kaggle.com/settings)
2. Click **API → Create New Token** — downloads `kaggle.json`
3. Place it at:
   ```
   C:\Users\<YourUsername>\.kaggle\kaggle.json
   ```

---

## 🚀 Usage

### Interactive Menu (Recommended)
```powershell
python main.py
```
Choose from a numbered menu to download, train, evaluate, visualize, or predict.

---

### CLI Mode

| Command | Description |
|---|---|
| `python main.py --step download` | Download PlantVillage from Kaggle |
| `python main.py --step train` | Train custom CNN from scratch |
| `python main.py --step train_mobilenet` | Train MobileNetV2 (transfer learning) |
| `python main.py --step evaluate` | Evaluate on test set |
| `python main.py --step visualize` | Generate all plots |
| `python main.py --step predict --image leaf.jpg` | Predict single image |
| `python main.py --step predict --folder ./images/` | Predict folder of images |
| `python main.py --step predict --webcam` | Live webcam prediction |
| `python main.py --step all` | Full pipeline (download → train → evaluate → visualize) |

---

## 🧠 Model Architectures

### Custom CNN
```
Input (224×224×3)
→ Conv Block 1: Conv2D(32) × 2 + BN + MaxPool + Dropout
→ Conv Block 2: Conv2D(64) × 2 + BN + MaxPool + Dropout
→ Conv Block 3: Conv2D(128) × 2 + BN + MaxPool + Dropout
→ Conv Block 4: Conv2D(256) × 2 + BN + MaxPool + Dropout
→ GlobalAveragePooling2D
→ Dense(512) + BN + Dropout(0.5)
→ Dense(256) + BN + Dropout(0.5)
→ Dense(38, softmax)
```

### MobileNetV2 (Transfer Learning)
```
Phase 1 (frozen base, 10 epochs):
  MobileNetV2 (ImageNet weights, frozen)
  → GAP → Dense(512) → Dense(256) → Dense(38, softmax)

Phase 2 (fine-tuning, last 30 layers unfrozen, 25 epochs):
  MobileNetV2 (partial unfreeze, lr × 0.1)
  → GAP → Dense(512) → Dense(256) → Dense(38, softmax)
```

---

## 📊 Training Configuration

| Parameter | Value |
|---|---|
| Image Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 25 (+ 10 Phase 1 for MobileNet) |
| Learning Rate | 0.001 (Phase 1), 0.0001 (Phase 2) |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Train/Val/Test Split | 70% / 20% / 10% |
| Early Stopping Patience | 5 epochs |

### Data Augmentation
- Random horizontal & vertical flip
- Random brightness / contrast / saturation
- Random crop (after slight upscale)

---

## 📈 Outputs & Results

After training and evaluation, the `results/` folder will contain:

| File | Description |
|---|---|
| `training_history_*.png` | Accuracy & loss curves |
| `confusion_matrix.png` | Normalized confusion matrix heatmap |
| `per_class_accuracy.png` | Per-class accuracy bar chart |
| `classification_report.txt` | Precision / Recall / F1 per class |
| `class_distribution.png` | Dataset class distribution chart |
| `augmentation_gallery.png` | Sample augmented images |
| `prediction_grid.png` | Test set prediction grid |
| `last_prediction.png` | Most recent single-image prediction |

---

## 🌱 Plant Species Covered

Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato — covering **14 species** and **38 disease/healthy classes**.

---

## 📋 Requirements

```
tensorflow >= 2.10.0
numpy
matplotlib
seaborn
scikit-learn
Pillow
kagglehub
opencv-python
```
