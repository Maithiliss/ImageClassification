# Cat vs Dog Image Classifier

> Binary image classification using **VGG16 transfer learning** — 98.52% validation accuracy in 5 epochs.

[![Train Accuracy](https://img.shields.io/badge/Train%20Accuracy-98.74%25-E05A1E?style=flat-square)](./Untitled2.ipynb)
[![Val Accuracy](https://img.shields.io/badge/Val%20Accuracy-98.52%25-1E6FC5?style=flat-square)](./Untitled2.ipynb)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%2F%20Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

---

## Overview

This project tackles the classic cats-vs-dogs binary classification challenge using **transfer learning** with VGG16 — a 16-layer deep convolutional network pretrained on ImageNet's 1.2 million images. Rather than training from scratch, the convolutional base weights are frozen and a lightweight custom classifier is trained on top, achieving high accuracy with minimal compute.

**Key stats at a glance:**

| Metric | Value |
|---|---|
| Training accuracy | 98.74% |
| Validation accuracy | 98.52% |
| Training images | 20,000 |
| Test images | 5,000 |
| Input resolution | 224 × 224 px |
| Epochs | 5 |
| Batch size | 32 |

---

## Model Architecture

A Sequential pipeline built on a frozen VGG16 base:

```
Input (224×224×3)
    │
    ▼
[FROZEN] VGG16 Base — pretrained on ImageNet, 13 conv layers + max pooling
    │
    ▼
[FROZEN] Feature Maps
    │
    ▼
[TRAINED] Flatten
    │
    ▼
[TRAINED] Dense(128) + ReLU + Dropout(0.3)
    │
    ▼
[TRAINED] Dense(2) + Softmax → Cat / Dog
```

The frozen base acts as a feature extractor; only the top dense layers are trained on the cats-vs-dogs dataset.

---

## Dataset

Images are distributed as a zip archive with `train/` and `test/` directories, one subfolder per class.

| Split | Images | Classes |
|---|---|---|
| Train | 20,000 | Cats (10k), Dogs (10k) |
| Validation / Test | 5,000 | Cats, Dogs |

All images are resized to **224×224** and normalized using VGG16's `preprocess_input` (per-channel mean subtraction from ImageNet statistics). Data pipeline uses `tf.data` with `AUTOTUNE` prefetching and batch size 32.

---

## Results

**Final epoch performance:**

| Split | Accuracy | Loss |
|---|---|---|
| Training | **98.74%** | 0.0403 |
| Validation | **98.52%** | 0.0543 |

The minimal gap between training and validation accuracy indicates the model generalizes well with no sign of overfitting.

---

## Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Loss function | Sparse Categorical Crossentropy |
| Batch size | 32 |
| Epochs | 5 |
| Dropout rate | 0.3 |
| Environment | Google Colab (GPU) |

---

## Getting Started

The notebook runs in **Google Colab** with GPU acceleration.

**1. Install dependencies**

```bash
pip install tensorflow numpy
```

**2. Open in Colab and enable GPU**

```
Runtime → Change runtime type → Hardware accelerator → GPU
Runtime → Run all
```

Upload your dataset zip file when the file-upload cell prompts you.

**3. Predict on a new image**

```python
from keras.utils import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
import numpy as np

img  = load_img("your_image.jpg", target_size=(224, 224))
x    = preprocess_input(np.expand_dims(img_to_array(img), axis=0))
pred = model.predict(x)
print(["Cat", "Dog"][np.argmax(pred)])
```

---

## Tech Stack

- **TensorFlow / Keras** — deep learning framework and model API
- **VGG16** — pretrained convolutional base (ImageNet weights)
- **NumPy** — numerical computing
- **Google Colab** — cloud GPU runtime
- **Jupyter Notebook** — interactive development environment

---

## File Structure

```
ImageClassification/
└── Untitled2.ipynb   # Main notebook: data loading, model, training, inference
```

---

## License

This project is open source and available under the [MIT License](LICENSE).
