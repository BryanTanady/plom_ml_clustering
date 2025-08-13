# plom\_ml\_clustering

Machine Learning clustering model architectures and training for the [Plom](https://github.com/...) grading system.


As of Aug 2025 supports:

* **HME Clustering** — clusters handwritten mathematical expressions.
* **MCQ Clustering** — groups scanned handwritten multiple-choice answers (A–F / a–f, with 'C' and 'c' merged) into semantic clusters.

These systems are designed for integration with Plom's grading workflow, enabling fast, semi-automated grouping of student responses to reduce marking time and provide insights on students' performance.

---

## 👁️ Preview of Integration with Plom
<p align="center">
  <img src="assets/plom_preview.gif" width=800>
</p>

A longer version: [extended demo]()

---

## 🧠 Systems Overview

### 1. HME Clustering

* **Goal:** Cluster handwritten math expressions by their structural/semantic content.
* **Model:** ResNet34 feature extractor + TrOCR encoder for symbolic representation.
* **Clustering Strategy:**

  * Embeddings from the encoder pooled and normalized.
  * Agglomerative clustering with cosine distance or KMeans.
* **Dataset:**

  * Training/Validation:  [Mathwriting-2024](https://arxiv.org/abs/2404.10690).
  * Testing: [CROHME 2019](https://tc11.cvc.uab.es/datasets/ICDAR2019-CROHME-TDF_1%7D) and custom collected dataset.

### 2. MCQ Clustering

The MCQ pipeline has **two model variants**:

#### MCQ v1 — AttentionPooling + Classification Pretrain

* **Architecture:**

  * ResNet18 backbone (ImageNet-pretrained), grayscale input.
  * Custom AttentionPooling layer replaces global average pooling.
  * 11-class output head (A–F/a–f, with C/c merged).
* **Training:**

  * Purely supervised classification with cross-entropy.
* **Clustering Representation:**

  * Hellinger transformation of final softmax probabilities (optionally temperature-scaled).
* **Clustering Method:**

  * KMeans (k=11) or Agglomerative on probability vectors.

#### MCQ v2 — Projection Head + Center Loss

* **Architecture:**

  * ResNet18 backbone without AttentionPooling.
  * Projection head mapping features to a compact embedding space.
* **Training:**

  * Supervised classification with cross-entropy **plus Center Loss** to enforce intra-class compactness.
* **Clustering Representation:**

  * Final embeddings from projection head.
* **Clustering Method:**

  * KMeans (k=11) or Agglomerative on embeddings.

**Dataset:**

* Training/Validation: [EMNIST ByClass](https://www.nist.gov/itl/products-and-services/emnist-dataset), filtered for target classes, with augmentation for robustness.
* Testing: Custom collected dataset of handwritten MCQ answers.

---
## 📊 Evaluation

### HME

| Backbone         | Representation         | Metric | Score |
| ---------------- | ---------------------- | ------ | ----- |
| ResNet34 + TrOCR | PCA-reduced embeddings | Purity | TBD   |

### MCQ

| Variant   | Backbone                               | Representation            | EMNIST Purity | Custom Dataset Purity |
| --------- | -------------------------------------- | ------------------------- | ------------- | --------------------- |
| Version 1 | ResNet18 + AttentionPooling            | Probabilities (Hellinger) | 0.9651           | 0.9751                   |
| Version 2 | ResNet18 + ProjectionHead + CenterLoss | Embeddings (cosine similarity space)                | 0.9641           | 0.9751                   |


---

## 🚀 Getting Started

### Prerequisites

Install dependencies:

```bash
pip install -r requirements.txt
```

### Prepare training data
```bash
python3 -m scripts.data_prep.hme #HME
python3 -m scripts.data_prep.hme #MCQ
```


### Training (HME)
```bash
python3 -m training.HME_training 
```

### Training (MCQ)
```bash
python3 -m training.MCQ1_training.py # Trained purely for classification.
python3 -m training.MCQ1_training.py # Trained with clustering in mind (CenterLoss).
```


---
## 📥 Getting pretrained weights

### MCQ (V1)
```bash
python3 -m scripts.get_pretrained_weights.mcq
```

### HMESymbolic
```bash
python3 -m scripts.get_pretrained_weights.hme
```