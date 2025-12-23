
# Adaptive Vehicle Re-Identification via Operational Attention and Multi-Part Embedding

This repository contains the official implementation of the **Adaptive Vehicle Re-Identification via Operational Attention and Multi-Part Embedding framework**, designed for robust feature extraction. We introduce a unified Operational Transformer with a Global Fusion Module.

The code is structured in a modular and reproducible manner, following best practices for academic research.

## 📌 Overview

The proposed framework extracts discriminative embeddings by combining:

* Until now, in our vehicle reset studies, we have used the ConvNeXt-Large model instead of the ResNet model, which uses a backbone network.
* An Operational Transformer with Global-Fusion Attention Module (OT-GFAM)
* A Global-Local Multi-Part Embedding

All features are pooled using **Generalized Mean (GeM) pooling**, L2-normalized, and concatenated into a single embedding vector for retrieval-based person re-identification.


## 🧠 Model Architecture

* **Backbone**: ConvNeXt-Large (via `timm`)
* **Pooling**: Details will be shared along with the source code once the paper is accepted.
* **OT-GFAM**: Details will be shared along with the source code once the paper is accepted.
* **GLPE**:    Details will be shared along with the source code once the paper is accepted.

## 📂 Repository Structure

```text
project_root/
│
├── main.py
├── config.yaml
├── requirements.txt
├── README.md
│
├── models/
│   ├── __init__.py
│   ├── gem.py
│   └── convnext_part_model.py
│
├── data/
│   ├── __init__.py
│   └── data_utils.py
│
└── visualization/
    ├── __init__.py
    └── visualize.py
```

## ⚙️ Environment Setup

### Requirements

* Python ≥ 3.8
* PyTorch ≥ 2.0
* CUDA (A100 GPU, for GPU acceleration)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Preparation

The dataset should follow the structure below:

```text
image_test/
├── 0001_xxxx_xxxxx_0
├── 0001_xxxx_xxxxx_1
├── ...
```

A text file describing the test set must be provided in the following format:

```text
image_name_without_extension  vehicle_id
```

Example:

```text
0003_c014_00018740_0 3
```

## 🚀 Running the Demo

To run the visualization demo (query + top-5 gallery results):

```bash
python main.py
```

The script randomly samples queries and visualizes retrieval results in a **10 × 6 grid**, where:

* The first row corresponds to query images
* The remaining rows correspond to the top-5 retrieved gallery images

Correct matches are highlighted in **green**, and incorrect matches in **red**.

## 🔍 Evaluation Protocol

* One image per identity is randomly selected as a query
* The remaining images form the gallery
* Euclidean distance is used for feature matching
* No post-processing is applied unless explicitly stated

This protocol follows standard practices for evaluating vehicle re-identification.


## 🔁 Reproducibility

All dataset preprocessing, model configuration, and evaluation steps are fully specified in the codebase.

Random seeds are fixed where applicable to ensure consistent and reproducible results across runs.


## 📦 Pretrained Models

The pre-trained model weights used in the experiments are published at the following link: https://drive.google.com/drive/folders/13fLPFAgHe9FcT29Accc9xdqKcQAPPjxo?usp=sharing.
After downloading, please place the model file into the ./weights/ directory.

## 🧾 Code Availability

The full source code for this project will be made publicly available upon acceptance of the paper.



## 📬 Contact

For questions or issues related to the code, please open an issue in this repository.

## Dataset Accessibility

All experiments were conducted using publicly available vehicle re-identification datasets. Experiments on non-public datasets were conducted with legal permission.
Due to licensing restrictions, datasets cannot be redistributed; the publicly available VeRi-776 dataset should be obtained from its original sources via the link below.
After you download it, please place the model file into the ./data/ directory.

Download VeRi-776:
https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset
