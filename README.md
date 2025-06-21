# 🧪 Fertilizer Prediction using ML (Kaggle Playground Series - June 2025)

This repository contains my solution for the [Kaggle Playground Series - June 2025](https://www.kaggle.com/competitions/playground-series-s5e6), where the task was to predict **top-3 recommended fertilizers** given a set of **soil and crop conditions**. 
This project was a part of my ongoing effort to build end-to-end ML pipelines and experiment with ranking-based metrics like **MAP@3**.

---

## 🧠 Problem Statement

Given features such as:
- Soil properties (Nitrogen, Potassium, Phosphorous)
- Environmental factors (Temperature, Humidity, Moisture)
- Categorical columns (Soil Type, Crop Type)

## Objective:
To predict the **top-3 fertilizer names** used for optimal crop yield. The evaluation metric was **MAP@3**, emphasizing not just the top prediction, but the top 3 ranked predictions.

---

## 📊 Dataset

Synthetic dataset provided by Kaggle, comprising:
- **75,000 training samples**
- Mixture of numerical and categorical variables
- Balanced class distribution (multi-class classification)

Sample columns:
```

| ID | Temperature | Humidity | Moisture | Soil Type | Crop Type | Nitrogen | Potassium | Phosphorous | Fertilizer Name |

```

---

## 🛠️ Workflow Summary

### 🔍 EDA
- Checked for correlation across features
- Distribution plots (Box, Count, Pair, Heatmap)
- Outlier analysis
- Cluster scatterplot with PCA to visually inspect target separability (ineffective)

### ⚙️ Modeling Approaches

#### ✅ Model 1: Multinomial Logistic Regression
- Baseline multi-class classification
- One-hot encoding for categorical columns
- Achieved **MAP@3 score: 0.2908**

#### ✅ Model 2: Logistic Regression + PCA
- Applied PCA post-encoding to reduce dimensionality
- Retained components explaining ~90% variance
- MAP@3 unchanged → **PCA not useful** in this case

#### ✅ Model 3: LightGBM (Final)
- Gradient Boosting on categorical + numerical mix
- Achieved final MAP@3: **0.33145**
- Final Rank: ~900 / ~1900 participants

---

## 📈 Metric: MAP@3

The competition uses **Mean Average Precision at 3**, rewarding correct predictions appearing in the top-3 ranked outputs.

Each prediction vector (from softmax or probability outputs) is post-processed to retrieve the **top 3 class labels**, which are used for submission.

---

## 🧪 Exploratory Clustering

An experiment was conducted using unsupervised clustering (KMeans) based on:
- Target: Fertilizer Type
- Features: Nitrogen, Potassium, Phosphorous

After PCA@2, visual inspection showed no meaningful cluster separability → no further action taken from this.

---

## 📌 Key Learnings

- Boosting models like LightGBM outperform linear models on weakly structured synthetic datasets.
- PCA doesn't guarantee performance improvement, especially when feature-target relationships are non-existent.
- MAP@k as a metric needs ranking-aware prediction formatting (top-k extraction).
- Even “simple” datasets require attention to encoding, dimensionality, and metric-specific tweaks.

---

## 🧰 Tech Stack

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-Learn
- LightGBM
- PCA (Sklearn)
- Kaggle environment

---

## 📂 File Structure

```

📁 Fertilizer-Prediction-Kaggle
├── 📜 fertilizer\_prediction.ipynb  # Main notebook
├── 📁 data/                        # Data used for training and predictions
└── README.md                       # This file

```

---

## ⚠️ Limitations

- Synthetic dataset → limited real-world interpretability
- No hyperparameter tuning for LightGBM (to keep run-time short)
- Not a deployed/productionized pipeline

---

## 📌 Final Score: **MAP@3 = 0.33145**

> This project is one of many short-cycle ML experiments. Not meant to be impressive — meant to be consistent.
