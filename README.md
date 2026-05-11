# Hybrid Ensemble Intrusion Detection System
### Random Forest + XGBoost + Neural Network | CIC-IDS2017 + UNSW-NB15

---

## Overview

A machine learning-based Network Intrusion Detection System (IDS) that merges two
benchmark datasets and trains a soft-voting ensemble of three models to detect
cyberattacks in network traffic.

- **Binary classification** — Normal (0) vs Attack (1)
- **Multi-class classification** — Identifies specific attack type (DoS, Brute Force, etc.)

---

## Results

| Model | Binary Accuracy | F1 Score | AUC-ROC |
|---|---|---|---|
| Random Forest | 98.85% | 98.86% | 99.96% |
| XGBoost | 99.30% | 99.30% | 99.97% |
| Neural Network | 98.88% | 98.88% | 99.91% |
| **Ensemble** | **99.16%** | **99.17%** | **99.97%** |

Multi-class Ensemble: **98.75% Accuracy | 98.84% F1 | 99.96% AUC-ROC**

---

## Datasets

| Dataset | Rows | Features | Source |
|---|---|---|---|
| CIC-IDS2017 | 2,830,743 | 80 | Canadian Institute for Cybersecurity |
| UNSW-NB15 | 257,673 | 43 | University of New South Wales |
| **Merged** | **2,363,245** | **41** | This project |

> **Note:** Datasets are not included in this repo due to size.
> Download from:
> - [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)
> - [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

---

## Attack Categories Detected

`Normal` `DoS` `DDoS` `Brute Force` `Reconnaissance` `Exploits`
`Fuzzers` `Backdoor` `Worms` `Generic` `Web Attack` `Botnet`

---

## Project Structure

```
├── merge_datasets.py       # Merges CIC-IDS2017 + UNSW-NB15 into unified dataset
├── train_ensemble.py       # Trains RF + XGBoost + MLP with checkpointing
├── IDS_Ensemble.ipynb      # Full notebook with EDA, training, evaluation, demo
├── Code_Explanation.md     # Simple explanation of all code and libraries
├── Viva_Preparation.md     # 300+ Q&A for project defense and viva
├── Ensemble_Output/        # Saved .pkl model files (generated after training)
└── Merged_Output/          # Merged dataset CSV (generated after merge step)
```

---

## Setup

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib
pip install notebook nbformat nbclient ipykernel
```

---

## How to Run

### Step 1 — Merge the datasets
```bash
python merge_datasets.py
```
Outputs: `Merged_Output/merged_ids_dataset.csv`

### Step 2 — Train the ensemble
```bash
python train_ensemble.py
```
Outputs: 6 model `.pkl` files in `Ensemble_Output/`
Checkpointed — safe to restart if interrupted.

### Step 3 — Run the notebook
```bash
jupyter notebook IDS_Ensemble.ipynb
```
Or run all cells via: Kernel → Restart & Run All

---

## Inference — Predict on New Data

```python
import joblib, numpy as np, pandas as pd

# Load models
rf_bin  = joblib.load("Ensemble_Output/rf_binary.pkl")
xgb_bin = joblib.load("Ensemble_Output/xgb_binary.pkl")
mlp_bin = joblib.load("Ensemble_Output/mlp_binary.pkl")
rf_mc   = joblib.load("Ensemble_Output/rf_multiclass.pkl")
xgb_mc  = joblib.load("Ensemble_Output/xgb_multiclass.pkl")
mlp_mc  = joblib.load("Ensemble_Output/mlp_multiclass.pkl")
scaler  = joblib.load("Ensemble_Output/scaler.pkl")
le      = joblib.load("Ensemble_Output/label_encoder.pkl")

def predict_ensemble(X_new):
    X_raw = X_new.fillna(0)
    X_sc  = scaler.transform(X_raw)
    p_bin = (rf_bin.predict_proba(X_raw)[:,1] +
             xgb_bin.predict_proba(X_raw)[:,1] +
             mlp_bin.predict_proba(X_sc)[:,1]) / 3.0
    p_mc  = (rf_mc.predict_proba(X_raw) +
             xgb_mc.predict_proba(X_raw) +
             mlp_mc.predict_proba(X_sc)) / 3.0
    return pd.DataFrame({
        "binary_label":      (p_bin >= 0.5).astype(int),
        "attack_category":   le.inverse_transform(np.argmax(p_mc, axis=1)),
        "attack_confidence": p_bin.round(4),
    })
```

---

## Tech Stack

`Python 3` `scikit-learn` `XGBoost` `pandas` `NumPy`
`matplotlib` `seaborn` `joblib` `Jupyter Notebook`

---

## Models

| Model | Config |
|---|---|
| Random Forest | 200 trees, max_depth=20, class_weight=balanced |
| XGBoost | 300 trees, max_depth=8, lr=0.1, subsample=0.8 |
| MLP Neural Network | 256→128→64, ReLU, Adam, early stopping |
| Ensemble | Soft voting — average of predicted probabilities |
