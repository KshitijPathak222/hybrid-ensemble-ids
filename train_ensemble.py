"""
Ensemble IDS Model: Random Forest + XGBoost + Neural Network
Soft-voting meta-ensemble trained on the merged CIC-IDS2017 + UNSW-NB15 dataset.

Checkpointed: each model is saved immediately after training.
Re-running the script skips already-trained models.
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "Merged_Output/merged_ids_dataset.csv"
OUTPUT_DIR   = "Ensemble_Output"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "duration","src_pkts","dst_pkts","src_bytes","dst_bytes",
    "flow_bytes_per_sec","flow_pkts_per_sec",
    "flow_iat_mean","src_iat_mean","dst_iat_mean","src_iat_std","dst_iat_std",
    "src_psh_flags","dst_psh_flags","src_urg_flags","dst_urg_flags",
    "src_header_len","dst_header_len","src_pkts_per_sec","dst_pkts_per_sec",
    "pkt_len_min","pkt_len_max","pkt_len_mean","pkt_len_std","pkt_len_var",
    "fin_flag_cnt","syn_flag_cnt","rst_flag_cnt","psh_flag_cnt",
    "ack_flag_cnt","urg_flag_cnt","down_up_ratio","avg_pkt_size",
    "avg_src_seg_size","avg_dst_seg_size","src_win_bytes","dst_win_bytes",
    "idle_mean","idle_std","idle_max","idle_min",
]

def ckpt(name): return os.path.join(OUTPUT_DIR, name)

def load_or_train(pkl_name, model, X_tr, y_tr, label=""):
    path = ckpt(pkl_name)
    if os.path.exists(path):
        print(f"  [SKIP] {label} — loading from {pkl_name}")
        return joblib.load(path)
    print(f"  [TRAIN] {label} …")
    t = time.time()
    model.fit(X_tr, y_tr)
    joblib.dump(model, path)
    print(f"  [DONE]  {label} in {time.time()-t:.1f}s  → {pkl_name}")
    return model

def report(name, model, X_te, y_te, multiclass=False):
    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)
    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="weighted")
    if multiclass:
        auc = roc_auc_score(y_te, y_prob, multi_class="ovr", average="weighted")
    else:
        auc = roc_auc_score(y_te, y_prob[:, 1])
    print(f"  [{name}] Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
    return y_pred, y_prob

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading dataset …")
t0 = time.time()
df = pd.read_csv(DATA_PATH, low_memory=False)
print(f"  {len(df):,} rows  ({time.time()-t0:.1f}s)")

X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
y_bin = df["label"].astype(int).values

le_path = ckpt("label_encoder.pkl")
if os.path.exists(le_path):
    le = joblib.load(le_path)
    y_multi = le.transform(df["attack_category"].astype(str))
else:
    le = LabelEncoder()
    y_multi = le.fit_transform(df["attack_category"].astype(str))
    joblib.dump(le, le_path)
print(f"  Classes: {list(le.classes_)}")

# ── 2. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, yb_tr, yb_te, ym_tr, ym_te = train_test_split(
    X, y_bin, y_multi, test_size=TEST_SIZE,
    random_state=RANDOM_STATE, stratify=y_bin,
)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

sc_path = ckpt("scaler.pkl")
if os.path.exists(sc_path):
    scaler = joblib.load(sc_path)
    X_tr_sc = scaler.transform(X_train)
else:
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    joblib.dump(scaler, sc_path)
X_te_sc = scaler.transform(X_test)

# ── 3. Binary models ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Binary classification (Normal vs Attack) …")

rf_bin = load_or_train("rf_binary.pkl",
    RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=4,
        n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"),
    X_train, yb_tr, "RandomForest-Binary")

xgb_bin = load_or_train("xgb_binary.pkl",
    XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        n_jobs=-1, random_state=RANDOM_STATE, tree_method="hist"),
    X_train, yb_tr, "XGBoost-Binary")

mlp_bin = load_or_train("mlp_binary.pkl",
    MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
        solver="adam", max_iter=50, early_stopping=True,
        validation_fraction=0.1, random_state=RANDOM_STATE),
    X_tr_sc, yb_tr, "MLP-Binary")

print("\nIndividual binary model scores:")
_, p_rf  = report("RF",  rf_bin,  X_test,   yb_te)
_, p_xgb = report("XGB", xgb_bin, X_test,   yb_te)
_, p_mlp = report("MLP", mlp_bin, X_te_sc,  yb_te)

ens_prob = (p_rf[:,1] + p_xgb[:,1] + p_mlp[:,1]) / 3.0
ens_pred = (ens_prob >= 0.5).astype(int)
print(f"\n  [Ensemble-Binary] Acc={accuracy_score(yb_te,ens_pred):.4f}"
      f"  F1={f1_score(yb_te,ens_pred,average='weighted'):.4f}"
      f"  AUC={roc_auc_score(yb_te,ens_prob):.4f}")
print("\nClassification Report (Binary Ensemble):")
print(classification_report(yb_te, ens_pred, target_names=["Normal","Attack"]))

# ── 4. Multi-class models ─────────────────────────────────────────────────────
print("=" * 60)
print("Multi-class classification (attack category) …")

rf_mc = load_or_train("rf_multiclass.pkl",
    RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=4,
        n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"),
    X_train, ym_tr, "RandomForest-MultiClass")

xgb_mc = load_or_train("xgb_multiclass.pkl",
    XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, eval_metric="mlogloss",
        n_jobs=-1, random_state=RANDOM_STATE, tree_method="hist"),
    X_train, ym_tr, "XGBoost-MultiClass")

mlp_mc = load_or_train("mlp_multiclass.pkl",
    MLPClassifier(hidden_layer_sizes=(256,128,64), activation="relu",
        solver="adam", max_iter=50, early_stopping=True,
        validation_fraction=0.1, random_state=RANDOM_STATE),
    X_tr_sc, ym_tr, "MLP-MultiClass")

print("\nIndividual multi-class model scores:")
_, pm_rf  = report("RF-MC",  rf_mc,  X_test,  ym_te, multiclass=True)
_, pm_xgb = report("XGB-MC", xgb_mc, X_test,  ym_te, multiclass=True)
_, pm_mlp = report("MLP-MC", mlp_mc, X_te_sc, ym_te, multiclass=True)

ens_mc_prob = (pm_rf + pm_xgb + pm_mlp) / 3.0
ens_mc_pred = np.argmax(ens_mc_prob, axis=1)
print(f"\n  [Ensemble-MultiClass] F1={f1_score(ym_te,ens_mc_pred,average='weighted'):.4f}")
print("\nClassification Report (Multi-class Ensemble):")
print(classification_report(ym_te, ens_mc_pred, target_names=le.classes_))

print("\n" + "=" * 60)
print("All models saved to Ensemble_Output/")
print("Done.")
