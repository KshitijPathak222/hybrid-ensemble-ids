"""
Merges CIC-IDS2017 (MachineLearningCVE) and UNSW-NB15 datasets into a unified,
clean dataset with a common feature schema.
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = "Merged_Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── CIC-IDS2017 column mapping → unified names ───────────────────────────────
CIC_COL_MAP = {
    " Destination Port":        "dst_port",
    " Flow Duration":           "duration",
    " Total Fwd Packets":       "src_pkts",
    " Total Backward Packets":  "dst_pkts",
    " Total Length of Fwd Packets": "src_bytes",
    " Total Length of Bwd Packets": "dst_bytes",
    " Flow Bytes/s":            "flow_bytes_per_sec",
    " Flow Packets/s":          "flow_pkts_per_sec",
    " Flow IAT Mean":           "flow_iat_mean",
    " Flow IAT Std":            "flow_iat_std",
    " Flow IAT Max":            "flow_iat_max",
    " Flow IAT Min":            "flow_iat_min",
    " Fwd IAT Mean":            "src_iat_mean",
    " Fwd IAT Std":             "src_iat_std",
    " Fwd IAT Max":             "src_iat_max",
    " Fwd IAT Min":             "src_iat_min",
    " Bwd IAT Mean":            "dst_iat_mean",
    " Bwd IAT Std":             "dst_iat_std",
    " Bwd IAT Max":             "dst_iat_max",
    " Bwd IAT Min":             "dst_iat_min",
    " Fwd PSH Flags":           "src_psh_flags",
    " Bwd PSH Flags":           "dst_psh_flags",
    " Fwd URG Flags":           "src_urg_flags",
    " Bwd URG Flags":           "dst_urg_flags",
    " Fwd Header Length":       "src_header_len",
    " Bwd Header Length":       "dst_header_len",
    " Fwd Packets/s":           "src_pkts_per_sec",
    " Bwd Packets/s":           "dst_pkts_per_sec",
    " Min Packet Length":       "pkt_len_min",
    " Max Packet Length":       "pkt_len_max",
    " Packet Length Mean":      "pkt_len_mean",
    " Packet Length Std":       "pkt_len_std",
    " Packet Length Variance":  "pkt_len_var",
    " FIN Flag Count":          "fin_flag_cnt",
    " SYN Flag Count":          "syn_flag_cnt",
    " RST Flag Count":          "rst_flag_cnt",
    " PSH Flag Count":          "psh_flag_cnt",
    " ACK Flag Count":          "ack_flag_cnt",
    " URG Flag Count":          "urg_flag_cnt",
    " CWE Flag Count":          "cwe_flag_cnt",
    " ECE Flag Count":          "ece_flag_cnt",
    " Down/Up Ratio":           "down_up_ratio",
    " Average Packet Size":     "avg_pkt_size",
    " Avg Fwd Segment Size":    "avg_src_seg_size",
    " Avg Bwd Segment Size":    "avg_dst_seg_size",
    " Init_Win_bytes_forward":  "src_win_bytes",
    " Init_Win_bytes_backward": "dst_win_bytes",
    " act_data_pkt_fwd":        "src_act_data_pkts",
    " min_seg_size_forward":    "src_seg_size_min",
    " Idle Mean":               "idle_mean",
    " Idle Std":                "idle_std",
    " Idle Max":                "idle_max",
    " Idle Min":                "idle_min",
    " Label":                   "_raw_label",
}

# CIC attack category mapping (label string → unified category)
CIC_ATTACK_MAP = {
    "BENIGN":                   "Normal",
    "DDoS":                     "DoS",
    "DoS Hulk":                 "DoS",
    "DoS GoldenEye":            "DoS",
    "DoS slowloris":            "DoS",
    "DoS Slowhttptest":         "DoS",
    "Heartbleed":               "Exploits",
    "PortScan":                 "Reconnaissance",
    "FTP-Patator":              "Brute Force",
    "SSH-Patator":              "Brute Force",
    "Bot":                      "Backdoor",
    "Web Attack \x96 Brute Force": "Brute Force",
    "Web Attack \x96 XSS":     "Web Attack",
    "Web Attack \x96 Sql Injection": "Web Attack",
    "Web Attack – Brute Force": "Brute Force",
    "Web Attack – XSS":        "Web Attack",
    "Web Attack – Sql Injection": "Web Attack",
    "Infiltration":             "Reconnaissance",
}

# ─── UNSW-NB15 column mapping → unified names ─────────────────────────────────
UNSW_COL_MAP = {
    "dur":              "duration",
    "spkts":            "src_pkts",
    "dpkts":            "dst_pkts",
    "sbytes":           "src_bytes",
    "dbytes":           "dst_bytes",
    "rate":             "flow_pkts_per_sec",
    "sload":            "flow_bytes_per_sec",   # closest proxy
    "sinpkt":           "src_iat_mean",
    "dinpkt":           "dst_iat_mean",
    "sjit":             "src_iat_std",
    "djit":             "dst_iat_std",
    "swin":             "src_win_bytes",
    "dwin":             "dst_win_bytes",
    "smean":            "avg_src_seg_size",
    "dmean":            "avg_dst_seg_size",
    "sloss":            "src_psh_flags",        # packet loss → reuse slot
    "dloss":            "dst_psh_flags",
    "tcprtt":           "flow_iat_mean",
    "synack":           "syn_flag_cnt",
    "ackdat":           "ack_flag_cnt",
    "attack_cat":       "_raw_attack_cat",
    "label":            "_raw_label",
}

# UNSW attack category normalisation
UNSW_ATTACK_MAP = {
    "Normal":           "Normal",
    "Backdoor":         "Backdoor",
    "Analysis":         "Reconnaissance",
    "Fuzzers":          "Fuzzers",
    "Shellcode":        "Exploits",
    "Reconnaissance":   "Reconnaissance",
    "Exploits":         "Exploits",
    "DoS":              "DoS",
    "Generic":          "Generic",
    "Worms":            "Worms",
}

# Unified numeric columns (present in both after mapping)
UNIFIED_COLS = [
    "duration", "src_pkts", "dst_pkts", "src_bytes", "dst_bytes",
    "flow_bytes_per_sec", "flow_pkts_per_sec",
    "flow_iat_mean", "src_iat_mean", "dst_iat_mean",
    "src_iat_std", "dst_iat_std",
    "src_psh_flags", "dst_psh_flags",
    "src_urg_flags", "dst_urg_flags",
    "src_header_len", "dst_header_len",
    "src_pkts_per_sec", "dst_pkts_per_sec",
    "pkt_len_min", "pkt_len_max", "pkt_len_mean", "pkt_len_std", "pkt_len_var",
    "fin_flag_cnt", "syn_flag_cnt", "rst_flag_cnt",
    "psh_flag_cnt", "ack_flag_cnt", "urg_flag_cnt",
    "down_up_ratio", "avg_pkt_size",
    "avg_src_seg_size", "avg_dst_seg_size",
    "src_win_bytes", "dst_win_bytes",
    "idle_mean", "idle_std", "idle_max", "idle_min",
    # labels
    "attack_category", "label",
]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_numeric(df, cols):
    """Replace inf/nan with 0 and clip extreme values."""
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


# ─── Load & process CIC-IDS2017 ───────────────────────────────────────────────

def load_cic():
    csv_files = glob.glob("MachineLearningCVE/*.csv")
    print(f"[CIC] Found {len(csv_files)} files")
    frames = []
    for f in csv_files:
        print(f"  Loading {os.path.basename(f)} …")
        df = pd.read_csv(f, low_memory=False)
        # strip leading/trailing spaces from column names
        df.columns = df.columns.str.strip()
        # rename with leading-space-stripped map
        col_map = {k.strip(): v for k, v in CIC_COL_MAP.items()}
        df.rename(columns=col_map, inplace=True)
        frames.append(df)

    cic = pd.concat(frames, ignore_index=True)
    print(f"[CIC] Total rows: {len(cic):,}")

    # Build unified label columns
    cic["_raw_label"] = cic["_raw_label"].astype(str).str.strip()
    cic["attack_category"] = cic["_raw_label"].map(CIC_ATTACK_MAP).fillna("Unknown")
    cic["label"] = (cic["attack_category"] != "Normal").astype(int)

    # Keep only unified columns that exist
    keep = [c for c in UNIFIED_COLS if c in cic.columns]
    cic = cic[keep].copy()
    cic["source"] = "CIC-IDS2017"
    return cic


# ─── Load & process UNSW-NB15 ─────────────────────────────────────────────────

def load_unsw():
    train = pd.read_csv("UNSW CSV/Training and Testing Sets/UNSW_NB15_training-set.csv", low_memory=False)
    test  = pd.read_csv("UNSW CSV/Training and Testing Sets/UNSW_NB15_testing-set.csv",  low_memory=False)
    unsw  = pd.concat([train, test], ignore_index=True)
    print(f"[UNSW] Total rows: {len(unsw):,}")

    unsw.rename(columns=UNSW_COL_MAP, inplace=True)

    # Build unified label columns
    unsw["_raw_attack_cat"] = unsw["_raw_attack_cat"].astype(str).str.strip()
    unsw["attack_category"] = unsw["_raw_attack_cat"].map(UNSW_ATTACK_MAP).fillna("Unknown")
    unsw["label"] = unsw["_raw_label"].astype(int)

    keep = [c for c in UNIFIED_COLS if c in unsw.columns]
    unsw = unsw[keep].copy()
    unsw["source"] = "UNSW-NB15"
    return unsw


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cic  = load_cic()
    unsw = load_unsw()

    # Align columns — fill missing cols with 0
    all_cols = list(dict.fromkeys(["source"] + UNIFIED_COLS))
    for df, name in [(cic, "CIC"), (unsw, "UNSW")]:
        for col in all_cols:
            if col not in df.columns:
                df[col] = 0

    merged = pd.concat([cic[all_cols], unsw[all_cols]], ignore_index=True)
    print(f"\n[Merged] Total rows before cleaning: {len(merged):,}")

    # ── Clean numeric columns ──────────────────────────────────────────────────
    num_cols = [c for c in UNIFIED_COLS if c not in ("attack_category", "label")]
    merged = clean_numeric(merged, num_cols)

    # Drop duplicate rows
    before = len(merged)
    merged.drop_duplicates(inplace=True)
    print(f"[Merged] Dropped {before - len(merged):,} duplicate rows")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[Merged] Final shape: {merged.shape}")
    print("\nAttack category distribution:")
    print(merged["attack_category"].value_counts())
    print("\nLabel distribution:")
    print(merged["label"].value_counts())
    print("\nSource distribution:")
    print(merged["source"].value_counts())

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, "merged_ids_dataset.csv")
    merged.to_csv(out_path, index=False)
    print(f"\n✓ Saved to {out_path}")

    # Also save a smaller sample (100k rows) for quick inspection
    sample = merged.sample(n=min(100_000, len(merged)), random_state=42)
    sample_path = os.path.join(OUTPUT_DIR, "merged_ids_sample_100k.csv")
    sample.to_csv(sample_path, index=False)
    print(f"✓ Sample (100k) saved to {sample_path}")


if __name__ == "__main__":
    main()
