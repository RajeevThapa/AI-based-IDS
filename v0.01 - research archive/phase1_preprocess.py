"""
Phase 1b — Preprocessing Pipeline
CIC-YNU-IoTMal  |  Architecture: X86  |  Modalities: PCAP + SAR

Reads:
    dataset/X86/pcap.csv
    dataset/X86/sar.csv

Writes to ./outputs/:
    pcap_train.csv  /  pcap_test.csv
    sar_train.csv   /  sar_test.csv
    label_encoder.json
    scalers/pcap_scaler.pkl   pcap_imputer.pkl
    scalers/sar_scaler.pkl    sar_imputer.pkl

Run:  python phase1_preprocess.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings("ignore")

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = "./dataset"          # folder that contains X86/
ARCH         = "X86"
PCAP_FILE    = "pcap.csv"
SAR_FILE     = "sar.csv"
# LABEL_COL    = "label"              # <-- update if different in your CSV
LABEL_COL = "MalwareFamily"         # ← Change from "label"
TEST_SIZE    = 0.2
RANDOM_STATE = 42
APPLY_SMOTE  = True                 # set False if RAM is limited

OUTPUT_DIR   = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "scalers"), exist_ok=True)

# ── Identifier columns to drop from PCAP (prevent data leakage) ──────────────
# Add any extra columns you spotted in phase1_load_explore.py output
# PCAP_DROP_COLS = [
#     "src_ip", "dst_ip",
#     "src_port", "dst_port",
#     "src_mac", "dst_mac",
#     "timestamp", "flow_id",
#     "flow_start", "flow_end",
#     "flow_start_time", "flow_end_time",
# ]
PCAP_DROP_COLS = [
    "Hash",           # ← Flow identifier (leakage!)
    "Arch"         # ← Architecture (constant across dataset)
    # "MalwareFamily"   # ← Target (handled separately anyway)
]

# ── Keywords to auto-select resource columns from SAR ────────────────────────
# Add more if your SAR column names don't match these
SAR_FEATURE_KEYWORDS = [
    "cpu", "mem", "memory", "io", "disk",
    "net", "network", "load", "swap", "proc", "usr", "sys", "idle"
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_csv(path: str, name: str) -> pd.DataFrame:
    """Load CSV, strip column name whitespace, report shape."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found at: {path}\n"
            f"Update DATASET_ROOT or file name at the top of this script."
        )
    print(f"  Reading {name} ...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    print(f"  Loaded  {len(df):,} rows  {df.shape[1]} columns")
    return df


def fit_label_encoder(series: pd.Series) -> LabelEncoder:
    """Fit LabelEncoder and save int → class name mapping to JSON."""
    le = LabelEncoder()
    le.fit(series.astype(str))
    mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
    path = os.path.join(OUTPUT_DIR, "label_encoder.json")
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n  Label encoder fitted — {len(le.classes_)} classes:")
    for i, cls in enumerate(le.classes_):
        print(f"    {i}  →  {cls}")
    print(f"  Saved: {path}")
    return le


def apply_smote(X: np.ndarray, y: np.ndarray, name: str):
    """Oversample minority classes on the training set."""
    unique, counts = np.unique(y, return_counts=True)
    before = {int(u): int(c) for u, c in zip(unique, counts)}
    print(f"\n  SMOTE [{name}] before: {before}")

    min_samples = int(counts.min())
    k = min(5, min_samples - 1)
    if k < 1:
        print(f"  SMOTE [{name}] SKIPPED — a class has only {min_samples} sample(s)")
        return X, y

    sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
    X_res, y_res = sm.fit_resample(X, y)

    unique2, counts2 = np.unique(y_res, return_counts=True)
    after = {int(u): int(c) for u, c in zip(unique2, counts2)}
    print(f"  SMOTE [{name}] after : {after}")
    return X_res, y_res


def save_csv(X_tr, X_te, y_tr, y_te, name: str, feature_names: list):
    """Save train/test splits as CSV files."""
    train_df = pd.DataFrame(X_tr, columns=feature_names)
    train_df[LABEL_COL] = y_tr
    test_df = pd.DataFrame(X_te, columns=feature_names)
    test_df[LABEL_COL] = y_te

    train_path = os.path.join(OUTPUT_DIR, f"{name}_train.csv")
    test_path  = os.path.join(OUTPUT_DIR, f"{name}_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    print(f"\n  Saved: {train_path}  ({len(train_df):,} rows)")
    print(f"  Saved: {test_path}    ({len(test_df):,} rows)")


# ── PCAP PREPROCESSING ────────────────────────────────────────────────────────

# def preprocess_pcap(df: pd.DataFrame, le: LabelEncoder):
#     print(f"\n  Input shape : {df.shape}")

#     # Drop identifier / leakage columns
#     drop = [c for c in PCAP_DROP_COLS if c in df.columns]
#     if drop:
#         print(f"  Dropping leakage cols : {drop}")
#     df = df.drop(columns=drop)

#     # Check label column exists
#     if LABEL_COL not in df.columns:
#         raise ValueError(
#             f"Label column '{LABEL_COL}' not found.\n"
#             f"Update LABEL_COL. Available columns: {list(df.columns)}"
#         )

#     y_raw = df[LABEL_COL].astype(str)
#     X = df.drop(columns=[LABEL_COL])

def preprocess_pcap(df: pd.DataFrame, le: LabelEncoder):
    print(f"\n  Input shape : {df.shape}")

    # Drop identifier / leakage columns (EXCLUDE label!)
    drop = [c for c in PCAP_DROP_COLS if c in df.columns]  # ← NO CHANGE HERE
    if drop:
        print(f"  Dropping leakage cols : {drop}")
    df = df.drop(columns=drop)

    # Label check stays EXACTLY the same
    if LABEL_COL not in df.columns:
        raise ValueError(...)
    
    y_raw = df[LABEL_COL].astype(str)  # ← This works now!
    X = df.drop(columns=[LABEL_COL])   # ← Label removed here

    # Drop non-numeric columns (after label is separated)
    non_num = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        print(f"  Dropping non-numeric cols : {non_num}")
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)
    print(f"  Numeric features : {len(feature_names)}")

    # Replace inf values
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing with median
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    joblib.dump(imputer, os.path.join(OUTPUT_DIR, "scalers", "pcap_imputer.pkl"))

    # Encode labels
    y = le.transform(y_raw)

    # Stratified train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Train : {len(X_tr):,} rows   Test : {len(X_te):,} rows")

    # SMOTE on train only
    if APPLY_SMOTE:
        X_tr, y_tr = apply_smote(X_tr, y_tr, "PCAP")

    # Scale — fit on train, apply to both
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scalers", "pcap_scaler.pkl"))
    print(f"  Scaler saved → outputs/scalers/pcap_scaler.pkl")

    return X_tr, X_te, y_tr, y_te, feature_names


# ── SAR PREPROCESSING ─────────────────────────────────────────────────────────

def preprocess_sar(df: pd.DataFrame, le: LabelEncoder):
    print(f"\n  Input shape : {df.shape}")

    # Drop non-feature columns
    drop = [c for c in ["timestamp", "datetime", "time", "date"] if c in df.columns]
    if drop:
        print(f"  Dropping time cols : {drop}")
    df = df.drop(columns=drop)

    if LABEL_COL not in df.columns:
        raise ValueError(
            f"Label column '{LABEL_COL}' not found in SAR.\n"
            f"Update LABEL_COL. Available columns: {list(df.columns)}"
        )

    y_raw = df[LABEL_COL].astype(str)
    X = df.drop(columns=[LABEL_COL])

    # Select resource metric columns by keyword
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    resource_cols = [
        c for c in numeric_cols
        if any(kw in c.lower() for kw in SAR_FEATURE_KEYWORDS)
    ]

    if not resource_cols:
        print(f"  [!] No columns matched SAR_FEATURE_KEYWORDS.")
        print(f"      Using ALL {len(numeric_cols)} numeric columns.")
        print(f"      Add keywords to SAR_FEATURE_KEYWORDS if you want to filter.")
        resource_cols = numeric_cols
    else:
        print(f"  Resource cols selected : {len(resource_cols)}")
        for c in resource_cols:
            print(f"    {c}")

    X = X[resource_cols]
    feature_names = list(X.columns)

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    joblib.dump(imputer, os.path.join(OUTPUT_DIR, "scalers", "sar_imputer.pkl"))

    y = le.transform(y_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Train : {len(X_tr):,} rows   Test : {len(X_te):,} rows")

    if APPLY_SMOTE:
        X_tr, y_tr = apply_smote(X_tr, y_tr, "SAR")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scalers", "sar_scaler.pkl"))
    print(f"  Scaler saved → outputs/scalers/sar_scaler.pkl")

    return X_tr, X_te, y_tr, y_te, feature_names


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    base = os.path.join(DATASET_ROOT, ARCH)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Loading CSVs")
    print("="*65)
    pcap = load_csv(os.path.join(base, PCAP_FILE), "PCAP")
    sar  = load_csv(os.path.join(base, SAR_FILE),  "SAR")

    # ── Label encoder ─────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Fitting label encoder")
    print("="*65)
    all_labels = []
    for df in [pcap, sar]:
        if LABEL_COL in df.columns:
            all_labels.append(df[LABEL_COL].astype(str))
    if not all_labels:
        raise RuntimeError(f"Label column '{LABEL_COL}' not found in either CSV.")
    le = fit_label_encoder(pd.concat(all_labels))

    # ── PCAP ──────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Preprocessing PCAP")
    print("="*65)
    X_ptr, X_pte, y_ptr, y_pte, pcap_feats = preprocess_pcap(pcap, le)
    save_csv(X_ptr, X_pte, y_ptr, y_pte, "pcap", pcap_feats)
    del pcap, X_ptr, X_pte      # free RAM before SAR

    # ── SAR ───────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Preprocessing SAR")
    print("="*65)
    X_str, X_ste, y_str, y_ste, sar_feats = preprocess_sar(sar, le)
    save_csv(X_str, X_ste, y_str, y_ste, "sar", sar_feats)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  DONE")
    print("="*65)
    print("""
  Files created in ./outputs/:

    pcap_train.csv          scaled + SMOTE-balanced PCAP features
    pcap_test.csv           scaled PCAP test set (20%)
    sar_train.csv           scaled + SMOTE-balanced SAR features
    sar_test.csv            scaled SAR test set (20%)
    label_encoder.json      int ↔ malware family name
    scalers/pcap_scaler.pkl  ← keep these for inference later
    scalers/pcap_imputer.pkl
    scalers/sar_scaler.pkl
    scalers/sar_imputer.pkl

  Next step:  python phase1_eda.py   (recommended)
          or: python phase2_models.py
""")


if __name__ == "__main__":
    main()
