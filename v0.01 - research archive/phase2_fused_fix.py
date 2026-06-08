"""
Phase 2 — Scenario 3 FIXED: PCAP + SAR Fused
CIC-YNU-IoTMal  |  Architecture: X86

Problem with the original fusion:
  Joining on Hash then aggregating with mean() collapsed 455K/529K rows
  into only 4,435 samples. Minority classes (Gafgyt=10, Generic=12) had
  too few samples to learn from → F1=0.00 for those classes.

Fix: Row-level fusion
  1. Each PCAP flow belongs to a Hash (malware sample)
  2. Each SAR row belongs to a Hash (same sample, recorded every second)
  3. For every PCAP flow, append the MEAN SAR features for that same Hash
     This gives a 455K-row fused dataset where each PCAP flow has
     its corresponding system-level context attached

This preserves the full row count, fixes minority class starvation,
and gives the model both network AND system context per observation.

Run:  python phase2_fused_fix.py
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.95)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = "./dataset"
ARCH         = "X86"
OUTPUT_DIR   = "./outputs"
MODEL_DIR    = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR     = os.path.join(OUTPUT_DIR, "phase2_plots")
LABEL_COL    = "MalwareFamily"
RANDOM_STATE = 42
TEST_SIZE    = 0.2

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

RF_PARAMS = dict(
    n_estimators     = 200,
    max_depth        = 20,
    min_samples_leaf = 2,
    class_weight     = "balanced",
    n_jobs           = -1,
    random_state     = RANDOM_STATE,
)

PCAP_DROP = ["Arch", "LLC", "ARP", "Tot size", "fin_count",
             "rst_count", "syn_count", "Tot sum"]

SAR_DROP  = ["interrupts[4].CPU0", "disk[0].rkB",
             "network.net-ip.fragok", "network.net-udp.odgm",
             "timestamp", "interval", "Arch"]

SAR_KEYWORDS = ["cpu", "mem", "io", "disk", "net", "load", "swap",
                "proc", "usr", "sys", "idle", "paging", "queue",
                "memory", "network", "filesys", "interrupt", "softnet",
                "power", "serial", "hugepage", "kernel"]


def load_label_encoder() -> dict:
    with open(os.path.join(OUTPUT_DIR, "label_encoder.json")) as f:
        return {int(k): v for k, v in json.load(f).items()}


# ── STEP 1: Build SAR profile per Hash ───────────────────────────────────────

def build_sar_profile(sar_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean of all numeric SAR features per Hash.
    Returns DataFrame indexed by Hash with 'sar_' prefixed columns.
    Each PCAP row will look up its Hash here to get system context.
    """
    # Select numeric resource columns only
    num_cols = [
        c for c in sar_raw.select_dtypes(include=[np.number]).columns
        if any(kw in c.lower() for kw in SAR_KEYWORDS)
        and c not in SAR_DROP
    ]
    print(f"  SAR numeric resource cols selected: {len(num_cols)}")

    sar_profile = (sar_raw
                   .groupby("Hash")[num_cols]
                   .mean()
                   .reset_index()
                   .set_index("Hash"))
    sar_profile.columns = ["sar_" + c for c in sar_profile.columns]
    print(f"  SAR profile shape (one row per malware hash): {sar_profile.shape}")
    return sar_profile


# ── STEP 2: Attach SAR profile to every PCAP row ─────────────────────────────

def build_fused_dataset(pcap_raw: pd.DataFrame,
                        sar_profile: pd.DataFrame) -> pd.DataFrame:
    """
    For each PCAP flow row, attach the SAR mean profile for its Hash.
    Result: every PCAP row has both network AND system features.
    Preserves full ~455K row count.
    """
    pcap_drop_cols = [c for c in PCAP_DROP if c in pcap_raw.columns]
    pcap_raw = pcap_raw.drop(columns=pcap_drop_cols)
    pcap_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Left join: every PCAP row gets SAR context for its Hash
    fused = pcap_raw.set_index("Hash").join(sar_profile, how="left")
    fused = fused.reset_index(drop=True)

    # How many PCAP rows got SAR context?
    matched = fused[sar_profile.columns[0]].notna().sum()
    print(f"  PCAP rows matched with SAR context: {matched:,} / {len(fused):,} "
          f"({matched/len(fused)*100:.1f}%)")

    return fused


# ── STEP 3: Preprocess & split ────────────────────────────────────────────────

def preprocess_fused(fused: pd.DataFrame, le_map: dict):
    """
    Encode labels, impute, SMOTE, scale. Returns train/test arrays.
    """
    if LABEL_COL not in fused.columns:
        raise ValueError(f"Label column '{LABEL_COL}' not found.")

    y_raw = fused[LABEL_COL].astype(str)
    X = fused.drop(columns=[LABEL_COL])

    # Keep only numeric
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)
    print(f"  Total features after fusion: {len(feature_names)}")

    # Encode labels
    inv_le = {v: k for k, v in le_map.items()}
    y = y_raw.map(inv_le).values.astype(int)

    # Impute
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)
    joblib.dump(imputer, os.path.join(OUTPUT_DIR, "scalers", "fused_imputer.pkl"))

    # Stratified split BEFORE SMOTE
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_imp, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )
    print(f"  Train: {len(X_tr):,}   Test: {len(X_te):,}")

    # SMOTE on train only
    unique, counts = np.unique(y_tr, return_counts=True)
    print(f"  SMOTE before: { {int(u): int(c) for u, c in zip(unique, counts)} }")
    min_samples = int(counts.min())
    k = min(5, min_samples - 1) if min_samples <= 5 else 5
    if k >= 1:
        sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        unique2, counts2 = np.unique(y_tr, return_counts=True)
        print(f"  SMOTE after : { {int(u): int(c) for u, c in zip(unique2, counts2)} }")

    # Scale
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scalers", "fused_scaler_v2.pkl"))

    return X_tr, X_te, y_tr, y_te, feature_names


# ── STEP 4: Train & Evaluate ──────────────────────────────────────────────────

def train_and_evaluate(X_tr, X_te, y_tr, y_te,
                       feature_names: list, le_map: dict):
    class_names = [le_map[i] for i in sorted(le_map.keys())]

    print(f"\n  Training Random Forest [fused_v2] ...")
    t0 = time.time()
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr, y_tr)
    print(f"  Training time: {time.time() - t0:.1f}s")
    joblib.dump(clf, os.path.join(MODEL_DIR, "rf_fused_v2.pkl"))
    print(f"  Model saved → outputs/models/rf_fused_v2.pkl")

    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average="macro")
    f1_wtd = f1_score(y_te, y_pred, average="weighted")

    print(f"\n  ── Results: fused_v2 ────────────────────────────────────")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (macro)      : {f1_mac:.4f}")
    print(f"  F1 (weighted)   : {f1_wtd:.4f}")
    print()
    print(classification_report(y_te, y_pred, target_names=class_names))

    # Save report
    report_str = classification_report(y_te, y_pred, target_names=class_names)
    with open(os.path.join(PLOT_DIR, "report_fused_v2.txt"), "w") as f:
        f.write(f"Scenario: fused_v2 (row-level join)\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1 macro : {f1_mac:.4f}\n\n")
        f.write(report_str)

    # Confusion matrix
    cm      = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Counts", "Normalised (recall per class)"],
        ["d", ".2f"]
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, cbar=True, linewidths=0.4)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {title}\nfused_v2")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "confusion_fused_v2.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # Feature importance
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:20]
    top_names  = [feature_names[i] for i in idx]
    top_values = importances[idx]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top_names[::-1], top_values[::-1], color="#5b8db8", edgecolor="none")
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_title("Top 20 features — fused_v2 (PCAP + SAR row-level join)")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "importance_fused_v2.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return {"scenario": "fused_v2", "accuracy": round(acc, 4),
            "f1_macro": round(f1_mac, 4), "f1_weighted": round(f1_wtd, 4)}


# ── STEP 5: Updated comparison table ─────────────────────────────────────────

def print_updated_comparison(new_result: dict):
    """Load original results CSV and append the new fused_v2 result."""
    csv_path = os.path.join(OUTPUT_DIR, "scenario_comparison.csv")

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        # Remove old fused row if present, add fused_v2
        df = df[df.index != "fused_v2"]
        new_row = pd.DataFrame([new_result]).set_index("scenario")
        df = pd.concat([df, new_row])
    else:
        df = pd.DataFrame([new_result]).set_index("scenario")

    df.to_csv(csv_path)

    print("\n\n" + "="*65)
    print("  UPDATED ABLATION STUDY — ALL 4 SCENARIOS")
    print("="*65)
    print(f"\n  {'Scenario':<22} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*12}")
    for scenario, row in df.iterrows():
        marker = " ← NEW" if scenario == "fused_v2" else ""
        print(f"  {scenario:<22} {row['accuracy']:>10.4f} {row['f1_macro']:>10.4f} "
              f"{row['f1_weighted']:>12.4f}{marker}")
    print(f"\n  Best (F1 macro): {df['f1_macro'].idxmax()}")
    print(f"\n  Saved: {csv_path}")

    # Updated bar chart with 4 scenarios
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    titles  = ["Accuracy", "F1 Score (Macro)", "F1 Score (Weighted)"]
    colors  = ["#5b8db8", "#e07b54", "#6aaa7b"]
    scenario_colors = {
        "pcap_only":  "#aac4dd",
        "sar_only":   "#e07b54",
        "fused":      "#cc8866",
        "fused_v2":   "#3a7abf",
    }

    for ax, metric, title in zip(axes, metrics, titles):
        vals = df[metric]
        bar_colors = [scenario_colors.get(s, "#888") for s in vals.index]
        bars = ax.bar(vals.index, vals.values, color=bar_colors,
                      edgecolor="none", width=0.55)
        ax.set_title(title, pad=8)
        ax.set_ylim(max(0, vals.min() - 0.08), min(1.02, vals.max() + 0.06))
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=20, labelsize=8)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{x:.2f}")
        )
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.003,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(
        "Ablation Study — PCAP only | SAR only | Fused (Hash-agg) | Fused v2 (row-level)",
        fontsize=11, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "scenario_comparison_updated.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    base   = os.path.join(DATASET_ROOT, ARCH)
    le_map = load_label_encoder()

    # Load raw CSVs
    print("\n" + "="*65)
    print("  Loading raw CSVs")
    print("="*65)
    pcap_raw = pd.read_csv(os.path.join(base, "pcap.csv"), low_memory=False)
    sar_raw  = pd.read_csv(os.path.join(base, "sar.csv"),  low_memory=False)
    pcap_raw.columns = pcap_raw.columns.str.strip()
    sar_raw.columns  = sar_raw.columns.str.strip()
    print(f"  PCAP: {pcap_raw.shape}   SAR: {sar_raw.shape}")

    # Build SAR profile per Hash (mean of all numeric columns)
    print("\n" + "="*65)
    print("  Step 1 — Build SAR profile per Hash")
    print("="*65)
    sar_profile = build_sar_profile(sar_raw)

    # Attach SAR profile to every PCAP row (row-level join)
    print("\n" + "="*65)
    print("  Step 2 — Attach SAR context to each PCAP flow row")
    print("="*65)
    fused = build_fused_dataset(pcap_raw, sar_profile)
    print(f"  Fused dataset shape: {fused.shape}")
    print(f"  Label distribution:\n{fused[LABEL_COL].value_counts().to_string()}")

    # Preprocess
    print("\n" + "="*65)
    print("  Step 3 — Preprocess (impute → SMOTE → scale)")
    print("="*65)
    X_tr, X_te, y_tr, y_te, feat = preprocess_fused(fused, le_map)

    # Train and evaluate
    print("\n" + "="*65)
    print("  Step 4 — Train and evaluate")
    print("="*65)
    result = train_and_evaluate(X_tr, X_te, y_tr, y_te, feat, le_map)

    # Updated comparison
    print_updated_comparison(result)

    print("\n\n" + "="*65)
    print("  DONE")
    print("="*65)
    print("""
  New files:
    outputs/models/rf_fused_v2.pkl
    outputs/phase2_plots/confusion_fused_v2.png
    outputs/phase2_plots/importance_fused_v2.png
    outputs/phase2_plots/scenario_comparison_updated.png
    outputs/scenario_comparison.csv   (updated with fused_v2 row)
""")


if __name__ == "__main__":
    main()
