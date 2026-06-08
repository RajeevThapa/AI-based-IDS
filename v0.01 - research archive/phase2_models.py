"""
Phase 2 — ML Detection Models  |  Ablation Study (3 Scenarios)
CIC-YNU-IoTMal  |  Architecture: X86

Scenario 1 — PCAP only       (network traffic features)
Scenario 2 — SAR only        (system resource features)
Scenario 3 — PCAP + SAR      (fused: joined on Hash column from raw CSVs)

Model: Random Forest (fast, interpretable, strong baseline for tabular data)

Each scenario produces:
  - Classification report (precision / recall / F1 per class)
  - Confusion matrix plot
  - Feature importance plot
  - Saved model (.pkl)

Final output: comparison table across all 3 scenarios

Run:  python phase2_models.py
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    f1_score, accuracy_score
)
import joblib

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", font_scale=0.95)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT  = "./dataset"
ARCH          = "X86"
OUTPUT_DIR    = "./outputs"
MODEL_DIR     = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR      = os.path.join(OUTPUT_DIR, "phase2_plots")
LABEL_COL     = "MalwareFamily"   # original label column name in raw CSVs
ENCODED_LABEL = "label"           # integer label column name in preprocessed CSVs

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)

# Random Forest hyperparameters
RF_PARAMS = dict(
    n_estimators  = 200,
    max_depth     = 20,
    min_samples_leaf = 2,
    class_weight  = "balanced",    # extra safety on top of SMOTE
    n_jobs        = -1,            # use all CPU cores
    random_state  = 42,
)

# Highly correlated PCAP pairs to drop (one from each — identified in EDA)
PCAP_DROP_REDUNDANT = [
    "LLC",          # r=1.0 with IPv
    "ARP",          # r=-1.0 with IPv
    "Tot size",     # r=1.0 with AVG
    "fin_count",    # r=1.0 with fin_flag_number
    "rst_count",    # r=1.0 with rst_flag_number
    "syn_count",    # r=0.999 with syn_flag_number
    "Tot sum",      # r=0.989 with AVG
]

# Highly correlated SAR pairs to drop (one from each — identified in EDA)
SAR_DROP_REDUNDANT = [
    "interrupts[4].CPU0",   # r=1.0 with interrupts[3].CPU0
    "disk[0].rkB",          # r=1.0 with disk[0].rd_sec
    "network.net-ip.fragok", # r=0.999 with network.net-ip.fragcrt
    "network.net-udp.odgm", # r=0.982 with network.net-udp.idgm
]


# ── LABEL ENCODER ────────────────────────────────────────────────────────────

def load_label_encoder() -> dict:
    path = os.path.join(OUTPUT_DIR, "label_encoder.json")
    with open(path) as f:
        le = json.load(f)
    return {int(k): v for k, v in le.items()}   # {0: "Benign", 1: "DarkNexus", ...}


# ── DATA LOADING ─────────────────────────────────────────────────────────────

# def load_preprocessed(name: str) -> tuple:
    """Load train/test CSVs produced by phase1_preprocess.py."""
    train = pd.read_csv(os.path.join(OUTPUT_DIR, f"{name}_train.csv"))
    test  = pd.read_csv(os.path.join(OUTPUT_DIR, f"{name}_test.csv"))

    y_tr = train[ENCODED_LABEL].values
    y_te = test[ENCODED_LABEL].values
    X_tr = train.drop(columns=[ENCODED_LABEL])
    X_te = test.drop(columns=[ENCODED_LABEL])

    # Drop redundant correlated features
    drop_cols = PCAP_DROP_REDUNDANT if name == "pcap" else SAR_DROP_REDUNDANT
    drop_cols = [c for c in drop_cols if c in X_tr.columns]
    if drop_cols:
        X_tr = X_tr.drop(columns=drop_cols)
        X_te = X_te.drop(columns=drop_cols)
        print(f"  Dropped {len(drop_cols)} redundant columns from {name.upper()}")

    print(f"  {name.upper():4s} train: {X_tr.shape}   test: {X_te.shape}")
    return X_tr.values, X_te.values, y_tr, y_te, list(X_tr.columns)

def load_preprocessed(name: str) -> tuple:
    """Load train/test CSVs produced by phase1_preprocess.py."""
    train_path = os.path.join(OUTPUT_DIR, f"{name}_train.csv")
    test_path  = os.path.join(OUTPUT_DIR, f"{name}_test.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Preprocessed file missing: {train_path}\nRun phase1_preprocess.py first!")
    
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    
    print(f"  {name.upper()} train columns: {list(train.columns)}")
    print(f"  {name.upper()} test  columns: {list(test.columns)}")

    # FIXED: Use the SAME label column name as Phase 1
    label_col = LABEL_COL  # "MalwareFamily" — matches your Phase 1 output
    
    if label_col not in train.columns or label_col not in test.columns:
        raise KeyError(f"Label column '{label_col}' not found!\n"
                      f"Available columns: {list(train.columns)}")
    
    print(f"  Using label column: '{label_col}' (encoded integers)")
    
    y_tr = train[label_col].values
    y_te = test[label_col].values
    X_tr = train.drop(columns=[label_col])
    X_te = test.drop(columns=[label_col])

    # Drop redundant correlated features
    drop_cols = PCAP_DROP_REDUNDANT if name == "pcap" else SAR_DROP_REDUNDANT
    drop_cols = [c for c in drop_cols if c in X_tr.columns]
    if drop_cols:
        X_tr = X_tr.drop(columns=drop_cols)
        X_te = X_te.drop(columns=drop_cols)
        print(f"  Dropped {len(drop_cols)} redundant columns from {name.upper()}")

    print(f"  {name.upper():4s} train: {X_tr.shape}   test: {X_te.shape}")
    return X_tr.values, X_te.values, y_tr, y_te, list(X_tr.columns)

def load_fused() -> tuple:
    """
    Scenario 3: Join PCAP and SAR on Hash column from raw CSVs.
    Each Hash identifies one malware sample execution.
    We aggregate PCAP flows and SAR rows per Hash (mean),
    then join into a single wide feature table.
    """
    print("\n  Building fused dataset from raw CSVs (joining on Hash) ...")
    base = os.path.join(DATASET_ROOT, ARCH)

    pcap_raw = pd.read_csv(os.path.join(base, "pcap.csv"), low_memory=False)
    sar_raw  = pd.read_csv(os.path.join(base, "sar.csv"),  low_memory=False)
    pcap_raw.columns = pcap_raw.columns.str.strip()
    sar_raw.columns  = sar_raw.columns.str.strip()

    print(f"  Raw PCAP: {pcap_raw.shape}   Raw SAR: {sar_raw.shape}")

    # ── Prepare PCAP: keep numeric features + Hash + label ───────────────────
    pcap_drop = ["Arch"] + PCAP_DROP_REDUNDANT
    pcap_drop = [c for c in pcap_drop if c in pcap_raw.columns]
    pcap_raw  = pcap_raw.drop(columns=pcap_drop)
    pcap_raw.replace([np.inf, -np.inf], np.nan, inplace=True)

    pcap_num_cols = pcap_raw.select_dtypes(include=[np.number]).columns.tolist()
    # Aggregate flows per Hash → mean of all numeric flow features
    pcap_agg = (pcap_raw
                .groupby("Hash")[pcap_num_cols]
                .mean()
                .reset_index())
    # Bring label back
    pcap_labels = (pcap_raw[["Hash", LABEL_COL]]
                   .drop_duplicates("Hash")
                   .set_index("Hash"))
    pcap_agg = pcap_agg.set_index("Hash").join(pcap_labels)
    pcap_agg.columns = ["pcap_" + c for c in pcap_agg.columns[:-1]] + [LABEL_COL]
    print(f"  PCAP aggregated per Hash: {pcap_agg.shape}")

    # ── Prepare SAR: keep numeric resource cols + Hash ────────────────────────
    sar_keywords = ["cpu", "mem", "io", "disk", "net", "load", "swap",
                    "proc", "usr", "sys", "idle", "interrupt", "paging",
                    "queue", "memory", "network", "filesys"]
    sar_num_cols = [c for c in sar_raw.select_dtypes(include=[np.number]).columns
                    if any(kw in c.lower() for kw in sar_keywords)]
    sar_drop_red = [c for c in SAR_DROP_REDUNDANT if c in sar_raw.columns]
    sar_num_cols = [c for c in sar_num_cols if c not in sar_drop_red]

    sar_agg = (sar_raw
               .groupby("Hash")[sar_num_cols]
               .mean()
               .reset_index()
               .set_index("Hash"))
    sar_agg.columns = ["sar_" + c for c in sar_agg.columns]
    print(f"  SAR  aggregated per Hash: {sar_agg.shape}")

    # ── Inner join on Hash ────────────────────────────────────────────────────
    fused = pcap_agg.join(sar_agg, how="inner")
    fused = fused.dropna(subset=[LABEL_COL])
    print(f"  Fused (inner join): {fused.shape}")
    print(f"  Label distribution:\n{fused[LABEL_COL].value_counts().to_string()}")

    # ── Encode labels ─────────────────────────────────────────────────────────
    le_map = load_label_encoder()
    inv_le = {v: k for k, v in le_map.items()}   # "Benign" → 0
    fused[ENCODED_LABEL] = fused[LABEL_COL].map(inv_le)
    fused = fused.drop(columns=[LABEL_COL])
    fused = fused.dropna(subset=[ENCODED_LABEL])
    fused[ENCODED_LABEL] = fused[ENCODED_LABEL].astype(int)

    # Fill any remaining NaN in features
    fused = fused.fillna(fused.median(numeric_only=True))

    # ── Scale fused features ──────────────────────────────────────────────────
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    y = fused[ENCODED_LABEL].values
    X = fused.drop(columns=[ENCODED_LABEL]).values
    feature_names = [c for c in fused.columns if c != ENCODED_LABEL]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scalers", "fused_scaler.pkl"))

    print(f"  Fused train: {X_tr.shape}   test: {X_te.shape}")
    return X_tr, X_te, y_tr, y_te, feature_names


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train_random_forest(X_tr, y_tr, scenario_name: str) -> RandomForestClassifier:
    print(f"\n  Training Random Forest [{scenario_name}] ...")
    t0 = time.time()
    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"  Training time: {elapsed:.1f}s")
    joblib.dump(clf, os.path.join(MODEL_DIR, f"rf_{scenario_name}.pkl"))
    print(f"  Model saved → outputs/models/rf_{scenario_name}.pkl")
    return clf


# ── EVALUATION ────────────────────────────────────────────────────────────────

def evaluate(clf, X_te, y_te, scenario_name: str, le_map: dict) -> dict:
    """Run predictions, print report, save plots. Returns summary metrics dict."""
    class_names = [le_map[i] for i in sorted(le_map.keys())]

    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    f1_mac = f1_score(y_te, y_pred, average="macro")
    f1_wtd = f1_score(y_te, y_pred, average="weighted")

    print(f"\n  ── Results: {scenario_name} ──────────────────────────────────")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  F1 (macro)      : {f1_mac:.4f}")
    print(f"  F1 (weighted)   : {f1_wtd:.4f}")
    print()
    print(classification_report(y_te, y_pred, target_names=class_names))

    # Save classification report to text file
    report_str = classification_report(y_te, y_pred, target_names=class_names)
    with open(os.path.join(PLOT_DIR, f"report_{scenario_name}.txt"), "w") as f:
        f.write(f"Scenario: {scenario_name}\n")
        f.write(f"Accuracy : {acc:.4f}\n")
        f.write(f"F1 macro : {f1_mac:.4f}\n\n")
        f.write(report_str)

    # Confusion matrix
    _plot_confusion_matrix(y_te, y_pred, class_names, scenario_name)

    return {
        "scenario"    : scenario_name,
        "accuracy"    : round(acc,    4),
        "f1_macro"    : round(f1_mac, 4),
        "f1_weighted" : round(f1_wtd, 4),
    }


def _plot_confusion_matrix(y_te, y_pred, class_names, scenario_name):
    cm = confusion_matrix(y_te, y_pred)
    # Normalise per true class (row)
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
        ax.set_title(f"Confusion Matrix — {title}\n{scenario_name}")
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"confusion_{scenario_name}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(clf, feature_names: list, scenario_name: str, top_n: int = 20):
    importances = clf.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_names  = [feature_names[i] for i in idx]
    top_values = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_names[::-1], top_values[::-1], color="#5b8db8", edgecolor="none")
    ax.set_xlabel("Feature importance (mean decrease in impurity)")
    ax.set_title(f"Top {top_n} important features — {scenario_name}")
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, f"importance_{scenario_name}.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── COMPARISON TABLE ──────────────────────────────────────────────────────────

def print_comparison_table(results: list):
    df = pd.DataFrame(results)
    df = df.set_index("scenario")

    print("\n\n" + "="*65)
    print("  ABLATION STUDY — SCENARIO COMPARISON")
    print("="*65)
    print(f"\n  {'Scenario':<22} {'Accuracy':>10} {'F1 Macro':>10} {'F1 Weighted':>12}")
    print(f"  {'─'*22} {'─'*10} {'─'*10} {'─'*12}")
    for scenario, row in df.iterrows():
        print(f"  {scenario:<22} {row['accuracy']:>10.4f} {row['f1_macro']:>10.4f} {row['f1_weighted']:>12.4f}")

    # Find best scenario
    best = df["f1_macro"].idxmax()
    print(f"\n  Best scenario (F1 macro): {best}")

    # Save comparison table
    df.to_csv(os.path.join(OUTPUT_DIR, "scenario_comparison.csv"))
    print(f"\n  Saved: outputs/scenario_comparison.csv")

    # Bar chart comparison
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = ["accuracy", "f1_macro", "f1_weighted"]
    titles  = ["Accuracy", "F1 Score (Macro)", "F1 Score (Weighted)"]
    colors  = ["#5b8db8", "#e07b54", "#6aaa7b"]

    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        vals = df[metric]
        bars = ax.bar(vals.index, vals.values, color=color, edgecolor="none", width=0.5)
        ax.set_title(title, pad=8)
        ax.set_ylim(max(0, vals.min() - 0.05), min(1.0, vals.max() + 0.05))
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15, labelsize=9)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Ablation Study — PCAP only vs SAR only vs PCAP+SAR (fused)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "scenario_comparison.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    le_map  = load_label_encoder()
    results = []

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 1 — PCAP only
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  SCENARIO 1 — PCAP only")
    print("="*65)
    X_tr, X_te, y_tr, y_te, feat = load_preprocessed("pcap")
    clf1   = train_random_forest(X_tr, y_tr, "pcap_only")
    res1   = evaluate(clf1, X_te, y_te, "pcap_only", le_map)
    plot_feature_importance(clf1, feat, "pcap_only")
    results.append(res1)

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 2 — SAR only
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  SCENARIO 2 — SAR only")
    print("="*65)
    X_tr, X_te, y_tr, y_te, feat = load_preprocessed("sar")
    clf2   = train_random_forest(X_tr, y_tr, "sar_only")
    res2   = evaluate(clf2, X_te, y_te, "sar_only", le_map)
    plot_feature_importance(clf2, feat, "sar_only")
    results.append(res2)

    # ══════════════════════════════════════════════════════════════
    # SCENARIO 3 — PCAP + SAR fused
    # ══════════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  SCENARIO 3 — PCAP + SAR fused (join on Hash)")
    print("="*65)
    # X_tr, X_te, y_tr, y_te, fe  at = load_fused()
    X_tr, X_te, y_tr, y_te, feat = load_fused()  # FIXED: no extra "at"
    clf3   = train_random_forest(X_tr, y_tr, "fused")
    res3   = evaluate(clf3, X_te, y_te, "fused", le_map)
    plot_feature_importance(clf3, feat, "fused")
    results.append(res3)

    # ══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════
    print_comparison_table(results)

    print("\n\n" + "="*65)
    print("  DONE — Phase 2 complete")
    print("="*65)
    print("""
  Files created:

    outputs/models/
      rf_pcap_only.pkl          Random Forest trained on PCAP
      rf_sar_only.pkl           Random Forest trained on SAR
      rf_fused.pkl              Random Forest trained on PCAP+SAR

    outputs/phase2_plots/
      confusion_pcap_only.png   Confusion matrix (counts + normalised)
      confusion_sar_only.png
      confusion_fused.png
      importance_pcap_only.png  Top 20 feature importances
      importance_sar_only.png
      importance_fused.png
      report_*.txt              Full classification reports
      scenario_comparison.png   Bar chart comparing all 3 scenarios

    outputs/scenario_comparison.csv   Summary table for your report

  Next step:  python phase3_agent.py
""")


if __name__ == "__main__":
    main()
