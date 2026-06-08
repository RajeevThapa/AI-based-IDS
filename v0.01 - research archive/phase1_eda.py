"""
Phase 1c — Exploratory Data Analysis
CIC-YNU-IoTMal  |  Architecture: X86  |  Modalities: PCAP + SAR

Reads preprocessed CSVs from ./outputs/ — run AFTER phase1_preprocess.py

Saves PNG plots to ./outputs/eda_plots/:
  pcap_class_dist.png
  pcap_top_features.png
  pcap_correlation.png
  sar_class_dist.png
  sar_correlation.png
  sar_resource_boxplots.png

Run:  python phase1_eda.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

OUTPUT_DIR = "./outputs"
EDA_DIR    = os.path.join(OUTPUT_DIR, "eda_plots")
LABEL_COL  = "MalwareFamily"   # <-- update if your label column has a different name

os.makedirs(EDA_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.0)

# Load label encoder for readable class names
LE_PATH = os.path.join(OUTPUT_DIR, "label_encoder.json")
LE_MAP  = {}
if os.path.exists(LE_PATH):
    with open(LE_PATH) as f:
        LE_MAP = json.load(f)   # {"0": "Benign", "1": "Mirai", ...}


def label_to_name(val) -> str:
    """Convert integer label to human-readable family name."""
    return LE_MAP.get(str(int(val)), str(val))


# ── 1. CLASS DISTRIBUTION ────────────────────────────────────────────────────

def plot_class_dist(df: pd.DataFrame, title: str, filename: str):
    if LABEL_COL not in df.columns:
        print(f"  [!] No label column in {title}")
        return

    labels  = df[LABEL_COL].map(label_to_name)
    counts  = labels.value_counts().sort_values()
    colors  = sns.color_palette("muted", len(counts))

    fig, ax = plt.subplots(figsize=(9, max(4, len(counts) * 0.6)))
    bars = ax.barh(counts.index.astype(str), counts.values, color=colors)
    ax.set_xlabel("Sample count")
    ax.set_title(title, pad=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    for bar, val in zip(bars, counts.values):
        ax.text(
            bar.get_width() + counts.max() * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}", va="center", fontsize=9
        )
    plt.tight_layout()
    _save(filename)


# ── 2. TOP FEATURES BY VARIANCE ───────────────────────────────────────────────

def plot_top_features(df: pd.DataFrame, title: str, filename: str, top_n: int = 20):
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[LABEL_COL], errors="ignore")
    var = numeric.var().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(11, 5))
    var.plot(kind="bar", ax=ax, color="#5b8db8", edgecolor="none")
    ax.set_title(f"Top {top_n} features by variance — {title}", pad=10)
    ax.set_ylabel("Variance (post-scaling)")
    ax.tick_params(axis="x", rotation=50, labelsize=8)
    plt.tight_layout()
    _save(filename)


# ── 3. CORRELATION HEATMAP ────────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame, title: str, filename: str, top_n: int = 30):
    numeric = df.select_dtypes(include=[np.number]).drop(columns=[LABEL_COL], errors="ignore")
    if numeric.shape[1] < 2:
        print(f"  [!] Not enough numeric columns for correlation in {title}")
        return

    top_cols = numeric.var().sort_values(ascending=False).head(top_n).index
    corr     = numeric[top_cols].corr()

    # Report highly correlated pairs
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = [
        (c, r, upper.loc[r, c])
        for c in upper.columns for r in upper.index
        if pd.notna(upper.loc[r, c]) and abs(upper.loc[r, c]) >= 0.95
    ]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, ax=ax, cmap="coolwarm", center=0,
                square=True, linewidths=0.3, annot=False,
                cbar_kws={"shrink": 0.6})
    ax.set_title(f"{title}  (top {top_n} by variance)", pad=12)
    plt.tight_layout()
    _save(filename)

    if high_corr:
        print(f"  Highly correlated pairs (|r| ≥ 0.95) — consider dropping one:")
        for a, b, r in sorted(high_corr, key=lambda x: -abs(x[2]))[:10]:
            print(f"    {a:40s} ↔  {b:40s}  r={r:.3f}")
    else:
        print(f"  No highly correlated pairs found (threshold |r| ≥ 0.95)")


# ── 4. SAR BOX PLOTS ──────────────────────────────────────────────────────────

def plot_sar_boxplots(df: pd.DataFrame):
    """Box plots of each resource metric split by malware family."""
    resource_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if any(kw in c.lower() for kw in ["cpu", "mem", "io", "load", "swap", "usr", "sys"])
        and c != LABEL_COL
    ]
    if not resource_cols:
        print("  [!] No resource columns matched for SAR box plots")
        return

    df = df.copy()
    df["_family"] = df[LABEL_COL].map(label_to_name)

    plot_cols = resource_cols[:6]   # max 6 subplots
    ncols = min(3, len(plot_cols))
    nrows = -(-len(plot_cols) // ncols)   # ceiling division
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, col in zip(axes, plot_cols):
        order = (df.groupby("_family")[col]
                   .median()
                   .sort_values(ascending=False)
                   .index)
        sns.boxplot(data=df, x="_family", y=col, order=order,
                    ax=ax, palette="muted", fliersize=1.5, linewidth=0.8)
        ax.set_title(col, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40, labelsize=8)

    for ax in axes[len(plot_cols):]:
        ax.set_visible(False)

    fig.suptitle("SAR — resource metrics by malware family", fontsize=13, y=1.01)
    plt.tight_layout()
    _save("sar_resource_boxplots.png")


# ── SAVE HELPER ───────────────────────────────────────────────────────────────

def _save(filename: str):
    path = os.path.join(EDA_DIR, filename)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    print("\n[EDA] Reading preprocessed CSVs from ./outputs/\n")

    pcap_path = os.path.join(OUTPUT_DIR, "pcap_train.csv")
    sar_path  = os.path.join(OUTPUT_DIR, "sar_train.csv")

    if not os.path.exists(pcap_path) and not os.path.exists(sar_path):
        print("  No output CSVs found — run phase1_preprocess.py first.")
        return

    # ── PCAP plots ────────────────────────────────────────────────────────────
    if os.path.exists(pcap_path):
        print("── PCAP ──────────────────────────────────────────────────────")
        pcap = pd.read_csv(pcap_path)
        plot_class_dist(pcap,   "PCAP — class distribution (train set)", "pcap_class_dist.png")
        plot_top_features(pcap, "PCAP", "pcap_top_features.png")
        plot_correlation(pcap,  "PCAP", "pcap_correlation.png")
    else:
        print("  pcap_train.csv not found")

    # ── SAR plots ─────────────────────────────────────────────────────────────
    if os.path.exists(sar_path):
        print("\n── SAR ───────────────────────────────────────────────────────")
        sar = pd.read_csv(sar_path)
        plot_class_dist(sar,  "SAR — class distribution (train set)", "sar_class_dist.png")
        plot_correlation(sar, "SAR", "sar_correlation.png")
        plot_sar_boxplots(sar)
    else:
        print("  sar_train.csv not found")

    print(f"\n[EDA] All plots saved to {EDA_DIR}/")
    print("""
  What to look for before Phase 2:
  ┌─────────────────────────────────────────────────────────────┐
  │ class_dist      → still imbalanced after SMOTE?            │
  │ top_features    → near-zero variance features → safe to drop│
  │ correlation     → |r| ≥ 0.95 pairs → drop one from each    │
  │ sar_boxplots    → do metrics differ clearly across families?│
  │                   if all boxes overlap → SAR adds low signal│
  └─────────────────────────────────────────────────────────────┘

  Next step:  python phase2_models.py
""")


if __name__ == "__main__":
    main()
