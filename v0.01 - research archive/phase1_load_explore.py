"""
Phase 1a — Data Loading & Exploration
CIC-YNU-IoTMal  |  Architecture: X86  |  Modalities: PCAP + SAR

Reads:
    dataset/X86/pcap.csv
    dataset/X86/sar.csv

Run:  python phase1_load_explore.py
"""

import os
import pandas as pd
import numpy as np

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATASET_ROOT = "./dataset"          # folder that contains X86/
ARCH         = "X86"
PCAP_FILE    = "pcap.csv"
SAR_FILE     = "sar.csv"
# LABEL_COL    = "label"              # <-- update if your label column has a different name
LABEL_COL = "MalwareFamily"         # ← Change from "label"
OUTPUT_DIR   = "./outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def load_csv(path: str, name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"  [!] File not found: {path}")
        return pd.DataFrame()
    print(f"  Reading {name} from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    # Strip whitespace from column names (common CSV issue)
    df.columns = df.columns.str.strip()
    print(f"  Done — {len(df):,} rows  {df.shape[1]} columns")
    return df


def summarise(df: pd.DataFrame, name: str):
    print(f"\n{'─'*65}")
    print(f"  {name}")
    print(f"{'─'*65}")
    print(f"  Shape   : {df.shape}")
    print(f"  Memory  : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"  Dtypes  : {dict(df.dtypes.value_counts())}")

    # Missing values
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        print(f"  Missing values (top 10):")
        for col, cnt in missing.head(10).items():
            print(f"    {col:42s}  {cnt:>8,}  ({cnt/len(df)*100:.1f}%)")
    else:
        print(f"  Missing values : none")

    # Inf values in numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    inf_count = np.isinf(numeric_df.values).sum()
    print(f"  Inf values     : {inf_count:,}" +
          (" ← will be replaced with NaN during preprocessing" if inf_count else ""))

    # Label distribution
    if LABEL_COL in df.columns:
        dist = df[LABEL_COL].value_counts()
        max_count = dist.max()
        print(f"\n  Label distribution  (total {len(df):,} rows):")
        for lbl, cnt in dist.items():
            bar = "█" * int(cnt / max_count * 35)
            pct = cnt / len(df) * 100
            print(f"    {str(lbl):25s}  {cnt:>8,}  ({pct:5.1f}%)  {bar}")
        ratio = max_count / dist.min()
        print(f"\n  Imbalance ratio : {ratio:.1f}x  (largest / smallest class)")
        if ratio > 10:
            print(f"  [!] High imbalance detected — SMOTE recommended")
    else:
        print(f"\n  [!] Column '{LABEL_COL}' NOT FOUND.")
        print(f"      Update LABEL_COL in this script.")
        print(f"      Candidate columns: {[c for c in df.columns if 'label' in c.lower() or 'class' in c.lower() or 'type' in c.lower() or 'family' in c.lower()]}")


def print_columns(df: pd.DataFrame, name: str):
    print(f"\n  All columns — {name}  ({len(df.columns)} total)")
    print(f"  {'idx':>4}  {'column name':45}  {'dtype':12}  {'non-null':>10}  sample")
    print(f"  {'─'*4}  {'─'*45}  {'─'*12}  {'─'*10}  {'─'*25}")
    for i, col in enumerate(df.columns):
        dtype    = str(df[col].dtype)
        non_null = df[col].notna().sum()
        sample   = df[col].dropna()
        sample   = str(sample.iloc[0]) if not sample.empty else "N/A"
        print(f"  [{i:3d}]  {col:45}  {dtype:12}  {non_null:>10,}  {sample[:25]}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    base = os.path.join(DATASET_ROOT, ARCH)
    if not os.path.exists(base):
        raise FileNotFoundError(
            f"\nFolder not found: {base}\n"
            f"Update DATASET_ROOT at the top of this script.\n"
            f"Expected layout:  dataset/X86/pcap.csv  and  dataset/X86/sar.csv"
        )

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  Loading PCAP and SAR")
    print("="*65)
    pcap = load_csv(os.path.join(base, PCAP_FILE), "PCAP")
    sar  = load_csv(os.path.join(base, SAR_FILE),  "SAR")

    # ── Summaries ─────────────────────────────────────────────────────────────
    if not pcap.empty:
        summarise(pcap, "X86 / PCAP")
    if not sar.empty:
        summarise(sar, "X86 / SAR")

    # ── Column details ────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  COLUMN DETAILS")
    print("="*65)
    if not pcap.empty:
        print_columns(pcap, "PCAP")
    if not sar.empty:
        print_columns(sar, "SAR")

    # ── Action checklist ──────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  CHECKLIST — review before running phase1_preprocess.py")
    print("="*65)
    print(f"""
  1. Confirm LABEL_COL = '{LABEL_COL}' matches your actual label column name.
     Update it at the top of both this script and phase1_preprocess.py.

  2. PCAP — note any identifier/leakage columns:
       src_ip, dst_ip, src_port, dst_port, timestamp,
       flow_id, src_mac, dst_mac, or similar.
     Add them to PCAP_DROP_COLS in phase1_preprocess.py.

  3. SAR — confirm resource metric columns exist with keywords:
       cpu, mem, io, load, swap, net
     If your column names differ, add keywords to SAR_FEATURE_KEYWORDS.

  Next step:  python phase1_preprocess.py
""")
