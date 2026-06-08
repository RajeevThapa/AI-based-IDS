"""
utils/data_loader.py
Reads the pre-saved fused_pcap_test.csv which already contains all 402
features (32 PCAP + 370 SAR). No sar.csv needed at inference time.

Pre-requisite: run create_fused_test_csv.py once to generate
               outputs/fused_pcap_test.csv

FIXES vs previous version:
  1. No sar.csv dependency — reads self-contained fused test CSV
  2. true_label decoded from integer to family name string (e.g. "Mirai")
  3. PerformanceWarning removed
"""
import json
import numpy as np
import pandas as pd
from config.settings import OUTPUT_DIR, LABEL_ENC
import os

# Path to the pre-built fused test CSV
FUSED_TEST_CSV = os.path.join(OUTPUT_DIR, "fused_pcap_test.csv")

# Fallback: original pcap_test.csv (PCAP-only, lower accuracy)
PCAP_TEST_CSV  = os.path.join(OUTPUT_DIR, "pcap_test.csv")

PCAP_DROP_REDUNDANT = [
    "LLC", "ARP", "Tot size", "fin_count",
    "rst_count", "syn_count", "Tot sum",
]


def _load_label_encoder() -> dict:
    """Returns {int_index: family_name} e.g. {4: 'Mirai'}."""
    with open(LABEL_ENC) as f:
        return {int(k): v for k, v in json.load(f).items()}


def build_fused_test_rows(n: int = None) -> pd.DataFrame:
    """
    Load the pre-built fused test CSV (PCAP + SAR, 402 features).
    Falls back to pcap_test.csv if the fused version doesn't exist.

    Parameters
    ----------
    n : int | None
        Number of rows to load (None = all).

    Returns
    -------
    DataFrame with 402 features and a 'true_label' column containing
    decoded family name strings (e.g. "Mirai", "Benign").
    """
    le_map = _load_label_encoder()

    # ── Choose CSV to load ────────────────────────────────────────────────
    if os.path.exists(FUSED_TEST_CSV):
        csv_path = FUSED_TEST_CSV
        print(f"  [DataLoader] Loading fused test CSV (402 features): {os.path.basename(csv_path)}")
    else:
        csv_path = PCAP_TEST_CSV
        print(f"  [DataLoader] fused_pcap_test.csv not found.")
        print(f"  [DataLoader] Run create_fused_test_csv.py first for full accuracy.")
        print(f"  [DataLoader] Falling back to PCAP-only ({os.path.basename(csv_path)})")

    # df = pd.read_csv(csv_path, nrows=n, low_memory=False)
    # df.columns = df.columns.str.strip()
    # print(f"  [DataLoader] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    df = pd.read_csv(csv_path, nrows=n, low_memory=False)
    df.columns = df.columns.str.strip()

    # Replace inf values — raw fused CSV contains inf in Std/Variance columns
    df = df.replace([float('inf'), float('-inf')], float('nan'))

    print(f"  [DataLoader] Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    

    # ── Extract and decode true labels ────────────────────────────────────
    label_col = next(
        (c for c in ["MalwareFamily", "label"] if c in df.columns), None
    )

    if label_col is not None:
        raw = df[label_col].copy()
        try:
            true_labels = raw.apply(
                lambda x: le_map.get(int(float(x)), str(x))
            )
        except (ValueError, TypeError):
            true_labels = raw.astype(str)
        df = df.drop(columns=[label_col])
    else:
        true_labels = None

    # ── Drop any remaining non-feature columns ────────────────────────────
    drop = [c for c in ["Arch", "Hash"] + PCAP_DROP_REDUNDANT
            if c in df.columns]
    if drop:
        df = df.drop(columns=drop, errors="ignore")

    # ── Attach decoded true labels ─────────────────────────────────────────
    if true_labels is not None:
        df["true_label"] = true_labels.values[:len(df)]

    print(f"  [DataLoader] Feature columns: {df.shape[1] - (1 if true_labels is not None else 0)}")
    return df