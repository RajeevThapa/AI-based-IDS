"""
create_fused_test_csv.py
Run this ONCE to create outputs/fused_pcap_test.csv

Reads raw pcap.csv (has Hash) + sar.csv, reproduces the exact
train/test split from phase1_preprocess.py, joins SAR onto test
rows, saves a self-contained fused_pcap_test.csv with 402 features.

After this runs successfully, the pipeline needs no sar.csv at runtime.

Run:  python create_fused_test_csv.py
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_ROOT        = "./dataset/X86"
OUTPUT_DIR          = "./outputs"
RAW_PCAP_CSV        = os.path.join(DATASET_ROOT, "pcap.csv")
SAR_CSV             = os.path.join(DATASET_ROOT, "sar.csv")
LABEL_ENC           = os.path.join(OUTPUT_DIR, "label_encoder.json")
OUT_PATH            = os.path.join(OUTPUT_DIR, "fused_pcap_test.csv")
TEST_SIZE           = 0.2
RANDOM_STATE        = 42
LABEL_COL           = "MalwareFamily"

PCAP_DROP_COLS      = ["Hash", "Arch"]
PCAP_DROP_REDUNDANT = ["LLC", "ARP", "Tot size", "fin_count",
                       "rst_count", "syn_count", "Tot sum"]

# MUST match phase2_fused_fix.py exactly — added 'hugepage' and 'kernel'
SAR_KEYWORDS = [
    "cpu", "mem", "io", "disk", "net", "load", "swap",
    "proc", "usr", "sys", "idle", "paging", "queue",
    "memory", "network", "filesys", "interrupt", "softnet",
    "power", "serial", "hugepage", "kernel",
]
SAR_DROP_REDUNDANT  = ["interrupts[4].CPU0", "disk[0].rkB",
                       "network.net-ip.fragok", "network.net-udp.odgm"]

# ── Label encoder ──────────────────────────────────────────────────────────────
with open(LABEL_ENC) as f:
    le_map = {int(k): v for k, v in json.load(f).items()}
inv_le = {v: k for k, v in le_map.items()}

# ── Load raw PCAP (has Hash column) ───────────────────────────────────────────
print("Loading raw pcap.csv ...")
pcap = pd.read_csv(RAW_PCAP_CSV, low_memory=False)
pcap.columns = pcap.columns.str.strip()
print(f"  Shape: {pcap.shape}  |  Has Hash: {'Hash' in pcap.columns}")

# ── Reproduce exact same split as phase1_preprocess.py ────────────────────────
y_enc = pcap[LABEL_COL].astype(str).map(inv_le).values.astype(int)
_, test_idx = train_test_split(np.arange(len(pcap)), test_size=TEST_SIZE,
                               random_state=RANDOM_STATE, stratify=y_enc)
pcap_test = pcap.iloc[test_idx].copy().reset_index(drop=True)
print(f"  Test split: {len(pcap_test):,} rows  (expected 91,129)")

# ── Build SAR profile per Hash ─────────────────────────────────────────────────
print("\nLoading sar.csv ...")
sar = pd.read_csv(SAR_CSV, low_memory=False)
sar.columns = sar.columns.str.strip()
print(f"  Shape: {sar.shape}")

print("Building SAR profile per Hash ...")
sar_num_cols = [
    c for c in sar.select_dtypes(include=[np.number]).columns
    if any(kw in c.lower() for kw in SAR_KEYWORDS)
    and c not in SAR_DROP_REDUNDANT
]
# Select only needed cols before groupby — avoids PerformanceWarning
sar_subset  = sar[["Hash"] + sar_num_cols].copy()
sar_profile = sar_subset.groupby("Hash")[sar_num_cols].mean()
sar_profile.columns = ["sar_" + c for c in sar_profile.columns]
print(f"  SAR profile: {sar_profile.shape}  (expected 4435 rows, 370 cols)")

# ── Join SAR onto test rows ────────────────────────────────────────────────────
print("\nJoining SAR onto test PCAP rows ...")

# Decode labels to family name strings before dropping the column
labels_decoded = (pcap_test[LABEL_COL].astype(str)
                  .map(lambda x: le_map.get(inv_le.get(x, -1), x)))

# Drop leakage + redundant cols; keep Hash for join, will drop via set_index
drop_now = [c for c in PCAP_DROP_REDUNDANT + ["Arch", LABEL_COL]
            if c in pcap_test.columns]
pcap_feat = pcap_test.drop(columns=drop_now, errors="ignore")

# Join: set_index("Hash") makes Hash the index, join adds SAR cols,
# reset_index(drop=True) removes Hash from the result entirely
fused = (pcap_feat
         .set_index("Hash")
         .join(sar_profile, how="left")
         .reset_index(drop=True))

matched = fused[sar_profile.columns[0]].notna().sum()
print(f"  Rows with SAR context: {matched:,}/{len(fused):,} "
      f"({matched/len(fused)*100:.1f}%)")

# Attach decoded label — use pd.concat to avoid PerformanceWarning
fused = pd.concat([fused, labels_decoded.rename(LABEL_COL).reset_index(drop=True)],
                  axis=1)

# ── Validate ───────────────────────────────────────────────────────────────────
pcap_cols = [c for c in fused.columns if not c.startswith("sar_") and c != LABEL_COL]
sar_cols  = [c for c in fused.columns if c.startswith("sar_")]
print(f"\n  PCAP features : {len(pcap_cols)}  (expected 32)")
print(f"  SAR  features : {len(sar_cols)}   (expected 370)")
print(f"  Total columns : {fused.shape[1]}   (expected 403 = 402 features + 1 label)")
print(f"\n  Label distribution:")
print(fused[LABEL_COL].value_counts().to_string())

if len(pcap_cols) != 32 or len(sar_cols) != 370:
    print("\n  [!] Column count mismatch — check SAR_KEYWORDS and SAR_DROP_REDUNDANT")
else:
    print("\n  Column counts correct.")

# ── Save ───────────────────────────────────────────────────────────────────────
fused.to_csv(OUT_PATH, index=False)
print(f"\n  Saved: {OUT_PATH}  ({len(fused):,} rows x {fused.shape[1]} cols)")
print("""
Next steps:
  1. Replace utils/data_loader.py with data_loader_v2.py
  2. python pipeline.py --demo    # all 6 families, correct predictions
  3. python pipeline.py           # full batch with 402-feature accuracy
""")