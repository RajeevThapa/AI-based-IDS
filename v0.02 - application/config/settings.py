# """
# config/settings.py  —  All thresholds and paths in one place.
# Edit only this file to change behaviour — nothing else needs touching.
# """
# import os

# BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # ── Model artefacts (copied from Phase 2) ─────────────────────────────────
# MODEL_PATH   = os.path.join(BASE_DIR, "outputs", "models",  "rf_fused_v2.pkl")
# SCALER_PATH  = os.path.join(BASE_DIR, "outputs", "scalers", "fused_scaler_v2.pkl")
# IMPUTER_PATH = os.path.join(BASE_DIR, "outputs", "scalers", "fused_imputer.pkl")
# LABEL_ENC    = os.path.join(BASE_DIR, "outputs", "label_encoder.json")

# # ── Raw dataset (needed to attach SAR context at inference time) ───────────
# DATASET_DIR  = os.path.join(BASE_DIR, "dataset", "X86")
# PCAP_TEST    = os.path.join(BASE_DIR, "outputs", "pcap_test.csv")
# # SAR_CSV      = os.path.join(DATASET_DIR, "sar.csv")
# SAR_CSV = "/home/rajeev/Documents/Project/Project - 4 Dataset/v1x/dataset/X86/sar.csv"

# # ── Output files ──────────────────────────────────────────────────────────
# OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
# AUDIT_LOG    = os.path.join(OUTPUT_DIR, "audit_log.jsonl")
# THRESH_FILE  = os.path.join(BASE_DIR, "config", "thresholds.json")
# THRESH_HIST  = os.path.join(OUTPUT_DIR, "threshold_history.jsonl")

# # ── Decision thresholds ───────────────────────────────────────────────────
# #   threat_score = max class probability (0.0 – 1.0)
# #   score <  ALERT              →  log only
# #   ALERT  <= score <  BLOCK    →  alert admin
# #   BLOCK  <= score <  ISOLATE  →  block source IP
# #   score >= ISOLATE            →  isolate device
# THRESHOLD_ALERT   = 0.30
# THRESHOLD_BLOCK   = 0.60
# THRESHOLD_ISOLATE = 0.80

# BENIGN_LABEL  = "Benign"

# # ── Feedback adaptation parameters ───────────────────────────────────────
# FP_DELTA      = 0.05    # raise thresholds by this on false positive
# FN_DELTA      = 0.02    # lower thresholds by this on false negative
# THRESHOLD_MIN = 0.10
# THRESHOLD_MAX = 0.95

# # ── Response mode ─────────────────────────────────────────────────────────
# #   False  →  print what the command WOULD do  (safe, default)
# #   True   →  run real iptables / ip commands  (needs root)
# REAL_MODE = False

# # ── SAR feature keywords (same as Phase 1/2) ──────────────────────────────
# SAR_KEYWORDS = [
#     "cpu", "mem", "memory", "io", "disk", "net", "network",
#     "load", "swap", "proc", "usr", "sys", "idle", "paging",
#     "queue", "filesys", "interrupt", "softnet", "power", "serial",
# ]
# SAR_DROP_REDUNDANT = [
#     "interrupts[4].CPU0", "disk[0].rkB",
#     "network.net-ip.fragok", "network.net-udp.odgm",
# ]
# PCAP_DROP_REDUNDANT = [
#     "LLC", "ARP", "Tot size", "fin_count",
#     "rst_count", "syn_count", "Tot sum",
# ]


"""
config/settings.py  —  All thresholds and paths in one place.
Edit only this file to change behaviour — nothing else needs touching.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Model artefacts ───────────────────────────────────────────────────────────
MODEL_PATH   = os.path.join(BASE_DIR, "outputs", "models",  "rf_fused_v2.pkl")
SCALER_PATH  = os.path.join(BASE_DIR, "outputs", "scalers", "fused_scaler_v2.pkl")
IMPUTER_PATH = os.path.join(BASE_DIR, "outputs", "scalers", "fused_imputer.pkl")
LABEL_ENC    = os.path.join(BASE_DIR, "outputs", "label_encoder.json")

# ── Dataset path — searches common locations automatically ────────────────────
# Priority:
#   1. dataset/X86/ inside the project folder  (v0.02/dataset/X86/)
#   2. One level up from project               (../dataset/X86/)
#   3. Two levels up                           (../../dataset/X86/)
# If none are found, falls back to (1) and will print a clear error at runtime.

def _find_dataset_dir():
    candidates = [
        os.path.join(BASE_DIR, "dataset", "X86"),
        os.path.join(os.path.dirname(BASE_DIR), "dataset", "X86"),
        os.path.join(os.path.dirname(os.path.dirname(BASE_DIR)), "dataset", "X86"),
        # Also try "Project - 4 Dataset" style paths one level up
        os.path.join(os.path.dirname(BASE_DIR), "Project - 4 Dataset", "X86"),
    ]
    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "sar.csv")):
            return path
    # Return first candidate as default (will fail clearly at runtime if wrong)
    return candidates[0]

DATASET_DIR = _find_dataset_dir()
SAR_CSV     = os.path.join(DATASET_DIR, "sar.csv")
RAW_PCAP    = os.path.join(DATASET_DIR, "pcap.csv")

# ── Test data ─────────────────────────────────────────────────────────────────
# Use fused_pcap_test.csv if it exists (run create_fused_test_csv.py once).
# Falls back to pcap_test.csv (PCAP-only, reduced accuracy).
_fused_test  = os.path.join(BASE_DIR, "outputs", "fused_pcap_test.csv")
_pcap_test   = os.path.join(BASE_DIR, "outputs", "pcap_test.csv")
PCAP_TEST    = _fused_test if os.path.exists(_fused_test) else _pcap_test

# ── Output files ──────────────────────────────────────────────────────────────
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
AUDIT_LOG    = os.path.join(OUTPUT_DIR, "audit_log.jsonl")
THRESH_FILE  = os.path.join(BASE_DIR, "config", "thresholds.json")
THRESH_HIST  = os.path.join(OUTPUT_DIR, "threshold_history.jsonl")

# ── Decision thresholds ───────────────────────────────────────────────────────
#   threat_score = max class probability (0.0 – 1.0)
#   score <  ALERT              →  log only
#   ALERT  <= score <  BLOCK    →  alert admin
#   BLOCK  <= score <  ISOLATE  →  block source IP
#   score >= ISOLATE            →  isolate device
THRESHOLD_ALERT   = 0.30
THRESHOLD_BLOCK   = 0.60
THRESHOLD_ISOLATE = 0.80

BENIGN_LABEL  = "Benign"

# ── Feedback adaptation parameters ───────────────────────────────────────────
FP_DELTA      = 0.05
FN_DELTA      = 0.02
THRESHOLD_MIN = 0.10
THRESHOLD_MAX = 0.95

# ── Response mode ─────────────────────────────────────────────────────────────
REAL_MODE = False

# ── SAR / PCAP feature lists (shared across data_loader, phase2, etc.) ────────
SAR_KEYWORDS = [
    "cpu", "mem", "memory", "io", "disk", "net", "network",
    "load", "swap", "proc", "usr", "sys", "idle", "paging",
    "queue", "filesys", "interrupt", "softnet", "power", "serial",
]
SAR_DROP_REDUNDANT = [
    "interrupts[4].CPU0", "disk[0].rkB",
    "network.net-ip.fragok", "network.net-udp.odgm",
]
PCAP_DROP_REDUNDANT = [
    "LLC", "ARP", "Tot size", "fin_count",
    "rst_count", "syn_count", "Tot sum",
]

# ── Debug: print resolved paths on import ─────────────────────────────────────
if __name__ == "__main__":
    print(f"BASE_DIR    : {BASE_DIR}")
    print(f"DATASET_DIR : {DATASET_DIR}")
    print(f"SAR_CSV     : {SAR_CSV}  (exists: {os.path.exists(SAR_CSV)})")
    print(f"RAW_PCAP    : {RAW_PCAP}  (exists: {os.path.exists(RAW_PCAP)})")
    print(f"PCAP_TEST   : {PCAP_TEST}  (exists: {os.path.exists(PCAP_TEST)})")
    print(f"MODEL_PATH  : {MODEL_PATH}  (exists: {os.path.exists(MODEL_PATH)})")