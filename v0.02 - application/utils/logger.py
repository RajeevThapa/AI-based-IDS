"""
utils/logger.py
Structured JSON-lines audit logger.

Every agent decision writes exactly one line to audit_log.jsonl.
Format chosen so it can be imported into Splunk, ELK, or a
pandas DataFrame with a single read_json() call.
"""

import json
import os
from datetime import datetime, timezone
from collections import Counter
from config.settings import AUDIT_LOG


# ── Write ──────────────────────────────────────────────────────────────────

def log_decision(
    device_id:    str,
    src_ip:       str,
    family:       str,
    threat_score: float,
    action:       str,
    confidence:   dict,
    real_mode:    bool = False,
    extra:        dict = None,
) -> dict:
    """
    Append one decision record to the audit log.

    Returns the record dict (so callers can inspect it immediately).
    """
    os.makedirs(os.path.dirname(AUDIT_LOG), exist_ok=True)

    record = {
        "timestamp"    : datetime.now(timezone.utc).isoformat(),
        "device_id"    : device_id,
        "src_ip"       : src_ip,
        "family"       : family,
        "threat_score" : round(float(threat_score), 4),
        "action"       : action,
        "real_mode"    : real_mode,
        "confidence"   : {k: round(float(v), 4) for k, v in confidence.items()},
    }
    if extra:
        record.update(extra)

    with open(AUDIT_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


# ── Read ───────────────────────────────────────────────────────────────────

def read_log() -> list:
    """Return all audit log records as a list of dicts."""
    if not os.path.exists(AUDIT_LOG):
        return []
    with open(AUDIT_LOG) as f:
        return [json.loads(line) for line in f if line.strip()]


def summarise_log() -> dict:
    """Return high-level counts from the audit log."""
    records = read_log()
    if not records:
        return {"total": 0}
    return {
        "total"    : len(records),
        "actions"  : dict(Counter(r["action"]  for r in records)),
        "families" : dict(Counter(r["family"]  for r in records)),
    }
