"""
pipeline.py  —  Agentic IDS  |  Main orchestrator
CIC-YNU-IoTMal  |  X86  |  rf_fused_v2

Run modes
─────────
  python pipeline.py                  batch simulation  (default, 300 samples)
  python pipeline.py --samples 1000   larger batch
  python pipeline.py --demo           one real sample per family → all 4 actions
  python pipeline.py --feedback       threshold adaptation demo  (RQ 2.1)
  python pipeline.py --real           execute real iptables commands  (root only)

FIXES vs previous version
─────────────────────────
  1. Batch uses fused PCAP+SAR rows (not raw PCAP) → correct family predictions
  2. Demo uses one real test sample per malware family → all 4 response levels shown
  3. sklearn feature-name warning suppressed in detector.py
"""
import os, sys, json, time, argparse
import numpy as np
import pandas as pd
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.detector        import MalwareDetector
from agents.response_agent  import ResponseAgent
from agents.feedback_agent  import FeedbackAgent
from utils.logger           import summarise_log
from utils.data_loader      import build_fused_test_rows
from config.settings        import OUTPUT_DIR, LABEL_ENC


def _le():
    with open(LABEL_ENC) as f:
        return {int(k): v for k, v in json.load(f).items()}


def _header():
    print("\n" + "="*65)
    print("  Agentic AI IDS for IoT")
    print("  Phase 3 — Autonomous Detection and Response")
    print("  Dataset: CIC-YNU-IoTMal  |  Architecture: X86")
    print("="*65)


# ── MODE 1 — Batch simulation ──────────────────────────────────────────────

def run_batch(detector, agent, n: int):
    """
    Replay n fused (PCAP+SAR) test rows through detect → decide → respond.
    Reports containment rate (RQ 2.3) and latency (RQ 2.2).
    """
    print(f"\n{'='*65}")
    print(f"  BATCH SIMULATION  —  {n} fused test samples")
    print(f"{'='*65}\n")
    print("  Loading fused test features ...")

    fused = build_fused_test_rows(n=n)

    true_labels = fused.pop("true_label") if "true_label" in fused.columns else None

    print(f"\n  {'#':<5} {'True':<14} {'Predicted':<14} "
          f"{'Score':<7} {'Action':<10} {'ms':>6}")
    print(f"  {'─'*5} {'─'*14} {'─'*14} {'─'*7} {'─'*10} {'─'*6}")

    records, latencies = [], []

    for i, (_, row) in enumerate(fused.iterrows()):
        t0        = time.perf_counter()
        detection = detector.predict(row.to_dict())
        record    = agent.respond(
            detection = detection,
            device_id = f"iot-{(i % 12) + 1:02d}",
            src_ip    = f"10.0.{(i // 254) % 256}.{(i % 254) + 1}",
        )
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        true_lbl = str(true_labels.iloc[i]) if true_labels is not None else "?"
        print(f"  {i+1:<5} {true_lbl:<14} {detection['family']:<14} "
              f"{detection['threat_score']:.3f}  {record['action']:<10} "
              f"{elapsed:>5.1f}")

        records.append({
            "true"     : true_lbl,
            "predicted": detection["family"],
            "action"   : record["action"],
            "ms"       : elapsed,
        })

    # ── Statistics ──────────────────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print(f"  RESULTS")
    print(f"{'='*65}")

    avg_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    max_ms = float(np.max(latencies))

    print(f"\n  Samples processed  : {len(records):,}")
    print(f"  Avg latency        : {avg_ms:.1f} ms  "
          f"(P95={p95_ms:.1f} ms, max={max_ms:.1f} ms)")
    print(f"  → IoT target <50ms : {'✓ PASS' if p95_ms < 50 else '✗ above target'}")

    print(f"\n  Action breakdown:")
    action_counts = Counter(r["action"] for r in records)
    for action in ["log", "alert", "block", "isolate"]:
        count = action_counts.get(action, 0)
        pct   = count / len(records) * 100
        bar   = "█" * int(pct / 2)
        print(f"    {action:<10} {count:>5,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Predicted families:")
    for fam, cnt in Counter(r["predicted"] for r in records).most_common():
        pct = cnt / len(records) * 100
        print(f"    {fam:<16} {cnt:>5,}  ({pct:5.1f}%)")

    attacks   = [r for r in records if r["predicted"] != "Benign"]
    if attacks:
        contained = sum(1 for r in attacks if r["action"] in ("block", "isolate"))
        rate      = contained / len(attacks) * 100
        print(f"\n  ┌──────────────────────────────────────────────────────┐")
        print(f"  │  Containment rate (RQ 2.3)                           │")
        print(f"  │  {contained}/{len(attacks)} attacks blocked or isolated"
              + " " * max(0, 27 - len(f"{contained}/{len(attacks)}")) + "│")
        print(f"  │  Rate: {rate:.1f}%"
              + " " * max(0, 45 - len(f"{rate:.1f}")) + "│")
        print(f"  └──────────────────────────────────────────────────────┘")

    print(f"\n  Audit log: outputs/audit_log.jsonl")


# ── MODE 2 — Demo ──────────────────────────────────────────────────────────

def run_demo(detector, agent, n: int = 500):
    """
    FIX: Load real test samples and find ONE example per malware family.
    This guarantees all four response levels are shown naturally.
    """
    print(f"\n{'='*65}")
    print(f"  DEMO  —  one real sample per malware family")
    print(f"{'='*65}\n")
    print("  Loading fused test features to find family examples ...")

    fused = build_fused_test_rows(n=n)
    if "true_label" in fused.columns:
        true_labels = fused["true_label"].copy()
        fused = fused.drop(columns=["true_label"])
    else:
        true_labels = None

    if true_labels is None:
        print("  [!] No true_label column found — falling back to first 6 rows")
        samples = list(fused.iterrows())[:6]
        labels  = ["?"] * 6
    else:
        # Pick one row per family
        fused["_lbl"] = true_labels.values
        seen, samples, labels = set(), [], []
        for _, row in fused.iterrows():
            lbl = str(row["_lbl"])
            if lbl not in seen:
                seen.add(lbl)
                samples.append(row.drop("_lbl"))
                labels.append(lbl)
        fused = fused.drop(columns=["_lbl"])

    device_map = {
        "Benign"    : ("smart-tv-01",     "192.168.1.10"),
        "DarkNexus" : ("ip-camera-02",    "192.168.1.55"),
        "Gafgyt"    : ("smart-lock-03",   "10.0.0.22"),
        "Generic"   : ("smart-bulb-04",   "10.0.0.88"),
        "Mirai"     : ("router-05",       "172.16.0.7"),
        "Unknown"   : ("iot-hub-06",      "192.168.2.1"),
    }

    for i, (row, lbl) in enumerate(zip(samples, labels)):
        device, ip = device_map.get(lbl, (f"device-{i+1:02d}", f"10.0.{i}.1"))
        print(f"\n  ── Sample {i+1}  |  True family: {lbl}")
        detection = detector.predict(row.to_dict()
                                     if hasattr(row, "to_dict") else dict(row))
        agent.respond(detection=detection, device_id=device, src_ip=ip)

    print(f"\n\n  Demo complete.  Audit log: outputs/audit_log.jsonl")


# ── MODE 3 — Feedback demo ─────────────────────────────────────────────────

def run_feedback_demo(detector, agent):
    """Shows FeedbackAgent adapting thresholds. Addresses RQ 2.1."""
    fa = FeedbackAgent()

    print(f"\n{'='*65}")
    print(f"  FEEDBACK DEMO  —  threshold adaptation  (RQ 2.1)")
    print(f"{'='*65}\n")
    print(f"  Initial thresholds: {fa.get_thresholds()}\n")

    # One real detection
    fused = build_fused_test_rows(n=10)
    fused = fused.drop(columns=["true_label"], errors="ignore")  # safe drop
    row   = fused.iloc[0]
    detection  = detector.predict(row.to_dict())
    record     = agent.respond(detection, device_id="test-dev",
                                src_ip="10.0.0.99")

    print(f"\n  Operator feedback sequence:")

    print(f"\n  Step 1 → FALSE POSITIVE (system over-reacted)")
    fa.record_feedback(record, "fp")

    print(f"\n  Step 2 → FALSE POSITIVE again")
    fa.record_feedback(record, "fp")

    print(f"\n  Step 3 → TRUE POSITIVE (attack confirmed)")
    fa.record_feedback(record, "tp")

    print(f"\n  Step 4 → FALSE NEGATIVE (attack was missed)")
    fa.record_feedback(record, "fn")

    print(f"\n  Final thresholds  : {fa.get_thresholds()}")
    print(f"  Adaptation events : {len(fa.history())}")
    print(f"  History file      : outputs/threshold_history.jsonl")
    print(f"\n  → Demonstrates RQ 2.1: agent adapts based on feedback")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Agentic IDS — Phase 3")
    ap.add_argument("--demo",     action="store_true",
                    help="One real sample per family, all 4 response levels")
    ap.add_argument("--feedback", action="store_true",
                    help="Threshold adaptation demo (RQ 2.1)")
    ap.add_argument("--real",     action="store_true",
                    help="Execute real system commands (needs root)")
    ap.add_argument("--samples",  type=int, default=300,
                    help="Samples for batch simulation (default 300)")
    args = ap.parse_args()

    _header()
    detector = MalwareDetector()
    agent    = ResponseAgent(real_mode=args.real)

    if args.demo:
        run_demo(detector, agent, n=10000)
    elif args.feedback:
        run_feedback_demo(detector, agent)
    else:
        run_batch(detector, agent, n=args.samples)

    s = summarise_log()
    print(f"\n{'='*65}")
    print(f"  AUDIT SUMMARY")
    print(f"{'='*65}")
    print(f"  Total decisions : {s.get('total', 0)}")
    if "actions"  in s: print(f"  Actions         : {s['actions']}")
    if "families" in s: print(f"  Families        : {s['families']}")


if __name__ == "__main__":
    main()