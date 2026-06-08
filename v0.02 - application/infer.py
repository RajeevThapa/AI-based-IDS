"""
infer.py  —  Single-sample inference entry point.

Usage
─────
  python infer.py                           # random features (demo)
  python infer.py --features sample.json   # from JSON file
  python infer.py --features sample.json --device router-01 --ip 10.0.0.5

sample.json: flat dict of {feature_name: value}.
Missing features are imputed with the training-set median.
"""
import os, sys, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.detector       import MalwareDetector
from agents.response_agent import ResponseAgent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=None)
    ap.add_argument("--device",   default="device-unknown")
    ap.add_argument("--ip",       default="0.0.0.0")
    ap.add_argument("--iface",    default="eth0")
    ap.add_argument("--real",     action="store_true")
    args = ap.parse_args()

    detector = MalwareDetector()
    agent    = ResponseAgent(real_mode=args.real)

    if args.features and os.path.exists(args.features):
        with open(args.features) as f:
            features = json.load(f)
        print(f"\n  Features from: {args.features}  ({len(features)} keys)")
    else:
        print("\n  No --features file → using random vector (demo)")
        features = np.random.default_rng(42).standard_normal(detector.n_features)

    detection = detector.predict(features)

    print(f"\n  ── Detection ────────────────────────────────────────")
    print(f"  Family       : {detection['family']}")
    print(f"  Threat score : {detection['threat_score']:.4f}")
    print(f"\n  Class probabilities:")
    for fam, prob in sorted(detection["confidence"].items(), key=lambda x: -x[1]):
        print(f"    {fam:<16}  {prob:.4f}  {'█' * int(prob * 30)}")

    record = agent.respond(
        detection=detection, device_id=args.device,
        src_ip=args.ip, iface=args.iface,
    )
    print(f"\n  Decision  : {record['action'].upper()}")
    print(f"  Audit log : outputs/audit_log.jsonl")


if __name__ == "__main__":
    main()
