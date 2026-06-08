"""
agents/feedback_agent.py  —  Threshold adaptation from operator feedback.
Addresses RQ 2.1: real-time adaptation improves the IDS over time.
"""
import json, os
from datetime import datetime, timezone
from config.settings import (
    THRESHOLD_ALERT, THRESHOLD_BLOCK, THRESHOLD_ISOLATE,
    FP_DELTA, FN_DELTA, THRESHOLD_MIN, THRESHOLD_MAX,
    THRESH_FILE, THRESH_HIST,
)


class FeedbackAgent:

    def __init__(self):
        self.thresholds = self._load()
        print(f"  [FeedbackAgent] Thresholds: {self.thresholds}")

    def record_feedback(self, record: dict, feedback_type: str) -> dict:
        """
        Apply operator feedback.

        feedback_type : "fp" false positive | "fn" false negative | "tp" true positive
        """
        family = record.get("family", "?")
        action = record.get("action", "?")
        score  = record.get("threat_score", 0.0)

        if feedback_type == "fp":
            delta  = +FP_DELTA
            reason = (f"False positive — {family} (score={score:.3f}) → {action}. "
                      f"Raising thresholds +{FP_DELTA}")
        elif feedback_type == "fn":
            delta  = -FN_DELTA
            reason = f"False negative — attack missed. Lowering thresholds -{FN_DELTA}"
        elif feedback_type == "tp":
            delta  = 0
            reason = f"True positive confirmed — {family}. No change."
        else:
            print(f"  [FeedbackAgent] Unknown type: {feedback_type}")
            return self.thresholds

        if delta != 0:
            self._adjust(delta)
            self._save()

        print(f"  [FeedbackAgent] {reason}")
        print(f"  [FeedbackAgent] Thresholds now: {self.thresholds}")
        return dict(self.thresholds)

    def get_thresholds(self) -> dict:
        return dict(self.thresholds)

    def reset(self):
        self.thresholds = self._defaults()
        self._save()
        print(f"  [FeedbackAgent] Reset: {self.thresholds}")

    def history(self) -> list:
        if not os.path.exists(THRESH_HIST):
            return []
        with open(THRESH_HIST) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _adjust(self, delta):
        old = dict(self.thresholds)
        for k in ("alert", "block", "isolate"):
            self.thresholds[k] = round(
                max(THRESHOLD_MIN, min(THRESHOLD_MAX,
                    self.thresholds[k] + delta)), 4)
        os.makedirs(os.path.dirname(THRESH_HIST), exist_ok=True)
        with open(THRESH_HIST, "a") as f:
            f.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "delta": delta, "before": old,
                "after": dict(self.thresholds),
            }) + "\n")

    def _load(self):
        if os.path.exists(THRESH_FILE):
            with open(THRESH_FILE) as f:
                return json.load(f)
        return self._defaults()

    def _defaults(self):
        return {"alert": THRESHOLD_ALERT,
                "block": THRESHOLD_BLOCK,
                "isolate": THRESHOLD_ISOLATE}

    def _save(self):
        os.makedirs(os.path.dirname(THRESH_FILE), exist_ok=True)
        with open(THRESH_FILE, "w") as f:
            json.dump(self.thresholds, f, indent=2)
