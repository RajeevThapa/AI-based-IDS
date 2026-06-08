"""
agents/response_agent.py  —  Autonomous response agent.

Maps threat_score → one of four escalating actions and executes it.
Simulated by default (REAL_MODE = False in config/settings.py).

Thresholds (edit config/settings.py — not here):
  score <  0.30  →  log      (record, no network action)
  score <  0.60  →  alert    (notify admin)
  score <  0.80  →  block    (iptables DROP on src IP)
  score >= 0.80  →  isolate  (block + bring interface down)
"""
import subprocess
from datetime import datetime, timezone
from config.settings import (
    THRESHOLD_ALERT, THRESHOLD_BLOCK, THRESHOLD_ISOLATE,
    BENIGN_LABEL, REAL_MODE,
)
from utils.logger import log_decision


class ResponseAgent:

    LOG     = "log"
    ALERT   = "alert"
    BLOCK   = "block"
    ISOLATE = "isolate"

    def __init__(self, real_mode: bool = None):
        self.real_mode = real_mode if real_mode is not None else REAL_MODE
        mode = "REAL COMMANDS" if self.real_mode else "SIMULATED (safe)"
        print(f"  [ResponseAgent] Mode: {mode}")

    # ── Main entry point ───────────────────────────────────────────────────

    def respond(self, detection: dict, device_id="device-unknown",
                src_ip="0.0.0.0", iface="eth0") -> dict:
        """
        Decide and execute the appropriate response.

        Parameters
        ----------
        detection : output of MalwareDetector.predict()
        device_id : IoT device name / ID
        src_ip    : source IP from the network flow
        iface     : network interface (used only for isolate)

        Returns  the audit log record written for this decision.
        """
        family       = detection["family"]
        threat_score = detection["threat_score"]
        confidence   = detection["confidence"]

        # Determine action
        if family == BENIGN_LABEL:
            action = self.LOG
        elif threat_score >= THRESHOLD_ISOLATE:
            action = self.ISOLATE
        elif threat_score >= THRESHOLD_BLOCK:
            action = self.BLOCK
        elif threat_score >= THRESHOLD_ALERT:
            action = self.ALERT
        else:
            action = self.LOG

        # Execute
        self._execute(action, family, threat_score, src_ip, device_id, iface)

        # Write audit log
        record = log_decision(
            device_id    = device_id,
            src_ip       = src_ip,
            family       = family,
            threat_score = threat_score,
            action       = action,
            confidence   = confidence,
            real_mode    = self.real_mode,
            extra        = {"iface": iface},
        )

        # Console line
        icons = {self.LOG:"📋", self.ALERT:"⚠️ ",
                 self.BLOCK:"🚫", self.ISOLATE:"🔴"}
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        print(f"  {icons[action]}  [{ts}]  "
              f"{action.upper():<8}  {family:<14}  "
              f"score={threat_score:.3f}  "
              f"src={src_ip:<16} dev={device_id}")
        return record

    # ── Action implementations ─────────────────────────────────────────────

    def _execute(self, action, family, score, src_ip, device_id, iface):
        if action == self.LOG:
            return

        elif action == self.ALERT:
            print(f"\n    ⚠️  ALERT — {family} on {device_id}  "
                  f"score={score:.3f}  src={src_ip}")

        elif action == self.BLOCK:
            self._run(f"iptables -A INPUT -s {src_ip} -j DROP")
            print(f"\n    🚫  BLOCKED {src_ip} — {family} on {device_id}  "
                  f"score={score:.3f}")

        elif action == self.ISOLATE:
            self._run(f"iptables -A INPUT -s {src_ip} -j DROP")
            self._run(f"ip link set {iface} down")
            print(f"\n    🔴  ISOLATED {device_id} ({iface} down) — {family}  "
                  f"src={src_ip}  score={score:.3f}")

    def _run(self, cmd: str):
        if self.real_mode:
            try:
                r = subprocess.run(cmd.split(), capture_output=True,
                                   text=True, timeout=5)
                if r.returncode != 0:
                    print(f"    [CMD FAILED] {cmd}\n    {r.stderr.strip()}")
            except Exception as e:
                print(f"    [CMD ERROR]  {cmd}  →  {e}")
        else:
            print(f"    [SIM] {cmd}")
