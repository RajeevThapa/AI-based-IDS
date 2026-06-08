"""
models/detector.py  —  Inference wrapper for rf_fused_v2.pkl

FIX: The scaler was fitted WITHOUT feature names (from a numpy array in
phase2_fused_fix.py), so we must call scaler.transform() with a plain
numpy array, not a DataFrame. The imputer WAS fitted with a DataFrame
(feature_names_in_ exists), so it gets a DataFrame.
This completely silences the sklearn feature-name UserWarning.
"""
import json, warnings
import numpy as np
import pandas as pd
import joblib
from config.settings import MODEL_PATH, SCALER_PATH, IMPUTER_PATH, LABEL_ENC

# Suppress any residual sklearn internal warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class MalwareDetector:
    """
    Wraps rf_fused_v2.pkl for single-sample and batch inference.

    Usage
    -----
        d = MalwareDetector()
        result = d.predict(feature_dict_or_series)
        # → {"family": "Mirai", "threat_score": 0.97, "confidence": {...}}
    """

    def __init__(self):
        print("  [Detector] Loading model ...")
        self.model   = joblib.load(MODEL_PATH)
        self.scaler  = joblib.load(SCALER_PATH)
        self.imputer = joblib.load(IMPUTER_PATH)

        with open(LABEL_ENC) as f:
            self.le_map = {int(k): v for k, v in json.load(f).items()}
        self.inv_le = {v: k for k, v in self.le_map.items()}

        # Feature names come from the imputer (fitted with DataFrame in phase2_fused_fix.py)
        if hasattr(self.imputer, "feature_names_in_"):
            self.feature_names = list(self.imputer.feature_names_in_)
        elif hasattr(self.scaler, "feature_names_in_"):
            self.feature_names = list(self.scaler.feature_names_in_)
        else:
            self.feature_names = [f"f{i}" for i in range(self.scaler.n_features_in_)]

        self.n_features = len(self.feature_names)
        print(f"  [Detector] Ready  —  {len(self.le_map)} classes  "
              f"·  {self.n_features} features")

    # ── Public API ─────────────────────────────────────────────────────────

    def predict(self, features) -> dict:
        """
        Classify one observation. Missing features → imputed with median.

        Parameters
        ----------
        features : dict | pd.Series | np.ndarray
        """
        # Step 1: build a named DataFrame for the imputer
        df_named = self._to_dataframe(features)

        # Step 2: impute — imputer wants a DataFrame (it was fitted with one)
        imputed = self.imputer.transform(df_named)   # returns numpy array

        # Step 3: scale — scaler wants a numpy array (it was fitted with one)
        scaled  = self.scaler.transform(imputed)     # returns numpy array

        # Step 4: predict
        probs = self.model.predict_proba(scaled)[0]

        idx          = int(np.argmax(probs))
        family       = self.le_map[idx]
        threat_score = float(probs[idx])
        confidence   = {self.le_map[i]: float(p) for i, p in enumerate(probs)}

        return {"family": family,
                "threat_score": threat_score,
                "confidence": confidence}

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Classify a DataFrame; adds predicted_family and threat_score."""
        feat_df = df.reindex(columns=self.feature_names, fill_value=np.nan)
        imputed = self.imputer.transform(feat_df)    # DataFrame → numpy
        scaled  = self.scaler.transform(imputed)     # numpy → numpy
        probs   = self.model.predict_proba(scaled)
        idxs    = np.argmax(probs, axis=1)
        scores  = probs[np.arange(len(probs)), idxs]

        out = df.copy()
        out["predicted_family"] = [self.le_map[i] for i in idxs]
        out["threat_score"]     = scores
        return out

    # ── Private ────────────────────────────────────────────────────────────

    def _to_dataframe(self, features) -> pd.DataFrame:
        """Convert any input to a one-row DataFrame with correct column names."""
        if isinstance(features, pd.Series):
            row = {f: features.get(f, np.nan) for f in self.feature_names}
        elif isinstance(features, dict):
            row = {f: features.get(f, np.nan) for f in self.feature_names}
        elif isinstance(features, np.ndarray):
            arr = features.flatten()
            if len(arr) < self.n_features:
                arr = np.concatenate([arr,
                      np.full(self.n_features - len(arr), np.nan)])
            row = dict(zip(self.feature_names, arr[:self.n_features]))
        else:
            raise TypeError(f"Unsupported input type: {type(features)}")
        return pd.DataFrame([row], columns=self.feature_names)
