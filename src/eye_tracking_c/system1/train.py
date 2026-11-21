"""System 1: predict per-word Dundee reading times with linear regression (paper Sec 3).

Produces the paper's three System-1 tables:
  Table 1  Pearson correlation of each feature with each reading-time measure
  Table 2  FFD ablation: feature groups added in the paper's order; a group is kept
           only if it raises R^2 (on the dev split), as the paper describes
  Table 3  R^2 per measure with the selected features, raw and thresholded
           (predictions below a dev-tuned threshold set to 0 ms)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from eye_tracking_c.features.pipeline import FeaturePipeline, add_lexical_features

TARGETS = ("ffd", "fpd", "rpd", "td")

# Paper Table 2 rows, in order: (label, feature columns added at that step)
ABLATION_STEPS = (
    ("Word Length", ["word_len"]),
    ("Sentence Length", ["sentence_len"]),
    ("Word Frequency", ["wiki_log_freq"]),
    ("Mean AoA", ["aoa_mean"]),
    ("Std Dev AoA", ["aoa_sd"]),
    ("Total Surprisal", ["total_surprisal"]),
    ("Lexical Surprisal", ["lexical_surprisal"]),
    ("Syntactic Surprisal", ["syntactic_surprisal"]),
    ("Entropy Reduction", ["entropy_reduction"]),
    ("Embedding Depth", ["embedding_depth", "weighted_embedding_depth"]),
    ("Embedding Difference", ["embedding_difference"]),
    ("Hierarchical structure feats", ["p_F+L-", "p_F-L+", "p_F+L+", "p_F-L-",
                                      "best_F+L-", "best_F-L+", "best_F+L+", "best_F-L-"]),
)


def apply_ms_threshold(y_pred: np.ndarray, tau: float) -> np.ndarray:
    out = np.asarray(y_pred, dtype=np.float64).copy()
    if tau > 0:
        out[out < tau] = 0.0
    return out


def tune_threshold(y_true, y_pred, grid) -> tuple[float, float]:
    """Return (best_tau, best_r2_on_dev)."""
    scores = [(r2_score(y_true, apply_ms_threshold(y_pred, t)), t) for t in grid]
    best_r2, best_tau = max(scores)
    return float(best_tau), float(best_r2)


def _fit(train, other, cols, target):
    pipe = FeaturePipeline(tuple(cols))
    reg = LinearRegression().fit(pipe.fit_transform(train), train[target].values)
    return pipe, reg, reg.predict(pipe.transform(other))


def correlations(df: pd.DataFrame) -> dict:
    """Table 1: Pearson r of each feature with each measure (rows with the feature)."""
    out = {}
    for _label, cols in ABLATION_STEPS:
        for c in cols:
            if c not in df.columns:
                continue
            m = df[c].notna()
            out[c] = {t: float(np.corrcoef(df.loc[m, c], df.loc[m, t])[0, 1]) for t in TARGETS}
    return out


def ablation(train, dev, test, target="ffd") -> tuple[list[dict], list[str]]:
    """Table 2: add feature groups in paper order; keep a group only if dev R^2 rises."""
    kept: list[str] = []
    best_dev = -np.inf
    rows = []
    for label, cols in ABLATION_STEPS:
        cols = [c for c in cols if c in train.columns]
        if not cols:
            continue
        trial = kept + cols
        _, _, pred_dev = _fit(train, dev, trial, target)
        _, _, pred_test = _fit(train, test, trial, target)
        r2_dev = r2_score(dev[target], pred_dev)
        r2_test = r2_score(test[target], pred_test)
        retained = r2_dev > best_dev
        if retained:
            kept, best_dev = trial, r2_dev
        rows.append({"step": label, "r2_dev": float(r2_dev), "r2_test": float(r2_test),
                     "retained": bool(retained)})
    return rows, kept


def train_system1(table_path: Path, models_dir: Path, experiments_dir: Path,
                  threshold_grid_ms: dict[str, list[float]]) -> dict:
    df = add_lexical_features(pd.read_csv(table_path))
    train, dev, test = (df[df["split"] == s] for s in ("train", "dev", "test"))

    table2, selected = ablation(train, dev, test)

    models_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics = {
        "run_id": run_id,
        "n_words": len(df),
        "n_sentences": int(df[["itemno", "sentence_id"]].drop_duplicates().shape[0]),
        "table1_correlations": correlations(df),
        "table2_ablation_ffd": table2,
        "feature_columns": selected,
        "targets": {},
    }

    for tgt in TARGETS:
        pipe, reg, pred_dev = _fit(train, dev, selected, tgt)
        pred_test = reg.predict(pipe.transform(test))
        tau, r2_dev = tune_threshold(dev[tgt].values, pred_dev, threshold_grid_ms.get(tgt, [0]))
        metrics["targets"][tgt] = {
            "threshold_ms": tau,
            "r2_dev_clipped": r2_dev,
            "r2_test_raw": float(r2_score(test[tgt], pred_test)),
            "r2_test_clipped": float(r2_score(test[tgt], apply_ms_threshold(pred_test, tau))),
            "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
        }
        joblib.dump(reg, models_dir / f"s1_{tgt}_regressor.joblib")
        joblib.dump(pipe, models_dir / "s1_feature_pipeline.joblib")

    exp_dir = experiments_dir / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
