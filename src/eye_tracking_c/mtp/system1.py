"""System 1 (enhanced): reading-time regression with GPT-2 features instead of the parser.

Same model as the paper (standardized LinearRegression, dev-tuned threshold). The
ablation compares fixed feature sets side by side and reports Delta R^2 over the
baseline, so each row answers "what does this source add":

  A  baseline: length, sentence length, frequency, AoA mean/SD
  B  A + 15 PCFG parser features (the current model; needs main.py's parser cache)
  C  A + GPT-2 surprisal
  D  A + GPT-2 surprisal, entropy, previous-word surprisal   <- final, feeds System 2
  E  D + 15 PCFG parser features (needs the parser cache)
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from eye_tracking_c.features.pipeline import LOW_LEVEL_FEATURES, PARSER_FEATURES
from eye_tracking_c.system1.train import TARGETS, _fit, apply_ms_threshold, tune_threshold

GPT2 = ["gpt2_surprisal", "gpt2_entropy", "gpt2_prev_surprisal"]
BASE = list(LOW_LEVEL_FEATURES)
PARSER = list(PARSER_FEATURES)
ROWS = (
    ("A", "Baseline (length, frequency, AoA)", BASE),
    ("B", "+ PCFG parser (current model)", BASE + PARSER),
    ("C", "+ GPT-2 surprisal", BASE + GPT2[:1]),
    ("D", "+ GPT-2 surprisal, entropy, spillover (enhanced)", BASE + GPT2),
    ("E", "Enhanced + PCFG parser", BASE + GPT2 + PARSER),
)
FINAL = "D"


def train_system1_mtp(df: pd.DataFrame, models_dir: Path, threshold_grid_ms: dict,
                      current_models_dir: Path | None = None) -> dict:
    """df: word table with lexical + GPT-2 columns (+ parser columns when available)."""
    train, dev, test = (df[df["split"] == s] for s in ("train", "dev", "test"))
    has_parser = all(c in df.columns for c in PARSER)

    ablation = []
    for row, label, cols in ROWS:
        if not has_parser and any(c in PARSER for c in cols):
            print(f"  row {row} skipped: no parser features (run main.py once to create them)", flush=True)
            continue
        r2 = {t: float(r2_score(test[t], _fit(train, test, cols, t)[2])) for t in TARGETS}
        ablation.append({"row": row, "label": label, "features": cols, "r2_test": r2})
        print(f"  {row} {label:52s} " + " ".join(f"{t} {v:.3f}" for t, v in r2.items()), flush=True)
    base = ablation[0]["r2_test"]
    for a in ablation:
        a["delta_r2_test"] = {t: a["r2_test"][t] - base[t] for t in TARGETS}

    cols = dict((r, c) for r, _, c in ROWS)[FINAL]
    models_dir.mkdir(parents=True, exist_ok=True)
    targets, test_pred = {}, {}
    for t in TARGETS:
        pipe, reg, pred_dev = _fit(train, dev, cols, t)
        pred_test = reg.predict(pipe.transform(test))
        tau, r2_dev = tune_threshold(dev[t].values, pred_dev, threshold_grid_ms.get(t, [0]))
        targets[t] = {"threshold_ms": tau, "r2_dev_clipped": r2_dev,
                      "r2_test_raw": float(r2_score(test[t], pred_test)),
                      "r2_test_clipped": float(r2_score(test[t], apply_ms_threshold(pred_test, tau)))}
        test_pred[t] = pred_test
        joblib.dump(reg, models_dir / f"s1_{t}_regressor.joblib")
        if t == "ffd":
            coefs = dict(zip(cols, map(float, reg.coef_)))
    joblib.dump(pipe, models_dir / "s1_feature_pipeline.joblib")

    r_len = {"enhanced": float(np.corrcoef(test_pred["ffd"], test["word_len"])[0, 1])}
    old = (current_models_dir or Path()) / "s1_ffd_regressor.joblib"
    if has_parser and old.is_file():
        reg_old = joblib.load(old)
        pipe_old = joblib.load(old.parent / "s1_feature_pipeline.joblib")
        r_len["current"] = float(np.corrcoef(reg_old.predict(pipe_old.transform(test)), test["word_len"])[0, 1])

    return {"n_words": len(df), "n_sentences": int(df[["itemno", "sentence_id"]].drop_duplicates().shape[0]),
            "ablation": ablation, "final_row": FINAL, "feature_columns": cols, "targets": targets,
            "ffd_coefficients": coefs, "r_pred_ffd_wordlen": r_len}
