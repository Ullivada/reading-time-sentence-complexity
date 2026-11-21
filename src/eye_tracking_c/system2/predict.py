"""Predict per-word FFD for Wikipedia sentences using the trained System 1."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from eye_tracking_c.parser.run import PARSER_COLUMNS
from eye_tracking_c.system1.train import apply_ms_threshold


def predict_sentences_ffd(sentences, parses, regressor, feature_pipe, threshold_ms=0.0):
    """Predict thresholded FFD (ms) per token.

    sentences: list of sentence strings (whitespace-tokenized, PTB style)
    parses:    {sentence: (n_tokens x 15 parser matrix, log parse prob)}
    """
    frames = []
    for sid, sent in enumerate(sentences):
        words = sent.split()
        f = pd.DataFrame(parses[sent][0], columns=list(PARSER_COLUMNS))
        f["word"] = words
        f["word_len"] = [len(w) for w in words]
        f["sentence_len"] = len(words)
        f["sid"] = sid
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    preds = apply_ms_threshold(regressor.predict(feature_pipe.transform(df)), threshold_ms)
    df["rt"] = preds
    return [g["rt"].tolist() for _, g in df.groupby("sid", sort=True)]


def load_s1_ffd_predictor(models_dir: Path, experiments_dir: Path):
    """Load FFD regressor, feature pipeline, and the latest dev-tuned FFD threshold."""
    regressor = joblib.load(models_dir / "s1_ffd_regressor.joblib")
    feature_pipe = joblib.load(models_dir / "s1_feature_pipeline.joblib")
    threshold_ms = 0.0
    for run in sorted(Path(experiments_dir).glob("*/metrics.json"), reverse=True):
        tgt = json.loads(run.read_text(encoding="utf-8")).get("targets", {}).get("ffd")
        if tgt is not None:
            threshold_ms = float(tgt.get("threshold_ms", 0.0))
            break
    return regressor, feature_pipe, threshold_ms
