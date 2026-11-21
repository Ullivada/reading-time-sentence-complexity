"""System 2: which sentence of a Standard/Simple Wikipedia pair is simpler (paper Sec 4).

Two formulations, both scored by 10-fold cross-validation accuracy (paper Table 5):
  pairwise  logistic regression on 'sentence1_*' + 'sentence2_*' features
  svmrank   linear ranking SVM; with two sentences per pair this is a linear SVM on
            the difference of the two sentence vectors (Joachims 2006)
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from eye_tracking_c.system2.dataset import load_wiki_pairs
from eye_tracking_c.system2.featurize import sentence_features
from eye_tracking_c.system2.predict import load_s1_ffd_predictor, predict_sentences_ffd

# (model name, feature kind, classifier)
MODELS = (
    ("base_rt_svmrank", "base", "svmrank"),
    ("base_rt_pairwise", "base", "pairwise"),
    ("extended_rt_svmrank", "extended", "svmrank"),
    ("extended_rt_pairwise", "extended", "pairwise"),
    ("sentence_only_pairwise", "sentence", "pairwise"),
)


def _cv_accuracy(clf, X, y, n_folds, seed):
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def train_system2(pairs_path: Path, parses: dict, models_dir: Path, experiments_dir: Path,
                  *, n_folds=10, random_seed=42, max_pairs=None) -> dict:
    pairs = load_wiki_pairs(pairs_path, max_pairs=max_pairs, random_seed=random_seed)
    regressor, feature_pipe, threshold_ms = load_s1_ffd_predictor(models_dir, experiments_dir)

    # predicted FFD for every unique sentence
    unique = sorted(set(pairs["standard"]) | set(pairs["simple"]))
    rts = dict(zip(unique, predict_sentences_ffd(unique, parses, regressor, feature_pipe, threshold_ms)))

    # random order within each pair; label 1 = second sentence is the simple one
    rng = random.Random(random_seed)
    ordered, labels = [], []
    for std, simp in zip(pairs["standard"], pairs["simple"]):
        if rng.random() < 0.5:
            ordered.append((std, simp))
            labels.append(1)
        else:
            ordered.append((simp, std))
            labels.append(0)
    y = np.asarray(labels)

    results = {}
    for name, kind, method in MODELS:
        feats = {s: sentence_features(rts[s], parses[s], kind) for s in unique}
        if method == "pairwise":
            rows = [{**{f"sentence1_{k}": v for k, v in feats[a].items()},
                     **{f"sentence2_{k}": v for k, v in feats[b].items()}} for a, b in ordered]
            X = DictVectorizer().fit_transform(rows)
            clf = make_pipeline(MaxAbsScaler(), LogisticRegression(max_iter=2000))
        else:
            vec = DictVectorizer().fit([feats[s] for s in unique])
            X = vec.transform([feats[a] for a, _ in ordered]) - vec.transform([feats[b] for _, b in ordered])
            clf = make_pipeline(MaxAbsScaler(), LinearSVC(max_iter=5000))
        mean, std = _cv_accuracy(clf, X, y, n_folds, random_seed)
        results[name] = {"accuracy_mean": mean, "accuracy_std": std, "n_features": int(X.shape[1])}
        print(f"  {name:24s} {100 * mean:6.2f}%  (±{100 * std:.2f})", flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    metrics = {"run_id": run_id, "system": "s2", "n_pairs": len(pairs), "n_folds": n_folds,
               "ffd_threshold_ms": threshold_ms, "models": results}
    exp_dir = experiments_dir / run_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
