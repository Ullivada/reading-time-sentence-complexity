"""System 2 (enhanced): which sentence of a Standard/Simple Wikipedia pair is simpler.

Classical rows keep the paper's pairwise logistic regression, now fed by the enhanced
System 1. Each is scored two ways:
  cv_mean   10-fold CV, the same protocol as main.py and the paper
  test_acc  fit on train+dev, scored on a fixed pair-level test split, the protocol the
            cross-encoder and hybrid rows use (a transformer cannot be 10-fold trained here)

The test split is grouped by the standard sentence, so no standard sentence appears on
both sides. Features are built as sparse matrices; 'sentence1_word{k}' in the paper is
column k of the first block, so this equals the DictVectorizer encoding of main.py.
"""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eye_tracking_c.mtp.system1 import GPT2
from eye_tracking_c.system1.train import TARGETS, apply_ms_threshold
from eye_tracking_c.system2.train import _cv_accuracy

# (key, label, feature kind)
CLASSICAL = (
    ("length_control", "Length control (chars + words)", "length"),
    ("base_rt", "Base RT (predicted FFD per word)", "base"),
    ("extended_rt", "Extended RT (Base RT + sentence features)", "extended"),
    ("sentence_only", "Sentence-level features only", "sentence"),
    ("base_rt_4measures", "Base RT, all 4 measures per word", "base4"),
)


def order_pairs(pairs: pd.DataFrame, seed: int):
    """Random order within each pair, exactly as system2/train.py; y=1: second is simple."""
    rng = random.Random(seed)
    ordered, y = [], []
    for std, simp in zip(pairs["standard"], pairs["simple"]):
        if rng.random() < 0.5:
            ordered.append((std, simp))
            y.append(1)
        else:
            ordered.append((simp, std))
            y.append(0)
    return ordered, np.asarray(y)


def split_pairs(pairs: pd.DataFrame, seed: int, test_ratio=0.2, dev_ratio=0.2) -> dict:
    """Pair indices for train/dev/test, grouped by the standard sentence."""
    groups = pd.factorize(pairs["standard"])[0]
    perm = np.random.default_rng(seed).permutation(groups.max() + 1)
    n_test, n_dev = int(test_ratio * len(perm)), int(dev_ratio * len(perm))
    where = np.empty(len(perm), dtype="<U5")
    where[perm[:n_test]], where[perm[n_test : n_test + n_dev]], where[perm[n_test + n_dev :]] = "test", "dev", "train"
    side = where[groups]
    return {s: np.flatnonzero(side == s) for s in ("train", "dev", "test")}


def predict_rts(unique: list[str], lm_wiki: dict, models_dir: Path) -> dict:
    """{target: flat array of predicted ms over all words of `unique`} + 'lens'."""
    lens = np.asarray([len(s.split()) for s in unique])
    df = pd.DataFrame(np.vstack([lm_wiki[s] for s in unique]), columns=GPT2)
    df["word"] = [w for s in unique for w in s.split()]
    df["word_len"] = df["word"].str.len()
    df["sentence_len"] = np.repeat(lens, lens)
    pipe = joblib.load(models_dir / "s1_feature_pipeline.joblib")
    X = pipe.transform(df)
    out = {"lens": lens, "gpt2": df[GPT2].to_numpy()}
    for t in TARGETS:
        out[t] = joblib.load(models_dir / f"s1_{t}_regressor.joblib").predict(X)
    return out


def sentence_matrices(unique: list[str], rts: dict, ffd_threshold_ms: float) -> dict:
    """One sparse (n_sentences x d) matrix per feature kind."""
    lens = rts["lens"]
    ptr = np.r_[0, np.cumsum(lens)]
    pos = np.arange(ptr[-1]) - np.repeat(ptr[:-1], lens)  # word position inside its sentence
    width = int(lens.max())
    ffd = apply_ms_threshold(rts["ffd"], ffd_threshold_ms)
    base = sp.csr_matrix((ffd, pos, ptr), shape=(len(unique), width))
    data4 = np.stack([rts[t] for t in TARGETS], axis=1).ravel()
    idx4 = (pos[:, None] * 4 + np.arange(4)).ravel()
    base4 = sp.csr_matrix((data4, idx4, ptr * 4), shape=(len(unique), width * 4))

    sums = lambda v: np.add.reduceat(v, ptr[:-1])  # noqa: E731  per-sentence sum
    surp = np.nan_to_num(rts["gpt2"][:, 0])
    ent = np.nan_to_num(rts["gpt2"][:, 1])
    sent = np.stack([lens, sums(ffd) / lens, sums(surp), sums(ent) / lens], axis=1)
    chars = np.asarray([len(s.replace(" ", "")) for s in unique])
    return {"length": sp.csr_matrix(np.stack([chars, lens], axis=1).astype(float)),
            "base": base, "base4": base4, "sentence": sp.csr_matrix(sent),
            "extended": sp.hstack([base, sp.csr_matrix(sent)]).tocsr()}


def pair_matrix(S, ia, ib):
    return sp.hstack([S[ia], S[ib]]).tocsr()


def classifier():
    """Logistic regression on std-scaled features (no centering, so sparse stays sparse).

    main.py uses MaxAbsScaler, which squeezes counts like sentence length into [0, 1];
    separating two similar sentences then needs large weights, which the default L2
    penalty (C=1) holds back. That costs the 4-feature length control 1.7 points
    (73.45 vs 75.18 CV), so it would understate the bar the RT rows must beat.
    """
    return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=2000))


def mcnemar(a_ok: np.ndarray, b_ok: np.ndarray) -> float:
    b, c = int((a_ok & ~b_ok).sum()), int((~a_ok & b_ok).sum())
    return float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0


def run_classical(X_by_kind: dict, y: np.ndarray, idx: dict, n_folds: int, seed: int,
                  four_measures: bool = True):
    """Rows 1-5. Returns (results, test_correct) where test_correct[key] is a bool array."""
    results, correct = {}, {}
    fit_idx = np.r_[idx["train"], idx["dev"]]
    for key, label, kind in CLASSICAL:
        if kind == "base4" and not four_measures:
            continue
        X = X_by_kind[kind]
        cv_mean, cv_std = _cv_accuracy(classifier(), X, y, n_folds, seed)
        clf = classifier().fit(X[fit_idx], y[fit_idx])
        ok = clf.predict(X[idx["test"]]) == y[idx["test"]]
        results[key] = {"label": label, "cv_mean": cv_mean, "cv_std": cv_std,
                        "test_acc": float(ok.mean()), "n_features": int(X.shape[1])}
        correct[key] = ok
        print(f"  {label:44s} CV {100 * cv_mean:6.2f}% (±{100 * cv_std:.2f})  test {100 * ok.mean():6.2f}%",
              flush=True)
    return results, correct


def run_hybrid(logits: dict, X_by_kind: dict, y: np.ndarray, idx: dict):
    """Rows 7-8: logistic regression on [cross-encoder logit + features], fit on dev
    (where the logits are out-of-sample) and scored on test."""
    results, correct = {}, {}
    rows = (("hybrid_rt", "Hybrid: cross-encoder + Extended RT", "extended"),
            ("hybrid_length", "Hybrid control: cross-encoder + length", "length"))
    for key, label, kind in rows:
        X = X_by_kind[kind]
        Xd = sp.hstack([sp.csr_matrix(logits["dev"][:, None]), X[idx["dev"]]]).tocsr()
        Xt = sp.hstack([sp.csr_matrix(logits["test"][:, None]), X[idx["test"]]]).tocsr()
        ok = classifier().fit(Xd, y[idx["dev"]]).predict(Xt) == y[idx["test"]]
        results[key] = {"label": label, "test_acc": float(ok.mean()), "n_features": int(Xt.shape[1])}
        correct[key] = ok
        print(f"  {label:44s} test {100 * ok.mean():6.2f}%", flush=True)
    return results, correct


def length_matched(L, ia, ib) -> np.ndarray:
    """Pairs whose two sentences have about the same length: word counts differ by at
    most 1 and character counts by at most 10%. L is the 'length' sentence matrix."""
    L = L.toarray()
    a, b = L[ia], L[ib]
    return (np.abs(a[:, 1] - b[:, 1]) <= 1) & (np.abs(a[:, 0] - b[:, 0]) <= 0.1 * np.maximum(a[:, 0], b[:, 0]))


def on_subset(correct: dict, mask: np.ndarray) -> dict:
    """Accuracy of every row on a subset of the test pairs, with the same McNemar tests."""
    sub = {k: ok[mask] for k, ok in correct.items()}
    acc = {k: {"test_acc": float(ok.mean())} for k, ok in sub.items()}
    return {"n_pairs": int(mask.sum()), "acc": {k: v["test_acc"] for k, v in acc.items()},
            "mcnemar": significance(acc, sub)}


SIGNIFICANCE = (("base_rt", "length_control"), ("extended_rt", "length_control"),
                ("hybrid_rt", "cross_encoder"), ("hybrid_rt", "hybrid_length"))


def significance(results: dict, correct: dict) -> dict:
    out = {}
    for a, b in SIGNIFICANCE:
        if a in correct and b in correct:
            out[f"{a}_vs_{b}"] = {"p_value": mcnemar(correct[a], correct[b]),
                                  "delta_acc": results[a]["test_acc"] - results[b]["test_acc"]}
    return out
