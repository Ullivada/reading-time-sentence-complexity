"""Sentence-level train/dev/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def sentence_keys_df(df: pd.DataFrame) -> pd.DataFrame:
    """Unique (itemno, sentence_id) from word-level table."""
    return df[["itemno", "sentence_id"]].drop_duplicates().reset_index(drop=True)


def make_splits(
    keys: pd.DataFrame,
    train_ratio: float,
    dev_ratio: float,
    random_seed: int,
) -> dict[str, list[list[int]]]:
    """
    Return {"train": [[itemno, sent_id], ...], "dev": [...], "test": [...]}.
    """
    n = len(keys)
    idx = np.arange(n)
    rng = np.random.default_rng(random_seed)
    rng.shuffle(idx)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train_i = idx[:n_train]
    dev_i = idx[n_train : n_train + n_dev]
    test_i = idx[n_train + n_dev :]

    def pack(sub_idx: np.ndarray) -> list[list[int]]:
        out = []
        for j in sub_idx:
            row = keys.iloc[int(j)]
            out.append([int(row["itemno"]), int(row["sentence_id"])])
        return out

    return {"train": pack(train_i), "dev": pack(dev_i), "test": pack(test_i)}


def save_splits(path: Path, splits: dict[str, list[list[int]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def assign_split_column(df: pd.DataFrame, splits: dict[str, list[list[int]]]) -> pd.Series:
    train_s = {tuple(k) for k in splits["train"]}
    dev_s = {tuple(k) for k in splits["dev"]}
    test_s = {tuple(k) for k in splits["test"]}

    def label(row: pd.Series) -> str:
        t = (int(row["itemno"]), int(row["sentence_id"]))
        if t in train_s:
            return "train"
        if t in dev_s:
            return "dev"
        if t in test_s:
            return "test"
        return "unknown"

    return df.apply(label, axis=1)
