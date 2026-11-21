"""Loaders for paper-faithful Tier-B lexical resources.

These back the low-level predictors used in Singh et al. (2016), Table 2:
- Wikipedia unigram frequency  (data/enwiki_word_frequency.txt)
- Age-of-acquisition mean + SD  (data/aoa_kuperman_2012.csv, Kuperman 2012)

Both are simple word-keyed lookups so they work identically for System-1
training (Dundee words) and System-2 inference (Wikipedia sentence words).
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import pandas as pd

from eye_tracking_c.paths import repo_root


def _resources_dir() -> Path:
    return repo_root() / "data"


@functools.lru_cache(maxsize=1)
def load_wiki_log_freq(path: str | None = None) -> dict[str, float]:
    """Return {lowercased word -> log10(count)} from the enwiki unigram list.

    File format: ``word<space>count`` per line (IlyaSemenov/wikipedia-word-frequency,
    enwiki-2023-04-13), a genuine English-Wikipedia unigram count as the paper
    specifies ("unigram frequency in the entire English Wikipedia text").
    """
    p = Path(path) if path else _resources_dir() / "enwiki_word_frequency.txt"
    out: dict[str, float] = {}
    if not p.is_file():
        return out
    with p.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            word, count = parts
            try:
                c = int(count)
            except ValueError:
                continue
            if c <= 0:
                continue
            out[word.lower()] = math.log10(c)
    return out


@functools.lru_cache(maxsize=1)
def load_aoa(path: str | None = None) -> dict[str, tuple[float, float]]:
    """Return {lowercased word -> (Rating.Mean, Rating.SD)} from Kuperman (2012)."""
    p = Path(path) if path else _resources_dir() / "aoa_kuperman_2012.csv"
    if not p.is_file():
        return {}
    df = pd.read_csv(p, usecols=["Word", "Rating.Mean", "Rating.SD"])
    df = df.dropna(subset=["Word"])
    out: dict[str, tuple[float, float]] = {}
    for word, mean, sd in zip(df["Word"], df["Rating.Mean"], df["Rating.SD"]):
        try:
            m = float(mean)
            s = float(sd)
        except (TypeError, ValueError):
            continue
        if math.isnan(m):
            continue
        out[str(word).lower()] = (m, s if not math.isnan(s) else float("nan"))
    return out
