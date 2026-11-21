"""Load Standard/Simple Wikipedia sentence pairs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_wiki_pairs(path: Path, max_pairs: int | None = None, random_seed: int = 42) -> pd.DataFrame:
    """
    Load sentence pairs.

    Supported formats:
    - Headered TSV: sentence_standard, sentence_simple, label, pair_id
    - Ambati/Hwang TSV (no header): Standard, Simple, Score
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Wiki pairs file not found: {path}")

    with path.open(encoding="utf-8", errors="replace") as f:
        first = f.readline()
    has_header = first.lower().startswith("sentence")

    if has_header:
        df = pd.read_csv(path, sep="\t")
        required = {"sentence_standard", "sentence_simple"}
        if not required.issubset(df.columns):
            raise ValueError(f"Expected columns {required}, got {list(df.columns)}")
        df = df.rename(
            columns={
                "sentence_standard": "standard",
                "sentence_simple": "simple",
            }
        )
    else:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            names=["standard", "simple", "score"],
            compression=None,
        )

    df["standard"] = df["standard"].astype(str).str.strip()
    df["simple"] = df["simple"].astype(str).str.strip()
    df = df[df["standard"] != df["simple"]].reset_index(drop=True)

    if max_pairs is not None and len(df) > max_pairs:
        df = df.sample(n=max_pairs, random_state=random_seed).reset_index(drop=True)

    return df
