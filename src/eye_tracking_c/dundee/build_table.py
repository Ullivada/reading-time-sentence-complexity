"""Build pooled Dundee word table + splits."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from eye_tracking_c.dundee.io_lexical import load_all_lexical
from eye_tracking_c.dundee.io_rt1p import load_rt1p_events, parse_rt1p_filename
from eye_tracking_c.dundee.measures import ZERO, all_word_measures
from eye_tracking_c.dundee.splits import (
    assign_split_column,
    make_splits,
    save_splits,
    sentence_keys_df,
)


def _load_dundee_wlen(corpus_root: Path) -> pd.DataFrame:
    """Word length as given by the Dundee corpus (per-word-info column 9, WLEN).

    Paper Sec 3.2: "Word length has been taken as it is from the Dundee corpus."
    """
    rows = []
    for path in sorted((corpus_root / "per-word-info").glob("tx*wrdp.dat")):
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 14:
                    rows.append((int(parts[1]), int(parts[12]), int(parts[8])))
    return pd.DataFrame(rows, columns=["itemno", "wnum", "word_len"])


def _merge_parser_features(table: pd.DataFrame, build_dir: Path) -> pd.DataFrame:
    """Left-join per-word PCFG parser features (build/parser_features_dundee.csv)."""
    path = build_dir / "parser_features_dundee.csv"
    if not path.is_file():
        return table
    return table.merge(pd.read_csv(path), on=["itemno", "wnum"], how="left")


def build_pooled_word_table(
    corpus_root: Path,
    min_pooled_td_ms: float = 1.0,
    random_seed: int = 42,
    train_ratio: float = 0.6,
    dev_ratio: float = 0.2,
    build_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict, dict]:
    """
    Returns (word_table_df, splits_dict, manifest_dict).
    """
    build_dir = build_dir or corpus_root.parent / "build"
    rt_dir = corpus_root / "RT_dataset1P"
    lexical_dir = corpus_root / "per-word-lexical-annotation"

    lexical = load_all_lexical(lexical_dir)

    # One representative row per (itemno, wnum) for metadata
    meta = lexical.drop_duplicates(subset=["itemno", "wnum"], keep="first")[
        ["itemno", "wnum", "word", "sentence_id"]
    ].copy()

    sent_len = (
        lexical.groupby(["itemno", "sentence_id"], as_index=False)
        .size()
        .rename(columns={"size": "sentence_len"})
    )
    meta = meta.merge(sent_len, on=["itemno", "sentence_id"], how="left")

    wnums_by_item: dict[int, list[int]] = {
        int(itemno): sorted(g["wnum"].unique().tolist())
        for itemno, g in lexical.groupby("itemno")
    }

    per_subject_rows: list[dict] = []
    for path in sorted(rt_dir.glob("s*ma1p.dat")):
        parsed = parse_rt1p_filename(path.name)
        if not parsed:
            continue
        _subj, text_id = parsed
        events = load_rt1p_events(path)
        if not events:
            continue
        measures = all_word_measures(events)
        for w in wnums_by_item.get(text_id, []):
            m = measures.get(w, ZERO)
            per_subject_rows.append(
                {
                    "itemno": text_id,
                    "wnum": w,
                    "subject_file": path.name,
                    **m,
                }
            )

    subj_df = pd.DataFrame(per_subject_rows)
    if subj_df.empty:
        raise RuntimeError("No RT_dataset1P rows parsed; check corpus_root path.")

    pooled = (
        subj_df.groupby(["itemno", "wnum"], as_index=False)
        .agg(
            ffd=("ffd", "mean"),
            fpd=("fpd", "mean"),
            rpd=("rpd", "mean"),
            td=("td", "mean"),
            n_subjects=("ffd", "count"),
        )
    )

    table = meta.merge(pooled, on=["itemno", "wnum"], how="inner")
    table = table.merge(_load_dundee_wlen(corpus_root), on=["itemno", "wnum"], how="left")
    table = table[table["td"] >= min_pooled_td_ms].reset_index(drop=True)

    table = _merge_parser_features(table, build_dir)

    keys = sentence_keys_df(table)
    splits = make_splits(keys, train_ratio, dev_ratio, random_seed)
    table["split"] = assign_split_column(table, splits)

    manifest = {
        "n_lexical_words": len(lexical),
        "n_pooled_rows": len(table),
        "n_sentences_split": len(keys),
        "n_rt_files_used": int(subj_df["subject_file"].nunique()),
        "min_pooled_td_ms": min_pooled_td_ms,
    }

    return table, splits, manifest


def write_build_artifacts(
    table: pd.DataFrame,
    splits: dict,
    manifest: dict,
    build_dir: Path,
) -> None:
    build_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(build_dir / "dundee_word_table.csv", index=False)
    save_splits(build_dir / "splits.json", splits)
    (build_dir / "build_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
