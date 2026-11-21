"""Load per-word lexical annotation (text1.txt … text20.txt)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parse_lexical_line(line: str) -> dict | None:
    """
    Nine logical columns from README; surface `word` may contain spaces encoded
    as extra leading tab-separated fields (see e.g. text2 `garbage sentimental`).
    Parse from the right: deprel, head, cpos, word_in_sent, sentence_id, wnum,
    itemno, pos, then remainder joined as word.
    """
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 9:
        return None
    try:
        deprel = parts[-1]
        head = int(parts[-2])
        cpos = parts[-3]
        word_in_sent = int(parts[-4])
        sentence_id = int(parts[-5])
        wnum = int(parts[-6])
        itemno = int(parts[-7])
        pos = parts[-8]
        word = " ".join(p.strip() for p in parts[:-8] if p.strip()).strip()
        if not word:
            word = " ".join(parts[:-8]).strip()
    except (ValueError, IndexError):
        return None
    return {
        "word": word,
        "pos": pos,
        "itemno": itemno,
        "wnum": wnum,
        "sentence_id": sentence_id,
        "word_in_sent": word_in_sent,
        "cpos": cpos,
        "head": head,
        "deprel": deprel,
    }


def load_lexical_for_text(lexical_dir: Path, itemno: int) -> pd.DataFrame:
    path = lexical_dir / f"text{itemno}.txt"
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            rec = _parse_lexical_line(line)
            if rec:
                rows.append(rec)
    return pd.DataFrame(rows)


def load_all_lexical(lexical_dir: Path, n_texts: int = 20) -> pd.DataFrame:
    frames = [load_lexical_for_text(lexical_dir, i) for i in range(1, n_texts + 1)]
    return pd.concat(frames, ignore_index=True)
