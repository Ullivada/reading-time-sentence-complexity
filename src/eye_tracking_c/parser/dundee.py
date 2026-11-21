"""Parser features for the Dundee words (System 1 training data).

Dundee sentences are parsed on the PTB-style tokens of the lexical annotation.
Several tokens can share one Dundee word (WNUM), e.g. "existence" + "?". They are
merged per word: surprisals, entropy reduction and embedding difference are summed
(chain rule / net change); depth and memory-operation features are taken at the
word's last token.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from eye_tracking_c.dundee.io_lexical import load_all_lexical
from eye_tracking_c.parser.run import PARSER_COLUMNS, parse_sentences

SUMMED = {"total_surprisal", "lexical_surprisal", "syntactic_surprisal",
          "entropy_reduction", "embedding_difference"}


def build_dundee_parser_features(corpus_root: Path, out_path: Path, beam=100) -> pd.DataFrame:
    lex = load_all_lexical(corpus_root / "per-word-lexical-annotation")
    groups = list(lex.groupby(["itemno", "sentence_id"], sort=False))
    sentences = [g["word"].astype(str).tolist() for _, g in groups]
    print(f"Parsing {len(sentences)} Dundee sentences ({len(lex)} tokens), beam={beam} ...")
    results = parse_sentences(sentences, beam=beam)

    tok = pd.concat(
        [pd.DataFrame(m, columns=PARSER_COLUMNS).assign(itemno=g["itemno"].values, wnum=g["wnum"].values)
         for (_, g), (m, _logp) in zip(groups, results)],
        ignore_index=True,
    )
    agg = {c: ("sum" if c in SUMMED else "last") for c in PARSER_COLUMNS}
    words = tok.groupby(["itemno", "wnum"], as_index=False, sort=False).agg(agg)
    # a word whose tokens all failed to parse stays NaN (sum() would give 0)
    failed = tok.groupby(["itemno", "wnum"], sort=False)["total_surprisal"].apply(lambda s: s.isna().all()).values
    words.loc[failed, list(SUMMED)] = np.nan
    words.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(words)} words, {int(failed.sum())} unparsed)")
    return words
