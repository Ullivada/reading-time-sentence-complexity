"""Sentence features for System 2 (paper Sec 4.2).

base      'word{k}: predicted FFD' for every word position k           (Base RT)
sentence  sentence length, length-normalized sum of predicted RT, sums of total /
          lexical / syntactic surprisal, log parse probability     (sentence-level)
extended  base + sentence                                            (Extended RT)
"""

from __future__ import annotations

import math

import numpy as np

from eye_tracking_c.parser.run import PARSER_COLUMNS

_TOT, _LEX, _SYN = (PARSER_COLUMNS.index(c) for c in
                    ("total_surprisal", "lexical_surprisal", "syntactic_surprisal"))


def sentence_features(rts: list[float], parse, kind: str) -> dict[str, float]:
    feats: dict[str, float] = {}
    if kind in ("base", "extended"):
        for k, rt in enumerate(rts, start=1):
            feats[f"word{k}"] = rt
    if kind in ("sentence", "extended"):
        matrix, logp = parse
        n = len(rts)
        feats["sentence_length"] = float(n)
        feats["normalized_rt_sum"] = float(sum(rts)) / n if n else 0.0
        feats["sum_total_surprisal"] = float(np.nansum(matrix[:, _TOT]))
        feats["sum_lexical_surprisal"] = float(np.nansum(matrix[:, _LEX]))
        feats["sum_syntactic_surprisal"] = float(np.nansum(matrix[:, _SYN]))
        # if no complete parse survived the beam, fall back to the prefix log-prob
        feats["log_parse_prob"] = logp if not math.isnan(logp) else -feats["sum_total_surprisal"]
    return feats
