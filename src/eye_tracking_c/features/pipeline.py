"""Feature pipeline for System 1.

The paper adds features incrementally (Table 2). We group them so the ablation
can be reproduced and so optional parser features slot in later:

- LOW_LEVEL_FEATURES : word length, sentence length, Wikipedia log-frequency,
                       AoA mean, AoA SD  (paper Table 2 rows 1-5 -> R2 ~ 0.533)
- PARSER_FEATURES    : 15 incremental-PCFG-parser features (surprisals, entropy
                       reduction, embedding depth/difference, memory operations).

Word-keyed resource features (Wikipedia freq, AoA) are computed from the ``word``
column so the exact same code path serves training and System-2 inference.
Missing values are mean-imputed (fit on train only) then standardized, so an
out-of-vocabulary word contributes a neutral 0 after scaling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eye_tracking_c.features.resources import load_aoa, load_wiki_log_freq
from eye_tracking_c.parser.run import PARSER_COLUMNS

# Low-level predictors (Demberg & Keller-style) — obtainable without a parser.
LOW_LEVEL_FEATURES = (
    "word_len",
    "sentence_len",
    "wiki_log_freq",
    "aoa_mean",
    "aoa_sd",
)

# 15 parser features from the incremental left-corner PCFG parser
# (eye_tracking_c.parser), in the paper's Table 2 order.
PARSER_FEATURES = PARSER_COLUMNS

# All 20 features in the order the paper adds them (Table 2).
ALL_FEATURES = LOW_LEVEL_FEATURES + PARSER_FEATURES


def add_lexical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add resource-derived columns (wiki_log_freq, aoa_mean, aoa_sd) from ``word``.

    Raw values with NaN for out-of-vocabulary words; imputation happens in the
    pipeline. Returns a copy; existing columns are overwritten.
    """
    out = df.copy()
    words = out["word"].astype(str).str.lower()

    wiki = load_wiki_log_freq()
    aoa = load_aoa()

    out["wiki_log_freq"] = words.map(lambda w: wiki.get(w, np.nan))
    out["aoa_mean"] = words.map(lambda w: aoa.get(w, (np.nan, np.nan))[0])
    out["aoa_sd"] = words.map(lambda w: aoa.get(w, (np.nan, np.nan))[1])
    return out


class FeaturePipeline:
    """Impute (train-mean) + standardize a chosen feature column set."""

    def __init__(self, feature_columns: tuple[str, ...] = LOW_LEVEL_FEATURES) -> None:
        self.feature_columns = tuple(feature_columns)
        self._pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="mean")),
                ("scale", StandardScaler()),
            ]
        )
        self._fitted = False

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing and "word" in df.columns:
            df = add_lexical_features(df)
        # Any still-missing feature (e.g. surprisal at System-2 predict time on
        # novel text) becomes NaN and is mean-imputed -> neutral after scaling.
        still_missing = [c for c in self.feature_columns if c not in df.columns]
        if still_missing:
            df = df.copy()
            for c in still_missing:
                df[c] = np.nan
        return df.loc[:, list(self.feature_columns)].astype(np.float64).values

    def fit(self, df: pd.DataFrame) -> FeaturePipeline:
        self._pipe.fit(self._matrix(df))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        return self._pipe.transform(self._matrix(df))

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)
