"""Parse many sentences in parallel with the incremental PCFG parser."""

from __future__ import annotations

from contextlib import contextmanager
from multiprocessing import Pool

import numpy as np

from eye_tracking_c.parser.pcfg import load_grammar
from eye_tracking_c.paths import repo_root

# The paper's 15 parser features (Sec 3.2), in Table 2 order.
PARSER_COLUMNS = (
    "total_surprisal",
    "lexical_surprisal",
    "syntactic_surprisal",
    "entropy_reduction",
    "embedding_depth",
    "weighted_embedding_depth",
    "embedding_difference",
    "p_F+L-", "p_F-L+", "p_F+L+", "p_F-L-",
    "best_F+L-", "best_F-L+", "best_F+L+", "best_F-L-",
)

_GRAMMAR = None
_BEAM = 100


def _init(beam):
    global _GRAMMAR, _BEAM
    _GRAMMAR = load_grammar(repo_root() / "data")
    _BEAM = beam


def _parse_one(words):
    """Return (n_tokens x 15 float32 matrix, log2 parse probability)."""
    try:
        feats, logp = _GRAMMAR.parse(words, beam=_BEAM)
    except Exception:  # noqa: BLE001 - one bad sentence must not stop a long run
        feats, logp = [None] * len(words), float("nan")
    m = np.full((len(words), len(PARSER_COLUMNS)), np.nan, dtype=np.float32)
    for i, f in enumerate(feats):
        if f is not None:
            m[i] = [f[c] for c in PARSER_COLUMNS]
    return m, logp


def parse_sentences(sentences, beam=100, n_jobs=8, chunksize=50):
    """sentences: list of token lists. Returns list of (matrix, logp) in order."""
    with parser_pool(beam=beam, n_jobs=n_jobs) as pool:
        return parse_with_pool(pool, sentences, chunksize=chunksize)


@contextmanager
def parser_pool(beam=100, n_jobs=8):
    """A worker pool whose processes each build the grammar once.

    Building the PCFG from WSJ 02-21 costs ~20-30 s per process, so a caller that
    parses in several batches should hold one pool open across all of them rather
    than call `parse_sentences` per batch.
    """
    with Pool(n_jobs, initializer=_init, initargs=(beam,)) as pool:
        yield pool


def parse_with_pool(pool, sentences, chunksize=50):
    """Parse one batch on an already-initialized pool from `parser_pool`."""
    return pool.map(_parse_one, sentences, chunksize=chunksize)
