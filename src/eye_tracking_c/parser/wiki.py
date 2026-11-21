"""Parser features for the Wikipedia sentences (needed by System 2).

The paper parsed all Wikipedia sentences with the incremental parser (footnote 4:
100 parallel sections, ~10 h each). Here each unique sentence is parsed once.
Results are saved in chunk files under build/wiki_parse/; a rerun loads every
saved chunk and parses only the sentences still missing, so a long run can be
stopped and resumed.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from eye_tracking_c.parser.run import parse_with_pool, parser_pool

CHUNK = 5_000


def parse_wiki_sentences(sentences: list[str], out_dir: Path, beam=20, n_jobs=8) -> dict:
    """Return {sentence: (n_tokens x 15 float32 matrix, log2 parse prob)}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for path in sorted(out_dir.glob("chunk_*.pkl")):
        with path.open("rb") as f:
            done.update(pickle.load(f))

    todo = sorted(set(sentences) - set(done))
    print(f"  {len(set(sentences)):,} unique sentences, {len(todo):,} still to parse", flush=True)
    next_id = len(list(out_dir.glob("chunk_*.pkl")))
    if todo:
        # One pool for every chunk: the grammar is built once per worker instead of
        # once per 5,000 sentences, which saved ~25 min on the full corpus.
        with parser_pool(beam=beam, n_jobs=n_jobs) as pool:
            for start in range(0, len(todo), CHUNK):
                part = todo[start : start + CHUNK]
                print(f"  parsing {start:,}-{start + len(part):,} of {len(todo):,}", flush=True)
                chunk = dict(zip(part, parse_with_pool(pool, [s.split() for s in part])))
                with (out_dir / f"chunk_{next_id:04d}.pkl").open("wb") as f:
                    pickle.dump(chunk, f)
                next_id += 1
                done.update(chunk)
    return {s: done[s] for s in set(sentences)}
