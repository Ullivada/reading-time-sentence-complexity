"""Eye-movement measures from a time-ordered (WNUM, FDUR) fixation stream (paper Sec 3.1).

FFD  duration of the first fixation on the word
FPD  sum of fixations from first entering the word until it is left
RPD  sum of fixations from first entering the word until a word to its right is fixated
TD   sum of all fixations on the word
A word that was never fixated gets 0 for every measure.
"""

from __future__ import annotations


def all_word_measures(events: list[tuple[int, int]]) -> dict[int, dict[str, float]]:
    """Return {wnum: {ffd, fpd, rpd, td}} for every fixated word, in one pass."""
    first: dict[int, int] = {}
    td: dict[int, float] = {}
    for i, (w, d) in enumerate(events):
        first.setdefault(w, i)
        td[w] = td.get(w, 0.0) + d

    n = len(events)
    out = {}
    for w, i in first.items():
        fpd, k = 0.0, i
        while k < n and events[k][0] == w:
            fpd += events[k][1]
            k += 1
        rpd, k = 0.0, i
        while k < n and events[k][0] <= w:
            rpd += events[k][1]
            k += 1
        out[w] = {"ffd": float(events[i][1]), "fpd": fpd, "rpd": rpd, "td": td[w]}
    return out


ZERO = {"ffd": 0.0, "fpd": 0.0, "rpd": 0.0, "td": 0.0}
