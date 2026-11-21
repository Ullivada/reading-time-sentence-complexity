"""Incremental probabilistic left-corner PCFG parser (free stand-in for ModelBlocks).

The paper (Sec 3.2) takes 15 features from van Schijndel et al.'s (2013) incremental
left-corner PCFG parser trained on WSJ 2-21. The same model class is trained here on
those same sections of the Penn Treebank 3 WSJ corpus (data/penn_treebank_3, 39,832
gold trees); when that corpus is absent it falls back to the free NLTK WSJ sample
(3,914 trees) with word emissions from CoNLL-2000. Per token it emits:

  total / lexical / syntactic surprisal  (Hale 2001; Roark et al. 2009 split)
  entropy reduction                      (Hale 2006; entropy over the beam)
  embedding depth (best parse), weighted embedding depth (probability-weighted),
  embedding difference                   (Wu et al. 2010)
  4 memory operations F+L-, F-L+, F+L+, F-L- (van Schijndel & Schuler 2013),
  each as beam probability and as a 0/1 flag on the best parse -> 8 features
and, per sentence, the log parse probability.

Parser state = stack of levels (partial category, awaited category). A word first
picks a POS x under the top awaited goal g with P = R*[g,x] * P(w|x), where R* is
the left-corner closure of the grammar. It then climbs: it either completes the goal
(pop), attaches as left child of the goal's rule (join) or of a new constituent (push).
Every step divides by R*, so the product of steps equals the PCFG tree probability.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

UNK_MIN_COUNT = 2  # words seen fewer times also use the unknown-word signature

# Climbing-operation kinds. Small ints, not strings: the parse loop compares these
# millions of times per sentence and `is`/`==` on ints is markedly cheaper.
OP_DONE, OP_UP, OP_JOIN, OP_PUSH = 0, 1, 2, 3

# _FLAG[fork][joined] -> the memory-operation name behind the 8 hierarchical
# features. Indexed by bools so the hot loop allocates no tuple key.
_FLAG = (("F-L-", "F-L+"), ("F+L-", "F+L+"))


def _mass(kv):
    """Beam sort key: a state's total mass (kv = (state, [mass, best, flag])).

    Deliberately not itemgetter(1): comparing the value lists would break ties on
    the following elements and select a different beam, and it measured slower.
    """
    return kv[1][0]


def _clean(tree):
    """Drop traces, strip function tags, collapse phrasal unary chains, right-binarize."""
    from nltk import Tree

    def strip(label):
        if label in ("-LRB-", "-RRB-", "-NONE-"):
            return label
        return re.split(r"[-=]", label)[0] or label

    def rec(t):
        if isinstance(t[0], str):  # preterminal
            return None if t.label() == "-NONE-" else Tree(t.label(), [t[0]])
        kids = [k for k in (rec(c) for c in t) if k is not None]
        if not kids:
            return None
        label = strip(t.label())
        # collapse X -> Y when Y is phrasal (keeps grammar free of unary cycles)
        while len(kids) == 1 and not isinstance(kids[0][0], str):
            kids = list(kids[0])
        # right binarization with 1st-order horizontal markovization
        while len(kids) > 2:
            tail = Tree(f"@{label}|{kids[-2].label()}", kids[-2:])
            kids = kids[:-2] + [tail]
        return Tree(label, kids)

    body = rec(tree)
    return None if body is None else Tree("ROOT", [body])


def signature(word: str) -> str:
    """Unknown-word class (Berkeley-parser style, simplified)."""
    sig = "UNK"
    if word[0].isupper():
        sig += "-CAP"
    if any(c.isdigit() for c in word):
        sig += "-NUM"
    if "-" in word:
        sig += "-DASH"
    low = word.lower()
    for suf in ("ing", "ed", "ly", "s", "ion", "er", "est", "al", "ity", "y", "ic", "ous", "ble"):
        if low.endswith(suf) and len(low) > len(suf) + 2:
            return sig + "-" + suf
    return sig


class PCFG:
    def __init__(self, trees, tagged_words):
        binary, unary, lhs_count = Counter(), Counter(), Counter()
        tag_word = Counter()
        for t in trees:
            t = _clean(t)
            if t is None:
                continue
            for sub in t.subtrees():
                kids = [k for k in sub]
                if isinstance(kids[0], str):
                    tag_word[(sub.label(), kids[0])] += 1
                    continue
                lhs_count[sub.label()] += 1
                if len(kids) == 1:
                    unary[(sub.label(), kids[0].label())] += 1
                else:
                    binary[(sub.label(), kids[0].label(), kids[1].label())] += 1
        for w, tag in tagged_words:
            if tag != "-NONE-":
                tag_word[(tag, w)] += 1

        # rule probabilities
        self.binary = {r: c / lhs_count[r[0]] for r, c in binary.items()}
        self.unary = {r: c / lhs_count[r[0]] for r, c in unary.items()}
        cats = sorted({x for r in binary for x in r} | {x for r in unary for x in r})
        self.cats = cats
        self.idx = {c: i for i, c in enumerate(cats)}

        # left-corner closure R* = (I - P_L)^-1
        n = len(cats)
        pl = np.zeros((n, n))
        for (a, b, _c), p in self.binary.items():
            pl[self.idx[a], self.idx[b]] += p
        for (a, b), p in self.unary.items():
            pl[self.idx[a], self.idx[b]] += p
        self.R = np.linalg.inv(np.eye(n) - pl)
        # nested lists: R[i][j] from Python is far cheaper than numpy scalar indexing
        self.Rl = self.R.tolist()

        # rules indexed by left child
        self.by_left = defaultdict(list)  # c -> [(e, d_or_None, p)]
        for (a, b, c), p in self.binary.items():
            self.by_left[b].append((a, c, p))
        for (a, b), p in self.unary.items():
            self.by_left[b].append((a, None, p))

        # lexicon: P(tag | word) with signature smoothing, P(w|tag) via Bayes
        word_count, tag_count = Counter(), Counter()
        for (tag, w), c in tag_word.items():
            word_count[w] += c
            tag_count[tag] += c
        self.total = sum(word_count.values())
        sig_tag = Counter()
        for (tag, w), c in tag_word.items():
            if word_count[w] < UNK_MIN_COUNT:
                sig_tag[(signature(w), tag)] += c
        sig_total = Counter()
        for (s, _t), c in sig_tag.items():
            sig_total[s] += c
        self.tag_word, self.word_count, self.tag_count = tag_word, word_count, tag_count
        self.sig_tag, self.sig_total = sig_tag, sig_total
        self.tags_of = defaultdict(list)
        for (tag, w) in tag_word:
            self.tags_of[w].append(tag)
        self.open_tags = [t for t in ("NN", "NNS", "NNP", "NNPS", "JJ", "VB", "VBD", "VBG",
                                      "VBN", "VBP", "VBZ", "RB", "CD") if t in self.idx]
        self._ops = {}
        self._lex = {}

    # ---- lexical model -------------------------------------------------------
    def _p_tag_given_sig(self, sig, tag):
        s = sig if self.sig_total[sig] else "UNK"
        return (self.sig_tag[(s, tag)] + 0.1) / (self.sig_total[s] + 0.1 * len(self.open_tags))

    def emissions(self, word):
        """[(tag, P(word|tag))] for tags the word can take."""
        if word in self._lex:
            return self._lex[word]
        w = word if word in self.word_count or word.lower() not in self.word_count else word.lower()
        n = self.word_count[w]
        sig = signature(word)
        out = []
        if n >= UNK_MIN_COUNT:
            for tag in self.tags_of[w]:
                p_tag_w = self.tag_word[(tag, w)] / n
                out.append((tag, p_tag_w * (n / self.total) / (self.tag_count[tag] / self.total)))
        else:  # rare / unknown: signature model mixed with any observed counts
            p_w = (n + 0.5) / self.total
            tags = set(self.open_tags) | set(self.tags_of[w])
            for tag in tags:
                p_tag_w = (self.tag_word[(tag, w)] + self._p_tag_given_sig(sig, tag)) / (n + 1)
                out.append((tag, p_tag_w * p_w / (self.tag_count[tag] / self.total)))
        out = [(t, p) for t, p in out if t in self.idx and p > 0]
        self._lex[word] = out
        return out

    # ---- upward operations --------------------------------------------------
    def ops(self, g, c):
        """Normalized climbing options for complete category c under goal g.

        ('done', None, None, p): c is the goal (directly or via unary g -> c) -> pop
        ('up',   e,    None, p): unary e -> c, keep climbing
        ('join', g,    d,    p): rule g -> c d, top level now awaits d
        ('push', e,    d,    p): rule e -> c d, e below g, new level (e, d)
        """
        key = (g, c)
        if key in self._ops:
            return self._ops[key]
        idx = self.idx
        gi, ci = idx[g], idx[c]
        row = self.Rl[gi]
        norm = row[ci]
        out = []
        if norm > 0:
            done = (1.0 if g == c else 0.0) + self.unary.get((g, c), 0.0)
            if done > 0:
                out.append((OP_DONE, None, None, done / norm))
            for e, d, p in self.by_left[c]:
                below = row[idx[e]] - (1.0 if e == g else 0.0)
                if d is None:
                    if below > 1e-12:
                        out.append((OP_UP, e, None, p * below / norm))
                else:
                    if e == g:
                        out.append((OP_JOIN, g, d, p / norm))
                    if below > 1e-12:
                        out.append((OP_PUSH, e, d, p * below / norm))
        out.sort(key=lambda o: -o[3])
        # keep the options covering 99.9% of the mass (cap 40)
        kept, mass = [], 0.0
        for o in out:
            kept.append(o)
            mass += o[3]
            if mass > 0.999 or len(kept) >= 40:
                break
        self._ops[key] = kept
        return kept

    # ---- parsing --------------------------------------------------------------
    def parse(self, words, beam=100):
        """Return (per-token feature dicts, log parse probability in bits)."""
        beam_states = {(("", "ROOT"),): 1.0}  # stack of (partial, awaited)
        prefix = 1.0
        prev_h = 0.0
        prev_depth = 1.0
        feats = []
        final_log = float("nan")
        log_norm = 0.0  # sum of log2 normalizers of previous words
        # hoist every attribute lookup out of the hot loop
        idx = self.idx
        Rl = self.Rl
        ops_cache = self._ops
        ops_fn = self.ops
        emissions = self.emissions
        for k, word in enumerate(words):
            # state -> [summed mass, best single derivation, its memory-op name].
            # Merging what used to be two dicts halves the nested-tuple hashing,
            # which the profile showed to be the single biggest cost.
            states = {}
            syn_mass = 0.0
            done_mass = 0.0
            push_mass = 0.0
            word_emissions = emissions(word)  # same for every beam state
            for stack, p in beam_states.items():
                row = Rl[idx[stack[-1][1]]]
                for tag, p_emit in word_emissions:
                    p_lc = row[idx[tag]]
                    if p_lc <= 0:
                        continue
                    inv_emit = 1.0 / p_emit
                    # climb: (prob, stack, complete cat, forked?)
                    agenda = [(p * p_lc * p_emit, stack, tag, True)]
                    while agenda:
                        q, st, c, fork = agenda.pop()
                        top = st[-1]
                        key = (top[1], c)
                        opts = ops_cache.get(key)
                        if opts is None:
                            opts = ops_fn(top[1], c)
                        for kind, e, d, w in opts:
                            q2 = q * w
                            if q2 < 1e-300:
                                continue
                            if kind == OP_UP:
                                agenda.append((q2, st, e, fork))
                            elif kind == OP_DONE:
                                if len(st) == 1:  # ROOT complete: sentence could end here
                                    done_mass += q2
                                    syn_mass += q2 * inv_emit
                                    continue
                                agenda.append((q2, st[:-1], top[0], False))
                            else:
                                if kind == OP_JOIN:
                                    st2 = st[:-1] + ((top[0], d),)
                                else:
                                    st2 = st + ((e, d),)
                                syn_mass += q2 * inv_emit
                                push_mass += q2
                                entry = states.get(st2)
                                if entry is None:
                                    states[st2] = [q2, q2, _FLAG[fork][kind == OP_JOIN]]
                                elif q2 > entry[1]:
                                    entry[0] += q2
                                    entry[1] = q2
                                    entry[2] = _FLAG[fork][kind == OP_JOIN]
                                else:
                                    entry[0] += q2
            total_mass = push_mass + done_mass
            final_log = log_norm + math.log2(done_mass) if done_mass > 0 else float("nan")
            if total_mass <= 0:  # parse failure: restart from a fresh stack
                feats.append(None)
                beam_states = {(("", "ROOT"),): 1.0}
                prefix, prev_h, log_norm = 1.0, 0.0, 0.0
                continue
            best = heapq.nlargest(beam, states.items(), key=_mass)
            kept = sum(v[0] for _, v in best) or 1.0
            inv_kept = 1.0 / kept
            # one pass for entropy, weighted depth and the memory-op distribution
            h = 0.0
            wdepth = 0.0
            ops_p = defaultdict(float)
            log2 = math.log2
            for st2, v in best:
                pr = v[0] * inv_kept
                h -= pr * log2(pr + 1e-300)
                wdepth += pr * len(st2)
                ops_p[v[2]] += pr
            best_flag = best[0][1][2] if best else ""
            f = {
                "total_surprisal": -math.log2(total_mass / prefix),
                "syntactic_surprisal": -math.log2(syn_mass / prefix) if syn_mass > 0 else 0.0,
                "entropy_reduction": max(0.0, prev_h - h),
                "embedding_depth": float(len(best[0][0])) if best else 0.0,
                "weighted_embedding_depth": wdepth,
                "embedding_difference": wdepth - prev_depth,
            }
            f["lexical_surprisal"] = f["total_surprisal"] - f["syntactic_surprisal"]
            for op in ("F+L-", "F-L+", "F+L+", "F-L-"):
                f[f"p_{op}"] = ops_p[op]
                f[f"best_{op}"] = 1.0 if best_flag == op else 0.0
            feats.append(f)
            # renormalize the beam to keep numbers in range; track prefix in log space
            beam_states = {s: v[0] / total_mass for s, v in best}
            prefix = kept / total_mass
            log_norm += math.log2(total_mass)
            prev_h, prev_depth = h, wdepth
        return feats, final_log


# Sections the paper trains on (Sec 3.2, after van Schijndel et al. 2013).
WSJ_TRAIN_SECTIONS = tuple(f"{i:02d}" for i in range(2, 22))


def _wsj_parsed_sents(wsj_dir: Path, sections=WSJ_TRAIN_SECTIONS):
    """Lazily read the gold trees of the given WSJ sections from .mrg files."""
    from nltk.corpus.reader import BracketParseCorpusReader

    reader = BracketParseCorpusReader(str(wsj_dir), r".*/wsj_.*\.mrg")
    keep = set(sections)
    fileids = [f for f in reader.fileids() if f.split("/")[0] in keep]
    if not fileids:
        raise FileNotFoundError(f"no .mrg files for sections {sorted(keep)} under {wsj_dir}")
    return reader.parsed_sents(fileids)


def load_grammar(data_dir):
    """Train the PCFG on WSJ 2-21 from data/penn_treebank_3, else the NLTK sample.

    The full treebank already supplies ~950k tagged tokens, so no extra tagged-word
    corpus is mixed in; counting CoNLL-2000 on top would double-count sections 15-18
    and 20, which are part of the training range.
    """
    import nltk

    data_dir = Path(data_dir)
    # authorize both roots for NLTK's corpus-path sandbox
    for root in (data_dir, data_dir / "nltk_data"):
        if str(root) not in nltk.data.path:
            nltk.data.path.insert(0, str(root))

    wsj = data_dir / "penn_treebank_3" / "parsed" / "mrg" / "wsj"
    if wsj.is_dir():
        return PCFG(_wsj_parsed_sents(wsj), [])

    from nltk.corpus import conll2000, treebank

    return PCFG(treebank.parsed_sents(), conll2000.tagged_words())
