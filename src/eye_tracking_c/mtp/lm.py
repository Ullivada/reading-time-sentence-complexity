"""GPT-2 word-level surprisal and entropy (replaces the PCFG parser features).

Per word:
  gpt2_surprisal       -log2 P(word | preceding text), summed over the word's BPE tokens
  gpt2_entropy         entropy (bits) of the next-token distribution just before the word
  gpt2_prev_surprisal  previous word's surprisal in the same sentence (spillover)

Dundee is running text, so each sentence is scored with the preceding sentences of the
same text as context (up to GPT-2's 1,024 tokens). The Wikipedia pairs are isolated
sentences and are scored alone. Tokens are mapped to words by character offsets.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

from eye_tracking_c.dundee.io_lexical import load_all_lexical

hf_logging.set_verbosity_error()  # no weight-loading reports or progress bars in the run log
hf_logging.disable_progress_bar()

LN2 = math.log(2)
CHUNK = 5_000
MAX_TOKENS = 1024

NO_SPACE_BEFORE = {",", ".", ";", ":", "?", "!", ")", "]", "}", "%", "'s", "'S", "n't", "'re",
                   "'ve", "'ll", "'d", "'m", "''", "..."}
NO_SPACE_AFTER = {"(", "[", "{", "$", "``", "#"}
SURFACE = {"``": '"', "''": '"', "`": "'", "-LRB-": "(", "-RRB-": ")"}


def free_gpu_cache() -> None:
    """The MPS allocator keeps buffers of every batch shape it has seen; on a 16 GB Mac
    that grows into swap within a few thousand batches unless released."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def device_name() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_lm(name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    device = device_name()
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    return tok, model, device


def detok(words: list[str], glue: list[bool] | None = None):
    """Join tokens into text. Returns (text, starts, ends) with one char span per word.

    glue[i] True means no space before word i; default is the PTB spacing rules.
    """
    parts, starts, ends, pos = [], [], [], 0
    for i, w in enumerate(words):
        surface = SURFACE.get(w, w.replace("\\/", "/"))
        if glue is not None:
            space = i > 0 and not glue[i]
        else:
            space = i > 0 and w not in NO_SPACE_BEFORE and words[i - 1] not in NO_SPACE_AFTER
        if space:
            parts.append(" ")
            pos += 1
        starts.append(pos)
        parts.append(surface)
        pos += len(surface)
        ends.append(pos)
    return "".join(parts), np.asarray(starts), np.asarray(ends)


def _word_values(tok_ends, surp, ent, starts):
    """Sum token surprisal per word (token goes to the word holding its last char);
    entropy is taken at the first token reaching into the word."""
    word_of_tok = np.searchsorted(starts, tok_ends - 1, side="right") - 1
    s = np.bincount(word_of_tok, weights=surp, minlength=len(starts))
    first = np.minimum(np.searchsorted(tok_ends, starts, side="right"), len(tok_ends) - 1)
    return s, ent[first]


def _stats(logits: torch.Tensor, targets: torch.Tensor):
    """Per position: surprisal of the target and entropy of the distribution (bits)."""
    logp = torch.log_softmax(logits.float(), dim=-1)
    surp = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1) / LN2
    ent = -(logp.exp() * logp).sum(-1) / LN2
    return surp, ent


@torch.no_grad()
def dundee_lm_features(corpus_root: Path, out_csv: Path, lm_name: str = "gpt2") -> pd.DataFrame:
    """Score every Dundee sentence with its preceding text as context; one row per word."""
    if out_csv.is_file():
        print("  cached", flush=True)
        return pd.read_csv(out_csv)
    tok, model, device = load_lm(lm_name)
    tok.model_max_length = 10**9  # whole texts are tokenized once, then windowed to 1,024
    lex = load_all_lexical(corpus_root / "per-word-lexical-annotation")
    n_sent = lex[["itemno", "sentence_id"]].drop_duplicates().shape[0]
    rows, done = [], 0
    for itemno, g in lex.groupby("itemno", sort=True):
        wnum = g["wnum"].to_numpy()
        glue = np.r_[False, wnum[1:] == wnum[:-1]]
        text, starts, ends = detok(g["word"].tolist(), glue.tolist())
        enc = tok(text, return_offsets_mapping=True)
        ids = np.asarray(enc["input_ids"])
        tok_ends = np.asarray([e for _, e in enc["offset_mapping"]])
        sent = g["sentence_id"].to_numpy()
        for sid in pd.unique(sent):
            m = np.flatnonzero(sent == sid)
            c0, c1 = starts[m[0]], ends[m[-1]]
            t = np.flatnonzero((tok_ends > c0) & (tok_ends <= c1))  # tokens of this sentence
            lo = max(0, t[-1] + 1 - (MAX_TOKENS - 1))
            window = [tok.eos_token_id] + ids[lo : t[-1] + 1].tolist()
            x = torch.tensor([window], device=device)
            logits = model(x).logits[0]
            q = t - lo + 1  # positions of the sentence tokens in the window
            surp, ent = _stats(logits[q - 1], x[0, q])
            s, e = _word_values(tok_ends[t], surp.cpu().numpy(), ent.cpu().numpy(), starts[m])
            del logits
            free_gpu_cache()
            rows.append(pd.DataFrame({"itemno": itemno, "wnum": wnum[m], "sentence_id": sid,
                                      "s": s, "e": e}))
            done += 1
            if done % 200 == 0:
                print(f"  progress {done}/{n_sent}", flush=True)
    df = pd.concat(rows, ignore_index=True)
    # several PTB tokens can share one Dundee word: sum surprisal, keep the first entropy
    df = df.groupby(["itemno", "wnum"], as_index=False, sort=False).agg(
        sentence_id=("sentence_id", "first"), gpt2_surprisal=("s", "sum"), gpt2_entropy=("e", "first"))
    df["gpt2_prev_surprisal"] = df.groupby(["itemno", "sentence_id"])["gpt2_surprisal"].shift(1)
    df = df.drop(columns="sentence_id")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


@torch.no_grad()
def score_sentences(sentences, tok, model, device):
    """Score isolated sentences in one padded batch -> list of (n_words, 3) float32 arrays."""
    encs, spans = [], []
    for s in sentences:
        text, starts, _ = detok(s.split())
        enc = tok(text, return_offsets_mapping=True)
        # GPT-2 has 1,024 positions; the rare longer sentence is cut (later words get 0)
        enc = {k: enc[k][: MAX_TOKENS - 1] for k in ("input_ids", "offset_mapping")}
        encs.append(enc)
        spans.append(starts)
    width = min(MAX_TOKENS, -(-(1 + max(len(e["input_ids"]) for e in encs)) // 16) * 16)  # few distinct shapes
    x = torch.full((len(encs), width), tok.eos_token_id, dtype=torch.long)
    mask = torch.zeros_like(x)
    for i, e in enumerate(encs):
        n = len(e["input_ids"])
        x[i, 1 : n + 1] = torch.tensor(e["input_ids"])
        mask[i, : n + 1] = 1
    x, mask = x.to(device), mask.to(device)
    logits = model(input_ids=x, attention_mask=mask).logits[:, :-1]
    real = mask[:, 1:].bool()  # vocab-wide softmax only on real tokens, not padding
    surp, ent = _stats(logits[real], x[:, 1:][real])
    lens = real.sum(1).tolist()
    surp, ent = np.split(surp.cpu().numpy(), np.cumsum(lens)[:-1]), np.split(ent.cpu().numpy(), np.cumsum(lens)[:-1])
    out = []
    for i, (e, starts) in enumerate(zip(encs, spans)):
        tok_ends = np.asarray([b for _, b in e["offset_mapping"]])
        s, h = _word_values(tok_ends, surp[i], ent[i], starts)
        prev = np.r_[np.nan, s[:-1]]
        out.append(np.stack([s, h, prev], axis=1).astype(np.float32))
    del logits, surp, ent
    free_gpu_cache()
    return out


def wiki_lm_features(sentences: list[str], out_dir: Path, lm_name: str = "gpt2",
                     batch_tokens: int = 1024) -> dict:
    """{sentence: (n_words, 3) array of surprisal, entropy, previous-word surprisal}.

    Chunked cache in out_dir like parser/wiki.py: a rerun scores only missing sentences.
    Sentences are sorted by length and batched by a token budget to keep padding small.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for path in sorted(out_dir.glob("chunk_*.pkl")):
        with path.open("rb") as f:
            done.update(pickle.load(f))
    wanted = set(sentences)
    todo = sorted(wanted - set(done), key=lambda s: len(s.split()))
    print(f"  {len(wanted):,} unique sentences, {len(todo):,} still to score", flush=True)
    if todo:
        tok, model, device = load_lm(lm_name)
        next_id = len(list(out_dir.glob("chunk_*.pkl")))
        for c0 in range(0, len(todo), CHUNK):
            part, chunk, i = todo[c0 : c0 + CHUNK], {}, 0
            while i < len(part):
                # sorted by length, so part[j] is the longest; ~1.6 GPT-2 tokens per word
                j = i + 1
                while j < len(part) and (j + 1 - i) * (1.6 * len(part[j].split()) + 2) <= batch_tokens:
                    j += 1
                batch = part[i:j]
                chunk.update(zip(batch, score_sentences(batch, tok, model, device)))
                i = j
            with (out_dir / f"chunk_{next_id:04d}.pkl").open("wb") as f:
                pickle.dump(chunk, f)
            next_id += 1
            done.update(chunk)
            print(f"  progress {c0 + len(part)}/{len(todo)}", flush=True)
    return {s: done[s] for s in wanted}
