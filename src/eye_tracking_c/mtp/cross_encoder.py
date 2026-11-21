"""Cross-encoder for System 2: a fine-tuned DistilRoBERTa reads both sentences at once.

Input `<s> sentence1 </s></s> sentence2 </s>`, two labels (1 = sentence2 is the simple one),
plain PyTorch loop. The returned logit is log-odds(sentence2 simple), averaged over both
sentence orders. Dev and test logits
are cached in one .npz, so a rerun skips training unless retrain=True.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from eye_tracking_c.mtp.lm import detok, device_name, free_gpu_cache


def _text(s: str) -> str:
    return detok(s.split())[0]


def _batches(tok, pairs, labels, order, batch_size, max_len, device):
    for i in range(0, len(order), batch_size):
        b = order[i : i + batch_size]
        enc = tok([_text(pairs[j][0]) for j in b], [_text(pairs[j][1]) for j in b],
                  truncation=True, max_length=max_len, padding=True, pad_to_multiple_of=16,
                  return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        if labels is not None:
            enc["labels"] = torch.as_tensor(labels[b], device=device)
        yield b, enc


@torch.no_grad()
def predict_logits(model, tok, pairs, batch_size=64, max_len=128, device=None) -> np.ndarray:
    """log-odds that the second sentence is the simple one, for every pair."""
    device = device or next(model.parameters()).device
    model.eval()
    order = np.argsort([len(a) + len(b) for a, b in pairs])  # similar lengths -> less padding
    out = np.empty(len(pairs), dtype=np.float32)
    for b, enc in _batches(tok, pairs, None, order, batch_size, max_len, device):
        logits = model(**enc).logits.float()
        out[b] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    free_gpu_cache()
    return out


def predict_both_orders(model, tok, pairs, max_len=128) -> np.ndarray:
    """Score (a, b) and (b, a) and average. Swapping the sentences must flip the answer,
    so log-odds(b simple) = [f(a, b) - f(b, a)] / 2 removes any position bias."""
    swapped = [(b, a) for a, b in pairs]
    return (predict_logits(model, tok, pairs, max_len=max_len) - predict_logits(model, tok, swapped, max_len=max_len)) / 2


def train_or_load(pairs, y, idx, cfg, out_dir: Path, cache: Path, retrain=False, seed=42) -> dict:
    """Return {"dev": logits, "test": logits, "dev_acc": [...per epoch]}."""
    if cache.is_file() and not retrain:
        print("  cached logits", flush=True)
        return {k: v for k, v in np.load(cache).items()}

    torch.manual_seed(seed)
    device = device_name()
    tok = AutoTokenizer.from_pretrained(cfg["ce_model"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["ce_model"], num_labels=2).to(device)

    train = idx["train"]
    cap = cfg.get("ce_max_train_pairs")
    rng = np.random.default_rng(seed)
    if cap and len(train) > cap:
        train = rng.choice(train, cap, replace=False)
    dev_pairs = [pairs[i] for i in idx["dev"]]
    bs, max_len, epochs = cfg["ce_batch_size"], cfg["ce_max_len"], cfg["ce_epochs"]
    n_batches = (len(train) + bs - 1) // bs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["ce_lr"])
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * n_batches * epochs), n_batches * epochs)

    best_acc, best_state, dev_accs = -1.0, None, []
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        for i, (_, enc) in enumerate(_batches(tok, pairs, y, rng.permutation(train), bs, max_len, device)):
            loss = model(**enc).loss
            loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()
            if i % 50 == 0:
                free_gpu_cache()
                print(f"  progress {i}/{n_batches} epoch {epoch + 1}/{epochs} loss {loss.item():.3f} "
                      f"({(time.time() - t0) / (i + 1):.2f} s/batch)", flush=True)
        dev_logits = predict_both_orders(model, tok, dev_pairs, max_len)
        acc = float(((dev_logits > 0) == y[idx["dev"]]).mean())
        dev_accs.append(acc)
        print(f"  epoch {epoch + 1}: dev accuracy {100 * acc:.2f}%", flush=True)
        if acc > best_acc:
            best_acc, best_dev = acc, dev_logits
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print("  scoring test split", flush=True)
    test_logits = predict_both_orders(model, tok, [pairs[i] for i in idx["test"]], max_len)
    out = {"dev": best_dev, "test": test_logits, "dev_acc": np.asarray(dev_accs)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **out)
    return out
