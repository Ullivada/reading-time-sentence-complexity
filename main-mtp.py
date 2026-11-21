"""MTP extension of Singh et al. (2016): transformer features and models - entry point.

    python main-mtp.py                     # full pipeline
    python main-mtp.py --skip-s2           # System 1 only
    python main-mtp.py --max-pairs 5000    # quick System 2 check on a random subset
    python main-mtp.py --skip-ce           # System 2 without the cross-encoder
    python main-mtp.py --retrain-ce        # ignore cached cross-encoder logits

Steps (slow ones are cached in build/mtp/; the current pipeline's files are only read):
  1. Dundee word table (same table and split as main.py, without parser features)
  2. GPT-2 surprisal / entropy / spillover for every Dundee word
  3. System 1: fixed-set ablation A-E (Delta R^2), final model D
  4. GPT-2 features for every Wikipedia sentence (~35 min the first time on an M2)
  5. System 2 classical rows: length control, Base/Extended RT, sentence-only, 4 measures
  6. System 2 cross-encoder (DistilRoBERTa) + hybrid rows, McNemar tests
  7. Report: results_mtp.md and experiments_mtp/<run_id>/metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eye_tracking_c.dundee.build_table import build_pooled_word_table, write_build_artifacts
from eye_tracking_c.features.pipeline import add_lexical_features
from eye_tracking_c.mtp import cross_encoder, system2
from eye_tracking_c.mtp.lm import dundee_lm_features, wiki_lm_features
from eye_tracking_c.mtp.report import current_runs, pick_largest, write_report
from eye_tracking_c.mtp.system1 import train_system1_mtp
from eye_tracking_c.system2.dataset import load_wiki_pairs

STEPS = ("Dundee word table", "GPT-2 features for Dundee", "System 1 (enhanced)",
         "GPT-2 features for Wikipedia", "System 2 classical rows",
         "Cross-encoder + hybrid", "Report")


class Steps:
    """Prints '== k/7 name ==' markers (the Streamlit app reads them) and times each step."""

    def __init__(self):
        self.timings, self._name, self._t = {}, None, 0.0

    def start(self, k: int, skipped: bool = False):
        self.finish()
        print(f"== {k}/{len(STEPS)} {STEPS[k - 1]} ==", flush=True)
        if skipped:
            print("skipped", flush=True)
        else:
            self._name, self._t = STEPS[k - 1], time.time()

    def finish(self):
        if self._name:
            self.timings[self._name] = round(time.time() - self._t, 1)
            self._name = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/mtp.yaml")
    ap.add_argument("--skip-s2", action="store_true", help="stop after System 1")
    ap.add_argument("--max-pairs", type=int, help="random subset of pairs for a quick System 2 run")
    ap.add_argument("--skip-ce", action="store_true", help="System 2 without the cross-encoder rows")
    ap.add_argument("--retrain-ce", action="store_true", help="retrain the cross-encoder even if cached")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text()) | yaml.safe_load((ROOT / args.config).read_text())

    corpus = ROOT / cfg["corpus_root"]
    build = ROOT / cfg["build_dir"]
    models = ROOT / cfg["models_dir"]
    seed = cfg["random_seed"]
    build.mkdir(parents=True, exist_ok=True)
    steps = Steps()

    steps.start(1)
    # build/mtp has no parser_features_dundee.csv, so this table is parser-free
    table, splits, manifest = build_pooled_word_table(
        corpus, min_pooled_td_ms=cfg["min_pooled_td_ms"], random_seed=seed,
        train_ratio=cfg["train_ratio"], dev_ratio=cfg["dev_ratio"], build_dir=build)
    write_build_artifacts(table, splits, manifest, build)
    old_splits = ROOT / "build/splits.json"
    if old_splits.is_file():
        assert json.loads(old_splits.read_text()) == splits, "split differs from main.py's"
        print("  split identical to main.py's build/splits.json")
    print(manifest)

    steps.start(2)
    lm_dundee = dundee_lm_features(corpus, build / "gpt2_dundee.csv", cfg["lm_name"])
    table = add_lexical_features(table.merge(lm_dundee, on=["itemno", "wnum"], how="left"))
    old_gpt2 = ROOT / "data/gpt2_total_surprisal_dundee.csv"
    if old_gpt2.is_file():  # sanity check against an earlier, independent GPT-2 run
        ref = pd.read_csv(old_gpt2).rename(columns={"Article": "itemno", "WNUM": "wnum"})
        j = table.merge(ref, on=["itemno", "wnum"])
        print(f"  r(gpt2_surprisal, earlier GPT-2 file) = {np.corrcoef(j.gpt2_surprisal, j.Total_Surprisal)[0, 1]:.3f}")
    parser_csv = ROOT / "build/parser_features_dundee.csv"  # main.py's cache, read-only
    if parser_csv.is_file():
        table = table.merge(pd.read_csv(parser_csv), on=["itemno", "wnum"], how="left")

    steps.start(3)
    s1 = train_system1_mtp(table, models, cfg["threshold_grid_ms"], current_models_dir=ROOT / "models")

    s2 = None
    if args.skip_s2:
        for k in (4, 5, 6):
            steps.start(k, skipped=True)
    else:
        steps.start(4)
        max_pairs = args.max_pairs or cfg.get("s2_max_pairs")
        pairs = load_wiki_pairs(ROOT / cfg["wiki_pairs_path"], max_pairs=max_pairs, random_seed=seed)
        unique = sorted(set(pairs["standard"]) | set(pairs["simple"]))
        lm_wiki = wiki_lm_features(unique, build / "gpt2_wiki", cfg["lm_name"], cfg["lm_batch_tokens"])
        assert all(len(lm_wiki[s]) == len(s.split()) for s in unique), "GPT-2 word count mismatch"

        steps.start(5)
        rts = system2.predict_rts(unique, lm_wiki, models)
        S = system2.sentence_matrices(unique, rts, s1["targets"]["ffd"]["threshold_ms"])
        ordered, y = system2.order_pairs(pairs, seed)
        pos = {s: i for i, s in enumerate(unique)}
        ia, ib = np.asarray([pos[a] for a, _ in ordered]), np.asarray([pos[b] for _, b in ordered])
        X = {k: system2.pair_matrix(M, ia, ib) for k, M in S.items()}
        idx = system2.split_pairs(pairs, seed, cfg["s2_test_ratio"], cfg["s2_dev_ratio"])
        results, correct = system2.run_classical(X, y, idx, cfg["s2_n_folds"], seed, cfg["four_measure_row"])

        if args.skip_ce:
            steps.start(6, skipped=True)
        else:
            steps.start(6)
            tag = "" if max_pairs is None else f"_{len(pairs)}"  # subset runs never overwrite the full model
            ce = cross_encoder.train_or_load(ordered, y, idx, cfg, models / f"cross_encoder{tag}",
                                             build / f"ce_logits{tag}.npz", retrain=args.retrain_ce, seed=seed)
            ok = (ce["test"] > 0) == y[idx["test"]]
            results["cross_encoder"] = {"label": f"Cross-encoder ({cfg['ce_model']})", "test_acc": float(ok.mean()),
                                        "dev_acc": [float(a) for a in ce["dev_acc"]]}
            correct["cross_encoder"] = ok
            print(f"  {results['cross_encoder']['label']:44s} test {100 * ok.mean():6.2f}%", flush=True)
            hyb, hyb_ok = system2.run_hybrid(ce, X, y, idx)
            results |= hyb
            correct |= hyb_ok
        s2 = {"n_pairs": len(pairs), "n_folds": cfg["s2_n_folds"],
              "split_sizes": {k: int(len(v)) for k, v in idx.items()},
              "ffd_threshold_ms": s1["targets"]["ffd"]["threshold_ms"],
              "models": results, "mcnemar": system2.significance(results, correct),
              "length_matched": system2.on_subset(correct, system2.length_matched(S["length"], ia, ib)[idx["test"]])}

    steps.start(7)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    steps.finish()
    metrics = {"run_id": run_id, "pipeline": "mtp",
               "config": {"lm": cfg["lm_name"], "cross_encoder": None if args.skip_ce else cfg["ce_model"],
                          "max_pairs": args.max_pairs, "ce_epochs": cfg["ce_epochs"],
                          "ce_max_train_pairs": cfg.get("ce_max_train_pairs")},
               "timings_s": steps.timings, "s1": s1, "s2": s2}
    exp = ROOT / cfg["experiments_dir"] / run_id
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cur_s1, cur_s2 = current_runs(ROOT / "experiments")
    write_report(metrics, cur_s1[0] if cur_s1 else None, pick_largest(cur_s2), ROOT / "results_mtp.md")
    print(f"\nWrote results_mtp.md and {exp.relative_to(ROOT)}/metrics.json")


if __name__ == "__main__":
    main()
