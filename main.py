"""Replication of Singh et al. (2016), W16-4123 - single entry point.

    python main.py              # full pipeline
    python main.py --skip-s2    # System 1 only
    python main.py --s2-max-pairs 5000   # quick System 2 check on a random subset

Steps (parser outputs are cached in build/, so reruns are fast):
  1. parse the Dundee sentences with the incremental PCFG parser
  2. build the pooled Dundee word table + sentence-level splits
  3. System 1: Tables 1-3 (correlations, ablation, R^2 per reading-time measure)
  4. parse every Wikipedia sentence (slow the first time: ~1-2 h on 8 cores)
  5. System 2: Table 5 (readability accuracy, 10-fold CV)
Results are printed and written to results.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eye_tracking_c.dundee.build_table import (
    build_pooled_word_table,
    write_build_artifacts,
)
from eye_tracking_c.parser.dundee import build_dundee_parser_features
from eye_tracking_c.parser.wiki import parse_wiki_sentences
from eye_tracking_c.report import s1_report, s2_report
from eye_tracking_c.system1.train import train_system1
from eye_tracking_c.system2.dataset import load_wiki_pairs
from eye_tracking_c.system2.train import train_system2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--skip-s2", action="store_true", help="stop after System 1")
    ap.add_argument("--s2-max-pairs", type=int, help="random subset of pairs for a quick System 2 run")
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())

    corpus = ROOT / cfg["corpus_root"]
    build_dir = ROOT / cfg["build_dir"]
    models_dir = ROOT / cfg["models_dir"]
    exp_dir = ROOT / cfg["experiments_dir"]
    build_dir.mkdir(exist_ok=True)

    print("== 1. Parser features for Dundee ==")
    if (build_dir / "parser_features_dundee.csv").is_file():
        print("cached")
    else:
        build_dundee_parser_features(corpus, build_dir / "parser_features_dundee.csv",
                                     beam=cfg["parser_beam_dundee"])

    print("== 2. Dundee word table ==")
    table, splits, manifest = build_pooled_word_table(
        corpus, min_pooled_td_ms=cfg["min_pooled_td_ms"], random_seed=cfg["random_seed"],
        train_ratio=cfg["train_ratio"], dev_ratio=cfg["dev_ratio"], build_dir=build_dir)
    write_build_artifacts(table, splits, manifest, build_dir)
    print(manifest)

    print("== 3. System 1 ==")
    s1 = train_system1(build_dir / "dundee_word_table.csv", models_dir, exp_dir, cfg["threshold_grid_ms"])
    report = [s1_report(s1)]
    print(report[-1])

    if not args.skip_s2:
        print("== 4. Parser features for Wikipedia ==")
        pairs_path = ROOT / cfg["wiki_pairs_path"]
        max_pairs = args.s2_max_pairs or cfg.get("s2_max_pairs")
        pairs = load_wiki_pairs(pairs_path, max_pairs=max_pairs, random_seed=cfg["random_seed"])
        parses = parse_wiki_sentences(pairs["standard"].tolist() + pairs["simple"].tolist(),
                                      build_dir / "wiki_parse", beam=cfg["parser_beam_wiki"],
                                      n_jobs=cfg["parser_jobs"])
        print("== 5. System 2 ==")
        s2 = train_system2(pairs_path, parses, models_dir, exp_dir,
                           n_folds=cfg["s2_n_folds"], random_seed=cfg["random_seed"],
                           max_pairs=max_pairs)
        report.append(s2_report(s2))
        print(report[-1])

    (ROOT / "results.md").write_text("# Results vs paper (W16-4123)\n\n" + "\n\n".join(report) + "\n")
    print("\nWrote results.md")


if __name__ == "__main__":
    main()
