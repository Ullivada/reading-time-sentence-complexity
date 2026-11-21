"""results_mtp.md: Paper / Current (main.py) / Enhanced (main-mtp.py) side by side.

Also holds the metrics loaders shared with the Streamlit app (app.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from eye_tracking_c.report import PAPER_T3, PAPER_T5

MEASURES = ("ffd", "fpd", "rpd", "td")

# enhanced System 2 key -> (current main.py key, paper Table 5 key)
S2_MATCH = {
    "base_rt": ("base_rt_pairwise", "base_rt_pairwise"),
    "extended_rt": ("extended_rt_pairwise", "extended_rt_pairwise"),
    "sentence_only": ("sentence_only_pairwise", "sentence_only_pairwise"),
}


def load_runs(folder: Path) -> list[dict]:
    """Every <folder>/<run_id>/metrics.json, newest first; unreadable files are skipped."""
    runs = []
    for p in sorted(Path(folder).glob("*/metrics.json"), reverse=True):
        try:
            runs.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            pass
    return runs


def n_pairs(run: dict) -> int:
    s2 = run.get("s2", run)
    return (s2 or {}).get("n_pairs", 0)


def pick_largest(runs: list[dict]) -> dict | None:
    """Newest run among those with the most pairs, so a quick subset run never wins."""
    if not runs:
        return None
    most = max(n_pairs(r) for r in runs)
    return next(r for r in runs if n_pairs(r) == most)


def current_runs(experiments_dir: Path) -> tuple[list[dict], list[dict]]:
    """main.py writes separate S1 and S2 metrics files; split them."""
    runs = load_runs(experiments_dir)
    return [r for r in runs if "targets" in r], [r for r in runs if r.get("system") == "s2"]


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.2f}"


def write_report(m: dict, cur_s1: dict | None, cur_s2: dict | None, path: Path) -> None:
    s1 = m["s1"]
    out = ["# Results: transformer extension (main-mtp.py)", "",
           f"Run {m['run_id']}. Current = latest main.py run, Paper = Singh et al. (2016).", "",
           f"## System 1 ({s1['n_words']:,} words, {s1['n_sentences']:,} sentences)", "",
           "**Ablation: test R² (ΔR² over row A)**", "",
           "| Row | Features | FFD | FPD | RPD | TD |", "|---|---|---|---|---|---|"]
    for a in s1["ablation"]:
        cells = [f"{a['r2_test'][t]:.3f} ({a['delta_r2_test'][t]:+.3f})" for t in MEASURES]
        out.append(f"| {a['row']} | {a['label']} | " + " | ".join(cells) + " |")

    out += ["", "**Final model (row D) vs current and paper: test R², raw**", "",
            "| Measure | Paper | Current | Enhanced | Enhanced thresholded [ms] |", "|---|---|---|---|---|"]
    for t, (label, p_raw, _, _) in PAPER_T3.items():
        cur = cur_s1["targets"][t]["r2_test_raw"] if cur_s1 else None
        e = s1["targets"][t]
        out.append(f"| {label} | {p_raw:.3f} | {'-' if cur is None else f'{cur:.3f}'} | {e['r2_test_raw']:.3f} "
                   f"| {e['r2_test_clipped']:.3f} [{e['threshold_ms']:.0f}] |")

    r = s1["r_pred_ffd_wordlen"]
    out += ["", "**Is predicted FFD still just word length?** Pearson r(predicted FFD, word length), "
            "Dundee test split: " + ", ".join(f"{k} {v:.3f}" for k, v in r.items()), "",
            "**Standardized FFD coefficients (row D):** "
            + ", ".join(f"{k} {v:+.1f}" for k, v in sorted(s1["ffd_coefficients"].items(),
                                                             key=lambda kv: -abs(kv[1])))]

    s2 = m.get("s2")
    if s2:
        sz = s2["split_sizes"]
        out += ["", f"## System 2 ({s2['n_pairs']:,} pairs; CV = {s2['n_folds']}-fold, "
                f"test = {sz['test']:,} pairs, train {sz['train']:,} / dev {sz['dev']:,})", "",
                "| Model | Paper | Current (CV) | Enhanced (CV) | Enhanced (test) |", "|---|---|---|---|---|",
                "| Vajjala & Meurers 2016 | 74.58 | - | - | - |",
                "| Ambati et al. 2016 | 78.87 | - | - | - |"]
        for key, row in s2["models"].items():
            cur_key, paper_key = S2_MATCH.get(key, (None, None))
            cur = (cur_s2 or {}).get("models", {}).get(cur_key, {}).get("accuracy_mean")
            paper = f"{PAPER_T5[paper_key][1]:.2f}" if paper_key else "-"
            out.append(f"| {row['label']} | {paper} | {pct(cur)} | {pct(row.get('cv_mean'))} "
                       f"| {pct(row.get('test_acc'))} |")
        if s2.get("mcnemar"):
            out += ["", "**McNemar test on the test split**", "", "| Comparison | Δ accuracy | p |", "|---|---|---|"]
            for k, v in s2["mcnemar"].items():
                out.append(f"| {k.replace('_vs_', ' vs ')} | {100 * v['delta_acc']:+.2f} | {v['p_value']:.4f} |")

        lm = s2.get("length_matched")
        if lm:
            out += ["", f"**Length-matched test pairs** ({lm['n_pairs']:,} pairs: word counts within 1, "
                    "characters within 10%; same models, scored on this subset)", "",
                    "| Model | All test pairs | Length-matched |", "|---|---|---|"]
            for key, acc in lm["acc"].items():
                out.append(f"| {s2['models'][key]['label']} | {pct(s2['models'][key]['test_acc'])} | {pct(acc)} |")
            for k, v in lm["mcnemar"].items():
                out.append(f"\n{k.replace('_vs_', ' vs ')} on length-matched pairs: "
                           f"Δ {100 * v['delta_acc']:+.2f}, p = {v['p_value']:.4f}")

    t = m["timings_s"]
    out += ["", "Step timings (s): " + ", ".join(f"{k} {v:.0f}" for k, v in t.items())]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
