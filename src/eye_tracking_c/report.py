"""Markdown tables of our results next to the paper's (Singh et al. 2016, W16-4123)."""

from __future__ import annotations

# Table 1: feature -> paper r with (FFD, FPD, RPD, TD)
PAPER_T1 = {
    "word_len": (0.765, 0.722, 0.700, 0.668),
    "sentence_len": (-0.009, -0.008, -0.010, -0.011),
    "wiki_log_freq": (-0.142, -0.133, -0.129, -0.126),
    "aoa_mean": (0.048, 0.032, 0.028, 0.021),
    "aoa_sd": (-0.048, -0.058, -0.059, -0.067),
    "total_surprisal": (0.369, 0.368, 0.365, 0.358),
    "lexical_surprisal": (0.372, 0.372, 0.369, 0.363),
    "syntactic_surprisal": (-0.067, -0.074, -0.073, -0.072),
    "entropy_reduction": (-0.008, -0.013, -0.008, -0.013),
    "embedding_depth": (0.102, 0.092, 0.086, 0.084),
    "embedding_difference": (0.051, 0.046, 0.043, 0.045),
}

# Table 2: step -> paper R^2 (FFD)
PAPER_T2 = {
    "Word Length": 0.267, "Sentence Length": 0.500, "Word Frequency": 0.504,
    "Mean AoA": 0.532, "Std Dev AoA": 0.533, "Total Surprisal": 0.576,
    "Lexical Surprisal": 0.575, "Syntactic Surprisal": 0.579, "Entropy Reduction": 0.579,
    "Embedding Depth": 0.580, "Embedding Difference": 0.580,
    "Hierarchical structure feats": 0.585,
}

# Table 3: measure -> (label, paper raw R^2, paper thresholded R^2, paper threshold ms)
PAPER_T3 = {
    "ffd": ("First Fixation Duration", 0.585, 0.649, 84),
    "fpd": ("First Pass Duration", 0.549, 0.600, 88),
    "rpd": ("Regression Path Duration", 0.521, 0.570, 91),
    "td": ("Total Fixation Duration", 0.510, 0.516, 98),
}

# Table 5 (+ Sec 4.5 sentence-level-only model): model -> (label, paper accuracy %)
PAPER_T5 = {
    "base_rt_svmrank": ("Base RT - SVMrank", 73.79),
    "base_rt_pairwise": ("Base RT - Pairwise", 73.82),
    "extended_rt_svmrank": ("Extended RT - SVMrank", 75.09),
    "extended_rt_pairwise": ("Extended RT - Pairwise", 75.21),
    "sentence_only_pairwise": ("Sentence-level features only (Sec 4.5)", 73.6),
}


def s1_report(m: dict) -> str:
    out = [f"### System 1 ({m['n_words']:,} words, {m['n_sentences']:,} sentences)", "",
           "**Table 1 - Pearson r with reading times (ours / paper)**", "",
           "| Feature | FFD | FPD | RPD | TD |", "|---|---|---|---|---|"]
    for feat, rs in m["table1_correlations"].items():
        paper = PAPER_T1.get(feat)
        cells = [f"{r:+.3f}" + (f" / {p:+.3f}" if paper else "")
                 for r, p in zip(rs.values(), paper or [None] * 4)]
        out.append(f"| {feat} | " + " | ".join(cells) + " |")

    out += ["", "**Table 2 - FFD ablation, features added in paper order (test R^2)**", "",
            "| Step | Ours | Kept (dev R^2 rose) | Paper |", "|---|---|---|---|"]
    for row in m["table2_ablation_ffd"]:
        out.append(f"| {row['step']} | {row['r2_test']:.3f} | {'yes' if row['retained'] else 'no'} "
                   f"| {PAPER_T2.get(row['step'], float('nan')):.3f} |")

    out += ["", "**Table 3 - R^2 on test split (threshold ms in brackets)**", "",
            "| Measure | Ours raw | Ours thresholded | Paper raw | Paper thresholded |",
            "|---|---|---|---|---|"]
    for tgt, (label, p_raw, p_thr, p_tau) in PAPER_T3.items():
        t = m["targets"][tgt]
        out.append(f"| {label} | {t['r2_test_raw']:.3f} | {t['r2_test_clipped']:.3f} "
                   f"[{t['threshold_ms']:.0f}] | {p_raw:.3f} | {p_thr:.3f} [{p_tau}] |")
    out += ["", f"Selected features ({len(m['feature_columns'])}): {', '.join(m['feature_columns'])}"]
    return "\n".join(out)


def s2_report(m: dict) -> str:
    title = (f"### System 2 ({m['n_pairs']:,} pairs, {m['n_folds']}-fold CV, "
             f"FFD threshold {m['ffd_threshold_ms']:.0f} ms)")
    out = [title, "",
           "**Table 5 - accuracy (%)**", "",
           "| Model | Ours | Paper |", "|---|---|---|",
           "| Vajjala & Meurers 2016 (baseline) | - | 74.58 |",
           "| Ambati et al. 2016 (state of the art) | - | 78.87 |"]
    for key, (label, paper) in PAPER_T5.items():
        r = m["models"].get(key)
        if r:
            out.append(f"| {label} | {100 * r['accuracy_mean']:.2f} (±{100 * r['accuracy_std']:.2f}) | {paper:.2f} |")
    return "\n".join(out)
