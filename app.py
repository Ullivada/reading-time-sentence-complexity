"""Streamlit demo: the 2016 pipeline (main.py) vs the transformer extension (main-mtp.py).

    streamlit run app.py

Opens with the saved results (experiments/ and experiments_mtp/); nothing runs until a
Run button is pressed. Runs are subprocesses whose output goes to logs/<name>.log, so
progress survives a browser refresh.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from eye_tracking_c.mtp.report import current_runs, load_runs, n_pairs, pick_largest
from eye_tracking_c.report import PAPER_T1, PAPER_T2, PAPER_T3, PAPER_T5

COLORS = {"Paper": "#9AA0A6", "Current": "#1F77B4", "Enhanced": "#E8710A"}
LOGS = ROOT / "logs"
PIPELINES = {
    "current": {"script": "main.py", "title": "Current (main.py)", "button": "▶ Run current",
                "step": re.compile(r"^== (\d)\. (.+?) =="), "expect": "~2 min with parse caches, ~45 min cold"},
    "mtp": {"script": "main-mtp.py", "title": "Enhanced (main-mtp.py)", "button": "▶ Run enhanced",
            "step": re.compile(r"^== (\d)/7 (.+?) =="), "expect": "~7 min cached, System 1 only ~10 s; first run ~2 h"},
}
SUB = re.compile(r"parsing [\d,]+-([\d,]+) of ([\d,]+)|progress (\d+)/(\d+)")
MEASURES = {"ffd": "FFD", "fpd": "FPD", "rpd": "RPD", "td": "TD"}

st.set_page_config(page_title="Eye-tracking readability", layout="wide")


# ---------- results ----------

def folder_stamp(folder: Path) -> float:
    return sum(p.stat().st_mtime for p in folder.glob("*/metrics.json")) if folder.is_dir() else 0.0


@st.cache_data
def cached_runs(stamp_cur: float, stamp_mtp: float):
    s1, s2 = current_runs(ROOT / "experiments")
    return s1, s2, load_runs(ROOT / "experiments_mtp")


def run_label(r: dict) -> str:
    t = r["run_id"]
    label = f"{t[:4]}-{t[4:6]}-{t[6:8]} {t[9:11]}:{t[11:13]} UTC"
    return label + (f" · {n_pairs(r):,} pairs" if n_pairs(r) else "")


def select_run(title: str, runs: list[dict], default: dict | None, key: str) -> dict | None:
    if not runs:
        st.caption(f"{title}: none yet")
        return None
    labels = [run_label(r) for r in runs]
    if st.session_state.pop(f"latest_{key}", False) or key not in st.session_state:
        st.session_state[key] = labels[runs.index(default)] if default in runs else labels[0]
    return runs[labels.index(st.selectbox(title, labels, key=key))]


# ---------- running pipelines ----------

def state(name: str) -> dict | None:
    try:
        return json.loads((LOGS / f"{name}.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def log_lines(name: str) -> list[str]:
    try:
        return (LOGS / f"{name}.log").read_text(errors="replace").splitlines()
    except OSError:
        return []


def status(name: str) -> str | None:
    """None (never run), 'running', 'done', 'failed: <code>' or 'stopped'."""
    s = state(name)
    if s is None:
        return None
    ends = [ln for ln in log_lines(name) if ln.startswith("== EXIT")]
    if ends:
        code = ends[-1].split()[-1]
        return "done" if code == "0" else "stopped" if code == "stopped" else f"failed: exit {code}"
    try:
        os.kill(s["pid"], 0)
        return "running"
    except OSError:
        return "stopped"


def running() -> str | None:
    return next((n for n in PIPELINES if status(n) == "running"), None)


def launch(name: str, flags: list[str]) -> None:
    LOGS.mkdir(exist_ok=True)
    cmd = f'"{sys.executable}" -u {PIPELINES[name]["script"]} {shlex.join(flags)}; echo "== EXIT $?"'
    with (LOGS / f"{name}.log").open("w") as log:
        proc = subprocess.Popen(["/bin/sh", "-c", cmd], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    (LOGS / f"{name}.json").write_text(json.dumps({"pid": proc.pid, "started": time.time(), "flags": flags}))
    st.session_state["watching"] = name


def stop(name: str) -> None:
    try:
        os.killpg(state(name)["pid"], signal.SIGTERM)  # the shell and the pipeline under it
    except (OSError, TypeError):
        pass
    with (LOGS / f"{name}.log").open("a") as f:
        f.write("\n== EXIT stopped\n")


def progress(name: str, lines: list[str]) -> tuple[int, int, str, float, str]:
    """(step, total steps, step label, overall fraction, last sub-progress line of this step)."""
    flags = (state(name) or {}).get("flags", [])
    total = 7 if name == "mtp" else (3 if "--skip-s2" in flags else 5)
    step, label, frac, sub = 0, "starting", 0.0, ""
    for line in lines:
        if m := PIPELINES[name]["step"].match(line):
            step, label, frac, sub = int(m[1]), m[2], 0.0, ""
        elif m := SUB.search(line):
            done, n = (m[1], m[2]) if m[1] else (m[3], m[4])
            frac, sub = int(done.replace(",", "")) / max(1, int(n.replace(",", ""))), line.strip()
    return step, total, label, min(1.0, (max(step - 1, 0) + frac) / total), sub


def mmss(seconds: float) -> str:
    return f"{int(seconds // 60)}m {int(seconds % 60):02d}s"


@st.fragment(run_every="2s")
def live_panel() -> None:
    name = running()
    watching = st.session_state.get("watching")
    if name is None:
        if watching:  # a run started from this page just ended: show its results
            st.session_state.pop("watching")
            if status(watching) == "done":
                for key in ("cur_s1", "cur_s2", "mtp"):
                    st.session_state[f"latest_{key}"] = True
            st.cache_data.clear()
            st.rerun(scope="app")
        for n in PIPELINES:
            s = status(n)
            if s and s.startswith("failed"):
                st.error(f"{PIPELINES[n]['title']} {s}")
                st.code("\n".join(log_lines(n)[-40:]))
        st.caption("Idle")
        return
    st.session_state["watching"] = name
    lines = log_lines(name)
    step, total, label, overall, sub = progress(name, lines)
    st.markdown(f"**{PIPELINES[name]['title']}** · step {step}/{total}  \n{label}")
    st.progress(overall)
    st.caption(f"elapsed {mmss(time.time() - state(name)['started'])}" + (f" · {sub}" if sub else ""))
    if st.button("■ Stop", key="stop"):
        stop(name)
    with st.expander("Log (last 40 lines)"):
        st.code("\n".join(lines[-40:]) or "(no output yet)")


# ---------- charts ----------

def grouped_bars(df: pd.DataFrame, x: str, y: str, fmt: str = ".3f") -> alt.Chart:
    series = [s for s in COLORS if s in set(df["Series"])]
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:N", title=None, sort=None),
        xOffset=alt.XOffset("Series:N", sort=series),
        y=alt.Y(f"{y}:Q", title=y),
        color=alt.Color("Series:N", scale=alt.Scale(domain=series, range=[COLORS[s] for s in series])),
        tooltip=[x, "Series", alt.Tooltip(f"{y}:Q", format=fmt)])
    return (base.mark_bar() + base.mark_text(dy=-6, fontSize=11).encode(text=alt.Text(f"{y}:Q", format=fmt))).properties(height=300)


def hbar(df: pd.DataFrame, x: str, rule: float | None = None, fmt: str = ".2f") -> alt.Chart:
    series = [s for s in COLORS if s in set(df["Series"])]
    lo = float(df[x].min()) - 3
    base = alt.Chart(df).encode(
        y=alt.Y("Model:N", sort="-x", title=None),
        x=alt.X(f"{x}:Q", scale=alt.Scale(domain=[lo, float(df[x].max()) + 1.5]), title=x),
        color=alt.Color("Series:N", scale=alt.Scale(domain=series, range=[COLORS[s] for s in series])),
        tooltip=["Model", "Series", alt.Tooltip(f"{x}:Q", format=fmt)])
    chart = base.mark_bar(clip=True) + base.mark_text(align="left", dx=3).encode(text=alt.Text(f"{x}:Q", format=fmt))
    if rule is not None:
        chart += alt.Chart(pd.DataFrame({x: [rule]})).mark_rule(strokeDash=[6, 4], color="#D93025").encode(x=f"{x}:Q")
    return chart.properties(height=34 * len(df) + 40)


def coef_chart(coefs: dict) -> alt.Chart:
    df = pd.DataFrame({"Feature": list(coefs), "Weight": list(coefs.values())})
    df["Source"] = np.where(df["Feature"].str.startswith("gpt2"), "Enhanced", "Current")
    return alt.Chart(df).mark_bar().encode(
        y=alt.Y("Feature:N", sort=alt.EncodingSortField("Weight", op="max", order="descending"), title=None),
        x=alt.X("Weight:Q", title="standardized weight (ms per SD)"),
        color=alt.Color("Source:N", scale=alt.Scale(domain=["Current", "Enhanced"], range=[COLORS["Current"], COLORS["Enhanced"]]),
                        legend=alt.Legend(title=None, labelExpr="datum.label == 'Enhanced' ? 'GPT-2 feature' : 'Lexical feature'")),
        tooltip=["Feature", alt.Tooltip("Weight:Q", format=".2f")]).properties(height=260)


def acc(x) -> float | None:
    return None if x is None else round(100 * x, 2)


# ---------- tabs ----------

def tab_current(s1: dict | None, s2: dict | None, mtp: dict | None) -> None:
    if not s1:
        st.info("No main.py results yet. Use **▶ Run current** in the sidebar.")
        return
    best = max(s2["models"].values(), key=lambda m: m["accuracy_mean"]) if s2 else None
    c = st.columns(4)
    c[0].metric("System 1 FFD R² (test)", f"{s1['targets']['ffd']['r2_test_raw']:.3f}",
                f"{s1['targets']['ffd']['r2_test_raw'] - PAPER_T3['ffd'][1]:+.3f} vs paper")
    c[1].metric("Best System 2 accuracy", f"{100 * best['accuracy_mean']:.2f}%" if best else "-",
                f"{100 * best['accuracy_mean'] - 75.21:+.2f} vs paper" if best else None)
    c[2].metric("Pairs", f"{s2['n_pairs']:,}" if s2 else "-")
    c[3].metric("Words (Dundee)", f"{s1['n_words']:,}")

    if mtp and mtp.get("s2") and "length_control" in mtp["s2"]["models"]:
        lc = mtp["s2"]["models"]["length_control"]["cv_mean"]
        st.warning(f"A length-only control (characters + words of each sentence) scores **{100 * lc:.2f}%**, "
                   "about the same as the reading-time model. Predicted FFD is mostly word length.")

    st.subheader("System 1: test R² per reading-time measure")
    rows = [{"Measure": label, "Series": "Paper", "R²": p} for t, (label, p, _, _) in PAPER_T3.items()]
    rows += [{"Measure": PAPER_T3[t][0], "Series": "Current", "R²": s1["targets"][t]["r2_test_raw"]} for t in MEASURES]
    st.altair_chart(grouped_bars(pd.DataFrame(rows), "Measure", "R²"), width="stretch")

    if s2:
        st.subheader(f"System 2: readability accuracy ({s2['n_pairs']:,} pairs, {s2['n_folds']}-fold CV)")
        rows = [{"Model": "Vajjala & Meurers 2016", "Ours (%)": None, "Paper (%)": 74.58},
                {"Model": "Ambati et al. 2016", "Ours (%)": None, "Paper (%)": 78.87}]
        rows += [{"Model": label, "Ours (%)": acc(s2["models"].get(k, {}).get("accuracy_mean")), "Paper (%)": p}
                 for k, (label, p) in PAPER_T5.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption("System 2 was not run in this run.")

    with st.expander("Table 1: Pearson r of each feature with each measure (ours / paper)"):
        rows = [{"Feature": f, **{MEASURES[t]: f"{v:+.3f}" + (f" / {PAPER_T1[f][i]:+.3f}" if f in PAPER_T1 else "")
                                  for i, (t, v) in enumerate(rs.items())}}
                for f, rs in s1["table1_correlations"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Table 2: FFD ablation in the paper's order"):
        st.dataframe(pd.DataFrame([{"Step": r["step"], "Ours R²": round(r["r2_test"], 3), "Kept": r["retained"],
                                    "Paper R²": PAPER_T2.get(r["step"])} for r in s1["table2_ablation_ffd"]]),
                     hide_index=True, width="stretch")


def tab_enhanced(m: dict | None) -> None:
    if not m:
        st.info("No main-mtp.py results yet. Use **▶ Run enhanced** in the sidebar.")
        return
    s1, s2 = m["s1"], m.get("s2")
    rows = {a["row"]: a for a in s1["ablation"]}
    models = (s2 or {}).get("models", {})
    classical = [v["test_acc"] for k, v in models.items() if "cv_mean" in v and k != "length_control"]
    c = st.columns(5)
    c[0].metric("FFD R² (row D)", f"{rows['D']['r2_test']['ffd']:.3f}", f"{rows['D']['delta_r2_test']['ffd']:+.3f} over baseline")
    r = s1["r_pred_ffd_wordlen"]
    c[1].metric("r(pred FFD, word length)", f"{r['enhanced']:.3f}",
                f"{r['enhanced'] - r['current']:+.3f} vs current" if "current" in r else None, delta_color="inverse")
    c[2].metric("Best classical S2 (test)", f"{100 * max(classical):.2f}%" if classical else "-")
    c[3].metric("Cross-encoder (test)", f"{100 * models['cross_encoder']['test_acc']:.2f}%" if "cross_encoder" in models else "-")
    c[4].metric("Hybrid (test)", f"{100 * models['hybrid_rt']['test_acc']:.2f}%" if "hybrid_rt" in models else "-")

    st.subheader("System 1: ablation, test R² (ΔR² over row A)")
    st.dataframe(pd.DataFrame([{"Row": a["row"], "Features": a["label"],
                                **{MEASURES[t]: f"{a['r2_test'][t]:.3f} ({a['delta_r2_test'][t]:+.3f})" for t in MEASURES}}
                               for a in s1["ablation"]]), hide_index=True, width="stretch")
    if "B" not in rows:
        st.caption("Rows B and E need main.py's parser cache (build/parser_features_dundee.csv).")
    left, right = st.columns(2)
    with left:
        st.markdown("**ΔR² over baseline, FFD**")
        df = pd.DataFrame([{"Model": f"{a['row']} {a['label']}", "ΔR²": a["delta_r2_test"]["ffd"],
                            "Series": "Enhanced" if "GPT-2" in a["label"] or a["row"] == "E" else "Current"}
                           for a in s1["ablation"] if a["row"] != "A"])
        st.altair_chart(hbar(df, "ΔR²", fmt=".3f").properties(height=200), width="stretch")
    with right:
        st.markdown("**FFD model weights (row D)**")
        st.altair_chart(coef_chart(s1["ffd_coefficients"]), width="stretch")

    if not s2:
        st.caption("System 2 was not run in this run.")
        return
    sz = s2["split_sizes"]
    st.subheader(f"System 2: {s2['n_pairs']:,} pairs")
    st.caption(f"CV = {s2['n_folds']}-fold (comparable with main.py and the paper). Test = fixed split of "
               f"{sz['test']:,} pairs (train {sz['train']:,}, dev {sz['dev']:,}), comparable across all rows.")
    st.dataframe(pd.DataFrame([{"Model": v["label"], "10-fold CV (%)": acc(v.get("cv_mean")),
                                "Test split (%)": acc(v["test_acc"]), "Features": v.get("n_features")}
                               for v in models.values()]), hide_index=True, width="stretch")
    if s2.get("mcnemar"):
        st.markdown("**McNemar test on the test split**")
        st.dataframe(pd.DataFrame([{"Comparison": k.replace("_vs_", " vs "), "Δ accuracy (pts)": round(100 * v["delta_acc"], 2),
                                    "p": round(v["p_value"], 4), "p < 0.05": "✓" if v["p_value"] < 0.05 else ""}
                                   for k, v in s2["mcnemar"].items()]), hide_index=True, width="stretch")
    with st.expander("Run config and step timings"):
        st.json({"config": m["config"], "timings_s": m["timings_s"]})


def tab_compare(s1: dict | None, s2: dict | None, m: dict | None) -> None:
    if not (s1 and m):
        st.info("Needs both a main.py run and a main-mtp.py run.")
        return
    rows = [{"Measure": label, "Series": "Paper", "R²": p} for t, (label, p, _, _) in PAPER_T3.items()]
    rows += [{"Measure": PAPER_T3[t][0], "Series": "Current", "R²": s1["targets"][t]["r2_test_raw"]} for t in MEASURES]
    rows += [{"Measure": PAPER_T3[t][0], "Series": "Enhanced", "R²": m["s1"]["targets"][t]["r2_test_raw"]} for t in MEASURES]
    st.subheader("System 1: test R², paper vs current vs enhanced")
    st.altair_chart(grouped_bars(pd.DataFrame(rows), "Measure", "R²"), width="stretch")

    abl = {a["row"]: a for a in m["s1"]["ablation"]}
    notes = []
    if "B" in abl:
        notes.append("ΔR² over the no-surprisal baseline, PCFG parser (B) vs GPT-2 (D): " + ", ".join(
            f"{MEASURES[t]} {abl['B']['delta_r2_test'][t]:+.3f} vs {abl['D']['delta_r2_test'][t]:+.3f}" for t in MEASURES) + ".")
    if "E" in abl:
        notes.append("Adding the parser on top of GPT-2 (E vs D) still changes R² by " + ", ".join(
            f"{MEASURES[t]} {abl['E']['r2_test'][t] - abl['D']['r2_test'][t]:+.3f}" for t in MEASURES) + ".")

    m2 = (m.get("s2") or {}).get("models", {})
    if m2:
        df = []
        for key, cur_key in (("base_rt", "base_rt_pairwise"), ("extended_rt", "extended_rt_pairwise"),
                             ("sentence_only", "sentence_only_pairwise")):
            if s2 and cur_key in s2["models"]:
                df.append({"Model": f"Current · {m2[key]['label']}", "Series": "Current",
                           "Accuracy (%)": 100 * s2["models"][cur_key]["accuracy_mean"]})
        for v in m2.values():
            df.append({"Model": f"Enhanced · {v['label']}" + ("" if "cv_mean" in v else " [test]"),
                       "Series": "Enhanced", "Accuracy (%)": 100 * v.get("cv_mean", v["test_acc"])})
        df += [{"Model": "Paper · Vajjala & Meurers 2016", "Series": "Paper", "Accuracy (%)": 74.58},
               {"Model": "Paper · Ambati et al. 2016", "Series": "Paper", "Accuracy (%)": 78.87}]
        lc = 100 * m2["length_control"]["cv_mean"]
        st.subheader("System 2: accuracy (%)")
        st.caption(f"Classical rows: 10-fold CV. Transformer rows [test]: fixed test split. "
                   f"Dashed red line: length control {lc:.2f}%, the bar to beat.")
        st.altair_chart(hbar(pd.DataFrame(df), "Accuracy (%)", rule=lc), width="stretch")
        p = m["s2"].get("mcnemar", {})
        b = m2["base_rt"]["test_acc"] - m2["length_control"]["test_acc"]
        pb = p.get("base_rt_vs_length_control", {}).get("p_value")
        notes.append(f"Enhanced Base RT is {100 * b:+.2f} points vs the length control on the test split"
                     + (f" (McNemar p = {pb:.3g})." if pb is not None else "."))
        if "cross_encoder" in m2:
            notes.append(f"The cross-encoder reaches {100 * m2['cross_encoder']['test_acc']:.2f}% "
                         f"(Ambati et al. 2016: 78.87%).")
        if "hybrid_rt" in m2:
            d = m2["hybrid_rt"]["test_acc"] - m2["cross_encoder"]["test_acc"]
            ph = p.get("hybrid_rt_vs_cross_encoder", {}).get("p_value")
            notes.append(f"Adding predicted reading times to the cross-encoder changes accuracy by {100 * d:+.2f} points"
                         + (f" (p = {ph:.3g})." if ph is not None else "."))
    st.subheader("Takeaways")
    st.markdown("\n".join(f"- {n}" for n in notes) or "-")


@st.cache_resource
def try_models():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from eye_tracking_c.mtp.lm import load_lm
    lm = load_lm("gpt2")
    ce_dir = ROOT / "models/mtp/cross_encoder"
    ce = None
    if ce_dir.is_dir():
        ce = (AutoTokenizer.from_pretrained(ce_dir), AutoModelForSequenceClassification.from_pretrained(ce_dir).to(lm[2]))
    return lm, ce


def tab_try() -> None:
    if not (ROOT / "models/mtp/s1_ffd_regressor.joblib").is_file():
        st.info("Run the enhanced pipeline first (it trains the models used here).")
        return
    from eye_tracking_c.mtp.cross_encoder import predict_logits
    from eye_tracking_c.mtp.lm import score_sentences
    from eye_tracking_c.mtp.system2 import predict_rts

    with st.form("try"):
        a = st.text_input("Sentence A", "Under conditions of high humidity, the rate of evaporation of sweat from the skin decreases.")
        b = st.text_input("Sentence B", "With a higher humidity, the rate of evaporation is less.")
        go = st.form_submit_button("Predict reading times")
    if not go:
        return
    (tok, lm, device), ce = try_models()
    sents = [" ".join(re.findall(r"\w+(?:[-'.]\w+)*|[^\w\s]", s)) for s in (a, b)]
    feats = dict(zip(sents, score_sentences(sents, tok, lm, device)))
    rts = predict_rts(sents, feats, ROOT / "models/mtp")
    cols, start = st.columns(2), 0
    for col, name, s in zip(cols, "AB", sents):
        words = s.split()
        ffd = rts["ffd"][start : start + len(words)]
        start += len(words)
        df = pd.DataFrame({"i": range(len(words)), "Word": words, "Predicted FFD (ms)": ffd,
                           "GPT-2 surprisal (bits)": feats[s][:, 0]})
        col.markdown(f"**Sentence {name}**: mean predicted FFD {ffd.mean():.0f} ms, total {ffd.sum():.0f} ms")
        col.altair_chart(alt.Chart(df).mark_bar(color=COLORS["Enhanced"]).encode(
            x=alt.X("Word:N", sort=alt.EncodingSortField("i"), title=None), y="Predicted FFD (ms):Q",
            tooltip=["Word", alt.Tooltip("Predicted FFD (ms):Q", format=".0f"),
                     alt.Tooltip("GPT-2 surprisal (bits):Q", format=".1f")]).properties(height=260),
            width="stretch")
    if ce:
        p = 1 / (1 + np.exp(-predict_logits(ce[1], ce[0], [tuple(sents)])[0]))
        st.metric("Cross-encoder: probability that sentence B is the simpler one", f"{100 * p:.1f}%")
    else:
        st.caption("No trained cross-encoder in models/mtp/cross_encoder yet.")


# ---------- page ----------

cur_s1_runs, cur_s2_runs, mtp_runs = cached_runs(folder_stamp(ROOT / "experiments"), folder_stamp(ROOT / "experiments_mtp"))
busy = running()

with st.sidebar:
    st.header("Run pipelines")
    for name, p in PIPELINES.items():
        st.markdown(f"**{p['title']}**")
        st.caption(p["expect"])
        flags = []
        if st.checkbox("System 1 only", key=f"{name}_s1"):
            flags.append("--skip-s2")
        if name == "mtp":
            if st.checkbox("Skip cross-encoder", key="mtp_noce"):
                flags.append("--skip-ce")
            if st.checkbox("Retrain cross-encoder", key="mtp_retrain"):
                flags.append("--retrain-ce")
        n = st.number_input("Max pairs (0 = all)", 0, 200_000, 0, 1000, key=f"{name}_n")
        if n:
            flags += ["--s2-max-pairs" if name == "current" else "--max-pairs", str(int(n))]
        if name == "current":
            st.caption("Rewrites results.md and models/ (as running main.py always does).")
        if st.button(p["button"], key=f"run_{name}", disabled=busy is not None, width="stretch"):
            launch(name, flags)
            st.rerun()
        st.divider()
    st.header("Live run")
    live_panel()
    st.header("Results shown")
    cur_s1 = select_run("Current · System 1", cur_s1_runs, cur_s1_runs[0] if cur_s1_runs else None, "cur_s1")
    cur_s2 = select_run("Current · System 2", cur_s2_runs, pick_largest(cur_s2_runs), "cur_s2")
    mtp = select_run("Enhanced", mtp_runs, pick_largest(mtp_runs), "mtp")

st.title("Eye-tracking readability: pipeline vs"" transformers")
st.caption("Transformer extension")
t1, t2, t3, t4 = st.tabs(["1 · Current (2016)", "2 · Enhanced (transformer)", "3 · Comparison", "4 · Try it"])
with t1:
    tab_current(cur_s1, cur_s2, mtp)
with t2:
    tab_enhanced(mtp)
with t3:
    tab_compare(cur_s1, cur_s2, mtp)
with t4:
    tab_try()
