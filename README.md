# eye-tracking-c — replication of Singh et al. (2016), W16-4123

Replicates *"Quantifying sentence complexity based on eye-tracking measures"* (CL4LC 2016):

- **System 1** predicts per-word Dundee reading times (FFD, FPD, RPD, TD) with linear regression on 20 psycholinguistic features.
- **System 2** decides which sentence of a Standard/Simple Wikipedia pair is simpler, from the predicted reading times (pairwise logistic regression and a ranking SVM).

Our numbers next to the paper's are in [`results.md`](results.md) (written by every run) and discussed in [`docs/replication-status.md`](docs/replication-status.md).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py                      # everything: parse, build, System 1, System 2
python main.py --skip-s2            # System 1 only (about 20 s once the Dundee parse is cached)
python main.py --s2-max-pairs 5000  # quick System 2 check on a random subset of pairs
```

The first full run parses about 235k Wikipedia sentences with the incremental parser — roughly 25 minutes on 8 cores. Parser output is cached in `build/wiki_parse/` and a stopped run resumes from the chunks already written, so re-running after a completed parse is fast.

## Data

Every dataset the pipeline reads lives under `data/`; see [`data/README.md`](data/README.md)
for provenance of each file.

- `data/dundee_corpus/` — Dundee eye-tracking corpus (reading times, per-word info, PTB-tokenized lexical annotation).
- `data/penn_treebank_3/` — Penn Treebank 3 WSJ gold trees (sections 00–24). Sections 02–21 train the parser, as in the paper.
- `data/wiki_pairs_hwang_good.tsv` — Standard/Simple Wikipedia pairs (Hwang et al. 2015, cleaned as in Ambati et al. 2016); 117,712 pairs after removing identical ones.
- `data/aoa_kuperman_2012.csv` — Kuperman et al. (2012) age-of-acquisition norms.
- `data/enwiki_word_frequency.txt` — English Wikipedia unigram counts.
- `data/nltk_data/` — NLTK's free WSJ sample (3,914 trees) and CoNLL-2000. Fallback grammar only, used when `data/penn_treebank_3/` is absent.

## Features

The paper's 20 features: word length (from Dundee), sentence length, Wikipedia frequency, AoA mean and SD, and 15 features from an incremental left-corner PCFG parser. The parser features are total, lexical and syntactic surprisal, entropy reduction, embedding depth, weighted embedding depth, embedding difference, and 8 memory-operation features.

The paper's parser (van Schijndel et al. 2013, ModelBlocks) was trained on WSJ sections 2–21. [`src/eye_tracking_c/parser/pcfg.py`](src/eye_tracking_c/parser/pcfg.py) implements the same model class and trains it on those same sections of `data/penn_treebank_3/` (39,832 trees), falling back to NLTK's 3,914-tree sample when the treebank is absent. The remaining parser deviation is the paper's GCG re-annotation, which is not applied here. See [`docs/replication-status.md`](docs/replication-status.md) for this and every other deviation.

## Code map

| Path | What it does |
|---|---|
| `main.py` | Single entry point, runs every step |
| `src/eye_tracking_c/dundee/` | Reads Dundee, computes FFD/FPD/RPD/TD, pools subjects, splits sentences 60/20/20 |
| `src/eye_tracking_c/parser/` | Incremental PCFG parser and its runs over Dundee and Wikipedia |
| `src/eye_tracking_c/features/` | Word-level features, imputation and standardization |
| `src/eye_tracking_c/system1/` | Tables 1–3: correlations, ablation, R² per measure |
| `src/eye_tracking_c/system2/` | Table 5: Base RT, Extended RT and sentence-only models |
| `src/eye_tracking_c/report.py` | Our numbers next to the paper's |

## MTP extension: transformers

A second, independent pipeline replaces the PCFG parser with GPT-2 surprisal and adds a
fine-tuned cross-encoder to System 2. `main.py` is unchanged; the new pipeline only reads
its outputs and writes to its own folders (`build/mtp/`, `models/mtp/`, `experiments_mtp/`,
`results_mtp.md`).

```bash
pip install -r requirements-mtp.txt
python main-mtp.py          # enhanced pipeline (first run ~1.5 h on an M2, cached reruns minutes)
streamlit run app.py        # demo: saved results of both pipelines, run either from the browser
```

How to run everything: [`execution-guide.md`](execution-guide.md). Design:
[`docs/architecture_transformers.md`](docs/architecture_transformers.md),
[`docs/implementation_transformer.md`](docs/implementation_transformer.md),
[`docs/streamlit_transformer.md`](docs/streamlit_transformer.md).

| Path | What it does |
|---|---|
| `main-mtp.py` | Entry point of the enhanced pipeline, 7 steps |
| `configs/mtp.yaml` | Its settings, on top of `configs/default.yaml` |
| `src/eye_tracking_c/mtp/lm.py` | GPT-2 word surprisal, entropy, spillover (Dundee with context, Wikipedia alone) |
| `src/eye_tracking_c/mtp/system1.py` | Ablation rows A–E (ΔR²) and the final System 1 |
| `src/eye_tracking_c/mtp/system2.py` | Length control, RT rows, hybrid rows, McNemar |
| `src/eye_tracking_c/mtp/cross_encoder.py` | DistilRoBERTa pair classifier |
| `src/eye_tracking_c/mtp/report.py` | `results_mtp.md`, metrics loaders shared with the app |
| `app.py` | Streamlit demo |
