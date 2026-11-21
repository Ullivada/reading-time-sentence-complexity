# Reading-Time Driven Sentence Complexity

Machine-learning replication of Singh et al. (2016), *"Quantifying sentence complexity based on eye-tracking measures"*, CL4LC workshop at COLING 2016 ([W16-4123](https://aclanthology.org/W16-4123/)), plus an MTP extension that adds transformer models.

- **System 1** predicts per-word Dundee reading times (FFD, FPD, RPD, TD) with linear regression on 20 psycholinguistic features.
- **System 2** decides which sentence of a Standard/Simple Wikipedia pair is simpler, from the predicted reading times (pairwise logistic regression and a ranking SVM).
- **MTP extension** replaces the PCFG parser's surprisal with GPT-2 surprisal and adds a fine-tuned RoBERTa cross-encoder to System 2.

## 1. Setup

Python 3.10 or newer.

```bash
git clone https://github.com/Ullivada/reading-time-sentence-complexity.git
cd reading-time-sentence-complexity
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # replication (main.py)
pip install -r requirements-mtp.txt    # transformer extension (main-mtp.py, app.py)
```

## 2. Getting the data

No data is included in this repository. Several datasets are licensed and cannot be redistributed, so obtain each one from its source and place it under `data/` exactly as below. Provenance details are in [`data/README.md`](data/README.md).

| Put it at | Dataset | Where to get it |
|---|---|---|
| `data/dundee_corpus/` | Dundee eye-tracking corpus (Kennedy & Pynte 2005) | On request from the corpus authors (University of Dundee) |
| `data/penn_treebank_3/parsed/mrg/wsj/` | Penn Treebank 3, WSJ sections 00–24 | Linguistic Data Consortium, [LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42) (licence required) |
| `data/wiki_pairs_hwang_good.tsv` | Standard/Simple Wikipedia pairs, `aligned-good` (0.67 threshold) | Hwang et al. (2015); cleaned as in Ambati et al. (2016) |
| `data/aoa_kuperman_2012.csv` | Age-of-acquisition norms, 30,121 words (`Word`, `Rating.Mean`, `Rating.SD`) | Kuperman, Stadthagen-Gonzalez & Brysbaert (2012) supplement |
| `data/enwiki_word_frequency.txt` | English Wikipedia unigram counts, `word count` per line | [IlyaSemenov/wikipedia-word-frequency](https://github.com/IlyaSemenov/wikipedia-word-frequency), `enwiki-2023-04-13.txt` |
| `data/nltk_data/corpora/` | NLTK `treebank` sample and `conll2000` *(optional)* | `python -m nltk.downloader -d data/nltk_data treebank conll2000` |

Expected layout:

```
data/
├── dundee_corpus/
│   ├── RT_dataset1P/                  sa01ma1p.dat … (200 files)
│   ├── RT_dataset2P/                  sa01ma2p.dat … (200 files)
│   ├── per-word-info/                 tx01wrdp.dat … tx20wrdp.dat
│   ├── per-word-lexical-annotation/   text1.txt … text20.txt
│   └── Sentences/                     sent1.txt …
├── penn_treebank_3/parsed/mrg/wsj/    00/ … 24/
├── wiki_pairs_hwang_good.tsv          standard<TAB>simple<TAB>score
├── aoa_kuperman_2012.csv
├── enwiki_word_frequency.txt
└── nltk_data/corpora/                 optional fallback grammar
```

The parser is trained on WSJ sections 02–21, as in the paper. Without `penn_treebank_3/` it falls back to NLTK's free 3,914-tree sample. The pipeline still runs, but the parser features and System 1 numbers will differ from the ones below.

## 3. Reproduce the replication

```bash
python main.py                      # everything: parse, build, System 1, System 2
python main.py --skip-s2            # System 1 only
python main.py --s2-max-pairs 5000  # quick System 2 check on a random subset of pairs
```

| Step | Input | Output |
|---|---|---|
| 1. Parse Dundee | Dundee lexical annotation, Penn Treebank | `build/parser_features_dundee.csv` (15 parser features per word) |
| 2. Dundee word table | Dundee RTs, parser features, frequency, AoA | pooled word table and 60/20/20 sentence splits in `build/` |
| 3. System 1 | word table | Tables 1–3, models in `models/` |
| 4. Parse Wikipedia | Wikipedia pairs | `build/wiki_parse/` (cached, resumable) |
| 5. System 2 | Wikipedia pairs + System 1 model | Table 5 (10-fold CV) |

Results are printed and written to `results.md`. The first full run parses about 235k Wikipedia sentences, which takes roughly 25–40 minutes on 8 cores. Parser output is cached, so later runs take about 2 minutes. Settings such as the random seed (42), split ratios and parser beam are in [`configs/default.yaml`](configs/default.yaml).

## 4. Reproduce the transformer extension

`main.py` must have been run first; the extension reads its outputs and writes only to `build/mtp/`, `models/mtp/`, `experiments_mtp/` and `results_mtp.md`.

```bash
python main-mtp.py                  # full enhanced pipeline
python main-mtp.py --skip-ce        # without the cross-encoder (minutes instead of hours)
python main-mtp.py --max-pairs 5000 # quick System 2 check
streamlit run app.py                # demo: saved results of both pipelines
```

GPT-2 small and `roberta-base` are downloaded from Hugging Face on first use. Fine-tuning the cross-encoder took about 4.5 hours on an Apple M2 (MPS); everything else takes minutes. Settings are in [`configs/mtp.yaml`](configs/mtp.yaml).

## 5. Expected results

Numbers from our runs, so you can check yours. Full tables (feature correlations, ablations, McNemar tests, length-matched pairs) are written to `results.md` and `results_mtp.md`.

**System 1 — test R², raw reading times**

| Measure | Paper | Replication (PCFG) | Extension (GPT-2) |
|---|---|---|---|
| First Fixation Duration | 0.585 | 0.511 | 0.506 |
| First Pass Duration | 0.549 | 0.587 | 0.595 |
| Regression Path Duration | 0.521 | 0.264 | 0.267 |
| Total Fixation Duration | 0.510 | 0.560 | 0.577 |

**System 2 — accuracy (%) on 117,712 Wikipedia pairs**

| Model | Paper | Replication (10-fold CV) | Extension (held-out test) |
|---|---|---|---|
| Base RT — pairwise | 73.82 | 73.90 | 73.80 |
| Extended RT — pairwise | 75.21 | 75.10 | 75.46 |
| Sentence-level features only | 73.60 | 74.39 | 74.38 |
| Length control (characters + words) | — | — | 75.13 |
| Cross-encoder (roberta-base) | — | — | 84.77 |

System 2 accuracy matches the paper within 0.4 points. A plain length baseline reaches about the same accuracy as the reading-time models, which is why the extension adds a length control and a length-matched evaluation. System 1 differs from the paper mainly because the parser uses plain Penn Treebank labels rather than the paper's GCG re-annotation (van Schijndel et al. 2013).

## Features

The paper's 20 features: word length (from Dundee), sentence length, Wikipedia frequency, AoA mean and SD, and 15 features from an incremental left-corner PCFG parser ([`parser/pcfg.py`](src/eye_tracking_c/parser/pcfg.py)): total, lexical and syntactic surprisal, entropy reduction, embedding depth, weighted embedding depth, embedding difference, and 8 memory-operation features. The extension replaces the parser features with GPT-2 surprisal, entropy and previous-word surprisal ([`mtp/lm.py`](src/eye_tracking_c/mtp/lm.py)).

## Code map

| Path | What it does |
|---|---|
| `main.py` | Replication entry point, 5 steps |
| `main-mtp.py` | Extension entry point, 7 steps |
| `app.py` | Streamlit demo |
| `configs/` | `default.yaml` (replication), `mtp.yaml` (extension, layered on top) |
| `src/eye_tracking_c/dundee/` | Reads Dundee, computes FFD/FPD/RPD/TD, pools subjects, splits sentences 60/20/20 |
| `src/eye_tracking_c/parser/` | Incremental PCFG parser and its runs over Dundee and Wikipedia |
| `src/eye_tracking_c/features/` | Word-level features, imputation and standardization |
| `src/eye_tracking_c/system1/` | Tables 1–3: correlations, ablation, R² per measure |
| `src/eye_tracking_c/system2/` | Table 5: Base RT, Extended RT and sentence-only models |
| `src/eye_tracking_c/report.py` | Writes `results.md` |
| `src/eye_tracking_c/mtp/lm.py` | GPT-2 word surprisal, entropy, spillover |
| `src/eye_tracking_c/mtp/system1.py` | Ablation rows A–E (ΔR²) and the final System 1 |
| `src/eye_tracking_c/mtp/system2.py` | Length control, RT rows, hybrid rows, McNemar |
| `src/eye_tracking_c/mtp/cross_encoder.py` | RoBERTa pair classifier |
| `src/eye_tracking_c/mtp/report.py` | Writes `results_mtp.md`; metrics loaders shared with the app |

## Reference

Abhinav Deep Singh, Poojan Mehta, Samar Husain and Rajakrishnan Rajkumar. 2016. Quantifying sentence complexity based on eye-tracking measures. In *Proceedings of the Workshop on Computational Linguistics for Linguistic Complexity (CL4LC)*, Osaka, Japan. [aclanthology.org/W16-4123](https://aclanthology.org/W16-4123/)
