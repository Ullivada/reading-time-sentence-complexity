# Data

Every dataset the pipeline reads lives here. Nothing outside `data/` is required at
run time. Sources are the paper's (Singh et al. 2016, W16-4123) where available.

## Corpora

| Path | What it is | Source | Used for |
|------|------------|--------|----------|
| `dundee_corpus/` | Dundee eye-tracking corpus: reading times (`RT_dataset1P`, `RT_dataset2P`), per-word info, PTB-tokenized lexical annotation, sentences | Kennedy & Pynte (2005) | System 1 targets (FFD/FPD/RPD/TD) and word alignment |
| `penn_treebank_3/parsed/mrg/wsj/` | Penn Treebank 3, WSJ gold parse trees, sections 00–24 (2,312 files) | LDC99T42 | Grammar of the incremental PCFG parser — sections **02–21** only, as in the paper |
| `wiki_pairs_hwang_good.tsv` | Standard/Simple Wikipedia sentence pairs, `standard<TAB>simple<TAB>score`, 154,805 rows (117,712 after dropping identical pairs) | Hwang et al. (2015) `aligned-good` (0.67 threshold), cleaned as in Ambati et al. (2016) | System 2 readability ranking |
| `wiki_pairs.example.tsv` | 5-row sample of the above | — | Schema tests |

## Lexical resources

| Path | What it is | Source | Used for |
|------|------------|--------|----------|
| `enwiki_word_frequency.txt` | English Wikipedia unigram counts, `word count` per line | github.com/IlyaSemenov/wikipedia-word-frequency, `enwiki-2023-04-13.txt` | Wikipedia frequency (log10) |
| `aoa_kuperman_2012.csv` | Age-of-acquisition ratings for 30,121 content words (`Rating.Mean`, `Rating.SD`) | Kuperman, Stadthagen-Gonzalez & Brysbaert (2012) | AoA mean and SD — **the file the pipeline reads** |
| `aoa_kuperman_51715_full.csv` | The full Kuperman release, 51,715 words (`AoA_Kup`, `AoA_Kup_lem`, `Perc_known`, SUBTLEX frequency, POS) | Kuperman et al. (2012) supplement | Not read by the pipeline: it has wider coverage but **no per-word SD**, and the paper uses both mean and SD |

## Parser training data

| Path | What it is | Used for |
|------|------------|----------|
| `nltk_data/corpora/treebank/` | NLTK's free WSJ sample, 3,914 gold trees (sections 00–01) | Fallback grammar when `penn_treebank_3/` is absent |
| `nltk_data/corpora/conll2000/` | CoNLL-2000, 259k POS-tagged WSJ tokens (sections 15–18, 20) | Word emissions of the fallback grammar only |

`load_grammar` in [`../src/eye_tracking_c/parser/pcfg.py`](../src/eye_tracking_c/parser/pcfg.py)
prefers `penn_treebank_3/` and falls back to `nltk_data/` automatically. With the full
treebank it trains on 39,832 trees — the paper's WSJ 02–21 — instead of 3,914, and it
takes the word emissions from those trees rather than from CoNLL-2000, which would
otherwise double-count sections 15–18 and 20.

The remaining deviation from the paper is the **GCG re-annotation** used by
van Schijndel et al. (2013)'s ModelBlocks parser. This repo trains on the plain
Penn Treebank labels.

## Not used

| Path | Why not |
|------|---------|
| `gpt2_total_surprisal_dundee.csv` | GPT-2 surprisal from an earlier attempt; the PCFG parser supplies surprisal now. Kept for reference |

Files in the source Dataset folder that are deliberately **not** copied here:
`PTB/ptb.{train,valid,test}.txt` (Mikolov language-model preprocessing — lowercased,
`<unk>`, numbers as `N`, and no trees, so it cannot train a PCFG) and
`Wiki/aligned-good_partial(0.53).gz` (a lower alignment threshold than the 0.67 set
the paper's comparators use).

## Notes

Kuperman et al. rated content words only, so function words have no AoA value.
They are mean-imputed before standardization.
