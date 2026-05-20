# hsemasters

**A century of French song lyrics, computationally analysed (1925–2025).**

This repository holds the code, notebooks and thesis for a Master's project at HSE on the thematic and emotional evolution of French-language popular song between 1925 and 2025. The pipeline builds a 63,652-song corpus from MusicBrainz and Genius, classifies every song by topic and mood using multilingual zero-shot NLI, and validates the outputs through template sensitivity, cross-model agreement, human annotation, calibration, bootstrap intervals and genre-adjusted regression.

The public-facing version of the findings lives at **[daniilvoloshin.com/chanson](https://daniilvoloshin.com/chanson)**.
The full thesis (PDF) is in this repo: **[Voloshin_D_M_thesis.pdf](./Voloshin_D_M_thesis.pdf)**.

---

## What's in here

Four Jupyter notebooks, run in order, plus the thesis.

| # | Notebook | What it does |
|---|---|---|
| 00 | `00_musicbrainz_french_artist_registry.ipynb` | Loads MusicBrainz dump tables (tracks, recordings, releases, artist credits, countries, languages, genres). Joins them into one candidate set and filters down to likely-French records using a composite score over four signals (language metadata, release country, artist country, title cue). Output: ~4.2M French-likely candidate rows. |
| 01 | `01_musicbrainz_enrichment_metadata_lyrics.ipynb` | Walks the candidate registry and queries Genius via `lyricsgenius` to attach lyric texts. Uses weighted title/artist similarity (65/35), chunked execution, checkpointing, retries and a pre-1990 backfill pass. Deduplicates by `recording_id`. Output: ~1.08M enriched rows, of which 81,823 have lyrics. |
| 02 | `02_lyrics_eda_and_nlp_cleaning.ipynb` | EDA on the working dataset (decade distribution, genre composition, lexical patterns), plus the cleaning pipeline: keep three text columns (`lyrics_raw`, `lyrics_normalized`, `lyrics_clean`) for auditability, strip section tags (`[Chorus]`, `[Couplet]`), metadata lines, URLs and encoding artefacts. Output: `nlp_final_dataset.csv`, 65,230 rows with usable lyrics. |
| 03 | `03_topic_mood.ipynb` | Topic and mood classification. Builds an additional `lyrics_dedup` column (collapse repeated lines), splits each song into overlapping 400-character chunks, runs two zero-shot NLI models against French hypothesis templates, computes cross-model agreement (Cohen's κ, Spearman ρ, Jaccard), calibrates an ensemble with an AND-gate confidence policy, and aggregates by decade with Mann–Kendall + Sen's slope and an artist-clustered bootstrap. Final classification corpus: **63,652 songs**. |

The thesis (`Voloshin_D_M_thesis.pdf`) carries the full methodology, validation results, limitations and interpretation. The notebooks carry the implementation.

---

## Pipeline at a glance

The corpus is a funnel: early stages prioritise coverage, later stages prioritise reliability.

```
121,327,699   MusicBrainz tracks (initial pool)
  6,892,156   France-related subset (country filter)
  4,186,516   Likely-French candidates (composite score ≥ 3)
  1,082,070   Genius-enriched records
     81,823   Records with lyrics retrieved
     66,711   Curated NLP working dataset
     65,230   Cleaned, non-empty lyrics
     63,652   Deduplicated, token-filtered → classification-ready
```

Each filter is documented in the relevant notebook and in Chapter 2 of the thesis.

---

## Methodology in one paragraph

There is no labelled training set for French song lyrics. The pipeline uses **zero-shot natural language inference** with two multilingual transformer models — `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` as the primary classifier (M1) and `joeddav/xlm-roberta-large-xnli` for cross-validation (M2). Each lyric is treated as a premise; each label is inserted into a French hypothesis template (e.g. *"Cette chanson évoque des thèmes liés à {}."*). Eight topic labels and eight mood labels are tracked across eleven decades. Outputs are validated through five independent diagnostics — template sensitivity, cross-model agreement, a 182-song human-annotated gold set, confidence calibration, and genre-adjusted logistic regression — and decade-level shares are tested with Mann–Kendall, Sen's slope and Benjamini–Hochberg FDR correction. Uncertainty intervals come from a 1,000-iteration artist-clustered bootstrap.

### Taxonomy

**Topics:** `amour et désir`, `rupture, perte et deuil`, `fête, plaisir et célébration`, `nostalgie, mémoire et temps qui passe`, `révolte, critique sociale et engagement`, `introspection, quête de sens et spiritualité`, `voyage, lieux et appartenance`, `vie quotidienne, famille et société`.

**Moods:** `joyeux`, `tendre`, `apaisé ou plein d'espoir`, `nostalgique`, `triste`, `colérique ou rebelle`, `angoissé`, `ironique ou désabusé` (mapped to positive / negative / ambivalent valence buckets for coarse cross-model checks).

---

## Headline findings

After validation, six of eight topic labels and five of eight mood labels are reliable enough to interpret. Within that validated set:

- **Anxiety (`angoissé`) rises sharply** — roughly +5.0 percentage points per decade, the largest single shift in the corpus.
- **Tenderness, joy and direct sadness all decline** (`tendre` −1.6 pp/decade, `joyeux` −2.2 pp/decade, `triste` −1.6 pp/decade). French lyrics do not become uniformly "more negative"; the *form* of negative affect changes — from sorrow to pressure.
- **Nostalgia is the most robust thematic trend** — +1.7 pp/decade, and it *strengthens* under genre control.
- **The apparent rise of protest is a genre artefact.** Unadjusted, `révolte` rises across the century. Once genre is controlled for, the trend reverses — the rise is driven by rap's growing share of the corpus, not by within-genre change.
- **Love loses dominance but not importance** — `amour et désir` declines (−1.97 pp/decade unadjusted) but ~70% of that decline is explained by genre composition.

Full results, including the genre-confound table and the per-decade composition charts, are in Chapter 3 of the thesis.

---

## Data and credentials

The notebooks expect:

- A local **MusicBrainz database dump** (the standard PostgreSQL dump; the loader in notebook 00 reads the relevant tables).
- A **Genius API token** in a `.env` file at the repo root as `GENIUS_ACCESS_TOKEN=...`.

Neither the dump nor the token is in this repository.

The notebooks write intermediate artefacts to `data/processed/musicbrainz/` (paths are configurable at the top of each notebook). Checkpoints (`*_checkpoint.txt`) make every long-running stage resumable.

---

## Running it

The pipeline was developed on a mix of local CPU (stages 00–02) and Google Colab with an A100 GPU (stage 03). The classification notebook loads models in `float16` to fit on a single GPU.

Key Python dependencies: `transformers`, `pandas`, `scikit-learn`, `statsmodels`, `pymannkendall`, `lyricsgenius`, `requests`, `beautifulsoup4`, `tqdm`, `matplotlib`, `seaborn`.

Run the notebooks in order (00 → 01 → 02 → 03). Stage 01 is the longest — millions of Genius calls with rate limits — but it is checkpointed and resumable. Stage 03 takes a few hours on an A100 if both models are run on the full corpus.

---

## Limitations (read before citing)

- **Coverage is uneven.** The 2010s alone hold 31,616 songs; the 1920s–1940s together hold 165. Decade-level comparisons must account for unequal samples; early-period claims are weaker than recent ones.
- **Genius bias.** Songs available on Genius skew popular, recent, and toward genres with strong online lyric communities. Older chanson, archival recordings and regional repertoire are under-represented.
- **Zero-shot classification is noisy.** Topic κ against the gold standard is 0.36; mood κ is 0.18 (0.52 on the five validated labels). Three mood labels (`colérique ou rebelle`, `ironique ou désabusé`, `apaisé ou plein d'espoir`) and two topic labels (`révolte`, `vie quotidienne`) failed cross-model validation and are not interpreted as standalone findings.
- **Genre composition shifts.** Any decade-level trend must be checked against changing genre shares before being read as cultural change.
- **A song classified as `angoissé` is a song whose lyrics carry textual signals associated with anxiety.** It is not a measurement of the artist's state, the listener's state, or French society's state.

These limitations are discussed in detail in §2.13 of the thesis.

---

## Citation

If you use the code, the corpus construction logic or the findings, please cite the thesis:

> Voloshin, D. M. (2025). *The Thematic and Emotional Evolution of French Song Lyrics, 1925–2025: A Computational NLP Approach.* Master's thesis, HSE University.

---

## Links

- **Public project:** [daniilvoloshin.com/chanson](https://daniilvoloshin.com/chanson)
- **Thesis PDF:** [Voloshin_D_M_thesis.pdf](./Voloshin_D_M_thesis.pdf)
- **MusicBrainz:** [musicbrainz.org](https://musicbrainz.org)
- **Genius / lyricsgenius:** [github.com/johnwmillr/LyricsGenius](https://github.com/johnwmillr/LyricsGenius)

---

## License

Code released for academic and research use. Lyrics retrieved via Genius remain the property of their respective rights holders and are not redistributed here.
