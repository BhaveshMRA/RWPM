# Beyond Exact Copies: Detecting Edited Copies of AI Agent Skills

SSW 625 (AI for Software Engineering), Group 8: Bhavesh Maurya, Abhishek Thakkar, Mahima Jain, Pratiksha Vilas Pawar.

Agent skills (`SKILL.md` files) are reused by copying them between repositories, often with edits on the way. This repository holds our code, labels and results for detecting such edited copies in the GitSkills dataset and separating them from skills that only share a template. It contains:

- the scripts of the whole study, from extraction to the figures in the paper;
- the 400 labelled skill pairs (both annotators' sheets, adjudication and gold set) and the viewer we labelled them with;
- the generated tables and figures that the paper includes.

## Repository layout

```
code/                    scripts, run in the order given below
labeling/
  GUIDELINES.md          labelling rubric (derived / template sibling / unrelated)
  pairs.csv              the 400 sampled pairs, by content hash, with their similarity band
  candidates.csv         frozen candidate pairs of the sample run (8,084 pairs)
  sheet_<name>.csv       each annotator's labels
  adjudication.csv       decisions on the pairs where two annotators disagreed
  gold.csv               final labels used for training and evaluation
  html/                  side-by-side viewer, one complete page per pair
  validation/            kit for checking predicted links on the full dataset (created by step 15)
paper/
  results_full/          results of the full-dataset run: RQ1-RQ3, statistics, provenance, validation, figures
  results_sample_final/  final results of the sample-scale run (sample macros used in Sections 5-6 of the paper)
  results_phase3_semantic/  embedding experiment (Phase 3, not part of the final model)
  results/               default output folder of the scripts (currently a copy of results_sample_final/)
requirements-full.txt    pinned versions used for all runs
requirements-lock.txt    complete environment of our Windows machine (sample runs, embedding experiment)
requirements-mac.txt     complete environment of the machine that ran the full-dataset pipeline
```

## Setup

We used Python 3.14.2. Create a virtual environment and install the pinned versions:

```bash
python3.14 -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-full.txt
```

`requirements-full.txt` covers everything except the optional embedding experiment (step 5b), which also needs `sentence-transformers`; `requirements-lock.txt` lists our complete environment.

Two datasets are used:

- **Official sample** (13,000 distinct contents, same schema as the full release). Labels were drawn from it.
  ```bash
  git clone https://github.com/giuseppedestefanis/gitskills-sample
  cd gitskills-sample && unzip agent_skills_sample.zip && cd ..
  ```
- **Full release**, `agent_skills_release.db` (41 GB), from Zenodo: https://doi.org/10.5281/zenodo.21875637.

All scripts run from inside `code/`. They read and write three folders, which can be changed with environment variables:

| Variable | Default | Holds |
|---|---|---|
| `SKILL_WORK` | `work/` | intermediate files (large, never committed) |
| `SKILL_LABELS` | `labeling/` | pairs, sheets, gold set |
| `SKILL_RESULTS` | `paper/results/` | tables, macros, figures |

## Part 1: sample run (labelling and evaluation)

| Step | Command | Writes |
|---|---|---|
| 1 | `python 01_extract.py --db ../gitskills-sample/agent_skills_sample.db` | work/skills.pkl, work/occurrences.pkl |
| 2 | `python 02_candidates.py` | work/candidates.pkl (loaded from labeling/candidates.csv) |
| 3 | already done; do not run again | labeling/pairs.csv, sheets |
| 3b | `python 03b_render_viewer.py` | labeling/html/ (complete pages, no resampling) |
| 4 | `python 04_agreement.py` | labeling/adjudication.csv, labeling/gold.csv |
| 5 | `python 05_features.py --pairs gold`, then `python 05_features.py --pairs all` | work/features_gold.pkl, work/features_all.pkl |
| 6 | `python 06_rq1_rq2.py` | RQ1 and RQ2 tables, out-of-fold predictions |
| 7 | `python 07_rq3.py` | RQ3 table and macros |
| 8 | `python 08_stats.py` | McNemar's test, bootstrap confidence intervals |
| 9 | `python 09_whitebox.py` | fig_pair_features.pdf, fig_model.pdf |

Notes:

- Step 2 recomputes candidates only with `--rebuild`. By default it reloads the frozen record in `labeling/candidates.csv`, because the LSA blocker uses randomised SVD whose last digits differ between machines; the frozen record keeps the candidate set identical everywhere.
- Step 3 drew the labelled sample. Running it again would overwrite the annotators' sheets, so it refuses unless `--force` is given.
- If step 4 reports disagreements, fill `final_label` (D, T or U) for each row of `labeling/adjudication.csv` and run it again. Until then it removes `gold.csv`, so later steps cannot run on stale labels.
- Step 5b (`05b_embeddings.py`) is optional: it reproduces our Phase 3 experiment with multilingual sentence embeddings, which did not improve the classifier. Its results are in `paper/results_phase3_semantic/`. If `work/embeddings.npz` exists, steps 5-9 include the embedding features, so delete or rename that file to reproduce the final model.

## Part 2: full-dataset run

The full release does not fit in memory, so steps 10-13 stream it in chunks and produce the same files that steps 6-9 read. The labelled pairs stay the same: the official sample is a subset of the release.

Needs about 80 GB of free disk and 16 GB of RAM. Set the folders first (in every new terminal):

```bash
export SKILL_WORK=~/gitskills/work_full
export SKILL_RESULTS=../paper/results_full
mkdir -p "$SKILL_WORK" "$SKILL_RESULTS"
# Windows PowerShell: $env:SKILL_WORK="C:\gitskills\work_full"; $env:SKILL_RESULTS="..\paper\results_full"
```

| Step | Command | What it does | Time (MacBook Pro) |
|---|---|---|---|
| 10 | `python 10_full_extract.py --db ~/gitskills/agent_skills_release.db` | streams the release with the filters of step 1 | 1-2 h |
| 11 | `python 11_full_index.py` | boilerplate lines over the whole corpus, MinHash signatures | 1-2 h |
| 12 | `python 12_full_candidates.py` | same-name and MinHash candidates | 10-40 min |
| 13 | `python 13_full_features.py --pairs gold` | features of the 400 gold pairs with full-corpus statistics | 10-20 min |
| 13 | `python 13_full_features.py --pairs all --workers 6` | features of every candidate pair, in parallel | 1-3 h |
| 6-9 | `python 06_rq1_rq2.py`, `python 07_rq3.py`, `python 08_stats.py`, `python 09_whitebox.py` | as in Part 1 | 30-60 min |
| 14 | `python 14_provenance.py --db ~/gitskills/agent_skills_release.db` | provenance evidence for the labelled pairs | 1 min |
| 15 | `python 15_validate_edges.py --make --annotators Bhavesh,Abhishek,Mahima,Pratiksha` | validation kit for predicted links | minutes |
| 16 | `python 16_edge_diagnostics.py` | where the predicted links come from: bands, names, hubs, largest components | 10-20 min |
| 7 | `python 07_rq3.py --same-name-only` | conservative RQ3 variant: families from same-name links only (files `*_samename*`) | 10-20 min |

Differences from the sample run, all deliberate:

- **No LSA blocker.** At this scale we generate candidates with MinHash and same-name blocking only. In the labelled sample, the pairs found only by LSA (band B6) contain no derived pair.
- **Bucket caps.** A MinHash bucket with more than `--max-bucket` skills (default 50) is not expanded into all pairs; each member is paired with the `--anchors` lowest-id members (default 5). Large catalogs of near-identical skills would otherwise produce billions of pairs.
- **Corpus statistics.** Boilerplate lines are counted over all skills. TF-IDF, LSA and normalisation models are fitted once on 50,000 randomly chosen skills plus every gold skill (`--fit-size`), then applied to all pairs.

On the sample database, steps 10-13 reproduce the features of steps 1-5 exactly (same 11,009 skills, 2,476 boilerplate lines, 4,352 MinHash pairs, identical gold features), so the two runs differ only in scale.

If step 12 prints more than about 40 million candidate pairs, rerun it with `--max-bucket 30 --anchors 3` before step 13. If step 13 runs out of memory, lower `--workers`, or delete `work_full/full_models.pkl` and use a smaller `--fit-size`.

## Labelling

**Gold set (400 pairs).** Every pair was labelled by two of us independently with the rubric in `labeling/GUIDELINES.md`, using the viewer in `labeling/html/`. A pair is *derived* when the two skills share distinctive content that is unlikely to be written twice independently (renamed and translated copies included); *template* when only generic scaffolding is shared; *unrelated* otherwise. Disagreements were settled in discussion (`labeling/adjudication.csv`).

**Validation of predicted links (full dataset).** Step 15 draws 80 links the model predicts on the full dataset and 40 candidate pairs it predicts as not derived, shuffled together. Each annotator labels their sheet in `labeling/validation/` with the same rubric and viewer. Do not open `labeling/validation/pairs.csv` while labelling: it shows which pairs the model predicted. Afterwards:

```bash
python 15_validate_edges.py --score
```

reports the agreement, the precision of the predicted links with a 95% confidence interval and the share of derived pairs among the decoys. Disagreements are settled through `labeling/validation/adjudication.csv`, as for the gold set.

## Reproducibility

- Versions are pinned in `requirements-full.txt`; all random choices use a fixed seed (625).
- Candidate pairs of the sample run are frozen in `labeling/candidates.csv`; the labelled sample is stored by content hash in `labeling/pairs.csv`.
- Threshold tuning rounds scores before choosing a threshold, so tiny floating-point differences between machines cannot change baseline decisions.
- The labelling viewer shows complete skills; `03b_render_viewer.py` rebuilds it from `pairs.csv` without resampling.
- Fenced code blocks are parsed as in CommonMark (fence length and indentation respected, unclosed fences run to the end).
- The sample keeps each skill in at most two candidate pairs and each name in at most four; the 20 random pairs were drawn without the skill limit, so one skill appears in a third pair (P0044, P0145, P0096).
- The version of the code and results reported in the Phase 3 paper is tagged `phase3`.

## Data

GitSkills: Destefanis, Graziotin, Vaccargiu and Ortu, *GitSkills: A Dataset of Agent Skills on GitHub*, MSR 2027 (arXiv:2608.10906); release at https://doi.org/10.5281/zenodo.21875637; official sample at https://github.com/giuseppedestefanis/gitskills-sample.


## Use of AI tools

We used an AI assistant (Claude Opus 5.5, Anthropic) for brainstorming the research direction; for searching the literature, which led us to the Skill Constellations study (Seddik, arXiv:2610.11169); for help in debugging the analysis scripts, including the full-dataset scripts, and assisting with the approach overview figure; for a review of labels and notes after the annotators had finished labelling, which changed some confidence values and notes but no labels; and for suggestions during the adjudication of disagreements, where the final decisions were ours. All labels were assigned by the authors, and the authors take full responsibility for the content of this paper.
