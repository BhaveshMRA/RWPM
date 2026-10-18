# SSW 625 Group 8: Recovering the lineage of reused agent skills

```
paper/      Overleaf project (acmart). Upload this folder as-is.
  main.tex, macros_defaults.tex, references.bib
  sections/   01-05 = revised Phase 1, 06-08 = Phase 2
  figures/    approach_overview.pdf (used by the paper), .png (for slides), .tex (TikZ source)
  results/    tables + numbers written by the scripts; the paper reads them automatically
code/       pipeline, steps 01-07
labeling/   ready-to-use labeling kit (400 pairs from the official sample)
```

## 1. Overleaf

Create a new blank project, then upload the whole `paper/` folder (or zip it and use
"Upload Project"). It compiles as-is. Anything in **red [brackets]** is either a number the
scripts have not produced yet or a sentence you must write. Dataset numbers (Sections 5-6)
are already filled in from the official sample.

After you run steps 04-07, upload the new files from `paper/results/` to Overleaf's
`results/` folder and recompile. Tables and numbers fill in by themselves.

## 2. Setup

```bash
pip install -r code/requirements.txt          # Python 3.9+
git clone https://github.com/giuseppedestefanis/gitskills-sample
cd gitskills-sample && unzip agent_skills_sample.zip && cd ..
```

## 3. Run order (official sample: a few minutes in total)

Run from inside `code/`. Outputs go to `../work`, `../labeling`, `../paper/results`.

| Step | Command | Time | Writes |
|---|---|---|---|
| 1 | `python 01_extract.py --db ../gitskills-sample/agent_skills_sample.db` | ~10 s | work/skills.pkl, work/extracted.db |
| 2 | `python 02_candidates.py` | ~40 s | work/candidates.pkl |
| 3 | already done: the kit is in `labeling/`. Do **not** re-run. | | |
| 4 | label (see below), then `python 04_agreement.py` | | labeling/adjudication.csv |
| 4b | fill `final_label` in adjudication.csv, run `python 04_agreement.py` again | | labeling/gold.csv |
| 5 | `python 05_features.py --pairs gold` and `python 05_features.py --pairs all` | ~1.5 min each | work/features_*.pkl |
| 6 | `python 06_rq1_rq2.py` | ~1 min | RQ1 + RQ2 tables |
| 7 | `python 07_rq3.py` | ~30 s | RQ3 table |

Steps 1-3 are deterministic: you will get exactly 11,009 skills, 8,085 candidate pairs and
the same 400 pairs as in the shipped kit. Steps 1 and 2 must be run on your machine because
steps 5-7 need the `work/` files.

## 4. Labeling (the professor's required manual analysis)

1. Read `labeling/GUIDELINES.md` together and agree on it.
2. Each person opens `labeling/html/index.html` in a browser and fills **their own**
   `labeling/sheet_<name>.csv`: `label` = D (derived), T (template sibling), U (unrelated);
   `confidence` = 1-3; optional `note`. Each pair is assigned to two people
   (Bhavesh + Abhishek, Mahima + Pratiksha), 200 pairs each.
3. Do not look at each other's sheets until everyone is done.
4. Run step 4, discuss each disagreement, fill `final_label`, run step 4 again.
5. Keep notes while labeling: why pairs look similar. These become the "Synthesize"
   paragraph in Section 6.

`labeling/pairs.csv` holds the similarity band of each pair; don't open it while labeling.

## 5. What you still have to write in the paper

- Section 6, Explore: what each of you read; confirm the three template examples.
- Section 6, Synthesize: your list of reasons why pairs look similar.
- Section 7: error analysis (use `paper/results/rq1_oof_predictions.csv`) and one-line
  answers to RQ1-RQ3.

## Full dataset (Phase 3)

`01_extract.py` also takes the full 41 GB SQLite file. Use `--sha-max 1b` for a uniform
~10% subset of distinct skills (the same hash-prefix idea the dataset authors used).
