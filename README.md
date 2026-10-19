# SSW 625 Group 8:  Detecting Edited Copies of AI Agent Skills

## 1. Setup

```bash
pip install -r code/requirements.txt          # Python 3.9+
git clone https://github.com/giuseppedestefanis/gitskills-sample
cd gitskills-sample && unzip agent_skills_sample.zip && cd ..
```

## 2. Run order

Run from inside `code/`. Outputs go to `../work`, `../labeling`, `../paper/results`.

| Step | Command | Writes |
|---|---|---|
| 1 | `python 01_extract.py --db ../gitskills-sample/agent_skills_sample.db` | work/skills.pkl, work/extracted.db |
| 2 | `python 02_candidates.py` | work/candidates.pkl |
| 3 | already done: the kit is in `labeling/`. No need to run it. | |
| 4 | label (see below), then `python 04_agreement.py` | labeling/adjudication.csv |
| 4b | If require, fill `final_label` in adjudication.csv, run `python 04_agreement.py` again | labeling/gold.csv |
| 5 | `python 05_features.py --pairs gold` and `python 05_features.py --pairs all` | work/features_*.pkl |
| 6 | `python 06_rq1_rq2.py` | RQ1 + RQ2 tables |
| 7 | `python 07_rq3.py` | RQ3 table |

Steps 1-3 are deterministic: we get exactly 11,009 skills, 8,084 candidate pairs and
the same 400 pairs as in the shipped kit. Steps 1 and 2 must be run on your machine because
steps 5-7 need the `work/` files.

## 3. Labeling

1. Read `labeling/GUIDELINES.md` together and agree on it.
2. Each person opens `labeling/html/index.html` in a browser and fills **their own**
   `labeling/sheet_<name>.csv`: `label` = D (derived), T (template sibling), U (unrelated);
   `confidence` = 1-3; optional `note`. Each pair is assigned to two people
   (Bhavesh + Abhishek, Mahima + Pratiksha), 200 pairs each.
3. Do not look at each other's sheets until everyone is done.
4. Run step 4, discuss each disagreement, fill `final_label`, if require, then run step 4 again.
5. The "Synthesize" paragraph in Section 6 is the description of why pairs look similar.

`labeling/pairs.csv` holds the similarity band of each pair.

## Full dataset (Phase 3)

`01_extract.py` also takes the full SQLite file. Use `--sha-max 1b` for a uniform
~10% subset of distinct skills (the same hash-prefix idea the dataset authors used).
