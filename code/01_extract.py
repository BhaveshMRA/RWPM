"""Step 1 - Dataset extraction.

Reads the GitSkills SQLite file (the 277 MB official sample or the full
41 GB file) and writes:
  work/skills.pkl       one row per in-scope distinct skill (the representative),
                        split into channels: name, description, prose, code, headings
  work/occurrences.pkl  every file occurrence of an in-scope skill (sha, repo, owner)
  work/extracted.db     the same two tables as SQLite (skills, occurrences)
  work/extract_stats.json  numbers for the Dataset Extraction section

Scope (population P): representatives whose exact basename is SKILL.md, whose
front matter is valid, and whose body has at least MIN_BODY_CHARS characters.
Working set W: P, optionally restricted to file_sha < --sha-max (a uniform,
reproducible hash-prefix subset, the same idea the dataset authors used for
their sample). Use --sha-max only on the full dataset.

Usage:
  python 01_extract.py --db agent_skills_sample.db
  python 01_extract.py --db gitskills.db --sha-max 1b      # ~10.5% of P
"""
import argparse
import json
import posixpath
import re
import sqlite3
import time

import pandas as pd

from common import (MIN_BODY_CHARS, WORK, headings, norm_name, save_json,
                    split_code, split_frontmatter)

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--sha-max", default=None,
                help="keep file_sha < this hex prefix (e.g. 1b), full dataset only")
args = ap.parse_args()

t0 = time.time()
con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
stats = {}

# ---- 1. all occurrences (no content) ---------------------------------------
occ = pd.read_sql_query(
    "SELECT file_sha AS sha, repo_full_name AS repo, filename, location_class, "
    "dedup_primary, frontmatter_valid, body_chars FROM artifacts", con)
occ["owner"] = occ["repo"].str.split("/").str[0]
stats["occurrences_total"] = len(occ)
stats["distinct_contents_total"] = occ["sha"].nunique()
stats["repos_total"] = occ["repo"].nunique()
stats["owners_total"] = occ["owner"].nunique()

reps = occ[occ.dedup_primary == 1]
funnel = [("Distinct contents (representatives)", len(reps))]
reps = reps[reps.filename == "SKILL.md"]
funnel.append(("Exact basename SKILL.md", len(reps)))
reps = reps[reps.frontmatter_valid == 1]
funnel.append(("Valid YAML front matter", len(reps)))
reps = reps[reps.body_chars.fillna(0) >= MIN_BODY_CHARS]
funnel.append((f"Body >= {MIN_BODY_CHARS} characters", len(reps)))
pop_shas = set(reps.sha)

pop_occ = occ[occ.sha.isin(pop_shas)][["sha", "repo", "owner", "location_class"]]
pop_occ.to_pickle(WORK / "population_occurrences.pkl")
stats["population_contents"] = len(pop_shas)
stats["population_occurrences"] = len(pop_occ)

# ---- 2. representatives of the working set, with text ------------------------
where = ("dedup_primary = 1 AND filename = 'SKILL.md' AND frontmatter_valid = 1 "
         "AND content IS NOT NULL")
params = []
if args.sha_max:
    where += " AND file_sha < ?"
    params.append(args.sha_max.lower())
skills = pd.read_sql_query(
    "SELECT file_sha AS sha, repo_full_name AS repo, path, location_class, name, "
    "description, content, first_commit_at, history_fetched, has_scripts, "
    f"has_references FROM artifacts WHERE {where}", con, params=params)
skills = skills[skills.sha.isin(pop_shas)].copy()

fm_body = skills["content"].map(split_frontmatter)
skills["body"] = fm_body.str[1]
skills["fm_fields"] = fm_body.str[0].map(
    lambda fm: len(re.findall(r"^[A-Za-z][\w-]*\s*:", fm, re.M)))
skills["code_langs"] = skills["body"].map(
    lambda b: sorted(set(x.lower() for x in re.findall(r"^\s*(?:```|~~~)\s*([\w+#-]+)", b, re.M))))
skills = skills[skills["body"].str.len() >= MIN_BODY_CHARS].copy()
pc = skills["body"].map(split_code)
skills["prose"] = pc.str[0]
skills["code_blocks"] = pc.str[1]
skills["headings"] = skills["prose"].map(headings)
skills["nname"] = skills["name"].map(norm_name)
skills["description"] = skills["description"].fillna("").astype(str)
skills["dirname"] = skills["path"].map(lambda p: posixpath.basename(posixpath.dirname(p)).lower())
skills["owner"] = skills["repo"].str.split("/").str[0]
skills = skills.drop(columns=["content"]).reset_index(drop=True)
if args.sha_max:
    funnel.append((f"Hash-prefix working subset (sha < {args.sha_max})", len(skills)))
elif len(skills) != funnel[-1][1]:
    funnel.append(("Body still >= threshold after re-parsing front matter", len(skills)))

# ---- 3. bundled file names of each representative ----------------------------
sib = pd.read_sql_query(
    "SELECT repo_full_name AS repo, artifact_path AS path, entry_name FROM artifact_siblings",
    con)
sib = sib.merge(skills[["repo", "path"]], on=["repo", "path"])
sib_map = sib.groupby(["repo", "path"])["entry_name"].apply(
    lambda s: sorted(set(x.lower() for x in s if x.lower() != "skill.md")))
skills["siblings"] = [sib_map.get((r, p), []) for r, p in zip(skills.repo, skills.path)]

w_shas = set(skills.sha)
w_occ = pop_occ[pop_occ.sha.isin(w_shas)]
skills.to_pickle(WORK / "skills.pkl")
w_occ.to_pickle(WORK / "occurrences.pkl")

# Same extraction as a small SQLite database (lists stored as JSON text).
out_db = WORK / "extracted.db"
out_db.unlink(missing_ok=True)
with sqlite3.connect(out_db) as xcon:
    flat = skills.copy()
    for col in ("code_blocks", "headings", "code_langs", "siblings"):
        flat[col] = flat[col].map(json.dumps)
    flat.to_sql("skills", xcon, index=False)
    w_occ.to_sql("occurrences", xcon, index=False)
    xcon.execute("CREATE INDEX idx_occ_sha ON occurrences(sha)")

stats["funnel"] = funnel
stats["working_contents"] = len(skills)
stats["working_occurrences"] = len(w_occ)
stats["working_repos"] = w_occ["repo"].nunique()
stats["working_owners"] = w_occ["owner"].nunique()
stats["working_with_history"] = int(skills["first_commit_at"].notna().sum())
stats["working_location_class"] = skills["location_class"].value_counts().to_dict()
stats["working_with_siblings"] = int((skills["siblings"].str.len() > 0).sum())
stats["distinct_names_working"] = int(skills["nname"].nunique())
stats["names_with_2plus_contents"] = int((skills.groupby("nname").size() >= 2).sum())
save_json(stats, WORK / "extract_stats.json")

from common import RESULTS, fmt_int  # noqa: E402
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven"]
mac = {"ResOccTotal": stats["occurrences_total"], "ResContentsTotal": stats["distinct_contents_total"],
       "ResReposTotal": stats["repos_total"], "ResOwnersTotal": stats["owners_total"],
       "ResWorkContents": stats["working_contents"], "ResWorkOcc": stats["working_occurrences"],
       "ResWorkRepos": stats["working_repos"], "ResWorkOwners": stats["working_owners"],
       "ResWorkHistory": stats["working_with_history"], "ResWorkSiblings": stats["working_with_siblings"],
       "ResNamesTwoPlus": stats["names_with_2plus_contents"]}
for k, (_, v) in enumerate(funnel):
    mac[f"ResFunnel{WORDS[k]}"] = v
(RESULTS / "macros_extract.tex").write_text(
    "\n".join(rf"\renewcommand{{\{k}}}{{{fmt_int(v)}}}" for k, v in mac.items()) + "\n")

for k, v in stats.items():
    print(f"{k}: {v}")
print(f"done in {time.time() - t0:.0f}s")
