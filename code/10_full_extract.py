"""Step 10 (full dataset) - Streaming extraction of the working set.

  python 10_full_extract.py --db /path/to/agent_skills_release.db

Applies exactly the scope filters of step 1 (exact basename SKILL.md, valid front matter, body >= MIN_BODY_CHARS after re-parsing) to the full GitSkills release, but streams the file contents in chunks instead of loading them all into memory. Writes to the work folder (set SKILL_WORK to keep it apart from the sample run):
  full_skills.db              one row per working-set skill (text + metadata)
  skills.pkl                  light table for steps 7 and 9 (no text except gold skills)
  occurrences.pkl             occurrences of working-set skills
  population_occurrences.pkl  occurrences of the in-scope population and results/macros_full_extract.tex with the full-dataset counts.
"""
import argparse
import json
import posixpath
import re
import sqlite3
import time

import pandas as pd

from common import (LABEL_DIR, MIN_BODY_CHARS, RESULTS, WORK, fmt_int, headings, norm_name, save_json, split_code, split_frontmatter)

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True)
ap.add_argument("--chunk", type=int, default=2000)
args = ap.parse_args()

t0 = time.time()
WORK.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
stats = {}

# 1. all occurrences (no content): same funnel as step 1
occ = pd.read_sql_query(
    "SELECT file_sha AS sha, repo_full_name AS repo, filename, location_class, "
    "dedup_primary, frontmatter_valid, body_chars FROM artifacts", con)
occ["owner"] = occ["repo"].str.split("/").str[0]
stats["occurrences_total"] = len(occ)
stats["distinct_contents_total"] = int(occ["sha"].nunique())
stats["repos_total"] = int(occ["repo"].nunique())
stats["owners_total"] = int(occ["owner"].nunique())
reps = occ[occ.dedup_primary == 1]
funnel = [("Distinct contents (representatives)", len(reps))]
reps = reps[reps.filename == "SKILL.md"]
funnel.append(("Exact basename SKILL.md", len(reps)))
reps = reps[reps.frontmatter_valid == 1]
funnel.append(("Valid YAML front matter", len(reps)))
reps = reps[reps.body_chars.fillna(0) >= MIN_BODY_CHARS]
funnel.append((f"Body >= {MIN_BODY_CHARS} characters", len(reps)))
pop_shas = set(reps.sha)
del reps
pop_occ = occ[occ.sha.isin(pop_shas)][["sha", "repo", "owner", "location_class"]].reset_index(drop=True)
del occ
pop_occ.to_pickle(WORK / "population_occurrences.pkl")
stats["population_contents"] = len(pop_shas)
stats["population_occurrences"] = len(pop_occ)
owners_of = pop_occ.groupby("sha")["owner"].agg(lambda s: sorted(set(s))).to_dict()
print(f"population: {len(pop_shas):,} contents, {len(pop_occ):,} occurrences ({time.time() - t0:.0f}s)")

# 2. stream the representatives with text into full_skills.db
out_db = WORK / "full_skills.db"
out_db.unlink(missing_ok=True)
out = sqlite3.connect(out_db)
out.execute("""CREATE TABLE skills (id INTEGER PRIMARY KEY, sha TEXT, repo TEXT, path TEXT,
               location_class TEXT, name TEXT, nname TEXT, description TEXT, body TEXT,
               fm_fields INTEGER, code_langs TEXT, headings TEXT, dirname TEXT, owner TEXT,
               owners TEXT, first_commit_at TEXT, siblings TEXT DEFAULT '[]')""")
cur = con.execute(
    "SELECT file_sha, repo_full_name, path, location_class, name, description, content, first_commit_at "
    "FROM artifacts WHERE dedup_primary = 1 AND filename = 'SKILL.md' AND frontmatter_valid = 1 "
    "AND content IS NOT NULL")
fence_lang = re.compile(r"^ {0,3}(?:`{3,}|~{3,})[ \t]*([\w+#-]+)", re.M)
n = 0
while True:
    rows = cur.fetchmany(args.chunk)
    if not rows:
        break
    batch = []
    for sha, repo, path, loc, name, desc, content, first in rows:
        if sha not in pop_shas:
            continue
        fm, body = split_frontmatter(content)
        if len(body) < MIN_BODY_CHARS:
            continue
        prose, _ = split_code(body)
        batch.append((n, sha, repo, path, loc, name, norm_name(name), "" if desc is None else str(desc), body,
                      len(re.findall(r"^[A-Za-z][\w-]*\s*:", fm, re.M)),
                      json.dumps(sorted(set(x.lower() for x in fence_lang.findall(body)))),
                      json.dumps(headings(prose)),
                      posixpath.basename(posixpath.dirname(path or "")).lower(),
                      repo.split("/")[0], json.dumps(owners_of.get(sha, [])), first))
        n += 1
    out.executemany("INSERT INTO skills (id, sha, repo, path, location_class, name, nname, description, body, "
                    "fm_fields, code_langs, headings, dirname, owner, owners, first_commit_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
    out.commit()
    print(f"\r{n:,} skills extracted ({time.time() - t0:.0f}s)", end="", flush=True)
print()
if n != funnel[-1][1]:
    funnel.append(("Body still >= threshold after re-parsing front matter", n))
out.execute("CREATE INDEX idx_sha ON skills(sha)")

# 3. bundled file names of each representative (streamed)
key_to_id = {(r, p): i for i, r, p in out.execute("SELECT id, repo, path FROM skills")}
sib = {}
cur = con.execute("SELECT repo_full_name, artifact_path, entry_name FROM artifact_siblings")
while True:
    rows = cur.fetchmany(200_000)
    if not rows:
        break
    for repo, path, entry in rows:
        i = key_to_id.get((repo, path))
        if i is not None and entry and entry.lower() != "skill.md":
            sib.setdefault(i, set()).add(entry.lower())
out.executemany("UPDATE skills SET siblings = ? WHERE id = ?", [(json.dumps(sorted(v)), i) for i, v in sib.items()])
out.commit()
del key_to_id

# 4. light table for steps 7 and 9, and working-set occurrences
light = pd.read_sql_query("SELECT id, sha, nname, owner, first_commit_at FROM skills ORDER BY id", out)
gold_shas = set()
if (LABEL_DIR / "gold.csv").exists():
    g = pd.read_csv(LABEL_DIR / "gold.csv")
    gold_shas = set(g.sha_a) | set(g.sha_b)
light["body"] = ""
light["name"] = ""
light["description"] = ""
if gold_shas:
    q = ",".join("?" * len(gold_shas))
    gtxt = pd.read_sql_query(f"SELECT id, body, name, description FROM skills WHERE sha IN ({q})", out,
                             params=list(gold_shas)).set_index("id")
    for col in ("body", "name", "description"):
        light.loc[gtxt.index, col] = gtxt[col].values
light = light.drop(columns=["id"])
light.to_pickle(WORK / "skills.pkl")
w_shas = set(light.sha)
w_occ = pop_occ[pop_occ.sha.isin(w_shas)].reset_index(drop=True)
w_occ.to_pickle(WORK / "occurrences.pkl")

stats["funnel"] = funnel
stats["working_contents"] = n
stats["working_occurrences"] = len(w_occ)
stats["working_repos"] = int(w_occ["repo"].nunique())
stats["working_owners"] = int(w_occ["owner"].nunique())
stats["working_with_history"] = int(light["first_commit_at"].notna().sum())
stats["working_with_siblings"] = len(sib)
stats["distinct_names_working"] = int(light["nname"].nunique())
stats["names_with_2plus_contents"] = int((light[light.nname != ""].groupby("nname").size() >= 2).sum())
save_json(stats, RESULTS / "full_extract_stats.json")
words = ["Zero", "One", "Two", "Three", "Four"]
mac = {"ResFullOccTotal": stats["occurrences_total"], "ResFullContentsTotal": stats["distinct_contents_total"],
       "ResFullReposTotal": stats["repos_total"], "ResFullOwnersTotal": stats["owners_total"],
       "ResFullWorkContents": n, "ResFullWorkOcc": stats["working_occurrences"],
       "ResFullWorkRepos": stats["working_repos"], "ResFullWorkOwners": stats["working_owners"],
       "ResFullWorkHistory": stats["working_with_history"], "ResFullWorkSiblings": stats["working_with_siblings"],
       "ResFullNamesTwoPlus": stats["names_with_2plus_contents"]}
for k, (_, v) in enumerate(funnel[:5]):
    mac[f"ResFullFunnel{words[k]}"] = v
(RESULTS / "macros_full_extract.tex").write_text(
    "\n".join(rf"\renewcommand{{\{k}}}{{{fmt_int(v)}}}" for k, v in mac.items()) + "\n")
for k, v in stats.items():
    print(f"{k}: {v}")
print(f"done in {time.time() - t0:.0f}s")
