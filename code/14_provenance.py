"""Step 14 - Provenance evidence for the labelled pairs.

  python 14_provenance.py --db /path/to/agent_skills_release.db   (after step 4)

Our labels judge content against a rubric; they do not observe copying. This
step checks the labelled pairs against evidence that does not come from the
similarity of the two texts:
  * link: one file names a repository (owner/repo) or GitHub account in which
    the other skill occurs and the file's own skill does not;
  * attribution: one file names an explicit source - a phrase such as "based
    on", "adapted from", "forked from" or "credit" followed on the same line by
    a URL, an @handle or an owner/repo reference;
  * bundled file: the two skill folders share a bundled file whose name is not
    generic (README, LICENSE, ...);
  * history: both skills have a recorded first commit, so the pair can be
    ordered in time.
If derived labels reflect real copying, derived pairs should carry such
evidence far more often than template or unrelated pairs.
Writes results/provenance.tex, results/macros_provenance.tex and
results/provenance_pairs.csv.
"""
import argparse
import re
import sqlite3

import pandas as pd

from common import LABEL_DIR, RESULTS, fmt_pct

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True, help="the GitSkills database (sample or full release)")
args = ap.parse_args()

gold = pd.read_csv(LABEL_DIR / "gold.csv")
shas = sorted(set(gold.sha_a) | set(gold.sha_b))
con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
q = ",".join("?" * len(shas))
rep = pd.read_sql_query(f"SELECT file_sha AS sha, repo_full_name AS repo, path, content, first_commit_at "
                        f"FROM artifacts WHERE dedup_primary = 1 AND file_sha IN ({q})", con, params=shas)
occ = pd.read_sql_query(f"SELECT file_sha AS sha, repo_full_name AS repo FROM artifacts WHERE file_sha IN ({q})",
                        con, params=shas)
sib = pd.read_sql_query("SELECT s.repo_full_name AS repo, s.artifact_path AS path, s.entry_name "
                        "FROM artifact_siblings s JOIN (SELECT repo_full_name, path FROM artifacts "
                        f"WHERE dedup_primary = 1 AND file_sha IN ({q})) r "
                        "ON s.repo_full_name = r.repo_full_name AND s.artifact_path = r.path", con, params=shas)

text = dict(zip(rep.sha, rep.content.fillna("").str.lower()))
first = dict(zip(rep.sha, rep.first_commit_at))
repos = occ.groupby("sha").repo.apply(lambda s: {r.lower() for r in s}).to_dict()
owners = {s: {r.split("/")[0] for r in rs} for s, rs in repos.items()}
GENERIC = {"readme.md", "license", "license.md", "license.txt", ".ds_store", "skill.md", "readme", ".gitkeep",
           "references", "scripts", "assets", "examples", "templates"}
key = dict(zip(zip(rep.repo, rep.path), rep.sha))
files = sib.assign(sha=[key.get(k) for k in zip(sib.repo, sib.path)]).dropna(subset=["sha"])
files = files[~files.entry_name.str.lower().isin(GENERIC)].groupby("sha").entry_name.apply(
    lambda s: {x.lower() for x in s}).to_dict()
ATTR = re.compile(r"(\b(based on|adapted from|derived from|forked from|ported from|inspired by|"
                  r"original(ly)? (written |created |authored )?(by|from)|credits?:?|courtesy of)\b|"
                  r"沿襲|基于|改编自|出典)[^\n]{0,80}?(https?://|github\.com|@[\w-]{2,}|\b[\w.-]+/[\w.-]+\b)")


def link(x, y):
    """x's file names a repository or GitHub account of y that x itself does not live in."""
    t = text.get(x, "")
    other_repos = repos.get(y, set()) - repos.get(x, set())
    other_owners = {o for o in owners.get(y, set()) - owners.get(x, set()) if len(o) >= 3}
    return any(r in t for r in other_repos) or any(f"github.com/{o}" in t for o in other_owners)


rows = []
for r in gold.itertuples():
    a, b = r.sha_a, r.sha_b
    rows.append({"pair_id": r.pair_id, "label": r.label,
                 "link": link(a, b) or link(b, a),
                 "attribution": bool(ATTR.search(text.get(a, ""))) or bool(ATTR.search(text.get(b, ""))),
                 "bundled_file": bool(files.get(a, set()) & files.get(b, set())),
                 "history": pd.notna(first.get(a)) and pd.notna(first.get(b))})
E = pd.DataFrame(rows)
E.to_csv(RESULTS / "provenance_pairs.csv", index=False)

cls = ["derived", "template", "unrelated"]
labels = {"attribution": "Explicit attribution to a source", "link": "Names the other skill's repository or account",
          "bundled_file": "Shared non-generic bundled file", "history": "Both have a recorded first commit"}
tab = E.groupby("label")[list(labels)].mean().reindex(cls)
n = E.label.value_counts().reindex(cls)
rows = [r"\begin{tabular}{lccc}", r"\toprule",
        "Evidence & " + " & ".join(f"{c} ($n={n[c]}$)" for c in cls) + r" \\", r"\midrule"]
for k, lab in labels.items():
    if k == "history":
        rows.append(r"\midrule")
    rows.append(f"{lab} & " + " & ".join(fmt_pct(tab.loc[c, k]) for c in cls) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "provenance.tex").write_text("\n".join(rows) + "\n")
mac = {f"ResProv{k.title().replace('_', '')}{c.title()}": fmt_pct(tab.loc[c, k]) for k in labels for c in cls}
(RESULTS / "macros_provenance.tex").write_text(
    "\n".join(rf"\renewcommand{{\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
print(tab.round(3).to_string())
print("wrote provenance.tex, macros_provenance.tex, provenance_pairs.csv")
