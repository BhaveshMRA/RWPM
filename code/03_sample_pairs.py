"""Step 3 - Sample candidate pairs for manual labeling.

Draws a stratified sample across similarity bands (plus random easy negatives),
assigns every pair to two annotators, and writes a labeling kit:
  labeling/pairs.csv              pair list with assignment (keep private: has bands)
  labeling/sheet_<annotator>.csv  one sheet per annotator (blind to bands/scores)
  labeling/html/index.html        side-by-side viewer, one page per pair
  labeling/GUIDELINES.md          the labeling rubric

Usage: python 03_sample_pairs.py --annotators Bhavesh,Abhishek,Mahima,Pratiksha
"""
import argparse
import difflib
import html
import shutil

import numpy as np
import pandas as pd

from common import LABEL_DIR, RESULTS, SEED, WORK

DEFAULT_TARGETS = "B1:60,B2:80,B3:80,B4:80,B5:50,B6:30,B7:20"

ap = argparse.ArgumentParser()
ap.add_argument("--annotators", default="A,B,C,D",
                help="comma-separated, even count; consecutive names form labeling teams")
ap.add_argument("--targets", default=DEFAULT_TARGETS)
ap.add_argument("--max-per-skill", type=int, default=2)
ap.add_argument("--max-per-name", type=int, default=4)
ap.add_argument("--max-lines", type=int, default=300)
ap.add_argument("--force", action="store_true", help="overwrite an existing labeling kit")
args = ap.parse_args()
if (LABEL_DIR / "pairs.csv").exists() and not args.force:
    raise SystemExit(f"{LABEL_DIR}/pairs.csv already exists. Re-running would overwrite the "
                     "annotators' sheets. Use --force only if nobody has started labeling.")
rng = np.random.default_rng(SEED)

skills = pd.read_pickle(WORK / "skills.pkl")
cand = pd.read_pickle(WORK / "candidates.pkl")
occ = pd.read_pickle(WORK / "occurrences.pkl")
copies = occ.groupby("sha").size()
targets = {k: int(v) for k, v in (t.split(":") for t in args.targets.split(","))}

use_skill, use_name, chosen = {}, {}, []


def take(df, band, k):
    got = 0
    for r in df.sample(frac=1, random_state=int(rng.integers(1e9))).itertuples():
        if got >= k:
            break
        a, b = r.i, r.j
        na, nb = skills.nname[a], skills.nname[b]
        if use_skill.get(a, 0) >= args.max_per_skill or use_skill.get(b, 0) >= args.max_per_skill:
            continue
        if na and use_name.get(na, 0) >= args.max_per_name:
            continue
        chosen.append((a, b, band))
        use_skill[a] = use_skill.get(a, 0) + 1
        use_skill[b] = use_skill.get(b, 0) + 1
        if na:
            use_name[na] = use_name.get(na, 0) + 1
        got += 1
    return got


report = {}
for band, k in targets.items():
    if band == "B7":
        continue
    report[band] = (take(cand[cand.band == band], band, k), k)

# B7: random pairs that no blocker proposed (easy negatives)
existing = set(zip(cand.i, cand.j))
need, got, tries = targets.get("B7", 0), 0, 0
while got < need and tries < 100_000:
    tries += 1
    a, b = sorted(rng.choice(len(skills), 2, replace=False))
    if (a, b) in existing:
        continue
    chosen.append((int(a), int(b), "B7"))
    got += 1
report["B7"] = (got, need)
for b, (g, k) in report.items():
    print(f"{b}: sampled {g} of target {k}")
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven"]
mac = {f"ResSample{WORDS[int(b[1])]}": g for b, (g, _) in report.items()}
mac["ResSampleTotal"] = sum(g for g, _ in report.values())
(RESULTS / "macros_sample.tex").write_text(
    "\n".join(rf"\renewcommand{{\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")

pairs = pd.DataFrame(chosen, columns=["i", "j", "band"])
pairs = pairs.sample(frac=1, random_state=SEED).reset_index(drop=True)
# randomise left/right so annotators cannot infer anything from order
swap = rng.random(len(pairs)) < 0.5
pairs.loc[swap, ["i", "j"]] = pairs.loc[swap, ["j", "i"]].values
pairs["pair_id"] = [f"P{k:04d}" for k in range(len(pairs))]
pairs["sha_a"] = skills.sha.values[pairs.i]
pairs["sha_b"] = skills.sha.values[pairs.j]

ann = [a.strip() for a in args.annotators.split(",") if a.strip()]
assert len(ann) >= 2 and len(ann) % 2 == 0, "need an even number of annotators"
teams = [(ann[k], ann[k + 1]) for k in range(0, len(ann), 2)]
pairs["ann_1"] = [teams[k % len(teams)][0] for k in range(len(pairs))]
pairs["ann_2"] = [teams[k % len(teams)][1] for k in range(len(pairs))]
pairs.to_csv(LABEL_DIR / "pairs.csv", index=False)

for a in ann:
    mine = pairs[(pairs.ann_1 == a) | (pairs.ann_2 == a)][["pair_id"]].copy()
    mine["label"] = ""        # D = derived, T = template sibling, U = unrelated
    mine["confidence"] = ""   # 1 = unsure, 2 = fairly sure, 3 = sure
    mine["note"] = ""
    mine.to_csv(LABEL_DIR / f"sheet_{a}.csv", index=False)

# ---- side-by-side viewer -----------------------------------------------------
hdir = LABEL_DIR / "html"
shutil.rmtree(hdir, ignore_errors=True)
hdir.mkdir(parents=True)
owners = occ.groupby("sha")["repo"].apply(lambda s: sorted(set(s))[:3])


def render(idx):
    s = skills.iloc[idx]
    head = f"name: {s['name']}\ndescription: {s.description}\n---\n"
    return (head + s.body).splitlines()[: args.max_lines]


def meta(idx):
    s = skills.iloc[idx]
    reps = ", ".join(owners.get(s.sha, []))
    return (f"<b>{html.escape(str(s['name']))}</b><br>copies: {copies.get(s.sha, 1)} | "
            f"first commit: {s.first_commit_at or 'n/a'}<br>repos: {html.escape(reps)}")


hd = difflib.HtmlDiff(wrapcolumn=72)
links = []
for r in pairs.itertuples():
    table = hd.make_table(render(r.i), render(r.j), "Skill A", "Skill B", context=False)
    page = (f"<html><head><meta charset='utf-8'><title>{r.pair_id}</title><style>"
            "body{font-family:sans-serif;font-size:13px} table.diff{font-family:monospace;"
            "font-size:12px} .diff_add{background:#cfc}.diff_chg{background:#ffa}"
            ".diff_sub{background:#fcc} td{vertical-align:top}</style></head><body>"
            f"<h2>{r.pair_id}</h2><table width='100%'><tr><td width='50%'>A: {meta(r.i)}</td>"
            f"<td>B: {meta(r.j)}</td></tr></table>{table}</body></html>")
    (hdir / f"{r.pair_id}.html").write_text(page, encoding="utf8")
    links.append(f"<li><a href='{r.pair_id}.html'>{r.pair_id}</a> ({r.ann_1}, {r.ann_2})</li>")
(hdir / "index.html").write_text("<html><body><h2>Pairs to label</h2><ol>"
                                 + "".join(links) + "</ol></body></html>", encoding="utf8")

(LABEL_DIR / "GUIDELINES.md").write_text("""# Labeling guidelines: is skill B derived from skill A (or vice versa)?

Open html/<pair_id>.html, read both skills, and fill your sheet with one label.
Do not look at the other annotator's sheet. Do not use similarity scores.

D - DERIVED VARIANT. The two skills share *distinctive* content that is unlikely
    to be written twice independently: verbatim or lightly reworded instruction
    passages, the same task-specific steps, the same examples, commands, file
    paths, configuration values, or idiosyncratic phrasing. One is a copied and
    edited version of the other, or both were copied from a common source.
    Translations of the same skill count as D. Renamed copies count as D.

T - TEMPLATE SIBLING. The overlap is limited to generic scaffolding: shared
    section headings ("When to use this skill", "Quality checklist"), boilerplate
    guidance lines, placeholder text, or a catalog's house style - while the
    task-specific instructions are different. Typical sources: skills generated
    from the same scaffold or the same catalog generator, or one author's
    personal template applied to different tasks.

U - UNRELATED. Neither of the above. Same topic alone ("both are code-review
    skills") is U, not D, unless the content itself was copied.

Decision order: look for copied distinctive content first (-> D). If the only
shared material is scaffolding (-> T). Otherwise U.
Confidence: 1 unsure, 2 fairly sure, 3 sure. Use the note column for anything
odd (e.g. one skill is a strict subset of the other, mixed languages).
""", encoding="utf8")
print(f"wrote {len(pairs)} pairs and {len(ann)} sheets to {LABEL_DIR}/")
