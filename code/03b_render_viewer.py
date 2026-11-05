"""Step 3b - Rebuild the side-by-side labeling viewer from labeling/pairs.csv.

  python 03b_render_viewer.py          -> labeling/html/index.html, one page per pair

Unlike step 3 this never resamples: it reads the saved sample (pairs are identified by content hash) and renders the complete text of both skills, including front-matter name and description, without truncation.
"""
import difflib
import html
import shutil

import pandas as pd

from common import LABEL_DIR, WORK

skills = pd.read_pickle(WORK / "skills.pkl")
occ = pd.read_pickle(WORK / "occurrences.pkl")
pairs = pd.read_csv(LABEL_DIR / "pairs.csv")
pos = {s: k for k, s in enumerate(skills.sha)}
missing = set(pairs.sha_a).union(pairs.sha_b) - set(pos)
if missing:
    raise SystemExit(f"{len(missing)} skills of pairs.csv are not in work/skills.pkl; run 01_extract.py first")
copies = occ.groupby("sha").size()
owners = occ.groupby("sha")["repo"].apply(lambda s: sorted(set(s))[:3])


def render(sha):
    s = skills.iloc[pos[sha]]
    return (f"name: {s['name']}\ndescription: {s.description}\n---\n" + s.body).splitlines()


def meta(sha):
    s = skills.iloc[pos[sha]]
    reps = ", ".join(owners.get(sha, []))
    return (f"<b>{html.escape(str(s['name']))}</b><br>copies: {copies.get(sha, 1)} | "
            f"first commit: {s.first_commit_at or 'n/a'}<br>repos: {html.escape(reps)}")


hdir = LABEL_DIR / "html"
shutil.rmtree(hdir, ignore_errors=True)
hdir.mkdir(parents=True)
hd = difflib.HtmlDiff(wrapcolumn=72)
links = []
for r in pairs.itertuples():
    table = hd.make_table(render(r.sha_a), render(r.sha_b), "Skill A", "Skill B", context=False)
    page = (f"<html><head><meta charset='utf-8'><title>{r.pair_id}</title><style>"
            "body{font-family:sans-serif;font-size:13px} table.diff{font-family:monospace;"
            "font-size:12px} .diff_add{background:#cfc}.diff_chg{background:#ffa}"
            ".diff_sub{background:#fcc} td{vertical-align:top}</style></head><body>"
            f"<h2>{r.pair_id}</h2><table width='100%'><tr><td width='50%'>A: {meta(r.sha_a)}</td>"
            f"<td>B: {meta(r.sha_b)}</td></tr></table>{table}</body></html>")
    (hdir / f"{r.pair_id}.html").write_text(page, encoding="utf8")
    links.append(f"<li><a href='{r.pair_id}.html'>{r.pair_id}</a> ({r.ann_1}, {r.ann_2})</li>")
(hdir / "index.html").write_text("<html><body><h2>Pairs to label</h2><ol>" + "".join(links) + "</ol></body></html>", encoding="utf8")
print(f"wrote {len(pairs)} complete pages to {hdir}")
