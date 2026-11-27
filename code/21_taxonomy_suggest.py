"""Step 21 - Reproduce the suggested taxonomy sub-categories that the annotators checked.

    python 21_taxonomy_suggest.py --db path/to/agent_skills_sample.db

Section 7 of the paper: rules applied to our labelling notes proposed one sub-category per gold pair, and
the two annotators of each pair confirmed or corrected it (labeling/taxonomy/). This script re-applies the
same rules, in the same order of precedence (keywords in the notes, whether the front-matter names differ,
the band), plus the nine documented overrides of cases where a keyword occurred only incidentally, and
compares the result with the suggestions in the annotators' sheets. Names come from the GitSkills database.
Writes results/taxonomy_suggestions.csv.
"""
import argparse
import re
import sqlite3

import pandas as pd

from common import LABEL_DIR, RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--db", required=True, help="GitSkills SQLite database (sample or full release)")
args = ap.parse_args()
L = LABEL_DIR
g = pd.read_csv(L / "gold.csv")
N = pd.concat([pd.read_csv(f, dtype=str).fillna("")[["pair_id", "note"]] for f in sorted(L.glob("sheet_*.csv"))])
g["notes"] = g.pair_id.map(N.groupby("pair_id").note.apply(lambda x: " || ".join(v for v in x if v)))
con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
nm = dict(con.execute("SELECT file_sha, name FROM artifacts WHERE name IS NOT NULL"))
g["name_a"], g["name_b"] = g.sha_a.map(nm).fillna(""), g.sha_b.map(nm).fillna("")
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())
g["same_name"] = [norm(a) == norm(b) and norm(a) != "" for a, b in zip(g.name_a, g.name_b)]


def has(t, pat): return re.search(pat, t, re.I) is not None
def classify(r):
    t = r.notes or ''
    if r.label == 'derived':
        if has(t, r'translat|chinese|japanese|korean|spanish|german version|in (english|chinese)|沿襲'): return 'D-translated', 'note: translation'
        if has(t, r'windsurf|cursor|codex|gemini|opencode|copilot|installer|ported|port\b|adapter|adapted for|paths? (were )?rewritten|rewritten paths|\.agents|agent tool|gsd-|windows'): return 'D-ported', 'note: tool/path port'
        if has(t, r'rework|no verbatim|without copying|restructur|rewr(o|i)te|paraphras|reworded'): return 'D-reworked', 'note: rework'
        if not r.same_name: return 'D-renamed', 'names differ'
        return 'D-edited', 'same name, copy with edits'
    if r.label == 'template':
        if has(t, r'preamble|house protocol|protocol|framework|gstack|gsd|spec-?kit|harness|shared (repository|project)|wrapper around|omni'): return 'T-preamble', 'note: shared preamble/protocol'
        if has(t, r'generat|stub|placeholder|slot|u-?series|winbda|catalog stub|numbered|scaffold with'): return 'T-generator', 'note: generator/slots'
        if has(t, r'composio|rube|membrane|integration|api|toolkit|connector|alibaba|clawlink|mcp|service|wrap'): return 'T-integration', 'note: integration catalog'
        if has(t, r'series|legal|german|author|domain|same (owner|scaffold)|structure'): return 'T-series', 'note: domain series'
        return 'T-?', 'no rule matched'
    if r.band == 'B7': return 'U-different', 'random pair'
    if r.same_name or has(t, r'same (skill )?name|same topic|both (discuss|address|explain|cover|are about|describe)|same domain|similar topic'): return 'U-samename', 'same name/topic'
    return 'U-different', 'different topics'
g[["suggested", "rule"]] = g.apply(lambda r: pd.Series(classify(r)), axis=1)
OVERRIDES = {  # pair: (sub-category, reason); the rule fired on an incidental keyword
    "P0121": ("D-reworked", "paper: reworks another skill's steps without copying a line"),
    "P0335": ("D-reworked", "paper: keeps only the description, no shared text"),
    "P0021": ("D-renamed", "Chinese only in an added reference; renamed copy"),
    "P0139": ("D-renamed", "Chinese only as added options; renamed copy"),
    "P0292": ("D-renamed", "only the description is in Chinese; renamed copy"),
    "P0204": ("D-edited", '"Copilot" is an added note, not a port'),
    "P0219": ("D-edited", '"adapter" refers to minor additions, not a port'),
    "P0116": ("T-generator", "FLAG: shares only a common documentation idiom; generator is the closest sub-category"),
    "P0185": ("T-series", "FLAG: same collection, generated section schema (paper rule P0185)"),
}
for p, (c, why) in OVERRIDES.items():
    g.loc[g.pair_id == p, ["suggested", "rule"]] = [c, "override: " + why]
g.loc[g.pair_id == "P0222", "rule"] = "FLAG: Korean documentation copy AND Codex install with rewritten paths; translated vs ported"
g[["pair_id", "label", "suggested", "rule"]].to_csv(RESULTS / "taxonomy_suggestions.csv", index=False)
issued = pd.concat([pd.read_csv(f, dtype=str)[["pair_id", "suggested"]] for f in sorted((L / "taxonomy").glob("taxonomy_*.csv"))]).drop_duplicates("pair_id")
m = g.merge(issued, on="pair_id", suffixes=("", "_issued"))
print(f"suggestions reproduced for {int((m.suggested == m.suggested_issued).sum())} of {len(m)} gold pairs")
print(g.suggested.value_counts().sort_index().to_string())
