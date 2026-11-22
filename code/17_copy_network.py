"""Step 17 - Compare our derivation links with the dated copy network of Seddik (Skill Constellations).

    python 17_copy_network.py                 # gold set and validation sample (any machine)
    python 17_copy_network.py --all-links     # also every predicted link (needs the full-run work folder)

The copy network (release data-v1.0, CC BY 4.0) lists every change to a SKILL.md in the git history of the GitSkills repositories, with the old and new git blob hash. GitSkills identifies a file by the same git blob hash, so our pairs join to it directly. Two skills are connected in the network when an edit path in that history joins their blobs; this is an upper bound on what its lineages link, because a lineage additionally keeps one front-matter name. Each skill is dated by the first commit that contains its blob. Writes results/copy_network.json, results/macros_copy_network.tex and results/copy_network_examples.csv.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common import ALL_FEATURES, LABEL_DIR as LABELS, RESULTS, SEED, WORK

URL = "https://github.com/FahdSeddik/Skill-Constellations/releases/download/data-v1.0/history.parquet"
ZERO = "0" * 40
ap = argparse.ArgumentParser()
ap.add_argument("--history", default=str(WORK / "seddik_history.parquet"), help="local copy of history.parquet")
ap.add_argument("--all-links", action="store_true", help="also analyse every predicted link (full-run work folder)")
args = ap.parse_args()
t0 = time.time()


def key(s):
    return int(s[:15], 16)


hist = Path(args.history)
if not hist.exists():
    hist.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} (591 MB) ...")
    urllib.request.urlretrieve(URL, hist)

# pair sets
gold = pd.read_csv(LABELS / "pairs.csv")[["pair_id", "sha_a", "sha_b", "band"]].merge(
    pd.read_csv(LABELS / "gold.csv")[["pair_id", "label"]], on="pair_id")
V = LABELS / "validation"
val = pd.read_csv(V / "pairs.csv")
lab = {}
for f in sorted(V.glob("sheet_*.csv")):
    s = pd.read_csv(f, dtype=str).fillna("")
    for p, l in zip(s.pair_id, s.label.str.strip().str.upper()):
        if l:
            lab.setdefault(p, l)
adj = V / "adjudication.csv"
if adj.exists():
    a = pd.read_csv(adj, dtype=str).fillna("")
    for p, l in zip(a.pair_id, a.final_label.str.strip().str.upper()):
        if l:
            lab[p] = l
val["final"] = val.pair_id.map(lab)
sets = {"gold": gold, "validation": val}

links = None
if args.all_links:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    best = json.load(open(RESULTS / "rq1_rq2.json"))
    make = {"Ours (logistic regression)": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
            "Ours (random forest)": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
            "Ours (gradient boosting)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)}
    fg = pd.read_pickle(WORK / "features_gold.pkl")
    fa = pd.read_pickle(WORK / "features_all.pkl")
    sk = pd.read_pickle(WORK / "skills.pkl")
    y = fg.label.values if best["three_class"] else np.where(fg.label == "derived", "derived", "other")
    model = make[best["best_ours"]]().fit(fg[ALL_FEATURES].values, y)
    X = fa[ALL_FEATURES].values
    pred = np.concatenate([model.predict(X[s:s + 1_000_000]) == "derived" for s in range(0, len(X), 1_000_000)])
    E = fa[pred]
    if "i" in E.columns:
        sha = sk.sha.values
        links = pd.DataFrame({"sha_a": sha[E.i.values], "sha_b": sha[E.j.values], "name_equal": E.name_equal.values})
    else:
        links = pd.DataFrame({"sha_a": E.sha_a.values, "sha_b": E.sha_b.values, "name_equal": E.name_equal.values})
    sets["all_links"] = links
    print(f"{len(links):,} predicted links ({time.time() - t0:.0f}s)")

wanted = set()
for d in sets.values():
    wanted |= set(d.sha_a) | set(d.sha_b)

# stream the history: edit edges and first appearance of the wanted blobs
pf = pq.ParquetFile(hist)
src, dst, firsts, rows = [], [], [], 0
for g in range(pf.num_row_groups):
    t = pf.read_row_group(g, columns=["old_blob", "new_blob", "commit_time"]).to_pandas()
    rows += len(t)
    e = t[(t.old_blob != ZERO) & (t.new_blob != ZERO)]
    src.append(np.fromiter((key(x) for x in e.old_blob), np.int64, len(e)))
    dst.append(np.fromiter((key(x) for x in e.new_blob), np.int64, len(e)))
    h = t[t.new_blob.isin(wanted)]
    firsts.append(h.groupby("new_blob").commit_time.min())
src, dst = np.concatenate(src), np.concatenate(dst)
first = pd.concat(firsts).groupby(level=0).min()
nodes, inv = np.unique(np.concatenate([src, dst]), return_inverse=True)
n_comp, comp = connected_components(coo_matrix((np.ones(len(src), np.int8), (inv[:len(src)], inv[len(src):])),
                                               shape=(len(nodes), len(nodes))), directed=False)
print(f"history rows {rows:,}; edit edges {len(src):,}; components {n_comp:,} ({time.time() - t0:.0f}s)")


def component(shas):
    k = np.array([key(s) for s in shas], dtype=np.int64)
    j = np.clip(np.searchsorted(nodes, k), 0, len(nodes) - 1)
    return np.where(nodes[j] == k, comp[j], -1)


def describe(d):
    ca, cb = component(d.sha_a), component(d.sha_b)
    ta, tb = d.sha_a.map(first), d.sha_b.map(first)
    dated = ta.notna().values & tb.notna().values
    conn = (ca >= 0) & (ca == cb)
    gap = (pd.to_datetime(tb[dated], utc=True) - pd.to_datetime(ta[dated], utc=True)).abs().dt.total_seconds() / 86400
    return {"n": int(len(d)), "dated": int(dated.sum()), "connected": int(conn.sum()),
            "median_gap_days": float(gap.median()) if len(gap) else None,
            "same_day": int((gap < 1).sum())}, conn, dated


out, examples = {}, []
for lab_, part in gold.groupby("label"):
    out[f"gold_{lab_}"], conn, dated = describe(part)
    if lab_ == "derived":
        ex = part[~conn & dated].copy()
        ex["first_a"], ex["first_b"] = ex.sha_a.map(first), ex.sha_b.map(first)
        examples.append(ex.assign(set="gold"))
vp = val[val.predicted == "derived"]
out["validation_predicted"], _, _ = describe(vp)
out["validation_confirmed"], _, _ = describe(vp[vp.final == "D"])
out["validation_confirmed_diffname"], _, _ = describe(vp[(vp.final == "D") & (vp.name_equal == 0)])
if links is not None:
    out["all_links"], conn, dated = describe(links)
    for nm, m in (("all_links_samename", links.name_equal.values == 1), ("all_links_diffname", links.name_equal.values == 0)):
        out[nm], _, _ = describe(links[m])
out["history"] = {"rows": rows, "edit_edges": int(len(src)), "components": int(n_comp)}
RESULTS.mkdir(parents=True, exist_ok=True)
json.dump(out, open(RESULTS / "copy_network.json", "w"), indent=2)
pd.concat(examples).to_csv(RESULTS / "copy_network_examples.csv", index=False)


def pct(a, b):
    return f"{100 * a / b:.0f}\\%" if b else "--"


mac = {}
d = out["gold_derived"]
mac.update(ResCNGoldDerived=d["n"], ResCNGoldDerivedDated=d["dated"], ResCNGoldDerivedConnected=d["connected"],
           ResCNGoldDerivedMissedPct=pct(d["n"] - d["connected"], d["n"]))
d = out["validation_confirmed"]
mac.update(ResCNValConfirmed=d["n"], ResCNValConfirmedDated=d["dated"], ResCNValConfirmedConnected=d["connected"],
           ResCNValConfirmedMissedPct=pct(d["n"] - d["connected"], d["n"]))
if "all_links" in out:
    d = out["all_links"]
    mac.update(ResCNAllLinks=f"{d['n']:,}", ResCNAllDatedPct=pct(d["dated"], d["n"]),
               ResCNAllConnectedPct=pct(d["connected"], d["n"]), ResCNAllMissedPct=pct(d["n"] - d["connected"], d["n"]),
               ResCNDiffNameMissedPct=pct(out["all_links_diffname"]["n"] - out["all_links_diffname"]["connected"], out["all_links_diffname"]["n"]),
               ResCNSameNameMissedPct=pct(out["all_links_samename"]["n"] - out["all_links_samename"]["connected"], out["all_links_samename"]["n"]))
g = out["gold_derived"]["median_gap_days"]
mac["ResCNGoldMedianGapDays"] = f"{g:.0f}" if g is not None else "--"
mac["ResCNHistoryRows"] = f"{rows:,}"
(RESULTS / "macros_copy_network.tex").write_text("".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(json.dumps({k: v for k, v in out.items() if k != "history"}, indent=1))
print(f"wrote copy_network.json, macros_copy_network.tex, copy_network_examples.csv ({time.time() - t0:.0f}s)")
