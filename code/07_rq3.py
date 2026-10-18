"""Step 7 - RQ3: how concentrated is reuse once copies are grouped into families?

Trains the best RQ1 model on all gold pairs, scores every candidate pair
(work/features_all.pkl), keeps predicted-derived edges, and splits the
resulting graph into families with Louvain community detection (connected
components alone chain unrelated skills together through a few bridging
edges). Then compares exact-hash units with family units.
"""
import json

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ALL_FEATURES, RESULTS, SEED, WORK, fmt_int, fmt_pct

MAKE = {
    "Ours (logistic regression)": lambda: make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
    "Ours (random forest)": lambda: RandomForestClassifier(
        n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
        random_state=SEED, n_jobs=-1),
    "Ours (gradient boosting)": lambda: HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED),
}

best = json.load(open(RESULTS / "rq1_rq2.json"))
gold = pd.read_pickle(WORK / "features_gold.pkl")
allp = pd.read_pickle(WORK / "features_all.pkl")
skills = pd.read_pickle(WORK / "skills.pkl")
occ = pd.read_pickle(WORK / "occurrences.pkl")
pop = pd.read_pickle(WORK / "population_occurrences.pkl")

y = gold.label.values if best["three_class"] else np.where(gold.label == "derived", "derived", "other")
model = MAKE[best["best_ours"]]().fit(gold[ALL_FEATURES].values, y)
allp["pred"] = model.predict(allp[ALL_FEATURES].values)
edges = allp[allp.pred == "derived"]
print(f"{len(allp)} candidate pairs scored, {len(edges)} predicted derived")

G = nx.Graph()
G.add_nodes_from(skills.sha)
G.add_edges_from(zip(edges.sha_a, edges.sha_b))
comms = nx.community.louvain_communities(G, seed=SEED)
fam = {s: k for k, c in enumerate(comms) for s in c}
n_cc = nx.number_connected_components(G)
largest_cc = max(len(c) for c in nx.connected_components(G))


def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if n and x.sum() else 0.0


def unit_stats(o, key):
    g = o.groupby(key)
    occ_per = g.size()
    owners_per = g["owner"].nunique()
    k = max(1, int(np.ceil(0.01 * len(occ_per))))
    return {
        "units": len(occ_per), "occurrences": int(occ_per.sum()),
        "singleton_share": float((occ_per == 1).mean()),
        "max_occurrences": int(occ_per.max()), "gini": gini(occ_per.values),
        "top1pct_share": float(occ_per.sort_values(ascending=False).iloc[:k].sum() / occ_per.sum()),
        "cross_owner_units": float((owners_per >= 2).mean()),
        "cross_owner_occ": float(occ_per[owners_per >= 2].sum() / occ_per.sum()),
    }


occ = occ.copy()
occ["family"] = occ.sha.map(fam)
H = unit_stats(occ, "sha")
Fm = unit_stats(occ, "family")
P = unit_stats(pop, "sha")  # whole population, exact-hash level

sizes = pd.Series([len(c) for c in comms])
first = skills.set_index("sha")["first_commit_at"]
fa, fb = edges.sha_a.map(first), edges.sha_b.map(first)
orient = float((fa.notna() & fb.notna() & (fa != fb)).mean()) if len(edges) else float("nan")

top = sorted(comms, key=len, reverse=True)[:10]
nn = skills.set_index("sha")["nname"]
pd.DataFrame([{"family_size": len(c), "occurrences": int(occ[occ.sha.isin(c)].shape[0]),
               "top_names": ", ".join(nn.reindex(list(c)).value_counts().index[:3])} for c in top]
             ).to_csv(RESULTS / "rq3_top_families.csv", index=False)

rows = [r"\begin{tabular}{lrrr}", r"\toprule",
        r" & \multicolumn{2}{c}{Working set} & Population \\", r"\cmidrule(lr){2-3}",
        r"Metric & Exact hash & Family & Exact hash \\", r"\midrule"]
spec_rows = [("Units", "units", fmt_int), ("File occurrences", "occurrences", fmt_int),
             ("Units with one occurrence", "singleton_share", fmt_pct),
             ("Largest unit (occurrences)", "max_occurrences", fmt_int),
             ("Gini of occurrences per unit", "gini", lambda v: f"{v:.2f}"),
             ("Share of occurrences in top 1\\% units", "top1pct_share", fmt_pct),
             ("Units spanning $\\geq$2 owners", "cross_owner_units", fmt_pct),
             ("Occurrences in multi-owner units", "cross_owner_occ", fmt_pct)]
for label, key, f in spec_rows:
    rows.append(f"{label} & {f(H[key])} & {f(Fm[key])} & {f(P[key])} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "rq3_table.tex").write_text("\n".join(rows) + "\n")

mac = {"ResWorkHashes": fmt_int(H["units"]), "ResFamilies": fmt_int(Fm["units"]),
       "ResInflation": f"{H['units'] / Fm['units']:.2f}",
       "ResMultiFamilies": fmt_int((sizes >= 2).sum()),
       "ResLargestFamily": fmt_int(sizes.max()), "ResLargestCC": fmt_int(largest_cc),
       "ResDerivedEdges": fmt_int(len(edges)), "ResCandPairs": fmt_int(len(allp)),
       "ResGiniHash": f"{H['gini']:.2f}", "ResGiniFam": f"{Fm['gini']:.2f}",
       "ResCrossHash": fmt_pct(H["cross_owner_units"]), "ResCrossFam": fmt_pct(Fm["cross_owner_units"]),
       "ResOrient": fmt_pct(orient)}
(RESULTS / "macros_rq3.tex").write_text(
    "\n".join(rf"\renewcommand{{\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
json.dump({"hash": H, "family": Fm, "population_hash": P, "orientation_coverage": orient,
           "connected_components": n_cc, "largest_cc": largest_cc}, open(RESULTS / "rq3.json", "w"), indent=2)
print(json.dumps({"hash": H, "family": Fm, "orient": orient, "largest_cc": largest_cc,
                  "largest_family": int(sizes.max())}, indent=1))
