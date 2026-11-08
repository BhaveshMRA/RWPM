"""Step 16 - Diagnostics of the predicted derivation links (full dataset).

  python 16_edge_diagnostics.py   (after 07_rq3.py; same SKILL_WORK / SKILL_RESULTS)

Summarises where the predicted links come from, so that the RQ3 numbers can
be checked before they are reported: links per band, with equal or different
names and owners, links that look like template pairs (almost all shared lines
are boilerplate and no distinctive line is shared), how strongly links
concentrate on a few hub skills, and the composition of the largest connected
components. Writes results/edge_diagnostics.json and
results/edge_diagnostics_samples.csv (small enough to commit).
"""
import json

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ALL_FEATURES, RESULTS, SEED, WORK

MAKE = {"Ours (logistic regression)": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
        "Ours (random forest)": lambda: RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "Ours (gradient boosting)": lambda: HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)}
best = json.load(open(RESULTS / "rq1_rq2.json"))
gold = pd.read_pickle(WORK / "features_gold.pkl")
allp = pd.read_pickle(WORK / "features_all.pkl")
skills = pd.read_pickle(WORK / "skills.pkl")
if "i" not in allp.columns:  # sample run: map hashes to row ids
    pos = {s: k for k, s in enumerate(skills.sha)}
    allp["i"], allp["j"] = allp.sha_a.map(pos).values, allp.sha_b.map(pos).values
y = gold.label.values if best["three_class"] else np.where(gold.label == "derived", "derived", "other")
model = MAKE[best["best_ours"]]().fit(gold[ALL_FEATURES].values, y)
k = list(model.classes_).index("derived")
X = allp[ALL_FEATURES].values
proba = np.concatenate([model.predict_proba(X[s:s + 1_000_000])[:, k] for s in range(0, len(X), 1_000_000)])
pred = np.concatenate([model.predict(X[s:s + 1_000_000]) == "derived" for s in range(0, len(X), 1_000_000)])
E = allp[pred]
pE = proba[pred]
band = allp.band.astype(str)


def tlike(D):
    return (D.frac_shared_boiler >= 0.9) & (D.jac_distinct_lines == 0)


out = {"candidates": int(len(allp)), "predicted_derived": int(pred.sum())}
out["by_band"] = {b: {"candidates": int((band == b).sum()), "derived": int((band[pred] == b).sum())}
                  for b in sorted(band.unique())}
out["derived_same_name"] = float((E.name_equal == 1).mean())
out["derived_same_owner"] = float((E.same_owner == 1).mean())
out["derived_template_like"] = float(tlike(E).mean())
out["derived_proba_quantiles"] = {str(q): float(np.quantile(pE, q)) for q in (0.05, 0.25, 0.5, 0.75)}
out["derived_proba_ge_0.9"] = float((pE >= 0.9).mean())
deg_c = np.bincount(np.r_[allp.i.values, allp.j.values], minlength=len(skills))
deg_e = np.bincount(np.r_[E.i.values, E.j.values], minlength=len(skills))
top = np.argsort(-deg_e)[: max(1, len(skills) // 1000)]
hub = np.isin(E.i.values, top) | np.isin(E.j.values, top)
out["hubs"] = {"max_degree_candidates": int(deg_c.max()), "max_degree_derived": int(deg_e.max()),
               "share_of_derived_links_touching_top_0.1pct_skills": float(hub.mean())}
G = nx.Graph()
G.add_edges_from(zip(E.i.values, E.j.values))
comps = sorted(nx.connected_components(G), key=len, reverse=True)[:5]
nn = skills.nname.values
rows, samples = [], []
for c, comp in enumerate(comps):
    m = np.isin(E.i.values, list(comp))
    Ec, pc = E[m], pE[m]
    names = pd.Series(nn[list(comp)]).value_counts()
    rows.append({"size": len(comp), "links": int(len(Ec)), "distinct_names": int(len(names)),
                 "top_names": ", ".join(names.index[:5]), "share_same_name": float((Ec.name_equal == 1).mean()),
                 "share_template_like": float(tlike(Ec).mean()), "median_proba": float(np.median(pc)),
                 "share_links_touching_hubs": float((np.isin(Ec.i.values, top) | np.isin(Ec.j.values, top)).mean())})
    pick = Ec.sample(min(15, len(Ec)), random_state=SEED)
    for r, p in zip(pick.itertuples(), proba[pred][m][[Ec.index.get_loc(x) for x in pick.index]]):
        samples.append({"component": c, "name_a": nn[r.i], "name_b": nn[r.j], "sha_a": skills.sha.iat[r.i],
                        "sha_b": skills.sha.iat[r.j], "band": str(r.band), "proba": round(float(p), 3),
                        "jac_shingle": round(float(r.jac_shingle), 3),
                        "jac_distinct_lines": round(float(r.jac_distinct_lines), 3),
                        "frac_shared_boiler": round(float(r.frac_shared_boiler), 3),
                        "desc_tfidf": round(float(r.desc_tfidf), 3), "name_equal": int(r.name_equal)})
out["largest_components"] = rows
json.dump(out, open(RESULTS / "edge_diagnostics.json", "w"), indent=2)
pd.DataFrame(samples).to_csv(RESULTS / "edge_diagnostics_samples.csv", index=False)
print(json.dumps(out, indent=2))
