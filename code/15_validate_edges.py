"""Step 15 - Manual validation of predicted derivation links.

  python 15_validate_edges.py --make --annotators Bhavesh,Abhishek,Mahima,Pratiksha
  (label the sheets in labeling/validation/, as for the gold set)
  python 15_validate_edges.py --score

--make trains the best RQ1 model on the gold set (as step 7 does), draws
--edges random predicted links that are not in the gold set and --decoys
random candidate pairs predicted as not derived, shuffles them so annotators
cannot tell which is which, and writes a labelling kit (pairs, one sheet per
annotator, complete side-by-side viewer) to labeling/validation/.
--score reads the sheets and labeling/validation/adjudication.csv and reports
the agreement, the precision of the predicted links with a 95% Wilson
interval, the share of derived pairs among the decoys, and the share of
confirmed links whose names differ (results/macros_validation.tex).
"""
import argparse
import difflib
import html
import json
import shutil
import sqlite3

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ALL_FEATURES, LABEL_DIR, RESULTS, SEED, WORK, fmt_int, fmt_pct

ap = argparse.ArgumentParser()
ap.add_argument("--make", action="store_true")
ap.add_argument("--score", action="store_true")
ap.add_argument("--annotators", default="Bhavesh,Abhishek,Mahima,Pratiksha")
ap.add_argument("--edges", type=int, default=80)
ap.add_argument("--decoys", type=int, default=40)
args = ap.parse_args()
VDIR = LABEL_DIR / "validation"
MAP = {"d": "derived", "derived": "derived", "t": "template", "template": "template",
       "u": "unrelated", "unrelated": "unrelated"}

if args.make:
    MAKE = {"Ours (logistic regression)": lambda: make_pipeline(
                StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
            "Ours (random forest)": lambda: RandomForestClassifier(
                n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                random_state=SEED, n_jobs=-1),
            "Ours (gradient boosting)": lambda: HistGradientBoostingClassifier(
                max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)}
    best = json.load(open(RESULTS / "rq1_rq2.json"))
    gold = pd.read_pickle(WORK / "features_gold.pkl")
    allp = pd.read_pickle(WORK / "features_all.pkl")
    skills = pd.read_pickle(WORK / "skills.pkl")
    y = gold.label.values if best["three_class"] else np.where(gold.label == "derived", "derived", "other")
    model = MAKE[best["best_ours"]]().fit(gold[ALL_FEATURES].values, y)
    pred = model.predict(allp[ALL_FEATURES].values) == "derived"
    sha = skills.sha.values
    if "sha_a" not in allp.columns:
        allp["sha_a"], allp["sha_b"] = sha[allp.i.values], sha[allp.j.values]
    in_gold = set(zip(gold.sha_a, gold.sha_b)) | set(zip(gold.sha_b, gold.sha_a))
    fresh = ~np.array([(a, b) in in_gold for a, b in zip(allp.sha_a, allp.sha_b)])
    rng = np.random.default_rng(SEED)
    e_idx = rng.choice(np.flatnonzero(pred & fresh), min(args.edges, int((pred & fresh).sum())), replace=False)
    d_idx = rng.choice(np.flatnonzero(~pred & fresh), min(args.decoys, int((~pred & fresh).sum())), replace=False)
    V = pd.concat([allp.iloc[e_idx].assign(predicted="derived"), allp.iloc[d_idx].assign(predicted="other")])
    V = V.sample(frac=1, random_state=SEED).reset_index(drop=True)
    swap = rng.random(len(V)) < 0.5
    V.loc[swap, ["sha_a", "sha_b"]] = V.loc[swap, ["sha_b", "sha_a"]].values
    V["pair_id"] = [f"V{k:03d}" for k in range(len(V))]
    ann = [a.strip() for a in args.annotators.split(",") if a.strip()]
    teams = [(ann[k], ann[k + 1]) for k in range(0, len(ann), 2)]
    V["ann_1"] = [teams[k % len(teams)][0] for k in range(len(V))]
    V["ann_2"] = [teams[k % len(teams)][1] for k in range(len(V))]
    if VDIR.exists() and any(VDIR.glob("sheet_*.csv")) and not input("Overwrite existing validation kit? [y/N] ").lower().startswith("y"):
        raise SystemExit("kept the existing kit")
    shutil.rmtree(VDIR, ignore_errors=True)
    (VDIR / "html").mkdir(parents=True, exist_ok=True)
    V[["pair_id", "sha_a", "sha_b", "band", "predicted", "name_equal", "same_owner", "ann_1", "ann_2"]].to_csv(
        VDIR / "pairs.csv", index=False)
    for a in ann:
        s = V[(V.ann_1 == a) | (V.ann_2 == a)][["pair_id"]].copy()
        s["label"], s["confidence"], s["note"] = "", "", ""
        s.to_csv(VDIR / f"sheet_{a}.csv", index=False)
    if (LABEL_DIR / "GUIDELINES.md").exists():
        shutil.copy(LABEL_DIR / "GUIDELINES.md", VDIR / "GUIDELINES.md")
    con = sqlite3.connect(WORK / "full_skills.db")
    need = sorted(set(V.sha_a) | set(V.sha_b))
    T = pd.read_sql_query(f"SELECT sha, name, description, body, repo FROM skills WHERE sha IN ({','.join('?' * len(need))})",
                          con, params=need).set_index("sha")
    hd = difflib.HtmlDiff(wrapcolumn=72)
    links = []
    for r in V.itertuples():
        side = [(f"name: {T.loc[s, 'name']}\ndescription: {T.loc[s, 'description']}\n---\n{T.loc[s, 'body']}").splitlines()
                for s in (r.sha_a, r.sha_b)]
        meta = [f"<b>{html.escape(str(T.loc[s, 'name']))}</b><br>repo: {html.escape(str(T.loc[s, 'repo']))}"
                for s in (r.sha_a, r.sha_b)]
        table = hd.make_table(side[0], side[1], "Skill A", "Skill B", context=False)
        (VDIR / "html" / f"{r.pair_id}.html").write_text(
            f"<html><head><meta charset='utf-8'><title>{r.pair_id}</title><style>body{{font-family:sans-serif;"
            "font-size:13px}} table.diff{{font-family:monospace;font-size:12px}} .diff_add{{background:#cfc}}"
            ".diff_chg{{background:#ffa}} .diff_sub{{background:#fcc}} td{{vertical-align:top}}</style></head><body>"
            f"<h2>{r.pair_id}</h2><table width='100%'><tr><td width='50%'>A: {meta[0]}</td><td>B: {meta[1]}</td>"
            f"</tr></table>{table}</body></html>", encoding="utf8")
        links.append(f"<li><a href='{r.pair_id}.html'>{r.pair_id}</a> ({r.ann_1}, {r.ann_2})</li>")
    (VDIR / "html" / "index.html").write_text("<html><body><h2>Pairs to label</h2><ol>" + "".join(links)
                                              + "</ol></body></html>", encoding="utf8")
    print(f"wrote {len(V)} pairs ({len(e_idx)} predicted links, {len(d_idx)} decoys) to {VDIR}; "
          f"{len(V) * 2 // len(ann)} pairs per annotator")

if args.score:
    V = pd.read_csv(VDIR / "pairs.csv")
    lab = {}
    for a in set(V.ann_1) | set(V.ann_2):
        s = pd.read_csv(VDIR / f"sheet_{a}.csv", dtype=str).fillna("")
        lab[a] = {p: MAP.get(x.strip().lower()) for p, x in zip(s.pair_id, s.label) if x.strip()}
    V["l1"] = [lab[a].get(p) for a, p in zip(V.ann_1, V.pair_id)]
    V["l2"] = [lab[a].get(p) for a, p in zip(V.ann_2, V.pair_id)]
    if V.l1.isna().any() or V.l2.isna().any():
        raise SystemExit(f"{int(V.l1.isna().sum() + V.l2.isna().sum())} labels are missing or invalid")
    kappa = cohen_kappa_score(V.l1, V.l2)
    dis = V[V.l1 != V.l2]
    adj_path = VDIR / "adjudication.csv"
    adj = pd.read_csv(adj_path, dtype=str).fillna("") if adj_path.exists() else pd.DataFrame(
        columns=["pair_id", "label_1", "label_2", "final_label"])
    new = dis[~dis.pair_id.isin(adj.pair_id)][["pair_id", "l1", "l2"]].rename(columns={"l1": "label_1", "l2": "label_2"})
    adj = pd.concat([adj, new.assign(final_label="")], ignore_index=True)
    adj.to_csv(adj_path, index=False)
    final = dict(zip(adj.pair_id, adj.final_label.str.strip().str.lower().map(MAP)))
    if any(pd.isna(final.get(p)) for p in dis.pair_id):
        raise SystemExit(f"{len(dis)} disagreements: fill final_label (D/T/U) in {adj_path}, then run --score again")
    V["final"] = [a if a == b else final[p] for p, a, b in zip(V.pair_id, V.l1, V.l2)]
    E, D = V[V.predicted == "derived"], V[V.predicted != "derived"]
    k, n = int((E.final == "derived").sum()), len(E)

    def wilson(k, n, z=norm.ppf(0.975)):
        p = k / n
        c = (p + z * z / (2 * n)) / (1 + z * z / n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        return c - h, c + h

    lo, hi = wilson(k, n)
    conf = E[E.final == "derived"]
    rq3 = {}
    if (RESULTS / "rq3.json").exists():
        rq3 = json.load(open(RESULTS / "rq3.json"))
    mac = {"ResValPairs": fmt_int(len(V)), "ResValEdges": fmt_int(n), "ResValDecoys": fmt_int(len(D)),
           "ResValKappa": f"{kappa:.2f}", "ResValAgree": fmt_pct((V.l1 == V.l2).mean()),
           "ResValPrecision": f"{k / n:.2f}", "ResValPrecLo": f"{lo:.2f}", "ResValPrecHi": f"{hi:.2f}",
           "ResValDecoyDerived": fmt_pct((D.final == "derived").mean()) if len(D) else "--",
           "ResValConfirmedDiffName": fmt_pct((conf.name_equal == 0).mean()) if len(conf) else "--",
           "ResValConfirmedCrossOwner": fmt_pct((conf.same_owner == 0).mean()) if len(conf) else "--"}
    (RESULTS / "macros_validation.tex").write_text(
        "\n".join(rf"\renewcommand{{\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
    json.dump({"pairs": len(V), "edges": n, "confirmed": k, "precision": k / n, "ci": [lo, hi], "kappa": kappa,
               "decoys_derived": int((D.final == "derived").sum()), "decoys": len(D),
               "confirmed_diff_name": float((conf.name_equal == 0).mean()) if len(conf) else None},
              open(RESULTS / "validation.json", "w"), indent=2)
    print(f"kappa={kappa:.3f}; precision of predicted links = {k}/{n} = {k / n:.3f} [95% CI {lo:.3f}, {hi:.3f}]; "
          f"derived among decoys = {int((D.final == 'derived').sum())}/{len(D)}")
