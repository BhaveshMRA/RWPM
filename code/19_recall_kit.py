"""Step 19 - Estimate the recall of the predicted links on the full release (Phase 3 feedback, Fix 1).

    python 19_recall_kit.py --make     -> labeling/recall/ (sheets, viewer pages, pairs.csv, strata.json)
    (label the sheets in labeling/recall/ with the usual rubric)
    python 19_recall_kit.py --score    -> results/macros_recall.tex, results/recall.json

The validation decoys of step 15 were drawn at random from all pairs predicted not derived, so most were
easy unrelated pairs. This kit draws the pairs predicted not derived stratified by similarity band, with
most of them from the middle bands B2-B4, where missed derivations are likeliest, and mixes in pairs
predicted derived so that annotators cannot assume every pair is negative. Pairs of the gold set, the
validation sample and the stress test are excluded.

Recall is estimated within the candidate pairs: the missed derivations of band b are estimated as
N_b * d_b / n_b (N_b pairs of band b predicted not derived, n_b of them labelled, d_b labelled derived),
and the found derivations as the number of predicted links times their precision (validation sample plus
the predicted-derived pairs of this kit). Point estimate and 95% interval are the median and the 2.5-97.5%
range of the recall when the derived share of every band and the precision are redrawn 10,000 times from
their beta posteriors (Jeffreys prior), so that bands with few sampled pairs widen the interval honestly.
Most negatives come from the middle bands B2-B4, as the feedback asked, and from B5, the same-name pairs with
little shared text, where the stress test found its reworked and translated copies.
"""
import argparse
import difflib
import html
import json
import shutil
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ALL_FEATURES, LABEL_DIR, RESULTS, SEED, WORK

ap = argparse.ArgumentParser()
ap.add_argument("--make", action="store_true")
ap.add_argument("--score", action="store_true")
ap.add_argument("--annotators", default="Bhavesh,Abhishek,Mahima,Pratiksha")
ap.add_argument("--negatives", default="B0:6,B1:4,B2:12,B3:12,B4:12,B5:14", help="pairs predicted not derived per band")
ap.add_argument("--positives", type=int, default=20, help="pairs predicted derived, mixed in for blinding")
ap.add_argument("--kit", default="recall")
args = ap.parse_args()
KDIR = LABEL_DIR / args.kit
MAP = {"d": "derived", "derived": "derived", "t": "template", "template": "template", "u": "unrelated", "unrelated": "unrelated"}

if args.make:
    make = {"Ours (logistic regression)": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
            "Ours (random forest)": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
            "Ours (gradient boosting)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)}
    best = json.load(open(RESULTS / "rq1_rq2.json"))
    gold = pd.read_pickle(WORK / "features_gold.pkl")
    allp = pd.read_pickle(WORK / "features_all.pkl")
    sha = pd.read_pickle(WORK / "skills.pkl").sha.values
    if "sha_a" not in allp.columns:
        allp["sha_a"], allp["sha_b"] = sha[allp.i.values], sha[allp.j.values]
    y = gold.label.values if best["three_class"] else np.where(gold.label == "derived", "derived", "other")
    model = make[best["best_ours"]]().fit(gold[ALL_FEATURES].values, y)
    X = allp[ALL_FEATURES].values
    allp["pred"] = np.concatenate([model.predict(X[s:s + 1_000_000]) == "derived" for s in range(0, len(X), 1_000_000)])
    used = set(zip(gold.sha_a, gold.sha_b))
    for kit in ("validation", "stress"):
        f = LABEL_DIR / kit / "pairs.csv"
        if f.exists():
            k = pd.read_csv(f)
            used |= set(zip(k.sha_a, k.sha_b))
    used |= {(b, a) for a, b in used}
    free = ~np.array([(a, b) in used for a, b in zip(allp.sha_a, allp.sha_b)])
    rng = np.random.default_rng(SEED)
    strata, picks = {}, []
    for part in args.negatives.split(","):
        band, n = part.split(":")
        pool = np.flatnonzero(free & ~allp.pred.values & (allp.band.values == band))
        strata[band] = {"population": int((~allp.pred.values & (allp.band.values == band)).sum()), "sampled": min(int(n), len(pool))}
        picks.append(allp.iloc[rng.choice(pool, strata[band]["sampled"], replace=False)].assign(stratum="neg-" + band))
    pos = np.flatnonzero(free & allp.pred.values)
    picks.append(allp.iloc[rng.choice(pos, min(args.positives, len(pos)), replace=False)].assign(stratum="pos"))
    K = pd.concat(picks).sample(frac=1, random_state=SEED).reset_index(drop=True)
    K["predicted"] = np.where(K.pred, "derived", "other")
    swap = rng.random(len(K)) < 0.5
    K.loc[swap, ["sha_a", "sha_b"]] = K.loc[swap, ["sha_b", "sha_a"]].values
    K["pair_id"] = [f"R{k:03d}" for k in range(len(K))]
    ann = [a.strip() for a in args.annotators.split(",") if a.strip()]
    teams = [(ann[k], ann[k + 1]) for k in range(0, len(ann), 2)]
    K["ann_1"] = [teams[k % len(teams)][0] for k in range(len(K))]
    K["ann_2"] = [teams[k % len(teams)][1] for k in range(len(K))]
    if KDIR.exists() and any(KDIR.glob("sheet_*.csv")) and not input(f"Overwrite existing kit {KDIR}? [y/N] ").lower().startswith("y"):
        raise SystemExit("kept the existing kit")
    shutil.rmtree(KDIR, ignore_errors=True)
    (KDIR / "html").mkdir(parents=True, exist_ok=True)
    K[["pair_id", "sha_a", "sha_b", "band", "stratum", "predicted", "name_equal", "same_owner", "ann_1", "ann_2"]].to_csv(KDIR / "pairs.csv", index=False)
    json.dump({"strata": strata, "predicted_links": int(allp.pred.sum())}, open(KDIR / "strata.json", "w"), indent=2)
    for a in ann:
        s = K[(K.ann_1 == a) | (K.ann_2 == a)][["pair_id"]].copy()
        s["label"], s["confidence"], s["note"] = "", "", ""
        s.to_csv(KDIR / f"sheet_{a}.csv", index=False)
    if (LABEL_DIR / "GUIDELINES.md").exists():
        shutil.copy(LABEL_DIR / "GUIDELINES.md", KDIR / "GUIDELINES.md")
    con = sqlite3.connect(WORK / "full_skills.db")
    need = sorted(set(K.sha_a) | set(K.sha_b))
    T = pd.read_sql_query(f"SELECT sha, name, description, body, repo FROM skills WHERE sha IN ({','.join('?' * len(need))})",
                          con, params=need).set_index("sha")
    hd = difflib.HtmlDiff(wrapcolumn=72)
    links = []
    for r in K.itertuples():
        side = [(f"name: {T.loc[s, 'name']}\ndescription: {T.loc[s, 'description']}\n---\n{T.loc[s, 'body']}").splitlines() for s in (r.sha_a, r.sha_b)]
        meta = [f"<b>{html.escape(str(T.loc[s, 'name']))}</b><br>repo: {html.escape(str(T.loc[s, 'repo']))}" for s in (r.sha_a, r.sha_b)]
        table = hd.make_table(side[0], side[1], "Skill A", "Skill B", context=False)
        (KDIR / "html" / f"{r.pair_id}.html").write_text(
            f"<html><head><meta charset='utf-8'><title>{r.pair_id}</title><style>body{{font-family:sans-serif;"
            "font-size:13px}} table.diff{{font-family:monospace;font-size:12px}} .diff_add{{background:#cfc}}"
            ".diff_chg{{background:#ffa}} .diff_sub{{background:#fcc}} td{{vertical-align:top}}</style></head><body>"
            f"<h2>{r.pair_id}</h2><table width='100%'><tr><td width='50%'>A: {meta[0]}</td><td>B: {meta[1]}</td>"
            f"</tr></table>{table}</body></html>", encoding="utf8")
        links.append(f"<li><a href='{r.pair_id}.html'>{r.pair_id}</a> ({r.ann_1}, {r.ann_2})</li>")
    (KDIR / "html" / "index.html").write_text("<html><body><h2>Pairs to label</h2><ol>" + "".join(links) + "</ol></body></html>", encoding="utf8")
    print(json.dumps(strata, indent=1))
    print(f"wrote {len(K)} pairs to {KDIR}; {len(K) * 2 // len(ann)} pairs per annotator")

if args.score:
    K = pd.read_csv(KDIR / "pairs.csv")
    meta = json.load(open(KDIR / "strata.json"))
    lab = {}
    for a in set(K.ann_1) | set(K.ann_2):
        s = pd.read_csv(KDIR / f"sheet_{a}.csv", dtype=str).fillna("")
        lab[a] = {p: MAP.get(x.strip().lower()) for p, x in zip(s.pair_id, s.label) if x.strip()}
    K["l1"] = [lab[a].get(p) for a, p in zip(K.ann_1, K.pair_id)]
    K["l2"] = [lab[a].get(p) for a, p in zip(K.ann_2, K.pair_id)]
    if K.l1.isna().any() or K.l2.isna().any():
        raise SystemExit(f"{int(K.l1.isna().sum() + K.l2.isna().sum())} labels are missing or invalid")
    kappa = cohen_kappa_score(K.l1, K.l2)
    dis = K[K.l1 != K.l2]
    adj_path = KDIR / "adjudication.csv"
    adj = pd.read_csv(adj_path, dtype=str).fillna("") if adj_path.exists() else pd.DataFrame(columns=["pair_id", "label_1", "label_2", "final_label"])
    new = dis[~dis.pair_id.isin(adj.pair_id)][["pair_id", "l1", "l2"]].rename(columns={"l1": "label_1", "l2": "label_2"})
    adj = pd.concat([adj, new.assign(final_label="")], ignore_index=True)
    adj.to_csv(adj_path, index=False)
    final = dict(zip(adj.pair_id, adj.final_label.str.strip().str.lower().map(MAP)))
    if any(pd.isna(final.get(p)) for p in dis.pair_id):
        raise SystemExit(f"{len(dis)} disagreements: fill final_label (D/T/U) in {adj_path}, then run --score again")
    K["final"] = [a if a == b else final[p] for p, a, b in zip(K.pair_id, K.l1, K.l2)]
    K["derived"] = K.final == "derived"
    val = json.load(open(RESULTS / "validation.json"))
    pos = K[K.stratum == "pos"]
    prec_k, prec_n = int(val["confirmed"]) + int(pos.derived.sum()), int(val["edges"]) + len(pos)
    links = meta["predicted_links"]
    bands, rows = sorted(meta["strata"]), {}
    for b in bands:
        part = K[K.stratum == "neg-" + b]
        rows[b] = {"population": meta["strata"][b]["population"], "sampled": len(part), "derived": int(part.derived.sum())}
    rng = np.random.default_rng(SEED)
    B = 10_000
    tp_s = links * rng.beta(prec_k + 0.5, prec_n - prec_k + 0.5, B)
    fn_s = sum(r["population"] * rng.beta(r["derived"] + 0.5, r["sampled"] - r["derived"] + 0.5, B) for r in rows.values() if r["sampled"])
    rec_s = tp_s / (tp_s + fn_s)
    tp, fn = float(np.median(tp_s)), float(np.median(fn_s))
    recall, lo, hi = float(np.median(rec_s)), float(np.percentile(rec_s, 2.5)), float(np.percentile(rec_s, 97.5))
    out = {"kappa": float(kappa), "pairs": len(K), "precision": [prec_k, prec_n], "bands": rows,
           "tp_estimate": tp, "fn_estimate": fn, "recall": recall, "recall_ci": [lo, hi]}
    json.dump(out, open(RESULTS / f"{args.kit}.json", "w"), indent=2)
    mac = {"ResRCPairs": len(K), "ResRCKappa": f"{kappa:.2f}", "ResRCNeg": int((K.stratum != 'pos').sum()),
           "ResRCNegDerived": int(K[K.stratum != 'pos'].derived.sum()), "ResRCPos": len(pos), "ResRCPosDerived": int(pos.derived.sum()),
           "ResRCRecall": f"{recall:.2f}", "ResRCRecallLo": f"{lo:.2f}", "ResRCRecallHi": f"{hi:.2f}", "ResRCMissed": f"{fn:,.0f}"}
    for b, r in rows.items():
        mac[f"ResRC{b}Sampled"], mac[f"ResRC{b}Derived"] = r["sampled"], r["derived"]
    (RESULTS / f"macros_{args.kit}.tex").write_text("".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    for b, r in rows.items():
        print(f"{b}: {r['derived']}/{r['sampled']} sampled pairs predicted not derived are derived (population {r['population']:,})")
    print(f"predicted-derived pairs of the kit confirmed: {int(pos.derived.sum())}/{len(pos)}; precision used {prec_k}/{prec_n}")
    print(f"estimated recall within candidates {recall:.3f} [95% {lo:.3f}, {hi:.3f}]; estimated missed derivations {fn:,.0f}; kappa {kappa:.3f}")
