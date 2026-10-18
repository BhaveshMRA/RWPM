"""Step 6 - RQ1 (effectiveness) and RQ2 (which signals matter).

5-fold StratifiedGroupKFold on the gold pairs. Groups are connected components
of the labelled-pair graph, so no skill appears in both a training and a test
fold. Baselines that output a score get their threshold tuned on the training
folds only. Writes LaTeX tables + macros into results/.
"""
import json

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import ALL_FEATURES, FEATURE_GROUPS, RESULTS, SEED, SR_FEATURES, WORK

F = pd.read_pickle(WORK / "features_gold.pkl").reset_index(drop=True)
y3 = F["label"].values
yb = (y3 == "derived").astype(int)
three = (y3 == "template").sum() >= 15
yc = y3 if three else np.where(yb == 1, "derived", "other")
print(f"{len(F)} gold pairs, derived={yb.sum()}, three-class mode={three}")

# ---- grouped folds -------------------------------------------------------------
G = nx.Graph()
G.add_edges_from(zip(F.sha_a, F.sha_b))
comp = {s: k for k, c in enumerate(nx.connected_components(G)) for s in c}
groups = F.sha_a.map(comp).values
largest = pd.Series(groups).value_counts().iloc[0] / len(F)
grouping = "connected components of the labelled-pair graph"
if largest > 0.25:
    groups = pd.factorize(F[["sha_a", "sha_b"]].min(axis=1))[0]
    grouping = "the lexicographically smaller skill of each pair (component too large)"
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, yb, groups))
hi = F.band.isin(["B1", "B2", "B3"]).values  # high-similarity bands (J >= 0.5)


def tune(s, y):
    cands = np.unique(np.quantile(s, np.linspace(0, 1, 201)))
    return max(cands, key=lambda t: f1_score(y, s >= t, zero_division=0))


def oof_threshold(col):
    p = np.zeros(len(F), int)
    for tr, te in folds:
        t = tune(F[col].values[tr], yb[tr])
        p[te] = F[col].values[te] >= t
    return p


def oof_model(make, feats, binary=False, tune_thr=False):
    target = yb if binary else yc
    pb = np.zeros(len(F), int)
    pc = np.empty(len(F), dtype=object)
    for tr, te in folds:
        m = make().fit(F[feats].values[tr], target[tr])
        if binary:
            if tune_thr:
                t = tune(m.predict_proba(F[feats].values[tr])[:, 1], yb[tr])
                pb[te] = m.predict_proba(F[feats].values[te])[:, 1] >= t
            else:
                pb[te] = m.predict(F[feats].values[te])
            pc[te] = np.where(pb[te] == 1, "derived", "other")
        else:
            pc[te] = m.predict(F[feats].values[te])
            pb[te] = pc[te] == "derived"
    return pb, pc


def metrics(p):
    fold_f1 = [f1_score(yb[te], p[te], zero_division=0) for _, te in folds]
    neg_hi = hi & (yb == 0)
    return {"P": precision_score(yb, p, zero_division=0), "R": recall_score(yb, p, zero_division=0),
            "F1": f1_score(yb, p, zero_division=0), "MCC": matthews_corrcoef(yb, p),
            "F1_sd": float(np.std(fold_f1)),
            "hiFP": float(p[neg_hi].mean()) if neg_hi.any() else float("nan")}


LR = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000))
RF = lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
                                    random_state=SEED, n_jobs=-1)
HGB = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                             random_state=SEED)
OURS = {"Ours (logistic regression)": LR, "Ours (random forest)": RF, "Ours (gradient boosting)": HGB}

res, preds = {}, {}
preds["Normalised-text hash"] = F["norm_hash_equal"].values.astype(int)
preds["Shingle Jaccard (MinHash), tuned threshold"] = oof_threshold("jac_shingle")
preds["TF-IDF cosine, tuned threshold"] = oof_threshold("tfidf_body")
preds["SkillReuse-style (re-implemented)"] = oof_model(LR, SR_FEATURES, binary=True, tune_thr=True)[0]
BASELINES = list(preds)
cls_preds = {}
for name, make in OURS.items():
    preds[name], cls_preds[name] = oof_model(make, ALL_FEATURES)
for k, p in preds.items():
    res[k] = metrics(p)
    print(f"{k:45s} " + " ".join(f"{m}={v:.3f}" for m, v in res[k].items()))

best_base = max(BASELINES, key=lambda k: res[k]["F1"])
best_ours = max(OURS, key=lambda k: res[k]["F1"])
labels = ["derived", "template", "unrelated"] if three else ["derived", "other"]
yc_true = y3 if three else np.where(yb == 1, "derived", "other")
cm = confusion_matrix(yc_true, cls_preds[best_ours].astype(str), labels=labels)
macro3 = f1_score(yc_true, cls_preds[best_ours].astype(str), labels=labels, average="macro")

# ---- RQ2: ablation by feature group -------------------------------------------
abl = {"All features": res[best_ours]}
for g, cols in FEATURE_GROUPS.items():
    feats = [f for f in ALL_FEATURES if f not in cols]
    abl[f"without {g}"] = metrics(oof_model(OURS[best_ours], feats)[0])
for g in ("lexical",):
    abl[f"{g} only"] = metrics(oof_model(OURS[best_ours], FEATURE_GROUPS[g])[0])
drops = {k: abl["All features"]["F1"] - v["F1"] for k, v in abl.items() if k.startswith("without")}
top_drop = max(drops, key=drops.get)

# ---- write tables --------------------------------------------------------------
f3 = lambda x: "--" if x != x else f"{x:.2f}"
rows = [r"\begin{tabular}{lccccc}", r"\toprule",
        r"Method & P & R & F1 & MCC & FP rate (hi-sim) \\", r"\midrule"]
for k in BASELINES + [r"\midrule"] + list(OURS):
    if k == r"\midrule":
        rows.append(k)
        continue
    m = res[k]
    name = rf"\textbf{{{k}}}" if k == best_ours else k
    rows.append(f"{name} & {f3(m['P'])} & {f3(m['R'])} & {f3(m['F1'])}$\\pm${m['F1_sd']:.2f} & "
                f"{f3(m['MCC'])} & {f3(m['hiFP'])} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "rq1_main.tex").write_text("\n".join(rows) + "\n")

BAND_DESC = {"B1": r"$J \geq 0.9$", "B2": r"$0.7 \leq J < 0.9$", "B3": r"$0.5 \leq J < 0.7$",
             "B4": r"$0.3 \leq J < 0.5$", "B5": r"same name, $J<0.3$", "B6": r"LSA neighbour, $J<0.3$",
             "B7": "random pair"}
rows = [r"\begin{tabular}{llrrcccc}", r"\toprule",
        r" & & & & \multicolumn{2}{c}{Best baseline} & \multicolumn{2}{c}{Ours} \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Band & Definition & Pairs & Derived & P & R & P & R \\", r"\midrule"]
for bnd in sorted(F.band.unique()):
    s = (F.band == bnd).values
    pb, po = preds[best_base][s], preds[best_ours][s]
    yy = yb[s]
    pr = lambda p: precision_score(yy, p, zero_division=0) if p.sum() else float("nan")
    rc = lambda p: recall_score(yy, p, zero_division=0) if yy.sum() else float("nan")
    rows.append(f"{bnd} & {BAND_DESC.get(bnd, bnd)} & {s.sum()} & {yy.sum()} & {f3(pr(pb))} & "
                f"{f3(rc(pb))} & {f3(pr(po))} & {f3(rc(po))} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "rq1_bands.tex").write_text("\n".join(rows) + "\n")

hdr = " & ".join(f"\\textit{{{l}}}" for l in labels)
rows = [r"\begin{tabular}{l" + "r" * len(labels) + "}", r"\toprule",
        rf"True $\downarrow$ / Predicted $\rightarrow$ & {hdr} \\", r"\midrule"]
for l, r in zip(labels, cm):
    rows.append(rf"\textit{{{l}}} & " + " & ".join(str(v) for v in r) + r" \\")
rows += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "rq1_confusion.tex").write_text("\n".join(rows) + "\n")

rows = [r"\begin{tabular}{lccc}", r"\toprule",
        r"Configuration & F1 & $\Delta$F1 & FP rate (hi-sim) \\", r"\midrule"]
for k, m in abl.items():
    d = "--" if k == "All features" else f"{m['F1'] - abl['All features']['F1']:+.2f}"
    rows.append(f"{k} & {f3(m['F1'])} & {d} & {f3(m['hiFP'])} \\\\")
rows += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "rq2_ablation.tex").write_text("\n".join(rows) + "\n")

mac = {
    "ResOursName": best_ours.replace("Ours (", "").rstrip(")"),
    "ResOursP": f"{res[best_ours]['P']:.2f}", "ResOursR": f"{res[best_ours]['R']:.2f}",
    "ResOursF": f"{res[best_ours]['F1']:.2f}", "ResOursMCC": f"{res[best_ours]['MCC']:.2f}",
    "ResOursHiFP": f"{res[best_ours]['hiFP']:.2f}",
    "ResBaseName": best_base, "ResBaseF": f"{res[best_base]['F1']:.2f}",
    "ResBaseHiFP": f"{res[best_base]['hiFP']:.2f}",
    "ResSRF": f"{res['SkillReuse-style (re-implemented)']['F1']:.2f}",
    "ResMacroF": f"{macro3:.2f}", "ResTopAblation": top_drop.replace("without ", ""),
    "ResTopAblationDrop": f"{drops[top_drop]:.2f}", "ResGrouping": grouping,
}
(RESULTS / "macros_rq1.tex").write_text(
    "\n".join(rf"\renewcommand{{\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
json.dump({"best_ours": best_ours, "three_class": bool(three), "results": res, "ablation": abl,
           "confusion": cm.tolist(), "labels": labels}, open(RESULTS / "rq1_rq2.json", "w"), indent=2)
out = F[["pair_id", "sha_a", "sha_b", "band", "label"]].copy()
for k in preds:
    out[k] = preds[k]
out["ours_class"] = cls_preds[best_ours]
out.to_csv(RESULTS / "rq1_oof_predictions.csv", index=False)
print(f"best baseline: {best_base} | best ours: {best_ours} | top ablation drop: {top_drop}")
