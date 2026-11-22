"""Step 18 - Stress test on cross-language and heavily reworked copies.

    python 18_stress_kit.py --make     -> labeling/stress/ (sheets, viewer pages, pairs.csv)
    (label the sheets in labeling/stress/ with the usual rubric)
    python 18_stress_kit.py --score    -> results/macros_stress.tex, results/stress.json

At full scale, translated or heavily reworked copies share too little text for MinHash, so they can only become candidates through same-name blocking. The kit therefore samples same-name candidate pairs outside the gold set and the validation sample, of two pre-set types:
  * cross-language: the prose of one skill is mostly in a non-Latin script (CJK, kana, Hangul, Cyrillic, Arabic, Hebrew, Devanagari, Thai) and the prose of the other mostly Latin;
  * reworked: both skills mostly Latin, little shared text (shingle Jaccard < 0.1), similar descriptions (description TF-IDF cosine >= 0.5).
The model's prediction is stored in pairs.csv but not shown to the annotators. --score reports the agreement, how many pairs are derived, and the recall and precision of the model on each type.
"""
import argparse
import difflib
import html
import json
import re
import shutil
import sqlite3

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from common import ALL_FEATURES, LABEL_DIR, RESULTS, SEED, WORK

ap = argparse.ArgumentParser()
ap.add_argument("--make", action="store_true")
ap.add_argument("--score", action="store_true")
ap.add_argument("--annotators", default="Bhavesh,Abhishek,Mahima,Pratiksha")
ap.add_argument("--per-type", type=int, default=30)
ap.add_argument("--kit", default="stress")
args = ap.parse_args()
SDIR = LABEL_DIR / args.kit
MAP = {"d": "derived", "derived": "derived", "t": "template", "template": "template", "u": "unrelated", "unrelated": "unrelated"}
NONLATIN = re.compile(r"[\u0400-\u04ff\u0590-\u05ff\u0600-\u06ff\u0900-\u097f\u0e00-\u0e7f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
LATIN = re.compile(r"[A-Za-z]")
FENCE = re.compile(r"```.*?```", re.S)


def nonlatin_share(body):
    t = FENCE.sub(" ", body or "")[:4000]
    n, l = len(NONLATIN.findall(t)), len(LATIN.findall(t))
    return n / (n + l) if n + l else 0.0


def wilson(k, n, z=norm.ppf(0.975)):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


if args.make:
    make = {"Ours (logistic regression)": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
            "Ours (random forest)": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
            "Ours (gradient boosting)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)}
    best = json.load(open(RESULTS / "rq1_rq2.json"))
    gold = pd.read_pickle(WORK / "features_gold.pkl")
    allp = pd.read_pickle(WORK / "features_all.pkl")
    skills = pd.read_pickle(WORK / "skills.pkl")
    sha = skills.sha.values
    if "sha_a" not in allp.columns:
        allp["sha_a"], allp["sha_b"] = sha[allp.i.values], sha[allp.j.values]
    used = set(zip(gold.sha_a, gold.sha_b))
    vp = LABEL_DIR / "validation" / "pairs.csv"
    if vp.exists():
        v = pd.read_csv(vp)
        used |= set(zip(v.sha_a, v.sha_b))
    used |= {(b, a) for a, b in used}
    pool = allp[(allp.name_equal == 1) & ~np.array([(a, b) in used for a, b in zip(allp.sha_a, allp.sha_b)])].copy()
    con = sqlite3.connect(WORK / "full_skills.db")
    need = sorted(set(pool.sha_a) | set(pool.sha_b))
    share = {}
    for k in range(0, len(need), 900):
        part = need[k:k + 900]
        for s, body in con.execute(f"SELECT sha, body FROM skills WHERE sha IN ({','.join('?' * len(part))})", part):
            share[s] = nonlatin_share(body)
    sa, sb = pool.sha_a.map(share).fillna(0).values, pool.sha_b.map(share).fillna(0).values
    cross = ((sa >= 0.3) & (sb <= 0.05)) | ((sb >= 0.3) & (sa <= 0.05))
    rework = (sa <= 0.05) & (sb <= 0.05) & (pool.jac_shingle.values < 0.1) & (pool.desc_tfidf.values >= 0.5)
    rng = np.random.default_rng(SEED)
    picks = []
    for kind, m in (("cross-language", cross), ("reworked", rework)):
        idx = np.flatnonzero(m)
        print(f"{kind}: {len(idx):,} eligible same-name pairs")
        picks.append(pool.iloc[rng.choice(idx, min(args.per_type, len(idx)), replace=False)].assign(kind=kind))
    S = pd.concat(picks)
    y = gold.label.values if best["three_class"] else np.where(gold.label == "derived", "derived", "other")
    model = make[best["best_ours"]]().fit(gold[ALL_FEATURES].values, y)
    S["predicted"] = np.where(model.predict(S[ALL_FEATURES].values) == "derived", "derived", "other")
    S = S.sample(frac=1, random_state=SEED).reset_index(drop=True)
    swap = rng.random(len(S)) < 0.5
    S.loc[swap, ["sha_a", "sha_b"]] = S.loc[swap, ["sha_b", "sha_a"]].values
    S["pair_id"] = [f"S{k:03d}" for k in range(len(S))]
    ann = [a.strip() for a in args.annotators.split(",") if a.strip()]
    teams = [(ann[k], ann[k + 1]) for k in range(0, len(ann), 2)]
    S["ann_1"] = [teams[k % len(teams)][0] for k in range(len(S))]
    S["ann_2"] = [teams[k % len(teams)][1] for k in range(len(S))]
    if SDIR.exists() and any(SDIR.glob("sheet_*.csv")) and not input(f"Overwrite existing kit {SDIR}? [y/N] ").lower().startswith("y"):
        raise SystemExit("kept the existing kit")
    shutil.rmtree(SDIR, ignore_errors=True)
    (SDIR / "html").mkdir(parents=True, exist_ok=True)
    S[["pair_id", "sha_a", "sha_b", "band", "kind", "predicted", "name_equal", "same_owner", "ann_1", "ann_2"]].to_csv(SDIR / "pairs.csv", index=False)
    for a in ann:
        s = S[(S.ann_1 == a) | (S.ann_2 == a)][["pair_id"]].copy()
        s["label"], s["confidence"], s["note"] = "", "", ""
        s.to_csv(SDIR / f"sheet_{a}.csv", index=False)
    if (LABEL_DIR / "GUIDELINES.md").exists():
        shutil.copy(LABEL_DIR / "GUIDELINES.md", SDIR / "GUIDELINES.md")
    need = sorted(set(S.sha_a) | set(S.sha_b))
    T = pd.read_sql_query(f"SELECT sha, name, description, body, repo FROM skills WHERE sha IN ({','.join('?' * len(need))})",
                          con, params=need).set_index("sha")
    hd = difflib.HtmlDiff(wrapcolumn=72)
    links = []
    for r in S.itertuples():
        side = [(f"name: {T.loc[s, 'name']}\ndescription: {T.loc[s, 'description']}\n---\n{T.loc[s, 'body']}").splitlines() for s in (r.sha_a, r.sha_b)]
        meta = [f"<b>{html.escape(str(T.loc[s, 'name']))}</b><br>repo: {html.escape(str(T.loc[s, 'repo']))}" for s in (r.sha_a, r.sha_b)]
        table = hd.make_table(side[0], side[1], "Skill A", "Skill B", context=False)
        (SDIR / "html" / f"{r.pair_id}.html").write_text(
            f"<html><head><meta charset='utf-8'><title>{r.pair_id}</title><style>body{{font-family:sans-serif;"
            "font-size:13px}} table.diff{{font-family:monospace;font-size:12px}} .diff_add{{background:#cfc}}"
            ".diff_chg{{background:#ffa}} .diff_sub{{background:#fcc}} td{{vertical-align:top}}</style></head><body>"
            f"<h2>{r.pair_id}</h2><table width='100%'><tr><td width='50%'>A: {meta[0]}</td><td>B: {meta[1]}</td>"
            f"</tr></table>{table}</body></html>", encoding="utf8")
        links.append(f"<li><a href='{r.pair_id}.html'>{r.pair_id}</a> ({r.ann_1}, {r.ann_2})</li>")
    (SDIR / "html" / "index.html").write_text("<html><body><h2>Pairs to label</h2><ol>" + "".join(links) + "</ol></body></html>", encoding="utf8")
    print(f"wrote {len(S)} pairs to {SDIR}; {len(S) * 2 // len(ann)} pairs per annotator")

if args.score:
    S = pd.read_csv(SDIR / "pairs.csv")
    lab = {}
    for a in set(S.ann_1) | set(S.ann_2):
        s = pd.read_csv(SDIR / f"sheet_{a}.csv", dtype=str).fillna("")
        lab[a] = {p: MAP.get(x.strip().lower()) for p, x in zip(s.pair_id, s.label) if x.strip()}
    S["l1"] = [lab[a].get(p) for a, p in zip(S.ann_1, S.pair_id)]
    S["l2"] = [lab[a].get(p) for a, p in zip(S.ann_2, S.pair_id)]
    if S.l1.isna().any() or S.l2.isna().any():
        raise SystemExit(f"{int(S.l1.isna().sum() + S.l2.isna().sum())} labels are missing or invalid")
    kappa = cohen_kappa_score(S.l1, S.l2)
    dis = S[S.l1 != S.l2]
    adj_path = SDIR / "adjudication.csv"
    adj = pd.read_csv(adj_path, dtype=str).fillna("") if adj_path.exists() else pd.DataFrame(columns=["pair_id", "label_1", "label_2", "final_label"])
    new = dis[~dis.pair_id.isin(adj.pair_id)][["pair_id", "l1", "l2"]].rename(columns={"l1": "label_1", "l2": "label_2"})
    adj = pd.concat([adj, new.assign(final_label="")], ignore_index=True)
    adj.to_csv(adj_path, index=False)
    final = dict(zip(adj.pair_id, adj.final_label.str.strip().str.lower().map(MAP)))
    if any(pd.isna(final.get(p)) for p in dis.pair_id):
        raise SystemExit(f"{len(dis)} disagreements: fill final_label (D/T/U) in {adj_path}, then run --score again")
    S["final"] = [a if a == b else final[p] for p, a, b in zip(S.pair_id, S.l1, S.l2)]
    out, mac = {"kappa": float(kappa), "pairs": len(S)}, {"ResSTPairs": len(S), "ResSTKappa": f"{kappa:.2f}"}
    for key, part in (("All", S), ("Cross", S[S.kind == "cross-language"]), ("Rework", S[S.kind == "reworked"])):
        d = part.final == "derived"
        p = part.predicted == "derived"
        tp = int((d & p).sum())
        rec = tp / d.sum() if d.sum() else float("nan")
        prec = tp / p.sum() if p.sum() else float("nan")
        lo, hi = wilson(tp, int(d.sum())) if d.sum() else (float("nan"), float("nan"))
        out[key] = {"n": len(part), "derived": int(d.sum()), "predicted": int(p.sum()), "true_positive": tp, "recall": rec, "precision": prec, "recall_ci": [lo, hi]}
        mac.update({f"ResST{key}N": len(part), f"ResST{key}Derived": int(d.sum()), f"ResST{key}TP": tp,
                    f"ResST{key}Recall": f"{rec:.2f}" if d.sum() else "--", f"ResST{key}Precision": f"{prec:.2f}" if p.sum() else "--",
                    f"ResST{key}RecallLo": f"{lo:.2f}" if d.sum() else "--", f"ResST{key}RecallHi": f"{hi:.2f}" if d.sum() else "--"})
        print(f"{key:7s}: {len(part)} pairs, {int(d.sum())} derived; model finds {tp} (recall {rec:.2f}), precision {prec:.2f}")
    (RESULTS / f"macros_{args.kit}.tex").write_text("".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
    json.dump(out, open(RESULTS / f"{args.kit}.json", "w"), indent=2)
    print(f"kappa={kappa:.3f}; wrote macros_{args.kit}.tex and {args.kit}.json")
