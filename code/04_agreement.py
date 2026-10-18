"""Step 4 - Agreement, adjudication, gold set.

Reads labeling/sheet_*.csv, computes Cohen's kappa (3-class and derived-vs-rest),
writes labeling/adjudication.csv for the disagreements, and - once every
disagreement has a final_label - writes labeling/gold.csv.
Run it twice: once after labeling, once after the team fills adjudication.csv.
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import LABEL_DIR, RESULTS, save_json

MAP = {"d": "derived", "derived": "derived", "t": "template", "template": "template",
       "u": "unrelated", "unrelated": "unrelated"}

pairs = pd.read_csv(LABEL_DIR / "pairs.csv")
labels = {}
for a in sorted(set(pairs.ann_1) | set(pairs.ann_2)):
    sh = pd.read_csv(LABEL_DIR / f"sheet_{a}.csv", dtype=str).fillna("")
    sh["label"] = sh["label"].str.strip().str.lower().map(MAP)
    bad = sh[sh.label.isna()]
    if len(bad):
        raise SystemExit(f"{a}: {len(bad)} rows missing/invalid label, e.g. {bad.pair_id.head(5).tolist()}")
    labels[a] = dict(zip(sh.pair_id, sh.label))

pairs["label_1"] = [labels[a][p] for a, p in zip(pairs.ann_1, pairs.pair_id)]
pairs["label_2"] = [labels[a][p] for a, p in zip(pairs.ann_2, pairs.pair_id)]


def kappas(df):
    b1, b2 = df.label_1.eq("derived"), df.label_2.eq("derived")
    return {"n": len(df),
            "agreement": float((df.label_1 == df.label_2).mean()),
            "kappa_3class": float(cohen_kappa_score(df.label_1, df.label_2)),
            "kappa_binary": float(cohen_kappa_score(b1, b2))}


out = {"overall": kappas(pairs)}
for (a1, a2), g in pairs.groupby(["ann_1", "ann_2"]):
    out[f"{a1}+{a2}"] = kappas(g)
out["label_counts_ann1"] = pairs.label_1.value_counts().to_dict()
save_json(out, RESULTS / "agreement.json")
print(out)

dis = pairs[pairs.label_1 != pairs.label_2]
adj_path = LABEL_DIR / "adjudication.csv"
if not adj_path.exists():
    d = dis[["pair_id", "ann_1", "label_1", "ann_2", "label_2"]].copy()
    d["final_label"] = ""
    d.to_csv(adj_path, index=False)
    print(f"\n{len(dis)} disagreements written to {adj_path}. Discuss each, fill final_label "
          f"(D/T/U), then run this script again.")
    raise SystemExit(0)

adj = pd.read_csv(adj_path, dtype=str).fillna("")
adj["final_label"] = adj.final_label.str.strip().str.lower().map(MAP)
missing = set(dis.pair_id) - set(adj.loc[adj.final_label.notna(), "pair_id"])
if missing:
    raise SystemExit(f"{len(missing)} disagreements still lack a final_label in {adj_path}")
final = dict(zip(adj.pair_id, adj.final_label))
pairs["label"] = [final.get(p, l1) for p, l1 in zip(pairs.pair_id, pairs.label_1)]
gold = pairs[["pair_id", "i", "j", "sha_a", "sha_b", "band", "label"]]
gold.to_csv(LABEL_DIR / "gold.csv", index=False)

k = out["overall"]
summary = {"gold_pairs": len(gold), **{f"gold_{c}": int((gold.label == c).sum())
                                         for c in ("derived", "template", "unrelated")},
           "disagreements": len(dis)}
save_json({**out, **summary}, RESULTS / "agreement.json")
macros = [r"\renewcommand{\ResGoldPairs}{%d}" % len(gold),
          r"\renewcommand{\ResGoldDerived}{%d}" % summary["gold_derived"],
          r"\renewcommand{\ResGoldTemplate}{%d}" % summary["gold_template"],
          r"\renewcommand{\ResGoldUnrelated}{%d}" % summary["gold_unrelated"],
          r"\renewcommand{\ResKappaThree}{%.2f}" % k["kappa_3class"],
          r"\renewcommand{\ResKappaBin}{%.2f}" % k["kappa_binary"],
          r"\renewcommand{\ResAgree}{%.1f\%%}" % (100 * k["agreement"]),
          r"\renewcommand{\ResDisagree}{%d}" % len(dis)]
(RESULTS / "macros_agreement.tex").write_text("\n".join(macros) + "\n")
print(f"gold set: {summary}")
