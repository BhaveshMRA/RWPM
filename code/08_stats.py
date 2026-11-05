"""Step 8 - Significance and confidence intervals for RQ1.

  python 08_stats.py   (after 06_rq1_rq2.py)

Reads results/rq1_oof_predictions.csv and reports, for the derived class:
  * 95% bootstrap confidence intervals of P, R and F1 for every method (resampling labelled pairs with replacement, 2,000 times);
  * McNemar's exact test between our best model and the strongest baseline, and between our best model and the Phase 2 feature set (when the semantic group was used).
Writes results/rq1_stats.tex and results/macros_stats.tex.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from common import RESULTS, SEED

B = 2000
oof = pd.read_csv(RESULTS / "rq1_oof_predictions.csv")
info = json.load(open(RESULTS / "rq1_rq2.json"))
best = info["best_ours"]
meta = {"pair_id", "sha_a", "sha_b", "band", "label", "ours_class"}
methods = [c for c in oof.columns if c not in meta and not c.startswith("ablation: ")]
baselines = [m for m in methods if not m.startswith("Ours")]
y = (oof.label == "derived").astype(int).values
n = len(y)


def prf(t, p):
    tp = int(((t == 1) & (p == 1)).sum())
    P = tp / p.sum() if p.sum() else 0.0
    R = tp / t.sum() if t.sum() else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)


rng = np.random.default_rng(SEED)
idx = rng.integers(0, n, size=(B, n))
stats = {}
for m in methods:
    p = oof[m].values.astype(int)
    point = prf(y, p)
    boot = np.array([prf(y[i], p[i]) for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    stats[m] = {"point": point, "lo": lo, "hi": hi}

best_base = max(baselines, key=lambda m: stats[m]["point"][2])


def mcnemar(a, b):
    ca, cb = oof[a].values.astype(int) == y, oof[b].values.astype(int) == y
    only_a, only_b = int((ca & ~cb).sum()), int((~ca & cb).sum())
    pval = binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue if only_a + only_b else 1.0
    return only_a, only_b, pval


def fmt_p(p):
    return r"$<$0.001" if p < 0.001 else f"{p:.3f}"


diff = np.array([prf(y[i], oof[best].values[i])[2] - prf(y[i], oof[best_base].values[i])[2] for i in idx])
b1, c1, p1 = mcnemar(best, best_base)
mac = {
    "ResOursFLo": f"{stats[best]['lo'][2]:.2f}", "ResOursFHi": f"{stats[best]['hi'][2]:.2f}",
    "ResBaseFLo": f"{stats[best_base]['lo'][2]:.2f}", "ResBaseFHi": f"{stats[best_base]['hi'][2]:.2f}",
    "ResDiffFLo": f"{np.percentile(diff, 2.5):.2f}", "ResDiffFHi": f"{np.percentile(diff, 97.5):.2f}",
    "ResMcNemarOnlyOurs": str(b1), "ResMcNemarOnlyBase": str(c1), "ResMcNemarP": fmt_p(p1),
}
print(f"{best} vs {best_base}: ours right only {b1}, baseline right only {c1}, McNemar p={p1:.2g}")
print(f"F1 difference 95% CI: [{np.percentile(diff, 2.5):.3f}, {np.percentile(diff, 97.5):.3f}]")

phase2 = "ablation: without semantic"
if phase2 in oof.columns:
    p2 = oof[phase2].values.astype(int)
    P2 = prf(y, p2)
    boot2 = np.array([prf(y[i], p2[i]) for i in idx])
    b2, c2, pv2 = mcnemar(best, phase2)
    mac.update({"ResPhaseTwoFLo": f"{np.percentile(boot2[:, 2], 2.5):.2f}",
                "ResPhaseTwoFHi": f"{np.percentile(boot2[:, 2], 97.5):.2f}",
                "ResPhaseMcNemarOnlyNew": str(b2), "ResPhaseMcNemarOnlyOld": str(c2),
                "ResPhaseMcNemarP": fmt_p(pv2)})
    print(f"with vs without semantic: new right only {b2}, old right only {c2}, McNemar p={pv2:.2g}")

rows = [r"\begin{tabular}{lccc}", r"\toprule",
        r"Method & P [95\% CI] & R [95\% CI] & F1 [95\% CI] \\", r"\midrule"]
for m in methods:
    s = stats[m]
    cells = [f"{s['point'][k]:.2f} [{s['lo'][k]:.2f}, {s['hi'][k]:.2f}]" for k in range(3)]
    rows.append(f"{m} & " + " & ".join(cells) + r" \\")
    if m == baselines[-1]:
        rows.append(r"\midrule")
rows += [r"\bottomrule", r"\end{tabular}"]
(RESULTS / "rq1_stats.tex").write_text("\n".join(rows) + "\n")
(RESULTS / "macros_stats.tex").write_text("\n".join(rf"\renewcommand{{\{k}}}{{{v}}}" for k, v in mac.items()) + "\n")
json.dump({m: {k: [float(v) for v in stats[m][k]] for k in ("point", "lo", "hi")} for m in methods}, open(RESULTS / "rq1_stats.json", "w"), indent=2)
print("wrote rq1_stats.tex, macros_stats.tex")
