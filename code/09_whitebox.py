"""Step 9 - White-box figures.

  python 09_whitebox.py [--pairs P0118,P0087]   (after 06_rq1_rq2.py)

Figure 1 (results/fig_pair_features.pdf): how two labelled pairs become features. For each skill, its normalised lines are split into lines shared with the other skill that are distinctive, shared lines that are boilerplate, and lines it does not share; the key feature values are printed beside them. By default it contrasts a template pair with high text overlap and a derived pair with lower overlap.

Figure 2 (results/fig_model.pdf): what the model relies on. (a) Coefficients of a logistic regression on standardised features for the derived class. (b) Group permutation importance of the best model: the drop in derived-class F1 on held-out folds when all features of one group are shuffled together.
"""
import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from common import ALL_FEATURES, FEATURE_GROUPS, RESULTS, SEED, WORK, norm_lines  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", default="P0206,P0087", help="two gold pair ids to draw")
ap.add_argument("--boiler-k", type=int, default=5)
ap.add_argument("--repeats", type=int, default=10)
args = ap.parse_args()

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "pdf.fonttype": 42})
COLORS = {"lexical": "#4C72B0", "boilerplate": "#DD8452", "metadata": "#55A868", "code": "#8172B3", "structure": "#937860", "repository": "#C44E52", "semantic": "#2AA3B5"}
LABELS = {"jac_shingle": "shingle Jaccard", "cont_shingle": "shingle containment", "tfidf_body": "TF-IDF cosine (body)", "lev_ratio": "edit similarity", "jac_distinct_lines": "Jaccard, distinctive lines", "cont_distinct_lines": "containment, distinctive lines", "jac_boiler_lines": "Jaccard, boilerplate lines", "frac_shared_boiler": "share of shared lines that are boilerplate", "name_equal": "same name", "name_sim": "name similarity", "desc_tfidf": "TF-IDF cosine (description)", "code_jac": "code identifier Jaccard", "code_both": "both contain code", "head_seq_sim": "heading-sequence similarity", "len_ratio": "length ratio", "heads_ratio": "heading-count ratio", "same_owner": "same owner", "same_dirname": "same folder name", "siblings_jac": "bundled-file Jaccard", "emb_body": "embedding cosine (body)", "emb_desc": "embedding cosine (description)"}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

skills = pd.read_pickle(WORK / "skills.pkl")
F = pd.read_pickle(WORK / "features_gold.pkl").reset_index(drop=True)
info = json.load(open(RESULTS / "rq1_rq2.json"))

# Figure 1: one pair, line by line
lines = [norm_lines(b) for b in skills.body]
name_key = [n if n else s for n, s in zip(skills.nname, skills.sha)]
names_per_line = defaultdict(set)
for nk, ls in zip(name_key, lines):
    for ln in ls:
        names_per_line[ln].add(nk)
boiler = {ln for ln, s in names_per_line.items() if len(s) >= args.boiler_k}

ids = args.pairs.split(",")
fig, axes = plt.subplots(1, len(ids), figsize=(7.0, 2.1))
for ax, pid in zip(np.atleast_1d(axes), ids):
    r = F[F.pair_id == pid].iloc[0]
    A, Bk = int(r.a), int(r.b)
    shared = lines[A] & lines[Bk]
    sd, sb = len(shared - boiler), len(shared & boiler)
    for row, k in ((1, A), (0, Bk)):
        own = len(lines[k]) - sd - sb
        ax.barh(row, sd, color="#2E7D32", label="shared, distinctive" if row else None)
        ax.barh(row, sb, left=sd, color="#F9A825", label="shared, boilerplate" if row else None)
        ax.barh(row, own, left=sd + sb, color="#BDBDBD", label="not shared" if row else None)
    ax.set_yticks([1, 0], ["skill A", "skill B"])
    ax.set_xlabel("normalised lines")
    ax.set_title(f"{pid}: labelled {r.label}", fontsize=8)
    txt = (f"shingle Jaccard {r.jac_shingle:.2f}\n"
           f"distinctive-line Jaccard {r.jac_distinct_lines:.2f}\n"
           f"boilerplate share {r.frac_shared_boiler:.2f}\n"
           f"same name {int(r.name_equal)}, same owner {int(r.same_owner)}")
    ax.text(1.02, 0.5, txt, transform=ax.transAxes, va="center", fontsize=7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
handles, labels_ = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(handles, labels_, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.08, 0.86, 1), w_pad=8)
fig.savefig(RESULTS / "fig_pair_features.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 2: what the model relies on
y3 = F.label.values
yb = (y3 == "derived").astype(int)
three = info["three_class"]
yc = y3 if three else np.where(yb == 1, "derived", "other")
X = F[ALL_FEATURES].values

lr = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)).fit(X, yc)
clf = lr[-1]
row = list(clf.classes_).index("derived")
coef = clf.coef_[row] if clf.coef_.shape[0] > 1 else clf.coef_[0] * (1 if clf.classes_[1] == "derived" else -1)

MAKE = {"Ours (logistic regression)": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
        "Ours (random forest)": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
        "Ours (gradient boosting)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)}
G = nx.Graph()
G.add_edges_from(zip(F.sha_a, F.sha_b))
comp = {s: k for k, c in enumerate(nx.connected_components(G)) for s in c}
groups = F.sha_a.map(comp).values
if pd.Series(groups).value_counts().iloc[0] / len(F) > 0.25:
    groups = pd.factorize(F[["sha_a", "sha_b"]].min(axis=1))[0]
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(F, yb, groups))
rng = np.random.default_rng(SEED)
imp = {g: [] for g in FEATURE_GROUPS}
for tr, te in folds:
    m = MAKE[info["best_ours"]]().fit(X[tr], yc[tr])
    base = f1_score(yb[te], m.predict(X[te]) == "derived", zero_division=0)
    for g, cols in FEATURE_GROUPS.items():
        ci = [ALL_FEATURES.index(c) for c in cols]
        for _ in range(args.repeats):
            Xp = X[te].copy()
            Xp[:, ci] = Xp[rng.permutation(len(te))][:, ci]
            imp[g].append(base - f1_score(yb[te], m.predict(Xp) == "derived", zero_division=0))
imp = {g: float(np.mean(v)) for g, v in imp.items()}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.5, 1]})
order = np.argsort(coef)
a1.barh([LABELS.get(ALL_FEATURES[i], ALL_FEATURES[i]) for i in order], coef[order], color=[COLORS[GROUP_OF[ALL_FEATURES[i]]] for i in order])
a1.axvline(0, color="black", lw=0.6)
a1.set_xlabel("coefficient for derived (standardised features)")
a1.set_title("(a) logistic regression", fontsize=8)
gs = sorted(imp, key=imp.get)
a2.barh(gs, [imp[g] for g in gs], color=[COLORS[g] for g in gs])
a2.axvline(0, color="black", lw=0.6)
a2.set_xlabel("drop in derived-class F1 when shuffled")
a2.set_title(f"(b) {info['best_ours'].replace('Ours (', '').rstrip(')')}, held-out folds", fontsize=8)
for ax in (a1, a2):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.tight_layout()
fig.savefig(RESULTS / "fig_model.pdf", bbox_inches="tight")
plt.close(fig)
json.dump({"lr_coef_derived": dict(zip(ALL_FEATURES, map(float, coef))), "group_permutation_importance": imp}, open(RESULTS / "whitebox.json", "w"), indent=2)
print("group permutation importance:", {g: round(v, 3) for g, v in sorted(imp.items(), key=lambda kv: -kv[1])})
print("wrote fig_pair_features.pdf, fig_model.pdf")
