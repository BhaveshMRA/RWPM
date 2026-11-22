"""find_relatives - list the likely edited relatives of a skill in the GitSkills release.

    python find_relatives.py path/to/skill-folder [--owner GITHUB_ACCOUNT] [--top 20] [--out relatives.csv]

Given a SKILL.md file (in a folder, as skills are distributed), the tool finds the skills of the
release that are probably edited copies of it or sources it was copied from. It applies the method
of the paper unchanged: candidates come from the same two blockers (MinHash estimate of the shingle
Jaccard >= 0.3, and the same normalised front-matter name), every candidate pair gets the 19
features of step 13 (computed by step 13's own code, with the same boilerplate set and text models),
and the best classifier of RQ1, trained on all gold pairs, gives the probability that the pair is
derived. It needs the work folder of a full-pipeline run (steps 10-13) and the results of step 6:
SKILL_WORK and SKILL_RESULTS point to them, as for the other steps.
"""
import argparse
import hashlib
import importlib
import json
import pickle
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash

from common import ALL_FEATURES, MIN_BODY_CHARS, RESULTS, SEED, WORK, headings, norm_name, shingles, split_code, split_frontmatter

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("skill", help="a skill folder, or the SKILL.md file inside it")
ap.add_argument("--owner", default="", help="GitHub account that owns the skill (for the same-owner feature)")
ap.add_argument("--top", type=int, default=20)
ap.add_argument("--min-jaccard", type=float, default=0.3, help="MinHash candidate threshold (as in step 12)")
ap.add_argument("--max-candidates", type=int, default=300)
ap.add_argument("--out", help="also write all scored candidates to this CSV file")
args = ap.parse_args()
f13 = importlib.import_module("13_full_features")

# ---- the query skill, built exactly like a row of step 10 -------------------------------------
path = Path(args.skill)
if path.is_dir():
    path = path / "SKILL.md"
if not path.is_file():
    sys.exit(f"no SKILL.md found at {path}: give the skill's folder or its SKILL.md file")
raw = path.read_bytes()
sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
text = raw.decode("utf8", errors="replace")
fm, body = split_frontmatter(text)


def field(key):
    m = re.search(rf"^{key}\s*:\s*(.*)$", fm, re.M)
    return m.group(1).strip().strip("'\"") if m else ""


name, description = field("name"), field("description")
if len(body) < MIN_BODY_CHARS:
    print(f"warning: the body has {len(body)} characters; the paper's scope requires at least {MIN_BODY_CHARS}")
fence_lang = re.compile(r"^ {0,3}(?:`{3,}|~{3,})[ \t]*([\w+#-]+)", re.M)
prose, code_blocks = split_code(body)
q = pd.DataFrame([{
    "id": -1, "sha": sha, "name": name, "nname": norm_name(name), "description": description, "body": body,
    "fm_fields": len(re.findall(r"^[A-Za-z][\w-]*\s*:", fm, re.M)),
    "code_langs": sorted(set(x.lower() for x in fence_lang.findall(body))), "headings": headings(prose),
    "dirname": path.resolve().parent.name.lower(), "owners": [args.owner] if args.owner else [],
    "siblings": sorted(p.name for p in path.resolve().parent.iterdir() if p.name != path.name),
    "prose": prose, "code_blocks": code_blocks}])

# ---- candidates: the same two blockers as the pipeline -----------------------------------------
con = sqlite3.connect(f"file:{WORK / 'full_skills.db'}?mode=ro", uri=True)
same = con.execute("SELECT COUNT(*) FROM skills WHERE sha = ?", (sha,)).fetchone()[0]
sig = np.load(WORK / "minhash.npy", mmap_mode="r")
qsig = MinHash.bulk([[x.encode("utf8") for x in shingles(body)] or [b""]], num_perm=sig.shape[1], seed=SEED)[0].hashvalues
est = np.concatenate([(np.asarray(sig[k:k + 200_000]) == qsig).mean(1) for k in range(0, len(sig), 200_000)])
lsh = np.flatnonzero(est >= args.min_jaccard)
lsh = lsh[np.argsort(-est[lsh])][:args.max_candidates]
by_name = [r[0] for r in con.execute("SELECT id FROM skills WHERE nname = ? AND nname != '' LIMIT ?", (q.nname[0], args.max_candidates))]
ids = [int(i) for i in dict.fromkeys(list(lsh) + by_name)]
ids = [i for i in ids if con.execute("SELECT sha FROM skills WHERE id = ?", (i,)).fetchone()[0] != sha]
if not ids:
    print("no candidates: no skill in the release shares enough text or the name with this one")
    sys.exit(0)

# ---- features (step 13's own code) and the paper's best model ----------------------------------
m = pickle.load(open(WORK / "full_models.pkl", "rb"))
S = pd.concat([q, f13.load(con, ids)], ignore_index=True)
F = f13.features(m, S, np.zeros(len(ids), dtype=int), np.arange(1, len(ids) + 1), sr=False)
cache = WORK / "find_relatives_model.pkl"
if cache.exists():
    model = pickle.load(open(cache, "rb"))
else:
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    make = {"Ours (logistic regression)": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=5000)),
            "Ours (random forest)": lambda: RandomForestClassifier(n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1),
            "Ours (gradient boosting)": lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=SEED)}
    best = json.load(open(RESULTS / "rq1_rq2.json"))
    gold = pd.read_pickle(WORK / "features_gold.pkl")
    y = gold.label.values if best["three_class"] else np.where(gold.label == "derived", "derived", "other")
    model = make[best["best_ours"]]().fit(gold[ALL_FEATURES].values, y)
    pickle.dump(model, open(cache, "wb"))
proba = model.predict_proba(F[ALL_FEATURES].values)
cls = list(model.classes_)
C = S.iloc[1:].reset_index(drop=True)
repos = {i: r for i, r in con.execute(f"SELECT id, repo FROM skills WHERE id IN ({','.join('?' * len(ids))})", ids)}
R = pd.DataFrame({"p_derived": proba[:, cls.index("derived")], "predicted": np.array(cls)[proba.argmax(1)],
                  "name": C.name.values, "repo": [repos[i] for i in C.id.values], "sha": C.sha.values,
                  "shingle_jaccard": F.jac_shingle.values, "distinct_line_jaccard": F.jac_distinct_lines.values,
                  "same_name": F.name_equal.values.astype(int)})
R = R.assign(_p=R.p_derived.round(3)).sort_values(["_p", "shingle_jaccard"], ascending=False).drop(columns="_p")
print(f"{path}: name '{name}', blob {sha[:12]}; {'this exact file occurs in the release' if same else 'not in the release as an exact copy'}")
print(f"{len(ids)} candidates ({len(lsh)} by MinHash, {len(by_name)} by name); {int((R.predicted == 'derived').sum())} predicted derived\n")
with pd.option_context("display.width", 160, "display.max_colwidth", 40):
    print(R.head(args.top).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
if args.out:
    R.to_csv(args.out, index=False)
    print(f"\nwrote {len(R)} scored candidates to {args.out}")
