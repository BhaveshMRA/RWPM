"""Step 2 - Candidate pair generation.

Comparing every pair of skills is quadratic, so we only score pairs that at
least one cheap, recall-oriented blocker proposes:
  * lsh  : MinHash LSH on word 5-gram shingles of the body (estimated Jaccard >= 0.3)
  * name : two distinct contents with the same normalised front-matter name
  * lsa  : nearest neighbours in an LSA space of the body (cosine >= 0.9);
           catches reworded or translated copies that share few exact shingles

Output: work/candidates.pkl with one row per unordered pair (i < j) and a
similarity band used for stratified sampling in step 3:
  B1 J>=0.9 | B2 0.7-0.9 | B3 0.5-0.7 | B4 0.3-0.5 | B5 same name, J<0.3
  B6 LSA neighbour, J<0.3 | B0 other (LSH false positive, J<0.3)
"""
import argparse
import time

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import RESULTS, SEED, WORK, fmt_int, shingles

ap = argparse.ArgumentParser()
ap.add_argument("--num-perm", type=int, default=128)
ap.add_argument("--lsh-threshold", type=float, default=0.3)
ap.add_argument("--max-name-pairs", type=int, default=45,
                help="cap on pairs drawn from one same-name group")
ap.add_argument("--lsa-dims", type=int, default=128)
ap.add_argument("--lsa-k", type=int, default=3)
ap.add_argument("--lsa-min-cos", type=float, default=0.9)
args = ap.parse_args()
rng = np.random.default_rng(SEED)
t0 = time.time()

skills = pd.read_pickle(WORK / "skills.pkl")
n = len(skills)
print(f"{n} skills")

# ---- MinHash LSH -------------------------------------------------------------
sh = [shingles(b) for b in skills["body"]]
mhs = []
for s in range(0, n, 5000):
    batch = [[x.encode("utf8") for x in sh[i]] or [b""] for i in range(s, min(n, s + 5000))]
    mhs.extend(MinHash.bulk(batch, num_perm=args.num_perm, seed=SEED))
lsh = MinHashLSH(threshold=args.lsh_threshold, num_perm=args.num_perm)
for i, m in enumerate(mhs):
    lsh.insert(i, m)
pairs = {}
for i, m in enumerate(mhs):
    for j in lsh.query(m):
        if j > i:
            pairs.setdefault((i, j), set()).add("lsh")
print(f"lsh pairs: {sum('lsh' in v for v in pairs.values())}  ({time.time() - t0:.0f}s)")

# ---- same-name blocking ------------------------------------------------------
for name, idx in skills[skills.nname != ""].groupby("nname").indices.items():
    idx = list(idx)
    if len(idx) < 2:
        continue
    allp = [(a, b) for k, a in enumerate(idx) for b in idx[k + 1:]]
    if len(allp) > args.max_name_pairs:
        allp = [allp[k] for k in rng.choice(len(allp), args.max_name_pairs, replace=False)]
    for a, b in allp:
        pairs.setdefault((min(a, b), max(a, b)), set()).add("name")
print(f"after name blocking: {len(pairs)}")

# ---- LSA nearest neighbours --------------------------------------------------
tf = TfidfVectorizer(min_df=2, max_df=0.5, sublinear_tf=True, max_features=200_000)
X = tf.fit_transform(skills["body"])
dims = min(args.lsa_dims, X.shape[1] - 1, n - 1)
Z = normalize(TruncatedSVD(n_components=dims, random_state=SEED).fit_transform(X)).astype(np.float32)
lsa_cos = {}
for s in range(0, n, 2000):
    S = Z[s:s + 2000] @ Z.T
    for r in range(S.shape[0]):
        i = s + r
        S[r, i] = -1
        top = np.argpartition(-S[r], args.lsa_k)[:args.lsa_k]
        for j in top:
            if S[r, j] >= args.lsa_min_cos:
                key = (min(i, j), max(i, j))
                pairs.setdefault(key, set()).add("lsa")
                lsa_cos[key] = float(S[r, j])
print(f"after lsa: {len(pairs)}  ({time.time() - t0:.0f}s)")

# ---- assemble ----------------------------------------------------------------
rows = []
for (i, j), src in pairs.items():
    est = mhs[i].jaccard(mhs[j])
    rows.append((i, j, est, "lsh" in src, "name" in src, "lsa" in src, lsa_cos.get((i, j), np.nan)))
cand = pd.DataFrame(rows, columns=["i", "j", "jac_est", "src_lsh", "src_name", "src_lsa", "lsa_cos"])
cand["sha_a"] = skills["sha"].values[cand.i]
cand["sha_b"] = skills["sha"].values[cand.j]


def band(r):
    J = r.jac_est
    if J >= 0.9:
        return "B1"
    if J >= 0.7:
        return "B2"
    if J >= 0.5:
        return "B3"
    if J >= 0.3:
        return "B4"
    if r.src_name:
        return "B5"
    if r.src_lsa:
        return "B6"
    return "B0"


cand["band"] = cand.apply(band, axis=1)
cand.to_pickle(WORK / "candidates.pkl")
print(cand["band"].value_counts().sort_index().to_string())
vc = cand["band"].value_counts()
WORDS = ["Zero", "One", "Two", "Three", "Four", "Five", "Six", "Seven"]
mac = {"ResCandTotal": len(cand), "ResCandLSH": int(cand.src_lsh.sum()),
       "ResCandName": int(cand.src_name.sum()), "ResCandLSA": int(cand.src_lsa.sum())}
for k in range(7):
    mac[f"ResBand{WORDS[k]}"] = int(vc.get(f"B{k}", 0))
(RESULTS / "macros_candidates.tex").write_text(
    "\n".join(rf"\renewcommand{{\{k}}}{{{fmt_int(v)}}}" for k, v in mac.items()) + "\n")
print(f"done in {time.time() - t0:.0f}s")
