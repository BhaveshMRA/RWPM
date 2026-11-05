"""Step 12 (full dataset) - Candidate pairs.

  python 12_full_candidates.py   (after 11_full_index.py)

Same blockers as step 2 except LSA:
  * MinHash LSH with the band split that datasketch uses for a Jaccard
    threshold of 0.3 and 128 permutations (37 bands of 3 rows); two skills are
    candidates when one band matches. Buckets larger than --max-bucket are
    not expanded into all pairs: every member is paired with the --anchors
    lowest-id members of the bucket instead, which keeps near-duplicate groups
    connected while the number of pairs stays linear in the bucket size;
  * same-name blocking, capped at --max-name-pairs pairs per name.
The LSA blocker is not used at this scale; in the labelled sample, band B6
(pairs found only by LSA) contains no derived pair.
Writes work/candidates_full.npz and results/macros_full_candidates.tex.
"""
import argparse
import sqlite3
import time
from itertools import combinations

import numpy as np
import pandas as pd
from datasketch.lsh import _optimal_param

from common import RESULTS, SEED, WORK, fmt_int

ap = argparse.ArgumentParser()
ap.add_argument("--lsh-threshold", type=float, default=0.3)
ap.add_argument("--max-bucket", type=int, default=50)
ap.add_argument("--anchors", type=int, default=5)
ap.add_argument("--max-name-pairs", type=int, default=45)
args = ap.parse_args()

t0 = time.time()
rng = np.random.default_rng(SEED)
sig = np.load(WORK / "minhash.npy", mmap_mode="r")
N, perm = sig.shape
b, r = _optimal_param(args.lsh_threshold, perm, 0.5, 0.5)
print(f"{N:,} skills; LSH with {b} bands of {r} rows")


def bucket_pairs(members):
    m = np.sort(members)
    if len(m) <= args.max_bucket:
        return list(combinations(m.tolist(), 2))
    anchors = m[: args.anchors].tolist()
    out = list(combinations(anchors, 2))
    out += [(a, x) for x in m[args.anchors:].tolist() for a in anchors]
    return out


codes = []
big = 0
for band in range(b):
    cols = np.asarray(sig[:, band * r:(band + 1) * r])
    order = np.lexsort(cols.T[::-1])
    ks = cols[order]
    starts = np.flatnonzero(np.r_[True, (ks[1:] != ks[:-1]).any(axis=1)])
    sizes = np.diff(np.r_[starts, len(ks)])
    two = starts[sizes == 2]
    if len(two):
        x, y = order[two], order[two + 1]
        codes.append(np.minimum(x, y).astype(np.int64) * N + np.maximum(x, y))
    for s, z in zip(starts[sizes > 2], sizes[sizes > 2]):
        big += z > args.max_bucket
        pr = np.array(bucket_pairs(order[s:s + z]), dtype=np.int64)
        codes.append(pr[:, 0] * N + pr[:, 1])
    print(f"\rband {band + 1}/{b} ({time.time() - t0:.0f}s)", end="", flush=True)
print()
lsh = np.unique(np.concatenate(codes)) if codes else np.empty(0, np.int64)
del codes
print(f"lsh pairs: {len(lsh):,} ({big:,} oversized buckets capped)")

con = sqlite3.connect(WORK / "full_skills.db")
names = pd.read_sql_query("SELECT id, nname FROM skills WHERE nname != '' ORDER BY id", con)
name_codes = []
for _, idx in names.groupby("nname").id:
    idx = idx.to_numpy()
    k = len(idx)
    if k < 2:
        continue
    if k * (k - 1) // 2 <= args.max_name_pairs:
        pr = list(combinations(idx.tolist(), 2))
    else:
        seen = set()
        while len(seen) < args.max_name_pairs:
            a, c = rng.choice(k, 2, replace=False)
            seen.add((min(idx[a], idx[c]), max(idx[a], idx[c])))
        pr = sorted(seen)
    pr = np.array(pr, dtype=np.int64)
    name_codes.append(np.minimum(pr[:, 0], pr[:, 1]) * N + np.maximum(pr[:, 0], pr[:, 1]))
name = np.unique(np.concatenate(name_codes)) if name_codes else np.empty(0, np.int64)
allc = np.union1d(lsh, name)
i, j = allc // N, allc % N
src_lsh, src_name = np.isin(allc, lsh), np.isin(allc, name)

jac = np.empty(len(allc), dtype=np.float32)
for s in range(0, len(allc), 500_000):
    a, c = i[s:s + 500_000], j[s:s + 500_000]
    jac[s:s + 500_000] = (sig[a] == sig[c]).mean(axis=1)
band = np.select([jac >= 0.9, jac >= 0.7, jac >= 0.5, jac >= 0.3, src_name],
                 ["B1", "B2", "B3", "B4", "B5"], default="B0")
np.savez(WORK / "candidates_full.npz", i=i, j=j, jac_est=jac, src_lsh=src_lsh, src_name=src_name, band=band)
vc = pd.Series(band).value_counts()
words = ["Zero", "One", "Two", "Three", "Four", "Five"]
mac = {"ResFullCandTotal": len(allc), "ResFullCandLSH": int(src_lsh.sum()), "ResFullCandName": int(src_name.sum()),
       "ResFullMaxBucket": args.max_bucket, "ResFullAnchors": args.anchors, "ResFullBigBuckets": int(big)}
for k in range(6):
    mac[f"ResFullBand{words[k]}"] = int(vc.get(f"B{k}", 0))
(RESULTS / "macros_full_candidates.tex").write_text(
    "\n".join(rf"\renewcommand{{\{k}}}{{{fmt_int(v)}}}" for k, v in mac.items()) + "\n")
print(vc.sort_index().to_string())
print(f"{len(allc):,} candidate pairs in {time.time() - t0:.0f}s")
