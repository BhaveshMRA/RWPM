"""Step 11 (full dataset) - Boilerplate lines and MinHash signatures.

  python 11_full_index.py   (after 10_full_extract.py)

Streams every skill of work/full_skills.db once to compute
  * the boilerplate lines of step 5 over the whole corpus: a normalised line is boilerplate when it appears in skills with at least --boiler-k different front-matter names (written to work/boiler_lines.pkl);
  * the MinHash signature of every body, exactly as step 2 computes it (word 5-gram shingles, 128 permutations, same seed), written to work/minhash.npy for candidate generation.
"""
import argparse
import hashlib
import pickle
import sqlite3
import time

import numpy as np
from datasketch import MinHash

from common import RESULTS, SEED, WORK, fmt_int, norm_lines, shingles

ap = argparse.ArgumentParser()
ap.add_argument("--boiler-k", type=int, default=5)
ap.add_argument("--num-perm", type=int, default=128)
ap.add_argument("--chunk", type=int, default=5000)
args = ap.parse_args()

t0 = time.time()
con = sqlite3.connect(WORK / "full_skills.db")
N = con.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
sig = np.lib.format.open_memmap(WORK / "minhash.npy", mode="w+", dtype=np.uint64, shape=(N, args.num_perm))


def h64(text):
    return int.from_bytes(hashlib.blake2b(text.encode("utf8"), digest_size=8).digest(), "little")


line_parts, name_parts = [], []
cur = con.execute("SELECT id, sha, nname, body FROM skills ORDER BY id")
done = 0
while True:
    rows = cur.fetchmany(args.chunk)
    if not rows:
        break
    batch = [[x.encode("utf8") for x in shingles(body)] or [b""] for _, _, _, body in rows]
    mhs = MinHash.bulk(batch, num_perm=args.num_perm, seed=SEED)
    ids = [r[0] for r in rows]
    sig[ids[0]:ids[-1] + 1] = np.vstack([m.hashvalues for m in mhs])
    for _, sha, nname, body in rows:
        ls = norm_lines(body)
        if ls:
            nk = h64(nname if nname else sha)
            line_parts.append(np.fromiter((h64(x) for x in ls), dtype=np.uint64, count=len(ls)))
            name_parts.append(np.full(len(ls), nk, dtype=np.uint64))
    done += len(rows)
    print(f"\r{done:,}/{N:,} skills hashed ({time.time() - t0:.0f}s)", end="", flush=True)
print()
sig.flush()

# boilerplate: lines carried by >= k distinct names
pairs = np.empty(sum(len(p) for p in line_parts), dtype=[("line", np.uint64), ("name", np.uint64)])
pairs["line"] = np.concatenate(line_parts)
pairs["name"] = np.concatenate(name_parts)
del line_parts, name_parts
uniq = np.unique(pairs)
del pairs
lines_u, n_names = np.unique(uniq["line"], return_counts=True)
boiler_h = set(lines_u[n_names >= args.boiler_k].tolist())
del uniq, lines_u, n_names
print(f"boilerplate line hashes: {len(boiler_h):,} ({time.time() - t0:.0f}s)")

# second pass: keep the text of boilerplate lines (step 5 and step 9 use strings)
boiler = set()
cur = con.execute("SELECT body FROM skills")
while True:
    rows = cur.fetchmany(args.chunk)
    if not rows:
        break
    for (body,) in rows:
        for x in norm_lines(body):
            if x not in boiler and h64(x) in boiler_h:
                boiler.add(x)
pickle.dump(boiler, open(WORK / "boiler_lines.pkl", "wb"))
(RESULTS / "macros_full_features.tex").write_text(
    rf"\renewcommand{{\ResFullBoilerLines}}{{{fmt_int(len(boiler))}}}" + "\n")
print(f"boilerplate lines: {len(boiler):,}; wrote minhash.npy and boiler_lines.pkl in {time.time() - t0:.0f}s")
