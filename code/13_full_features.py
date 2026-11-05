"""Step 13 (full dataset) - Pair features with full-dataset corpus statistics.

  python 13_full_features.py --pairs gold   -> work/features_gold.pkl  (RQ1/RQ2)
  python 13_full_features.py --pairs all    -> work/features_all.pkl   (RQ3)

Computes the same features as step 5. The corpus-level parts are fitted once
and reused for every pair: boilerplate lines over the whole corpus (step 11),
and the TF-IDF, LSA and normalisation models of step 5 fitted on --fit-size
randomly chosen skills plus every skill of the gold set (models saved to
work/full_models.pkl). Pairs are then processed in chunks, in parallel.
Steps 6-9 run unchanged on the output.
"""
import argparse
import difflib
import json
import pickle
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import (ALL_FEATURES, LABEL_DIR, SEED, WORK, code_tokens, containment, jaccard, norm_lines,
                    norm_ws, shingles, split_code, tokens)

DB = WORK / "full_skills.db"
MODELS = WORK / "full_models.pkl"
COLS = "id, sha, name, nname, description, body, fm_fields, code_langs, headings, dirname, owners, siblings"


def load(con, ids):
    """Skills with the given ids, in the order of ids, with step-1 channels derived."""
    ids = [int(x) for x in ids]
    parts = [pd.read_sql_query(f"SELECT {COLS} FROM skills WHERE id IN ({','.join('?' * len(c))})", con, params=c)
             for c in (ids[k:k + 900] for k in range(0, len(ids), 900))]
    S = pd.concat(parts).set_index("id").loc[ids].reset_index()
    pc = S.body.map(split_code)
    S["prose"], S["code_blocks"] = pc.str[0], pc.str[1]
    for col in ("code_langs", "headings", "owners", "siblings"):
        S[col] = S[col].map(json.loads)
    S["description"] = S.description.fillna("").astype(str)
    S["name"] = S["name"].fillna("").astype(str)
    return S


def structure(S):
    body = S.body.tolist()
    blen = np.array([len(b) for b in body], dtype=float)
    ncb = np.array([len(c) for c in S.code_blocks], dtype=float)
    cchars = np.array([sum(map(len, c)) for c in S.code_blocks], dtype=float)
    return np.column_stack([[len(tokens(x)) for x in body], ncb, cchars / np.maximum(blen, 1),
                            cchars / np.maximum(ncb, 1), [len(x) for x in S.code_langs],
                            np.ones(len(S)), S.fm_fields.values, S.description.str.len().values])


def channels(S):
    yaml = (S["name"] + " " + S.description).tolist()
    code = [" ".join(c) for c in S.code_blocks]
    flat = [y + " " + b for y, b in zip(yaml, S.body)]
    return yaml, S.prose.tolist(), code, flat


def fit_models(con, gold_ids, fit_size):
    n = con.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
    rng = np.random.default_rng(SEED)
    ids = sorted(set(rng.choice(n, min(n, fit_size), replace=False).tolist()) | set(gold_ids))
    S = load(con, ids)
    yaml, nl, code, flat = channels(S)
    m = {"body": TfidfVectorizer(sublinear_tf=True, min_df=2, max_df=0.5, ngram_range=(1, 2),
                                 max_features=300_000).fit(S.body.tolist()),
         "desc": TfidfVectorizer(sublinear_tf=True, min_df=1, max_features=100_000).fit(S.description.tolist()),
         "flat": TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=300_000).fit(flat)}
    for ch, texts in (("y", yaml), ("n", nl), ("c", [t if t.strip() else "empty" for t in code])):
        v = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50_000, token_pattern=r"(?u)\b\w+\b")
        X = v.fit_transform(texts)
        d = max(1, min(128, X.shape[1] - 1, X.shape[0] - 1))
        m["lsa_" + ch] = (v, TruncatedSVD(n_components=d, random_state=SEED).fit(X))
    st = structure(S)
    m["smin"], m["smax"] = st.min(0), st.max(0)
    m["boiler"] = pickle.load(open(WORK / "boiler_lines.pkl", "rb"))
    m["fit_size"] = len(ids)
    return m


def features(m, S, a, b, sr):
    """Step-5 features for pairs (a[k], b[k]) of rows of S."""
    boiler = m["boiler"]
    body = S.body.tolist()
    sh = [shingles(x) for x in body]
    lines = [norm_lines(x) for x in body]
    name_key = [n if n else s for n, s in zip(S.nname, S.sha)]
    distinct = [ls - boiler for ls in lines]
    bl = [ls & boiler for ls in lines]
    code = [code_tokens(c) for c in S.code_blocks]
    heads = S.headings.tolist()
    nbody = [norm_ws(x)[:5000] for x in body]
    own = [set(o) for o in S.owners]
    sib = [set(x) for x in S.siblings]
    blen = np.array([len(x) for x in body], dtype=float)

    def cos(X):
        X = normalize(X)
        return np.asarray(X[a].multiply(X[b]).sum(axis=1)).ravel()

    F = {"tfidf_body": cos(m["body"].transform(body)), "desc_tfidf": cos(m["desc"].transform(S.description.tolist()))}
    if sr:
        yaml, nl, code_txt, flat = channels(S)
        F["sr_flat"] = cos(m["flat"].transform(flat))
        for col, ch, texts, min_tok in (("sr_y", "y", yaml, 1), ("sr_n", "n", nl, 5),
                                        ("sr_c", "c", [t if t.strip() else "empty" for t in code_txt], 1)):
            v, svd = m["lsa_" + ch]
            Z = normalize(svd.transform(v.transform(texts)))
            pres = np.array([len(tokens(t)) >= min_tok for t in (code_txt if ch == "c" else texts)])
            F[col] = np.where(pres[a] & pres[b], np.clip((Z[a] * Z[b]).sum(axis=1), 0, 1), 0.0)
        F["sr_yn"], F["sr_nc"], F["sr_yc"] = F["sr_y"] * F["sr_n"], F["sr_n"] * F["sr_c"], F["sr_y"] * F["sr_c"]
        st = (structure(S) - m["smin"]) / np.maximum(m["smax"] - m["smin"], 1e-9)
        F["sr_struct"] = 1 - np.abs(st[a] - st[b]).mean(axis=1)
    rows = []
    for x, y in zip(a, b):
        shared = lines[x] & lines[y]
        rows.append({
            "jac_shingle": jaccard(sh[x], sh[y]),
            "cont_shingle": containment(sh[x], sh[y]),
            "lev_ratio": fuzz.ratio(nbody[x], nbody[y]) / 100.0,
            "jac_distinct_lines": jaccard(distinct[x], distinct[y]),
            "cont_distinct_lines": containment(distinct[x], distinct[y]),
            "jac_boiler_lines": jaccard(bl[x], bl[y]),
            "frac_shared_boiler": len(shared & boiler) / len(shared) if shared else 0.0,
            "name_equal": float(bool(name_key[x]) and name_key[x] == name_key[y]),
            "name_sim": fuzz.ratio(S.nname.iat[x], S.nname.iat[y]) / 100.0,
            "code_jac": jaccard(code[x], code[y]),
            "code_both": float(bool(code[x]) and bool(code[y])),
            "head_seq_sim": (difflib.SequenceMatcher(None, heads[x], heads[y]).ratio()
                             if heads[x] and heads[y] else 0.0),
            "len_ratio": min(blen[x], blen[y]) / max(blen[x], blen[y]),
            "heads_ratio": (min(len(heads[x]), len(heads[y])) + 1) / (max(len(heads[x]), len(heads[y])) + 1),
            "same_owner": float(bool(own[x] & own[y])),
            "same_dirname": float(bool(S.dirname.iat[x]) and S.dirname.iat[x] == S.dirname.iat[y]),
            "siblings_jac": jaccard(sib[x], sib[y]),
            "norm_hash_equal": float(norm_ws(body[x]) == norm_ws(body[y])),
        })
    return pd.concat([pd.DataFrame(F), pd.DataFrame(rows)], axis=1)


_M, _CON = None, None


def _init():
    global _M, _CON
    _M = pickle.load(open(MODELS, "rb"))
    _CON = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)


def _chunk(args):
    i, j = args
    ids = np.unique(np.r_[i, j])
    S = load(_CON, ids)
    pos = {v: k for k, v in enumerate(ids.tolist())}
    a = np.array([pos[v] for v in i.tolist()])
    b = np.array([pos[v] for v in j.tolist()])
    F = features(_M, S, a, b, sr=False)[ALL_FEATURES].astype(np.float32)
    F.insert(0, "i", i.astype(np.int32))
    F.insert(1, "j", j.astype(np.int32))
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", choices=["gold", "all"], default="gold")
    ap.add_argument("--fit-size", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=20_000)
    args = ap.parse_args()
    t0 = time.time()
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    sha = pd.read_sql_query("SELECT sha FROM skills ORDER BY id", con).sha.to_numpy()
    id_of = {s: k for k, s in enumerate(sha)}
    gold = pd.read_csv(LABEL_DIR / "gold.csv")
    gold["a"], gold["b"] = gold.sha_a.map(id_of), gold.sha_b.map(id_of)
    missing = gold[gold.a.isna() | gold.b.isna()]
    if len(missing):
        raise SystemExit(f"{len(missing)} gold pairs reference skills outside the full working set")
    gold = gold.astype({"a": int, "b": int})
    if not MODELS.exists():
        print("fitting corpus models ...")
        m = fit_models(con, set(gold.a) | set(gold.b), args.fit_size)
        pickle.dump(m, open(MODELS, "wb"))
        print(f"models fitted on {m['fit_size']:,} skills; {len(m['boiler']):,} boilerplate lines "
              f"({time.time() - t0:.0f}s)")
    if args.pairs == "gold":
        m = pickle.load(open(MODELS, "rb"))
        ids = np.unique(np.r_[gold.a, gold.b])
        S = load(con, ids)
        pos = {v: k for k, v in enumerate(ids.tolist())}
        F = features(m, S, gold.a.map(pos).values, gold.b.map(pos).values, sr=True)
        F = pd.concat([gold.reset_index(drop=True), F], axis=1)
        F.to_pickle(WORK / "features_gold.pkl")
        print(f"wrote {WORK / 'features_gold.pkl'} ({len(F)} rows) in {time.time() - t0:.0f}s")
        return
    C = np.load(WORK / "candidates_full.npz")
    i, j = C["i"], C["j"]
    jobs = [(i[s:s + args.chunk], j[s:s + args.chunk]) for s in range(0, len(i), args.chunk)]
    parts = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        for k, part in enumerate(ex.map(_chunk, jobs), 1):
            parts.append(part)
            print(f"\rchunk {k}/{len(jobs)} ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    F = pd.concat(parts, ignore_index=True)
    F["band"] = pd.Categorical(C["band"])
    F["jac_est"] = C["jac_est"]
    F.to_pickle(WORK / "features_all.pkl")
    print(f"wrote {WORK / 'features_all.pkl'} ({len(F):,} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
