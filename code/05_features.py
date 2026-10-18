"""Step 5 - Pair features.

  python 05_features.py --pairs gold   -> work/features_gold.pkl  (labelled pairs, RQ1/RQ2)
  python 05_features.py --pairs all    -> work/features_all.pkl   (every candidate, RQ3)

Key idea (boilerplate-aware similarity): a line is treated as template
boilerplate when it appears in skills carrying at least --boiler-k different
names. A widely copied skill usually keeps one name, so its lines stay
"distinctive"; scaffold lines recur under many unrelated names. Derived
variants should share distinctive lines; template siblings mostly share
boilerplate lines.
"""
import argparse
import difflib
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from common import (LABEL_DIR, RESULTS, SEED, WORK, code_tokens, containment, fmt_int,
                    jaccard, norm_lines, norm_ws, shingles, tokens)

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", choices=["gold", "all"], default="gold")
ap.add_argument("--boiler-k", type=int, default=5)
args = ap.parse_args()
t0 = time.time()

skills = pd.read_pickle(WORK / "skills.pkl")
occ = pd.read_pickle(WORK / "occurrences.pkl")
idx_of = {s: k for k, s in enumerate(skills.sha)}
if args.pairs == "gold":
    P = pd.read_csv(LABEL_DIR / "gold.csv")
else:
    P = pd.read_pickle(WORK / "candidates.pkl")
P["a"] = P.sha_a.map(idx_of)
P["b"] = P.sha_b.map(idx_of)
P = P.dropna(subset=["a", "b"]).astype({"a": int, "b": int}).reset_index(drop=True)
print(f"{len(P)} pairs over {len(skills)} skills")

# ---- per-skill caches --------------------------------------------------------
body = skills["body"].tolist()
sh = [shingles(b) for b in body]
lines = [norm_lines(b) for b in body]
name_key = [n if n else s for n, s in zip(skills.nname, skills.sha)]
names_per_line = defaultdict(set)
for nk, ls in zip(name_key, lines):
    for ln in ls:
        names_per_line[ln].add(nk)
boiler = {ln for ln, s in names_per_line.items() if len(s) >= args.boiler_k}
del names_per_line
print(f"boilerplate lines: {len(boiler)}")
(RESULTS / "macros_features.tex").write_text(
    rf"\renewcommand{{\ResBoilerLines}}{{{fmt_int(len(boiler))}}}" + "\n"
    + rf"\renewcommand{{\ResBoilerK}}{{{args.boiler_k}}}" + "\n")
distinct = [ls - boiler for ls in lines]
bl = [ls & boiler for ls in lines]
code = [code_tokens(c) for c in skills.code_blocks]
heads = skills.headings.tolist()
nbody = [norm_ws(b)[:5000] for b in body]
owners = occ.groupby("sha")["owner"].apply(set).to_dict()
own = [owners.get(s, set()) for s in skills.sha]
sib = [set(x) for x in skills.siblings]
blen = np.array([len(b) for b in body], dtype=float)


def tfidf_cos(texts, a, b, **kw):
    X = normalize(TfidfVectorizer(sublinear_tf=True, **kw).fit_transform(texts))
    return np.asarray(X[a].multiply(X[b]).sum(axis=1)).ravel()


a, b = P.a.values, P.b.values
F = pd.DataFrame(index=P.index)
F["tfidf_body"] = tfidf_cos(body, a, b, min_df=2, max_df=0.5, ngram_range=(1, 2),
                            max_features=300_000)
F["desc_tfidf"] = tfidf_cos(skills.description.tolist(), a, b, min_df=1, max_features=100_000)

# ---- SkillReuse-style baseline features (Zhu et al., ASE'26), re-implemented ----
yaml_txt = (skills["name"].fillna("") + " " + skills.description).tolist()
nl_txt = skills.prose.tolist()
code_txt = [" ".join(c) for c in skills.code_blocks]
flat_txt = [y + " " + bo for y, bo in zip(yaml_txt, body)]
F["sr_flat"] = tfidf_cos(flat_txt, a, b, min_df=2, max_features=300_000)


def lsa(texts):
    X = TfidfVectorizer(sublinear_tf=True, min_df=2, max_features=50_000,
                        token_pattern=r"(?u)\b\w+\b").fit_transform(texts)
    d = max(1, min(128, X.shape[1] - 1, X.shape[0] - 1))
    return normalize(TruncatedSVD(n_components=d, random_state=SEED).fit_transform(X))


def present(texts, min_tok=5):
    return np.array([len(tokens(t)) >= min_tok for t in texts])


Zy, Zn, Zc = lsa(yaml_txt), lsa(nl_txt), lsa([t if t.strip() else "empty" for t in code_txt])
py_, pn, pc = present(yaml_txt, 1), present(nl_txt), present(code_txt, 1)
for col, Z, pr in (("sr_y", Zy, py_), ("sr_n", Zn, pn), ("sr_c", Zc, pc)):
    s = np.clip((Z[a] * Z[b]).sum(axis=1), 0, 1)
    F[col] = np.where(pr[a] & pr[b], s, 0.0)
F["sr_yn"], F["sr_nc"], F["sr_yc"] = F.sr_y * F.sr_n, F.sr_n * F.sr_c, F.sr_y * F.sr_c
ncb = np.array([len(c) for c in skills.code_blocks], dtype=float)
cchars = np.array([sum(map(len, c)) for c in skills.code_blocks], dtype=float)
S = np.column_stack([
    [len(tokens(x)) for x in body], ncb, cchars / np.maximum(blen, 1),
    cchars / np.maximum(ncb, 1), [len(x) for x in skills.code_langs],
    np.ones(len(skills)), skills.fm_fields.values, skills.description.str.len().values])
S = (S - S.min(0)) / np.maximum(S.max(0) - S.min(0), 1e-9)
F["sr_struct"] = 1 - np.abs(S[a] - S[b]).mean(axis=1)

# ---- pairwise loop -----------------------------------------------------------
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
        "name_sim": fuzz.ratio(skills.nname.iat[x], skills.nname.iat[y]) / 100.0,
        "code_jac": jaccard(code[x], code[y]),
        "code_both": float(bool(code[x]) and bool(code[y])),
        "head_seq_sim": (difflib.SequenceMatcher(None, heads[x], heads[y]).ratio()
                         if heads[x] and heads[y] else 0.0),
        "len_ratio": min(blen[x], blen[y]) / max(blen[x], blen[y]),
        "heads_ratio": (min(len(heads[x]), len(heads[y])) + 1) / (max(len(heads[x]), len(heads[y])) + 1),
        "same_owner": float(bool(own[x] & own[y])),
        "same_dirname": float(bool(skills.dirname.iat[x]) and skills.dirname.iat[x] == skills.dirname.iat[y]),
        "siblings_jac": jaccard(sib[x], sib[y]),
        "norm_hash_equal": float(norm_ws(body[x]) == norm_ws(body[y])),
    })
F = pd.concat([P, F, pd.DataFrame(rows, index=P.index)], axis=1)
out = WORK / f"features_{args.pairs}.pkl"
F.to_pickle(out)
print(f"wrote {out} ({len(F)} rows) in {time.time() - t0:.0f}s")
