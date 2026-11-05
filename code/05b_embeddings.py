"""Step 5b - Multilingual sentence embeddings.

  python 05b_embeddings.py        -> work/embeddings.npz, results/macros_embeddings.tex

Phase 2 errors were copies that share almost no text with their source: translations and heavy rewrites. Lexical features cannot see them, so we add one embedding per skill from a multilingual sentence-embedding model. The prose of each skill (body without code) is split into chunks of --words words; the first --chunks chunks are embedded and averaged. The description is embedded separately. Step 5 then adds two features, the cosine similarity of the body embeddings and of the description embeddings, as the "semantic" group.

The model is downloaded once from Hugging Face on the first run.
"""
import argparse
import time

import numpy as np
import pandas as pd

from common import EMBEDDINGS, RESULTS, WORK

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
ap.add_argument("--chunks", type=int, default=6, help="max chunks of prose per skill")
ap.add_argument("--words", type=int, default=100, help="words per chunk")
ap.add_argument("--batch", type=int, default=64)
args = ap.parse_args()

from sentence_transformers import SentenceTransformer

t0 = time.time()
skills = pd.read_pickle(WORK / "skills.pkl")
model = SentenceTransformer(args.model)


def chunks(prose, body):
    words = (prose if prose.strip() else body).split()
    out = [" ".join(words[i:i + args.words])
           for i in range(0, min(len(words), args.words * args.chunks), args.words)]
    return out or [""]


texts, owner = [], []
for k, (prose, body) in enumerate(zip(skills.prose, skills.body)):
    for c in chunks(prose, body):
        texts.append(c)
        owner.append(k)
owner = np.array(owner)
print(f"{len(skills)} skills, {len(texts)} chunks; encoding with {args.model}")
V = model.encode(texts, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True)
body = np.zeros((len(skills), V.shape[1]), dtype=np.float32)
np.add.at(body, owner, V)
body /= np.maximum(np.linalg.norm(body, axis=1, keepdims=True), 1e-9)

desc_txt = [(n or "") + ": " + (d or "") for n, d in zip(skills["name"], skills.description)]
desc = model.encode(desc_txt, batch_size=args.batch, normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)

np.savez_compressed(EMBEDDINGS, sha=np.array(skills.sha.tolist(), dtype="U64"), body=body, desc=desc)
short = args.model.split("/")[-1]
(RESULTS / "macros_embeddings.tex").write_text(
    rf"\renewcommand{{\ResEmbModel}}{{\texttt{{{short}}}}}" + "\n"
    + rf"\renewcommand{{\ResEmbChunks}}{{{args.chunks}}}" + "\n"
    + rf"\renewcommand{{\ResEmbWords}}{{{args.words}}}" + "\n")
print(f"wrote {EMBEDDINGS} ({body.shape[1]} dims) in {time.time() - t0:.0f}s")
