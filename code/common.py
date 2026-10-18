"""Shared helpers: paths, SKILL.md parsing, normalisation, shingling."""
import json
import os
import re
from pathlib import Path

# Default layout (project root = parent of code/):
#   work/      intermediate files (not shared)
#   labeling/  labeling kit and gold set
#   paper/results/  LaTeX tables and macros, picked up by paper/main.tex
ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("SKILL_WORK", ROOT / "work"))
LABEL_DIR = Path(os.environ.get("SKILL_LABELS", ROOT / "labeling"))
RESULTS = Path(os.environ.get("SKILL_RESULTS", ROOT / "paper" / "results"))
for d in (WORK, LABEL_DIR, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 625
SHINGLE_K = 5           # word n-gram size for near-duplicate detection
MIN_BODY_CHARS = 200    # drops stubs and symlink skills (content = link target)
CLASSES = ["derived", "template", "unrelated"]

_FM = re.compile(r"^\ufeff?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.S)
_FENCE = re.compile(r"(```|~~~)[^\n]*\n(.*?)(?:\1|\Z)", re.S)
_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.*\S)\s*$", re.M)
_WORD = re.compile(r"\w+", re.U)
_LETTER = re.compile(r"[^\W\d_]", re.U)
_LIST_MARK = re.compile(r"^(?:[-*+>]|\d+[.)])\s+")


def split_frontmatter(text: str):
    """Return (front_matter_text, body)."""
    text = text or ""
    m = _FM.match(text)
    if not m:
        return "", text
    return m.group(1), text[m.end():]


def split_code(body: str):
    """Return (prose_without_code, list_of_code_blocks)."""
    blocks = [m.group(2) for m in _FENCE.finditer(body)]
    prose = _FENCE.sub("\n", body)
    return prose, blocks


def headings(prose: str):
    return [h.strip().lower() for _, h in _HEADING.findall(prose)]


def norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def norm_name(name) -> str:
    return re.sub(r"[\s_]+", "-", str(name or "").strip().lower())


def tokens(text: str):
    return _WORD.findall((text or "").lower())


def shingles(text: str, k: int = SHINGLE_K):
    toks = tokens(text)
    if not toks:
        return set()
    if len(toks) < k:
        return {" ".join(toks)}
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def norm_lines(text: str, min_len: int = 20):
    """Normalised, de-duplicated content lines used for boilerplate analysis."""
    out = set()
    for ln in (text or "").splitlines():
        ln = _LIST_MARK.sub("", ln.strip().lower())
        ln = re.sub(r"\s+", " ", ln).strip()
        if len(ln) >= min_len and _LETTER.search(ln):
            out.add(ln)
    return out


def code_tokens(blocks):
    toks = set()
    for b in blocks:
        for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", b):
            for p in re.sub(r"([a-z])([A-Z])", r"\1 \2", t).replace("_", " ").split():
                if len(p) > 1:
                    toks.add(p.lower())
    return toks


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def load_json(path):
    return json.loads(Path(path).read_text())


def fmt_int(x) -> str:
    return f"{int(x):,}"


def fmt_pct(x, digits=1) -> str:
    return f"{100 * x:.{digits}f}\\%"


# Feature groups used by the classifier (steps 5-7). "sr" holds the
# SkillReuse-style baseline features and is never given to our models.
FEATURE_GROUPS = {
    "lexical": ["jac_shingle", "cont_shingle", "tfidf_body", "lev_ratio"],
    "boilerplate": ["jac_distinct_lines", "cont_distinct_lines", "jac_boiler_lines",
                    "frac_shared_boiler"],
    "metadata": ["name_equal", "name_sim", "desc_tfidf"],
    "code": ["code_jac", "code_both"],
    "structure": ["head_seq_sim", "len_ratio", "heads_ratio"],
    "repository": ["same_owner", "same_dirname", "siblings_jac"],
}
SR_FEATURES = ["sr_flat", "sr_y", "sr_n", "sr_c", "sr_yn", "sr_nc", "sr_yc", "sr_struct"]
ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]
