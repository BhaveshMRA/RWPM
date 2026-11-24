# Taxonomy check (Phase 4, "Example with description")

Each gold pair already has a label (D/T/U) from Phase 2-3. That label does **not** change here.
This check adds *why* two skills look alike: one sub-category per pair. Each of you checks the
200 pairs you labelled, in `taxonomy_<your name>.csv`. Expect 45-60 minutes.

## How to fill the sheet

For every row, read `notes` (your own earlier notes), open the pair in `labeling/html/<pair_id>.html`
only if unsure, and write in `verdict`:

- `OK` if the suggested sub-category is right, or
- the correct code from the list below (e.g. `D-renamed`).

Rows whose `rule` contains `FLAG` are genuinely mixed cases: decide them with your partner.
Do not edit any other column.

## Sub-categories (one per pair)

**Derived** - when several apply, take the first one in this order:

| Code | Name | When |
|---|---|---|
| `D-translated` | translated copy | the copy's instructions are in another language |
| `D-ported` | ported copy | paths, commands or adapters rewritten for another agent tool (Codex, Cursor, Windsurf, Gemini, installer copies) |
| `D-reworked` | reworked copy | distinctive steps reused, but (almost) no line copied verbatim |
| `D-renamed` | renamed copy | different front-matter names, text largely copied |
| `D-edited` | copy with edits | same name; trimmed, extended or lightly edited |

**Template**, in this order:

| Code | Name | When |
|---|---|---|
| `T-preamble` | shared project preamble | different skills of one project/framework sharing a long common preamble or protocol (gstack, GSD, spec-kit, house protocols) |
| `T-generator` | catalog generator | a generator filled different slots into one fixed text (Winbda, u-series, numbered stubs) |
| `T-integration` | integration catalog | one wrapper applied to many applications/APIs (Composio/Rube, Membrane, cloud services) |
| `T-series` | domain series | one author's series on different topics of one domain, same structure (German legal practice, a research collection) |

**Unrelated**:

| Code | Name | When |
|---|---|---|
| `U-samename` | same name or topic | same name or same topic, but independently written text |
| `U-different` | different topics | nothing beyond generic wording in common (random pairs, LSA neighbours) |

## Clarifications (added after the first sheet)

1. `T-generator` vs `T-series`: generator = the bodies are near-identical except the filled-in slots; series = a shared skeleton, but topic-specific content.
2. `T-integration`: any catalog wrapping different applications or APIs the same way (Composio/Rube, Membrane, cloud-service wrappers) is integration, even if the notes say "generated".

## Afterwards

Send the four filled sheets. Pairs where the two of you disagree get resolved together, as in the
gold set; the counts and one example per sub-category then go into the paper and the slides.
