# Labeling guidelines: is skill B derived from skill A (or vice versa)?

Open html/<pair_id>.html, read both skills, and fill your sheet with one label.
Do not look at the other annotator's sheet. Do not use similarity scores.

D - DERIVED VARIANT. The two skills share *distinctive* content that is unlikely
    to be written twice independently: verbatim or lightly reworded instruction
    passages, the same task-specific steps, the same examples, commands, file
    paths, configuration values, or idiosyncratic phrasing. One is a copied and
    edited version of the other, or both were copied from a common source.
    Translations of the same skill count as D. Renamed copies count as D.

T - TEMPLATE SIBLING. The overlap is limited to generic scaffolding: shared
    section headings ("When to use this skill", "Quality checklist"), boilerplate
    guidance lines, placeholder text, or a catalog's house style - while the
    task-specific instructions are different. Typical sources: skills generated
    from the same scaffold or the same catalog generator, or one author's
    personal template applied to different tasks.

U - UNRELATED. Neither of the above. Same topic alone ("both are code-review
    skills") is U, not D, unless the content itself was copied.

Decision order: look for copied distinctive content first (-> D). If the only
shared material is scaffolding (-> T). Otherwise U.
Confidence: 1 unsure, 2 fairly sure, 3 sure. Use the note column for anything
odd (e.g. one skill is a strict subset of the other, mixed languages).
