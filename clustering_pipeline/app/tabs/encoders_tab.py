"""Static analysis and source for modules/encoders.py."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ast_tools import analyze_module, get_registry_map
from app.paths import ENCODERS_PY, ROOT


def render_encoders_tab() -> None:
    st.subheader("Encoders module")
    st.caption(f"`{ENCODERS_PY.relative_to(ROOT)}` — design and registry without importing heavy dependencies.")

    if not ENCODERS_PY.is_file():
        st.error("encoders.py not found.")
        return

    struct = analyze_module(ENCODERS_PY)
    reg = get_registry_map(ENCODERS_PY)

    st.markdown("##### Architecture")
    st.markdown(
        """
**Design.** `BaseEncoder` defines `fit_transform(cleaned_texts=..., tokenized_texts=...)` and returns a dense
embedding matrix. This repo currently wires **TF-IDF** (with optional TruncatedSVD), **Doc2Vec**, and
**SBERT** (`SentenceTransformer`). `get_encoder(cfg)` picks the implementation from `cfg["encoder"]["name"]`
via the module registry.
        """.strip()
    )

    if reg:
        st.markdown("**Registry** (`name` → class)")
        st.table(
            [{"encoder key": k, "class": v} for k, v in sorted(reg.items(), key=lambda x: x[0])]
        )

    rows = []
    for c in struct.classes:
        rows.append(
            {
                "class": c.name,
                "bases": ", ".join(c.bases) or "—",
                "public methods": ", ".join(c.methods) or "—",
                "line": c.lineno,
            }
        )
    st.markdown("##### Classes (AST)")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    lines = ["classDiagram"]
    for c in struct.classes:
        for b in c.bases:
            if b in ("ABC", "object"):
                continue
            lines.append(f"    {b} <|-- {c.name}")
    if len(lines) > 1:
        st.markdown("##### Inheritance (Mermaid)")
        st.code("\n".join(lines), language="mermaid")

    with st.expander("Full source", expanded=False):
        st.code(ENCODERS_PY.read_text(encoding="utf-8"), language="python")
