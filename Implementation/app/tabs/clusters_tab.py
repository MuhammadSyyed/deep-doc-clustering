"""Static analysis and source for modules/clustering.py."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.ast_tools import analyze_module, get_registry_map
from app.paths import CLUSTERING_PY, ROOT


def render_clusters_tab() -> None:
    st.subheader("Clustering module")
    st.caption(f"`{CLUSTERING_PY.relative_to(ROOT)}` — clusterers and PyTorch helpers (structure via AST).")

    if not CLUSTERING_PY.is_file():
        st.error("clustering.py not found.")
        return

    struct = analyze_module(CLUSTERING_PY)
    reg = get_registry_map(CLUSTERING_PY)

    st.markdown("##### Architecture")
    st.markdown(
        """
**Design.** `BaseClusterer` exposes `fit_predict(embeddings, k)`. This module includes **KMeans** on the input
embeddings, an **autoencoder + KMeans** head (`AEClusterer`, config key `ae`), an **IDEC-style** deep clustering
model (`IDECClusterer`, registered as **`dec`** in the table below), and **GCNClusterer** (graph convolutions
then KMeans in the learned space). `get_clusterer(cfg)` uses `cfg["clustering"]["name"]` to select the entry
in `_REGISTRY`.
        """.strip()
    )

    if reg:
        st.markdown("**Registry** (`name` → class)")
        st.table(
            [{"clusterer key": k, "class": v} for k, v in sorted(reg.items(), key=lambda x: x[0])]
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
            if b in ("ABC", "object", "Module"):
                continue
            lines.append(f"    {b} <|-- {c.name}")
    if len(lines) > 1:
        st.markdown("##### Inheritance (Mermaid)")
        st.code("\n".join(lines), language="mermaid")

    st.markdown("##### Top-level functions")
    fn_df = pd.DataFrame([{"function": f.name, "line": f.lineno} for f in struct.functions])
    st.dataframe(fn_df, use_container_width=True, hide_index=True)

    with st.expander("Full source", expanded=False):
        st.code(CLUSTERING_PY.read_text(encoding="utf-8"), language="python")
