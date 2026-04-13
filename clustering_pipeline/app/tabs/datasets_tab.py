"""Browse datasets/*/metadata.json with tables and Plotly charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

from app.paths import DATASETS_DIR, ROOT


def _load_all_metadata() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not DATASETS_DIR.is_dir():
        return rows
    for sub in sorted(DATASETS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        meta_path = sub / "metadata.json"
        if not meta_path.is_file():
            continue
        try:
            data = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"name": sub.name, "error": "invalid JSON"}
        data["_folder"] = sub.name
        data["_has_data_csv"] = (sub / "data.csv").is_file()
        rows.append(data)
    return rows


def render_datasets_tab() -> None:
    st.subheader("Datasets")
    st.caption(f"Scanning `{DATASETS_DIR.relative_to(ROOT)}/*/metadata.json`.")

    metas = _load_all_metadata()
    if not metas:
        st.warning("No dataset folders with metadata.json found.")
        return

    summary_records = []
    for m in metas:
        lm = m.get("label_map")
        n_labels = len(lm) if isinstance(lm, dict) else m.get("n_classes")
        summary_records.append(
            {
                "folder": m.get("_folder"),
                "name": m.get("name"),
                "n_docs": m.get("n_docs"),
                "n_classes": m.get("n_classes"),
                "labels_in_map": len(lm) if isinstance(lm, dict) else None,
                "source": m.get("source"),
                "data.csv": m.get("_has_data_csv"),
            }
        )
    summary = pd.DataFrame(summary_records)

    st.markdown("##### Overview")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        fig_docs = px.bar(
            summary.sort_values("n_docs", ascending=False),
            x="name",
            y="n_docs",
            color="n_classes",
            title="Documents per dataset",
            labels={"n_docs": "Documents", "name": "Dataset"},
        )
        fig_docs.update_layout(xaxis_tickangle=-35, dragmode="pan", hovermode="x unified")
        fig_docs.update_traces(hovertemplate="%{x}<br>docs: %{y}<extra></extra>")
        st.plotly_chart(fig_docs, use_container_width=True)
    with c2:
        show_labels = st.checkbox("Show point labels", value=False, key="datasets_scatter_labels")
        fig_sc = px.scatter(
            summary,
            x="n_docs",
            y="n_classes",
            size="n_docs",
            size_max=40,
            hover_name="name",
            title="Classes vs documents (bubble size ∝ docs)",
            labels={"n_docs": "Documents", "n_classes": "Classes"},
        )
        if show_labels:
            fig_sc.update_traces(
                text=summary["name"],
                textposition="top center",
                textfont=dict(size=10),
            )
        fig_sc.update_traces(
            hovertemplate="%{hovertext}<br>docs: %{x:,}<br>classes: %{y}<extra></extra>"
        )
        fig_sc.update_layout(dragmode="zoom")
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("##### Per-dataset details")
    names = [m.get("name") or m.get("_folder") for m in metas]
    choice = st.selectbox("Dataset", options=range(len(metas)), format_func=lambda i: names[i])
    sel = metas[choice]

    col_a, col_b = st.columns(2)
    with col_a:
        st.json({k: v for k, v in sel.items() if not str(k).startswith("_")})
    with col_b:
        lm = sel.get("label_map")
        if isinstance(lm, dict) and lm:
            lf = pd.DataFrame(
                [{"id": k, "label": v} for k, v in sorted(lm.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])]
            )
            st.dataframe(lf, use_container_width=True, hide_index=True)
            lf_plot = lf.assign(weight=1)
            fig_lbl = px.bar(
                lf_plot,
                x="label",
                y="weight",
                title="Classes in metadata (no per-class document counts)",
                labels={"weight": "placeholder", "label": "Class"},
            )
            fig_lbl.update_traces(hovertemplate="%{x}<extra></extra>")
            fig_lbl.update_layout(showlegend=False, xaxis_tickangle=-45, yaxis_visible=False)
            st.plotly_chart(fig_lbl, use_container_width=True)
        else:
            st.info("No label_map in this metadata file.")

    # Source strings length comparison
    src_df = pd.DataFrame(
        [{"name": m.get("name"), "source": (m.get("source") or "")[:120]} for m in metas]
    )
    src_df["len"] = src_df["source"].str.len()
    fig_h = px.bar(
        src_df.sort_values("len", ascending=True),
        x="len",
        y="name",
        orientation="h",
        title="Source field length (truncated display)",
    )
    fig_h.update_layout(yaxis=dict(categoryorder="total ascending"))
    st.plotly_chart(fig_h, use_container_width=True)
