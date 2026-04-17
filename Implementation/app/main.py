"""
Clustering pipeline dashboard.

Run from repository root:

    streamlit run app/main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from app.tabs.clusters_tab import render_clusters_tab
from app.tabs.config_tab import render_config_tab
from app.tabs.datasets_tab import render_datasets_tab
from app.tabs.encoders_tab import render_encoders_tab
from app.tabs.results_tab import render_results_tab

st.set_page_config(
    page_title="Clustering pipeline",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Clustering pipeline")
st.markdown(
    "Inspect **configs**, **datasets**, **encoder / clusterer modules**, and **results** in one place."
)

tab_cfg, tab_data, tab_enc, tab_clu, tab_res = st.tabs(
    ["Configs", "Datasets", "Encoders", "Clusters", "Results"]
)

with tab_cfg:
    render_config_tab()

with tab_data:
    render_datasets_tab()

with tab_enc:
    render_encoders_tab()

with tab_clu:
    render_clusters_tab()

with tab_res:
    render_results_tab()
