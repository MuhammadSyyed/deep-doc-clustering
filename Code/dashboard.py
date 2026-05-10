from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / "datasets"
RESULTS_PATH = BASE_DIR / "outputs" / "results.csv"
PLOTS_DIR = BASE_DIR / "outputs" / "plots"


def normalize_name(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.lower()


st.set_page_config(page_title="Deep Document Clustering Dashboard", layout="wide")

# Dark, high-contrast styling for better readability.
st.markdown(
    """
    <style>
      .stApp {
        background-color: #0e1117;
        color: #f3f6fc;
      }
      [data-testid="stSidebar"] {
        background-color: #151a24;
      }
      .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #e9eef9;
      }
      h1, h2, h3, h4, h5, h6, p, label, div, span {
        color: #f3f6fc !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_dataset_metadata(datasets_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for metadata_file in sorted(datasets_dir.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        dataset_name = metadata.get("name", metadata_file.parent.name)
        labels = metadata.get("label_map", {})
        rows.append(
            {
                "dataset": dataset_name,
                "n_docs": metadata.get("n_docs"),
                "n_classes": metadata.get("n_classes"),
                "source": metadata.get("source"),
                "labels_preview": ", ".join(list(labels.values())[:5]),
                "data_path": str(metadata_file.parent / "data.csv"),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data
def build_plot_index(plots_dir: Path) -> pd.DataFrame:
    rows: list[dict] = []
    pattern = re.compile(
        r"^(?P<dataset>.+?)_(?P<encoder>[^_]+)_(?P<model>[^_]+)_clusters.*\.(png|jpg|jpeg|webp)$",
        re.IGNORECASE,
    )

    for image_path in sorted(plots_dir.glob("*")):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue

        parsed = pattern.match(image_path.name)
        if parsed:
            dataset = parsed.group("dataset")
            encoder = parsed.group("encoder")
            model = parsed.group("model")
        else:
            dataset, encoder, model = "unknown", "unknown", "unknown"

        rows.append(
            {
                "file_name": image_path.name,
                "path": str(image_path),
                "dataset": dataset,
                "encoder": encoder,
                "model": model,
                "size_mb": round(image_path.stat().st_size / (1024 * 1024), 2),
            }
        )

    return pd.DataFrame(rows)


st.title("Deep Document Clustering Dashboard")
st.caption("Interactive exploration of datasets, model results, and cluster visualizations.")
plotly_dark_layout = {
    "template": "plotly_dark",
    "plot_bgcolor": "#0e1117",
    "paper_bgcolor": "#0e1117",
    "font": {"color": "#f3f6fc"},
    "height": 420,
}

results_df = load_results(RESULTS_PATH)
datasets_df = load_dataset_metadata(DATASETS_DIR)
plots_df = build_plot_index(PLOTS_DIR)

if not results_df.empty:
    if "encoder" in results_df.columns:
        results_df["encoder"] = results_df["encoder"].map(normalize_name)
    if "model" in results_df.columns:
        results_df["model"] = results_df["model"].map(normalize_name)

if not plots_df.empty:
    if "encoder" in plots_df.columns:
        plots_df["encoder"] = plots_df["encoder"].map(normalize_name)
    if "model" in plots_df.columns:
        plots_df["model"] = plots_df["model"].map(normalize_name)

# Prefer actual processed sample size from results (e.g., dbpedia 150k) over raw source size.
if not datasets_df.empty and not results_df.empty and {"dataset", "rows"}.issubset(results_df.columns):
    sampled_rows_by_dataset = (
        results_df.dropna(subset=["dataset", "rows"])
        .groupby("dataset", as_index=False)["rows"]
        .max()
        .rename(columns={"rows": "processed_rows"})
    )
    datasets_df = datasets_df.merge(sampled_rows_by_dataset, on="dataset", how="left")
    datasets_df["used_docs"] = datasets_df["processed_rows"].fillna(datasets_df["n_docs"])
else:
    datasets_df["used_docs"] = datasets_df.get("n_docs")

all_datasets = sorted(
    set(datasets_df.get("dataset", pd.Series(dtype=str)).dropna().tolist())
    | set(results_df.get("dataset", pd.Series(dtype=str)).dropna().tolist())
    | set(plots_df.get("dataset", pd.Series(dtype=str)).dropna().tolist())
)
all_encoders = sorted(
    set(results_df.get("encoder", pd.Series(dtype=str)).dropna().tolist())
    | set(plots_df.get("encoder", pd.Series(dtype=str)).dropna().tolist())
)
all_models = sorted(
    set(results_df.get("model", pd.Series(dtype=str)).dropna().tolist())
    | set(plots_df.get("model", pd.Series(dtype=str)).dropna().tolist())
)

st.markdown("### Filters")
filter_col1, filter_col2, filter_col3 = st.columns(3)
with filter_col1:
    selected_datasets = st.multiselect("Datasets", options=all_datasets, default=all_datasets)
with filter_col2:
    selected_encoders = st.multiselect("Encoders", options=all_encoders, default=all_encoders)
with filter_col3:
    selected_models = st.multiselect("Clustering Models", options=all_models, default=all_models)


def in_selected(series: pd.Series, selected: list[str]) -> pd.Series:
    if series.empty:
        return pd.Series([], dtype=bool)
    if not selected:
        return pd.Series([False] * len(series), index=series.index)
    return series.isin(selected)


datasets_mask = (
    in_selected(datasets_df.get("dataset", pd.Series(dtype=str)), selected_datasets)
    if not datasets_df.empty
    else pd.Series([], dtype=bool)
)
results_mask = (
    in_selected(results_df.get("dataset", pd.Series(dtype=str)), selected_datasets)
    & in_selected(results_df.get("encoder", pd.Series(dtype=str)), selected_encoders)
    & in_selected(results_df.get("model", pd.Series(dtype=str)), selected_models)
    if not results_df.empty
    else pd.Series([], dtype=bool)
)
plots_mask = (
    in_selected(plots_df.get("dataset", pd.Series(dtype=str)), selected_datasets)
    & in_selected(plots_df.get("encoder", pd.Series(dtype=str)), selected_encoders)
    & in_selected(plots_df.get("model", pd.Series(dtype=str)), selected_models)
    if not plots_df.empty
    else pd.Series([], dtype=bool)
)

filtered_datasets = datasets_df[datasets_mask] if not datasets_df.empty else datasets_df
filtered_results = results_df[results_mask] if not results_df.empty else results_df
filtered_plots = plots_df[plots_mask] if not plots_df.empty else plots_df

tab_datasets, tab_results, tab_visuals = st.tabs(
    ["Datasets", "Results", "Visualizing Clusters"]
)

with tab_datasets:
    st.subheader("Dataset Details")

    if filtered_datasets.empty:
        st.warning("No dataset metadata matched the current filters.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Datasets", value=len(filtered_datasets))
        col2.metric("Total Documents", value=f"{int(filtered_datasets['used_docs'].fillna(0).sum()):,}")
        col3.metric("Total Classes", value=f"{int(filtered_datasets['n_classes'].fillna(0).sum()):,}")

        docs_fig = px.bar(
            filtered_datasets.sort_values("used_docs", ascending=False),
            x="dataset",
            y="used_docs",
            title="Documents per Dataset",
            text_auto=True,
        )
        docs_fig.update_layout(**plotly_dark_layout)
        st.plotly_chart(docs_fig, use_container_width=True)

        st.dataframe(
            filtered_datasets.sort_values(by=["dataset"], ascending=True)[
                ["dataset", "used_docs", "n_docs", "n_classes", "source", "labels_preview", "data_path"]
            ],
            use_container_width=True,
        )

with tab_results:
    st.subheader("Clustering Results")

    if filtered_results.empty:
        st.warning("No results matched the selected dataset/encoder/model filters.")
    else:
        metric = st.pills(
            "Metric",
            options=["All", "NMI", "ARI", "ACC", "Purity"],
            selection_mode="single",
            default="All",
        )
        if metric is None:
            metric = "All"

        if metric == "All":
            melted = filtered_results.melt(
                id_vars=["dataset", "encoder", "model", "rows", "pca_components"],
                value_vars=["NMI", "ARI", "ACC", "Purity"],
                var_name="metric",
                value_name="score",
            )
            summary_fig = px.bar(
                melted.sort_values(["dataset", "metric", "encoder"], ascending=True),
                x="dataset",
                y="score",
                color="metric",
                barmode="group",
                hover_data=["encoder", "model", "rows", "pca_components"],
                title="All Metrics by Dataset",
            )
        else:
            summary_fig = px.bar(
                filtered_results.sort_values(metric, ascending=False),
                x="dataset",
                y=metric,
                color="encoder",
                barmode="group",
                hover_data=["model", "rows", "pca_components"],
                title=f"{metric} by Dataset and Encoder",
            )
        summary_fig.update_layout(**plotly_dark_layout)
        st.plotly_chart(summary_fig, use_container_width=True)

        pivot = (
            filtered_results.pivot_table(
                index=["dataset", "encoder", "model"],
                values=["NMI", "ARI", "ACC", "Purity"],
                aggfunc="mean",
            )
            .reset_index()
            .sort_values(by=["dataset", "encoder", "model"], ascending=True)
        )
        st.dataframe(pivot, use_container_width=True)

with tab_visuals:
    st.subheader("Cluster Plot Gallery")

    if filtered_plots.empty:
        st.warning("No cluster plots matched the selected filters.")
    else:
        selection = st.selectbox("Choose Plot File", options=filtered_plots["file_name"].tolist())
        selected_row = filtered_plots[filtered_plots["file_name"] == selection].iloc[0]
        selected_path = Path(selected_row["path"])

        info_cols = st.columns(4)
        info_cols[0].metric("Dataset", selected_row["dataset"])
        info_cols[1].metric("Encoder", selected_row["encoder"])
        info_cols[2].metric("Model", selected_row["model"])
        info_cols[3].metric("Image Size (MB)", selected_row["size_mb"])

        left_spacer, image_col, right_spacer = st.columns([1, 3, 1])
        with image_col:
            st.image(str(selected_path), caption=selected_row["file_name"], width=850)

        with selected_path.open("rb") as image_file:
            st.download_button(
                "Download Selected Plot",
                data=image_file.read(),
                file_name=selected_row["file_name"],
                mime="image/png",
            )
