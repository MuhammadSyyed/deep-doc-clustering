"""Interactive views of results/results.csv (aggregated metrics + timings schema)."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.paths import RESULTS_CSV, ROOT


def _sorted_unique(series: pd.Series) -> list:
    return sorted(series.dropna().astype(str).unique().tolist())


def _pretty_metric(col: str) -> str:
    base = col.replace("_mean", "").replace("_", " ").strip()
    return base.upper() if base else col


def _metric_mean_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for c in df.columns:
        if not c.endswith("_mean"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        # e.g. clustering_time_mean ends with _mean but is a timing field, not a score
        if "time" in c.lower():
            continue
        cols.append(c)
    preferred = [
        "acc_mean",
        "nmi_mean",
        "ari_mean",
        "purity_mean",
        "silhouette_mean",
        "davies_bouldin_mean",
        "collapse_mean",
    ]
    ordered = [c for c in preferred if c in cols]
    rest = sorted(c for c in cols if c not in ordered)
    return ordered + rest


def _time_columns(df: pd.DataFrame) -> tuple[list[str], dict[str, str]]:
    """Return (column_names, display_labels)."""
    mapping: dict[str, str] = {}
    if "embedding_time" in df.columns:
        mapping["embedding_time"] = "Embedding"
    if "clustering_time_mean" in df.columns:
        mapping["clustering_time_mean"] = "Clustering (mean)"
    if "clustering_time_std" in df.columns and pd.api.types.is_numeric_dtype(df["clustering_time_std"]):
        if df["clustering_time_std"].fillna(0).abs().sum() > 1e-12:
            mapping["clustering_time_std"] = "Clustering (std)"
    if "total_time" in df.columns:
        mapping["total_time"] = "Total"
    legacy_enc = "encoding_time" if "encoding_time" in df.columns else None
    legacy_clu = "clustering_time" if "clustering_time" in df.columns else None
    if legacy_enc and legacy_enc not in mapping:
        mapping[legacy_enc] = "Encoding (legacy)"
    if legacy_clu and legacy_clu not in mapping:
        mapping[legacy_clu] = "Clustering (legacy)"
    return list(mapping.keys()), mapping


def _prepare_plot_id(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Pick x-axis: dataset alone, or composite run label when multiple experiments or duplicate datasets."""
    if df.empty or "dataset" not in df.columns:
        return df.copy() if not df.empty else df, "dataset"
    out = df.copy()
    multi_by_ds = len(out) > out["dataset"].nunique()
    multi_exp = "experiment_name" in out.columns and out["experiment_name"].nunique() > 1
    if not multi_by_ds and not multi_exp:
        return out, "dataset"
    parts: list[pd.Series] = []
    if "experiment_name" in out.columns and multi_exp:
        parts.append(out["experiment_name"].astype(str))
    parts.append(out["dataset"].astype(str))
    if "encoder" in out.columns:
        parts.append(out["encoder"].astype(str))
    if "clusterer" in out.columns:
        parts.append(out["clusterer"].astype(str))
    lab = parts[0].copy()
    for p in parts[1:]:
        lab = lab + " · " + p
    out["_run_label"] = lab
    return out, "_run_label"


def render_results_tab() -> None:
    st.subheader("Experiment results")
    st.caption(f"Source: `{RESULTS_CSV.relative_to(ROOT)}` — rows are aggregated runs (means/std where applicable).")

    if not RESULTS_CSV.is_file():
        st.warning("results.csv not found. Run the pipeline to populate it.")
        return

    df_all = pd.read_csv(RESULTS_CSV)
    if df_all.empty:
        st.info("CSV is empty.")
        return

    st.markdown("##### Filters")
    st.caption(
        "Restrict the table and all charts. Empty multiselect = no restriction on that axis. "
        "**Baseline category** is the `experiment_name` column (e.g. `baseline_known_k`)."
    )
    r1 = st.columns(4)
    with r1[0]:
        if "experiment_name" in df_all.columns:
            exp_opts = _sorted_unique(df_all["experiment_name"])
            exp_pick = st.multiselect(
                "Baseline category",
                options=exp_opts,
                default=exp_opts,
                key="res_filter_experiment",
                help="Matches CSV column `experiment_name` (experiment / baseline preset).",
            )
            if not exp_pick:
                exp_pick = exp_opts
        else:
            exp_pick = None
    with r1[1]:
        if "encoder" in df_all.columns:
            enc_opts = _sorted_unique(df_all["encoder"])
            enc_pick = st.multiselect("Encoder", options=enc_opts, default=enc_opts, key="res_filter_encoder")
            if not enc_pick:
                enc_pick = enc_opts
        else:
            enc_pick = None
    with r1[2]:
        if "clusterer" in df_all.columns:
            cl_opts = _sorted_unique(df_all["clusterer"])
            cl_pick = st.multiselect("Clusterer", options=cl_opts, default=cl_opts, key="res_filter_clusterer")
            if not cl_pick:
                cl_pick = cl_opts
        else:
            cl_pick = None
    with r1[3]:
        if "dataset" in df_all.columns:
            ds_opts = _sorted_unique(df_all["dataset"])
            ds_pick = st.multiselect("Dataset", options=ds_opts, default=ds_opts, key="res_filter_dataset")
            if not ds_pick:
                ds_pick = ds_opts
        else:
            ds_pick = None

    df = df_all.copy()
    if exp_pick is not None:
        df = df[df["experiment_name"].astype(str).isin(exp_pick)]
    if enc_pick is not None:
        df = df[df["encoder"].astype(str).isin(enc_pick)]
    if cl_pick is not None:
        df = df[df["clusterer"].astype(str).isin(cl_pick)]
    if ds_pick is not None:
        df = df[df["dataset"].astype(str).isin(ds_pick)]

    if df.empty:
        st.warning("No rows match the current filters.")
        return

    st.caption(f"Showing **{len(df)}** of **{len(df_all)}** run(s).")

    metric_cols = _metric_mean_columns(df)
    time_cols, time_labels = _time_columns(df)
    cat_cols = [c for c in ["experiment_name", "dataset", "encoder", "clusterer"] if c in df.columns]

    # Quick summary
    show_metrics = [c for c in ["acc_mean", "nmi_mean", "ari_mean"] if c in df.columns]
    if show_metrics:
        st.markdown("##### Summary (filtered mean)")
        sm = st.columns(len(show_metrics))
        for i, mc in enumerate(show_metrics):
            with sm[i]:
                st.metric(_pretty_metric(mc), f"{df[mc].mean():.4f}")

    with st.expander("Full results table", expanded=False):
        st.dataframe(df, use_container_width=True, hide_index=True)

    df_plot, x_cat = _prepare_plot_id(df)
    x_title = "Run (experiment · dataset · encoder · clusterer)" if x_cat == "_run_label" else "Dataset"

    st.markdown("##### Performance metrics")
    if metric_cols and x_cat in df_plot.columns:
        melt = df_plot.melt(
            id_vars=[x_cat],
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        ).dropna(subset=["value"])
        if not melt.empty:
            melt["metric"] = melt["metric"].map(lambda c: _pretty_metric(str(c)))
            fig_g = px.bar(
                melt,
                x=x_cat,
                y="value",
                color="metric",
                barmode="group",
                title="Metrics (mean per run)",
                labels={x_cat: x_title, "value": "Score"},
            )
            fig_g.update_layout(xaxis_tickangle=-40, hovermode="x unified", dragmode="zoom", legend_title_text="Metric")
            st.plotly_chart(fig_g, use_container_width=True)

        plot_metrics = [c for c in metric_cols if df_plot[c].notna().any()]
        if len(plot_metrics) > 1:
            heat = df_plot.set_index(x_cat)[plot_metrics]
            heat = heat.dropna(axis=1, how="all")
            if not heat.empty:
                heat_disp = heat.T
                heat_disp.index = [_pretty_metric(str(i)) for i in heat_disp.index]
                fig_h = px.imshow(
                    heat_disp,
                    labels=dict(x=x_title, y="Metric", color="Score"),
                    title="Metric heatmap (means)",
                    aspect="auto",
                )
                fig_h.update_layout(dragmode="zoom")
                st.plotly_chart(fig_h, use_container_width=True)
    elif not metric_cols:
        st.info("No `*_mean` numeric metric columns found in this CSV.")

    st.markdown("##### Timing")
    if time_cols and "dataset" in df_plot.columns:
        t_melt = df_plot.melt(
            id_vars=[x_cat],
            value_vars=time_cols,
            var_name="phase",
            value_name="seconds",
        )
        t_melt["phase"] = t_melt["phase"].map(lambda p: time_labels.get(str(p), str(p)))
        timing_view = st.radio(
            "Timing scale",
            options=["Normalized (0–1)", "Raw seconds"],
            horizontal=True,
            index=0,
            key="timing_scale_mode",
        )
        if timing_view == "Normalized (0–1)":
            grp = t_melt.groupby("phase")["seconds"]
            smin = grp.transform("min")
            smax = grp.transform("max")
            denom = (smax - smin).replace(0, 1.0)
            t_melt["value"] = (t_melt["seconds"] - smin) / denom
            y_col = "value"
            y_title = "Relative time (0 = fastest, 1 = slowest)"
            title = "Timing by phase (normalized within each phase)"
        else:
            y_col = "seconds"
            y_title = "Seconds"
            title = "Timing by phase (seconds)"
        fig_t = px.bar(
            t_melt,
            x=x_cat,
            y=y_col,
            color="phase",
            barmode="group",
            title=title,
            labels={x_cat: x_title, y_col: y_title},
        )
        fig_t.update_layout(xaxis_tickangle=-40, hovermode="x unified")
        if timing_view == "Normalized (0–1)":
            fig_t.update_yaxes(range=[0, 1], tickformat=".2f")
        fig_t.update_traces(hovertemplate="%{x}<br>%{fullData.name}: %{y}<extra></extra>")
        st.plotly_chart(fig_t, use_container_width=True)
    else:
        st.info("No timing columns found (`embedding_time`, `clustering_time_mean`, `total_time`, …).")

    st.markdown("##### Explore")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    x_opt = num_cols if num_cols else list(df.columns)
    y_opt = num_cols if num_cols else list(df.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        x_ax = st.selectbox("X axis", options=x_opt, index=0, key="res_explore_x")
    with c2:
        y_ix = min(1, len(y_opt) - 1) if len(y_opt) > 1 else 0
        y_ax = st.selectbox("Y axis", options=y_opt, index=y_ix, key="res_explore_y")
    with c3:
        color_ax = st.selectbox("Color", options=["(none)"] + cat_cols, index=0, key="res_explore_c")

    hover_cols = [
        c
        for c in [
            "experiment_name",
            "dataset",
            "encoder",
            "clusterer",
            "k_used",
            "k_true",
            "n_docs",
        ]
        if c in df.columns
    ]
    if x_ax and y_ax:
        color = None if color_ax == "(none)" else color_ax
        fig_s = px.scatter(
            df,
            x=x_ax,
            y=y_ax,
            color=color,
            size="n_docs" if "n_docs" in df.columns else None,
            hover_data=hover_cols,
            title=f"{y_ax} vs {x_ax}",
        )
        fig_s.update_layout(dragmode="zoom", hovermode="closest")
        st.plotly_chart(fig_s, use_container_width=True)

    pc_metrics = [
        c
        for c in metric_cols
        if df[c].notna().any() and df[c].notna().sum() >= 1
    ]
    if len(pc_metrics) >= 2 and "dataset" in df.columns:
        st.markdown("##### Parallel coordinates")
        st.caption(
            "Each line is one CSV row. **Davies–Bouldin** is lower-is-better; other shown metrics are typically higher-is-better. "
            "Drag along an axis to filter."
        )
        _tick_font = dict(size=12)
        dimensions: list[dict] = []
        for col in pc_metrics:
            ser = pd.to_numeric(df[col], errors="coerce")
            if ser.notna().sum() == 0:
                continue
            fill = float(ser.median()) if ser.notna().any() else 0.0
            vals = ser.fillna(fill).astype(float).tolist()
            lo, hi = float(min(vals)), float(max(vals))
            if hi <= 1.05 and lo >= -0.05:
                pad = 0.03 if hi > lo else 0.05
                lo = max(0.0, lo - pad)
                hi = min(1.0, hi + pad) if hi <= 1.0 else hi + pad
            elif hi > lo:
                span = hi - lo
                lo -= span * 0.05
                hi += span * 0.05
            tickformat = ".2f" if abs(hi) <= 2 and abs(lo) <= 2 else ",.2f"
            dimensions.append(
                dict(
                    label=_pretty_metric(col)[:24],
                    values=vals,
                    range=[lo, hi],
                    tickformat=tickformat,
                )
            )

        if "n_docs" in df.columns:
            cvals = df["n_docs"].astype(float)
            cmin, cmax = float(cvals.min()), float(cvals.max())
            if cmin == cmax:
                cmax = cmin + 1.0
            line = dict(
                color=cvals.tolist(),
                colorscale="Plasma",
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(
                    title=dict(text="n_docs", font=dict(size=14)),
                    tickfont=dict(size=12),
                    len=0.75,
                    thickness=18,
                    outlinewidth=0,
                ),
            )
        else:
            line = dict(color="rgba(55, 126, 184, 0.65)")

        fig_r = go.Figure(
            data=go.Parcoords(
                line=line,
                dimensions=dimensions,
                labelangle=-20,
                labelfont=dict(size=14, family="Arial, sans-serif"),
                tickfont=_tick_font,
                unselected=dict(line=dict(color="lightgray", opacity=0.35)),
            )
        )
        fig_r.update_layout(
            title=dict(
                text="Metrics across runs (means)",
                font=dict(size=17),
                x=0.02,
                xanchor="left",
            ),
            font=dict(size=13, family="Arial, sans-serif"),
            height=520,
            margin=dict(l=56, r=120, t=80, b=48),
            plot_bgcolor="#f8f9fb",
            paper_bgcolor="white",
        )
        st.plotly_chart(
            fig_r,
            use_container_width=True,
            config={"scrollZoom": True, "displayModeBar": True},
        )
