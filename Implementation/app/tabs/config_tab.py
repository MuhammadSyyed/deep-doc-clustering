"""Edit pipeline YAML (matches `config/default.yaml` layout)."""

from __future__ import annotations

from typing import Any, Dict

import streamlit as st
import yaml

from app.paths import CONFIG_LEGACY, CONFIG_PRIMARY, ROOT, ensure_configs_dir, resolve_config_path


def render_config_tab() -> None:
    st.subheader("Pipeline configuration")
    load_path = resolve_config_path()
    if not load_path.exists():
        st.error(f"No configuration file found. Expected `{CONFIG_PRIMARY}` or `{CONFIG_LEGACY}`.")
        return

    st.caption(
        f"Loaded from `{load_path.relative_to(ROOT)}`. Saves write to `{CONFIG_PRIMARY.relative_to(ROOT)}` "
        "(folder `configs/` is created if needed)."
    )

    raw_disk = load_path.read_text(encoding="utf-8")
    editor_key = "pipeline_yaml_editor_v2"
    if editor_key not in st.session_state:
        st.session_state[editor_key] = raw_disk

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reload from disk", key="cfg_reload"):
            st.session_state[editor_key] = load_path.read_text(encoding="utf-8")
            st.rerun()
    with c2:
        st.download_button(
            label="Download current editor text",
            data=st.session_state[editor_key],
            file_name="default.yaml",
            mime="text/yaml",
            key="cfg_download",
        )

    yaml_text = st.text_area(
        "Configuration (YAML)",
        height=520,
        key=editor_key,
        help="Structure follows the repo `config/default.yaml`: experiment, dataset, preprocessing, encoder, reduction, clustering, evaluation, logging.",
    )

    with st.expander("Parsed preview (read-only)", expanded=False):
        try:
            parsed: Dict[str, Any] = yaml.safe_load(yaml_text) or {}
            st.json(parsed)
        except yaml.YAMLError as e:
            st.error(f"Invalid YAML: {e}")

    err = st.empty()
    if st.button("Save to configs/default.yaml", type="primary", key="cfg_save"):
        try:
            new_cfg = yaml.safe_load(yaml_text)
            if not isinstance(new_cfg, dict):
                err.error("YAML root must be a mapping (dictionary).")
                return
        except yaml.YAMLError as e:
            err.error(f"Invalid YAML: {e}")
            return
        ensure_configs_dir()
        out_text = yaml.dump(new_cfg, default_flow_style=False, sort_keys=False, allow_unicode=True)
        CONFIG_PRIMARY.write_text(out_text, encoding="utf-8")
        st.session_state[editor_key] = out_text
        st.success(f"Wrote `{CONFIG_PRIMARY.relative_to(ROOT)}` — reloading editor.")
        st.rerun()

    with st.expander("Original file on disk (reference)", expanded=False):
        st.code(raw_disk, language="yaml")
