"""Project root and canonical paths for the Streamlit dashboard."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIG_PRIMARY = ROOT / "configs" / "default.yaml"
CONFIG_LEGACY = ROOT / "config" / "default.yaml"
DATASETS_DIR = ROOT / "datasets"
MODULES_DIR = ROOT / "modules"
ENCODERS_PY = MODULES_DIR / "encoders.py"
CLUSTERING_PY = MODULES_DIR / "clustering.py"
RESULTS_CSV = ROOT / "results" / "results.csv"


def resolve_config_path() -> Path:
    if CONFIG_PRIMARY.exists():
        return CONFIG_PRIMARY
    return CONFIG_LEGACY


def ensure_configs_dir() -> None:
    CONFIG_PRIMARY.parent.mkdir(parents=True, exist_ok=True)
