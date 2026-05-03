from pathlib import Path


def ensure_artifact_subfolders(root_path: Path | str = Path.cwd()) -> None:
    root = Path(root_path)
    artifacts_dir = root / "artifacts"
    subfolders = ["embeddings", "models", "training_metrics"]

    if not artifacts_dir.exists():
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    for subfolder in subfolders:
        folder = artifacts_dir / subfolder
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    ensure_artifact_subfolders()