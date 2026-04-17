from pathlib import Path
import copy
import json
import os
import pandas as pd

cols = ['experiment_name','dataset','encoder','clusterer','n_docs','k_used','k_true','embedding_time','clustering_time_mean','clustering_time_std','total_time','acc_mean','acc_std','nmi_mean','nmi_std','ari_mean','ari_std','purity_mean','purity_std','silhouette_mean','silhouette_std','davies_bouldin_mean','davies_bouldin_std','collapse_mean']
def save_results(result: dict, filename: str = "results.csv"):
    os.makedirs("../results", exist_ok=True)

    file_path = os.path.join("../results", filename)

    columns = cols

    row = {col: result.get(col) for col in columns}

    df = pd.DataFrame([row])

    if os.path.exists(file_path):
        df.to_csv(file_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_path, mode="w", header=True, index=False)

def dataset_table_path(REPO_ROOT,name):
    folder = REPO_ROOT / "datasets" / name
    for rel in (Path("data.csv"), Path(f"{name}.csv")):
        p = folder / rel
        if p.is_file():
            return p
    raise FileNotFoundError(folder)

def load_json(path):
    path = Path(path)  # normalize input

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix != ".json":
        raise ValueError(f"Expected a .json file, got: {path.suffix}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_metadata(REPO_ROOT,name):
    folder = REPO_ROOT / "datasets" / name
    for rel in (Path("metadata.json"), Path(f"{name}.json")):
        p = folder / rel
        if p.is_file():
            return load_json(p)
    raise FileNotFoundError(folder)


