import argparse
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path
import pandas as pd


def _check_pip(*packages):
    missing = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n  Missing packages: {', '.join(missing)}")
        print(f"  Run:  pip install {' '.join(missing)}\n")
        sys.exit(1)


GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(s): return print(f"  {GREEN}✓{RESET} {s}")
def warn(s): return print(f"  {YELLOW}⚠{RESET}  {s}")


def err(s): return print(f"  {RED}✗{RESET} {s}")
def hdr(s): return print(f"\n{BOLD}{'─'*55}\n  {s}\n{'─'*55}{RESET}")


def save_dataset(out_dir: Path, name: str, rows: list, label_map: dict, source: str):
    """Persist rows + metadata. rows = list of (id, text, label_int)."""
    folder = out_dir / name
    folder.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows, columns=["id", "text", "label"])
    df.to_csv(folder / "data.csv", index=False)

    meta = {
        "name":      name,
        "n_docs":    len(df),
        "n_classes": df["label"].nunique(),
        "label_map": label_map,       # int → human-readable class name
        "source":    source,
    }
    with open(folder / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    ok(f"Saved {len(df):,} docs · {df['label'].nunique()} classes  →  {folder / 'data.csv'}")
    return meta


def subsample(rows, max_samples, seed=42):
    if max_samples and len(rows) > max_samples:
        random.seed(seed)
        rows = random.sample(rows, max_samples)
    return rows


def download_20newsgroups(out_dir, max_samples):
    hdr("1 / 9  ·  20 Newsgroups")
    _check_pip("sklearn")
    from sklearn.datasets import fetch_20newsgroups

    print("  Fetching via sklearn …")
    raw = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),
        random_state=42,
    )
    rows = [
        (str(i), raw.data[i].strip(), int(raw.target[i]))
        for i in range(len(raw.data))
        if raw.data[i].strip()
    ]
    rows = subsample(rows, max_samples)
    label_map = {str(i): name for i, name in enumerate(raw.target_names)}
    return save_dataset(out_dir, "20newsgroups", rows, label_map,
                        source="sklearn.datasets.fetch_20newsgroups")


def download_reuters(out_dir, max_samples):
    hdr("2 / 9  ·  Reuters-8 (Reuters-21578 subset)")
    _check_pip("datasets")
    from datasets import load_dataset

    print("  Fetching via HuggingFace datasets …")
    raw = load_dataset("yangwang825/reuters-21578")

    label_feat = raw["train"].features["label"]
    if hasattr(label_feat, "names") and getattr(label_feat, "names", None):
        label_map = {str(i): n for i, n in enumerate(label_feat.names)}
    else:
        seen = sorted({int(ex["label"]) for split in raw for ex in raw[split]})
        label_map = {str(i): str(i) for i in seen}

    rows = []
    idx = 0
    for split in ("train", "test"):
        for ex in raw[split]:
            text = ex["text"].strip()
            if text:
                rows.append((str(idx), text, int(ex["label"])))
                idx += 1

    rows = subsample(rows, max_samples)
    return save_dataset(out_dir, "reuters", rows, label_map,
                        source="HuggingFace: yangwang825/reuters-21578 (Reuters-8)")


def download_agnews(out_dir, max_samples):
    hdr("3 / 9  ·  AG News")
    _check_pip("datasets")
    from datasets import load_dataset

    print("  Fetching via HuggingFace datasets …")
    raw = load_dataset("ag_news")

    LABEL_MAP = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    rows = []
    idx = 0
    for split in ("train", "test"):
        for ex in raw[split]:
            rows.append((str(idx), ex["text"].strip(), int(ex["label"])))
            idx += 1

    rows = subsample(rows, max_samples)
    return save_dataset(out_dir, "agnews", rows,
                        {str(k): v for k, v in LABEL_MAP.items()},
                        source="HuggingFace: ag_news")


def download_bbc_news(out_dir, max_samples):
    hdr("4 / 9  ·  BBC News")
    _check_pip("datasets")
    from datasets import load_dataset

    print("  Fetching via HuggingFace datasets …")
    raw = load_dataset("SetFit/bbc-news")

    label_feat = raw["train"].features["label"]
    if hasattr(label_feat, "names") and getattr(label_feat, "names", None):
        label_map = {str(i): n for i, n in enumerate(label_feat.names)}
    elif "label_text" in raw["train"].features:
        # SetFit/bbc-news uses int64 label + string label_text (no ClassLabel)
        by_id = {}
        for split in raw:
            for ex in raw[split]:
                by_id[int(ex["label"])] = ex["label_text"]
        label_map = {str(k): by_id[k] for k in sorted(by_id)}
    else:
        seen = sorted({int(ex["label"]) for split in raw for ex in raw[split]})
        label_map = {str(i): str(i) for i in seen}

    rows = []
    idx = 0
    for split in raw:
        for ex in raw[split]:
            text = ex.get("text", ex.get("body", "")).strip()
            if text:
                rows.append((str(idx), text, int(ex["label"])))
                idx += 1

    rows = subsample(rows, max_samples)
    return save_dataset(out_dir, "bbc_news", rows, label_map,
                        source="HuggingFace: SetFit/bbc-news")


def download_dbpedia(out_dir, max_samples):
    hdr("5 / 9  ·  DBpedia-14")
    _check_pip("datasets")
    from datasets import load_dataset

    print("  Fetching via HuggingFace datasets …")
    raw = load_dataset("dbpedia_14")

    label_names = raw["train"].features["label"].names
    label_map = {str(i): n for i, n in enumerate(label_names)}

    rows = []
    idx = 0
    for split in ("train", "test"):
        for ex in raw[split]:
            text = (ex.get("title", "") + " " + ex.get("content", "")).strip()
            if text:
                rows.append((str(idx), text, int(ex["label"])))
                idx += 1

    rows = subsample(rows, max_samples)
    return save_dataset(out_dir, "dbpedia", rows, label_map,
                        source="HuggingFace: dbpedia_14")


def download_trec(out_dir, max_samples):
    hdr("6 / 9  ·  TREC-6")
    _check_pip("datasets")
    from datasets import load_dataset

    print("  Fetching via HuggingFace datasets …")
    raw = load_dataset("trec")

    # Use coarse labels (6 classes)
    label_names = raw["train"].features["coarse_label"].names
    label_map = {str(i): n for i, n in enumerate(label_names)}

    rows = []
    idx = 0
    for split in ("train", "test"):
        for ex in raw[split]:
            text = ex["text"].strip()
            if text:
                rows.append((str(idx), text, int(ex["coarse_label"])))
                idx += 1

    rows = subsample(rows, max_samples)
    return save_dataset(out_dir, "trec", rows, label_map,
                        source="HuggingFace: trec (coarse_label)")


def download_yahoo_answers(out_dir, max_samples):
    hdr("7 / 9  ·  Yahoo Answers Topics")
    _check_pip("datasets")
    from datasets import load_dataset

    print("  Fetching via HuggingFace datasets …")
    raw = load_dataset("yahoo_answers_topics")

    LABEL_MAP = {
        "0": "Society & Culture",    "1": "Science & Mathematics",
        "2": "Health",               "3": "Education & Reference",
        "4": "Computers & Internet", "5": "Sports",
        "6": "Business & Finance",   "7": "Entertainment & Music",
        "8": "Family & Relationships", "9": "Politics & Government",
    }

    rows = []
    idx = 0
    for split in ("train", "test"):
        for ex in raw[split]:
            parts = [
                ex.get("question_title", ""),
                ex.get("question_content", ""),
                ex.get("best_answer", ""),
            ]
            text = " ".join(p for p in parts if p).strip()
            if text:
                rows.append((str(idx), text, int(ex["topic"])))
                idx += 1

    rows = subsample(rows, max_samples)
    return save_dataset(out_dir, "yahoo_answers", rows, LABEL_MAP,
                        source="HuggingFace: yahoo_answers_topics")


SNIPPET_CLASSES = {
    "business": 0, "computers": 1, "culture-arts-entertainment": 2,
    "education-science": 3, "engineering": 4, "health": 5,
    "politics-society": 6, "sports": 7,
}


def _download_snippets_split(url: str) -> list:
    """Download and parse one split of the SearchSnippets dataset."""
    rows = []
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        warn(f"Could not reach {url}: {e}")
        return []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: <label>\t<text>  OR  <text>\t<label>
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        # Try to detect which part is label
        if parts[0].lower() in SNIPPET_CLASSES:
            label_str, text = parts[0].lower(), "\t".join(parts[1:])
        elif parts[-1].lower() in SNIPPET_CLASSES:
            label_str, text = parts[-1].lower(), "\t".join(parts[:-1])
        else:
            continue
        rows.append((text.strip(), SNIPPET_CLASSES[label_str]))
    return rows


def download_searchsnippets(out_dir, max_samples):
    hdr("8 / 9  ·  Search Snippets")

    # Two known public mirrors for this research dataset
    URLS = [
        "https://raw.githubusercontent.com/zhitao-wang/PLSDM/master/data/SearchSnippets/train.txt",
        "https://raw.githubusercontent.com/zhitao-wang/PLSDM/master/data/SearchSnippets/test.txt",
    ]

    print("  Downloading from GitHub mirror …")
    all_rows = []
    for url in URLS:
        all_rows.extend(_download_snippets_split(url))

    if not all_rows:
        warn("SearchSnippets could not be downloaded automatically.")
        warn("Download manually from: http://jwebpro.sourceforge.net/data-web-snippets.tar.gz")
        warn("Then place train.txt / test.txt in datasets/searchsnippets/ and re-run.")
        return None

    rows = [(str(i), text, label) for i, (text, label) in enumerate(all_rows)]
    rows = subsample(rows, max_samples)
    label_map = {str(v): k for k, v in SNIPPET_CLASSES.items()}
    return save_dataset(out_dir, "searchsnippets", rows, label_map,
                        source="jwebpro.sourceforge.net / GitHub mirror")


STACKEXCHANGE_SITES = [
    "academia", "android", "apple", "askubuntu", "bicycles",
    "biology", "chemistry", "cooking", "crypto", "electronics",
]


def download_stackexchange(out_dir, max_samples):
    hdr("9 / 9  ·  StackExchange")
    _check_pip("datasets")
    from datasets import load_dataset

    print("  Fetching 10 StackExchange communities …")
    rows = []
    idx = 0
    label_map = {}

    for label_id, site in enumerate(STACKEXCHANGE_SITES):
        label_map[str(label_id)] = site
        try:
            raw = load_dataset(
                "flax-sentence-embeddings/stackexchange_titlebody_best_and_down_voted_answer_jsonl",
                data_files=f"{site}.jsonl",
                split="train",
                trust_remote_code=True,
            )
            count = 0
            per_site = (max_samples // len(STACKEXCHANGE_SITES)
                        ) if max_samples else 5000
            for ex in raw:
                text = (ex.get("title", "") + " " + ex.get("body", "")).strip()
                if text and count < per_site:
                    rows.append((str(idx), text, label_id))
                    idx += 1
                    count += 1
            ok(f"  {site}: {count} docs")
        except Exception as e:
            warn(f"  {site}: failed ({e})")

    if not rows:
        err("No StackExchange data could be fetched.")
        return None

    rows = subsample(rows, max_samples)
    return save_dataset(out_dir, "stackexchange", rows, label_map,
                        source="HuggingFace: flax-sentence-embeddings/stackexchange_*")


DOWNLOADERS = {
    "20newsgroups":  download_20newsgroups,
    "reuters":       download_reuters,
    "agnews":        download_agnews,
    "bbc_news":      download_bbc_news,
    "dbpedia":       download_dbpedia,
    "yahoo_answers": download_yahoo_answers
}


def print_summary(out_dir: Path, results: dict):
    print(f"\n\n{'═'*55}")
    print(f"  DOWNLOAD SUMMARY")
    print(f"{'═'*55}")
    print(f"  {'Dataset':<20} {'Docs':>8}  {'Classes':>8}  {'Status'}")
    print(f"  {'─'*20} {'─'*8}  {'─'*8}  {'─'*10}")

    for name, meta in results.items():
        if meta is None:
            print(f"  {name:<20} {'—':>8}  {'—':>8}  {RED}FAILED{RESET}")
        else:
            print(
                f"  {name:<20} {meta['n_docs']:>8,}  "
                f"{meta['n_classes']:>8}  {GREEN}OK{RESET}"
            )

    print(f"{'═'*55}")
    print(f"  Output directory: {out_dir.resolve()}")
    print(f"{'═'*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download all document clustering benchmark datasets."
    )
    parser.add_argument(
        "--datasets", nargs="*", default=list(DOWNLOADERS.keys()),
        help=f"Which datasets to download. Choose from: {list(DOWNLOADERS.keys())}",
    )
    parser.add_argument(
        "--output_dir", default="datasets",
        help="Root folder to save all datasets (default: ./datasets)",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Cap each dataset at N samples (useful for quick testing)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate requested dataset names
    invalid = [d for d in args.datasets if d not in DOWNLOADERS]
    if invalid:
        err(f"Unknown datasets: {invalid}")
        err(f"Valid options: {list(DOWNLOADERS.keys())}")
        sys.exit(1)

    print(f"\n{BOLD}Document Clustering Dataset Downloader{RESET}")
    print(f"Saving to : {out_dir.resolve()}")
    print(f"Datasets  : {args.datasets}")
    if args.max_samples:
        print(f"Cap       : {args.max_samples:,} samples per dataset")

    results = {}
    for name in args.datasets:
        try:
            meta = DOWNLOADERS[name](out_dir, args.max_samples)
            results[name] = meta
        except Exception as e:
            err(f"{name} failed with exception: {e}")
            results[name] = None

    print_summary(out_dir, results)

    # Save master index
    index_path = out_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(
            {k: v for k, v in results.items() if v is not None},
            f, indent=2,
        )
    print(f"  Master index → {index_path}\n")


if __name__ == "__main__":
    main()
