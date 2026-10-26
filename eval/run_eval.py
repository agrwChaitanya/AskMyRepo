"""Retrieval evaluation for AskMyRepo.

Run from the project root:   python eval/run_eval.py

For every config it (re)uses a cached index per repo, retrieves top-k chunks for each
question in questions.json, and checks whether an expected file shows up.

Metrics
  hit@k : share of questions where at least one expected file is in the retrieved chunks
  MRR   : mean of 1/rank of the first expected file (rank counted over unique files)
"""
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from indexer import index_repository, setup_llamaindex_settings  # noqa: E402
from retrieval import build_retriever, load_index_and_nodes, retrieve  # noqa: E402

ROOT = Path(__file__).resolve().parent
QUESTIONS_FILE = ROOT / "questions.json"
INDEX_DIR = ROOT / "indexes"
TOP_K = 5

CONFIGS = [
    {"name": "1. baseline (default chunks, vector)", "chunking": "default", "mode": "vector"},
    {"name": "2. + code-aware chunks", "chunking": "code", "mode": "vector"},
    {"name": "3. + hybrid (BM25 + vector)", "chunking": "code", "mode": "hybrid"},
    {"name": "4. + cross-encoder rerank", "chunking": "code", "mode": "hybrid_rerank"},
]


def slug(repo_url: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", repo_url.replace("https://github.com/", "")).strip("_")


def ensure_index(repo_url: str, chunking: str) -> str:
    path = INDEX_DIR / f"{slug(repo_url)}__{chunking}"
    if not (path / "docstore.json").exists():
        print(f"Indexing {repo_url} (chunking={chunking}) ...")
        index_repository(repo_url, persist_dir=str(path), chunking=chunking)
    return str(path)


def ranked_files(nodes) -> list[str]:
    files = []
    for n in nodes:
        fp = n.node.metadata.get("file_path")
        if fp and fp not in files:
            files.append(fp)
    return files


def score(expected: list[str], retrieved: list[str]) -> tuple[int, float]:
    ranks = [retrieved.index(f) + 1 for f in expected if f in retrieved]
    if not ranks:
        return 0, 0.0
    return 1, 1.0 / min(ranks)


def main():
    setup_llamaindex_settings()
    questions = json.loads(QUESTIONS_FILE.read_text())
    summary, per_question = [], []

    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        built = {}  # one retriever per repo for this config
        hits = mrr = 0.0

        for q in questions:
            repo = q["repo_url"]
            if repo not in built:
                index, nodes = load_index_and_nodes(ensure_index(repo, cfg["chunking"]))
                built[repo] = build_retriever(index, nodes, mode=cfg["mode"], top_k=TOP_K)
            retriever, post = built[repo]

            files = ranked_files(retrieve(q["question"], retriever, post))
            hit, rr = score(q["expected_files"], files)
            hits += hit
            mrr += rr
            per_question.append({
                "config": cfg["name"], "repo": repo, "type": q.get("type", ""),
                "question": q["question"], "hit": hit, "rr": round(rr, 3),
                "expected": "; ".join(q["expected_files"]), "retrieved": "; ".join(files),
            })
            time.sleep(0.2)  # be gentle with free-tier rate limits

        n = len(questions)
        summary.append((cfg["name"], hits / n, mrr / n))

    print(f"\n{'Config':42} {'hit@' + str(TOP_K):>8} {'MRR':>8}")
    print("-" * 60)
    for name, h, m in summary:
        print(f"{name:42} {h:8.1%} {m:8.3f}")

    out = ROOT / "results_per_question.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question[0].keys()))
        writer.writeheader()
        writer.writerows(per_question)
    print(f"\nPer-question results (use these to study the misses): {out}")


if __name__ == "__main__":
    main()