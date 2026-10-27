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


def score(expected: list[str], retrieved: list[str], k: int) -> dict:
    ret_k = retrieved[:k]
    hit_1 = 1.0 if (ret_k and ret_k[0] in expected) else 0.0
    hit_k = 1.0 if any(f in expected for f in ret_k) else 0.0
    ranks = [retrieved.index(f) + 1 for f in expected if f in retrieved]
    rr = 1.0 / min(ranks) if ranks else 0.0
    precision = (len([f for f in ret_k if f in expected]) / len(ret_k)) if ret_k else 0.0
    recall = (len([f for f in expected if f in ret_k]) / len(expected)) if expected else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "hit_1": hit_1,
        "hit_k": hit_k,
        "rr": rr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    setup_llamaindex_settings()
    questions = json.loads(QUESTIONS_FILE.read_text())
    summary, per_question = [], []

    for cfg in CONFIGS:
        print(f"\n=== {cfg['name']} ===")
        built = {}  # one retriever per repo for this config
        h1 = hk = mrr = prec = rec = f1 = 0.0

        for q in questions:
            repo = q["repo_url"]
            if repo not in built:
                persist_path = ensure_index(repo, cfg["chunking"])
                index, nodes = load_index_and_nodes(persist_path)
                built[repo] = build_retriever(index, nodes, mode=cfg["mode"], top_k=TOP_K, persist_dir=persist_path)
            retriever, post = built[repo]

            files = ranked_files(retrieve(q["question"], retriever, post))
            s = score(q["expected_files"], files, TOP_K)
            h1 += s["hit_1"]
            hk += s["hit_k"]
            mrr += s["rr"]
            prec += s["precision"]
            rec += s["recall"]
            f1 += s["f1"]

            per_question.append({
                "config": cfg["name"], "repo": repo, "type": q.get("type", ""),
                "question": q["question"], "hit_1": int(s["hit_1"]), "hit_k": int(s["hit_k"]),
                "rr": round(s["rr"], 3), "precision": round(s["precision"], 3),
                "recall": round(s["recall"], 3), "f1": round(s["f1"], 3),
                "expected": "; ".join(q["expected_files"]), "retrieved": "; ".join(files),
            })
            time.sleep(0.2)  # be gentle with free-tier rate limits

        n = len(questions)
        summary.append((cfg["name"], h1 / n, hk / n, mrr / n, prec / n, rec / n, f1 / n))

    print(f"\n{'Config':38} {'Hit@1':>7} {'Hit@' + str(TOP_K):>7} {'MRR':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print("-" * 88)
    for name, h1_val, hk_val, m, p, r, f in summary:
        print(f"{name:38} {h1_val:7.1%} {hk_val:7.1%} {m:7.3f} {p:7.1%} {r:7.1%} {f:7.3f}")

    out = ROOT / "results_per_question.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_question[0].keys()))
        writer.writeheader()
        writer.writerows(per_question)
    print(f"\nPer-question results (use these to study the misses): {out}")


if __name__ == "__main__":
    main()