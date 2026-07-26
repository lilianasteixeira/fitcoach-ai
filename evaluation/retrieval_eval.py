"""
Evaluates and compares three retrieval strategies (vector, text/BM25, and
hybrid) using hit-rate and MRR (Mean Reciprocal Rank), against the ground
truth generated in data/generate_ground_truth.py.

Usage:
    python evaluation/retrieval_eval.py --ground-truth data/ground_truth.json --top-k 5
"""

import argparse
import json

from rag.search import SEARCH_STRATEGIES


def evaluate_strategy(search_fn, ground_truth: list, top_k: int) -> dict:
    hits = 0
    reciprocal_ranks = []

    for item in ground_truth:
        results = search_fn(item["question"], top_k=top_k)
        result_ids = [r["doc_id"] for r in results]

        if item["doc_id"] in result_ids:
            hits += 1
            rank = result_ids.index(item["doc_id"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    n = len(ground_truth)
    return {
        "hit_rate": round(hits / n, 4),
        "mrr": round(sum(reciprocal_ranks) / n, 4),
        "n_queries": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/ground_truth.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--out", default="evaluation/retrieval_results.json")
    args = parser.parse_args()

    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    results = {}
    for name, search_fn in SEARCH_STRATEGIES.items():
        print(f"Evaluating strategy: {name}...")
        results[name] = evaluate_strategy(search_fn, ground_truth, args.top_k)
        print(f"  hit_rate={results[name]['hit_rate']}  mrr={results[name]['mrr']}")

    best = max(results, key=lambda k: results[k]["mrr"])
    results["_best_strategy"] = best
    print(f"\nBest strategy by MRR: {best}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
