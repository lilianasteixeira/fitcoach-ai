"""
Evaluates the quality of the LLM's final answers by comparing two prompt
templates ("concise" vs "detailed") using an LLM-as-a-judge
(rag/llm.py:judge_answer).

Samples a subset of the ground truth to keep runtime reasonable.

Usage:
    python evaluation/llm_eval.py --ground-truth data/ground_truth.json --sample-size 30
"""

import argparse
import json
import random

from rag.llm import judge_answer
from rag.pipeline import answer_question
from rag.prompts import PROMPT_TEMPLATES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth", default="data/ground_truth.json")
    parser.add_argument("--sample-size", type=int, default=30)
    parser.add_argument("--search-strategy", default="hybrid")
    parser.add_argument("--out", default="evaluation/llm_eval_results.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.ground_truth, encoding="utf-8") as f:
        ground_truth = json.load(f)

    random.seed(args.seed)
    sample = random.sample(ground_truth, min(args.sample_size, len(ground_truth)))

    results = {name: {"relevance": [], "faithfulness": []} for name in PROMPT_TEMPLATES}

    for i, item in enumerate(sample):
        question = item["question"]
        print(f"[{i + 1}/{len(sample)}] {question[:60]}")
        for template_name in PROMPT_TEMPLATES:
            rag_result = answer_question(
                question, search_strategy=args.search_strategy, prompt_template=template_name
            )
            context = "\n".join(s["question"] for s in rag_result["sources"])
            judgment = judge_answer(question, context, rag_result["answer"])
            if judgment["relevance"] is not None:
                results[template_name]["relevance"].append(judgment["relevance"])
                results[template_name]["faithfulness"].append(judgment["faithfulness"])

    summary = {}
    for template_name, scores in results.items():
        n = len(scores["relevance"]) or 1
        summary[template_name] = {
            "avg_relevance": round(sum(scores["relevance"]) / n, 3),
            "avg_faithfulness": round(sum(scores["faithfulness"]) / n, 3),
            "n_samples": len(scores["relevance"]),
        }

    best = max(summary, key=lambda k: summary[k]["avg_relevance"] + summary[k]["avg_faithfulness"])
    summary["_best_template"] = best
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
