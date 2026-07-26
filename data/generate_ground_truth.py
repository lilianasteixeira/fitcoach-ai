"""
Generates a "ground truth" set for retrieval evaluation: for each document
in the dataset, asks the LLM to paraphrase the original question in 3
different ways, simulating how a real user might ask the same thing. This
allows computing hit-rate and MRR by knowing which document is correct for
each generated question.

Usage:
    python data/generate_ground_truth.py --dataset data/dataset.json \
        --out data/ground_truth.json --n-variants 3
"""

import argparse
import json
import os
import re
import time

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

PROMPT_TEMPLATE = """Given the following original question about fitness/nutrition,
generate {n} alternative ways a user might ask the SAME question, using
different wording — some more casual, some more technical.

Original question: "{question}"

Return ONLY a JSON array of strings, with no extra text:
["variant 1", "variant 2", ...]
"""


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.8}},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def extract_json_array(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in response")
    return json.loads(match.group(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--out", default="data/ground_truth.json")
    parser.add_argument("--n-variants", type=int, default=3)
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    ground_truth = []
    for i, doc in enumerate(dataset):
        prompt = PROMPT_TEMPLATE.format(n=args.n_variants, question=doc["question"])
        print(f"[{i + 1}/{len(dataset)}] {doc['question'][:60]}")
        try:
            raw = call_ollama(prompt)
            variants = extract_json_array(raw)
        except Exception as e:
            print(f"  failed, falling back to original question: {e}")
            variants = [doc["question"]]

        for variant in variants:
            ground_truth.append({"doc_id": doc["id"], "question": variant})
        time.sleep(0.2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(ground_truth)} evaluation questions to {args.out}")


if __name__ == "__main__":
    main()
