"""
Generates a synthetic fitness & nutrition knowledge dataset using a local
LLM via Ollama. Each entry is a (question, answer) pair tied to a
topic/subtopic, in the same spirit as the FAQ used in the course, but for
a different domain.

Usage:
    python data/generate_dataset.py --out data/dataset.json --per-subtopic 8

Requires Ollama to be running (see docker-compose.yml) and the model set in
OLLAMA_MODEL to already be pulled (see Makefile: make pull-models).
"""

import argparse
import json
import os
import re
import time
import uuid

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))

# Domain structure: topic -> subtopics
TOPICS = {
    "strength_training": [
        "progressive overload for beginners",
        "squat and deadlift technique",
        "strength training for hypertrophy vs. max strength",
        "weekly training frequency and volume",
    ],
    "cardio": [
        "high-intensity interval training (HIIT)",
        "fasted cardio",
        "heart rate training zones",
        "running for beginners",
    ],
    "mobility_and_flexibility": [
        "static vs. dynamic stretching",
        "joint mobility for lifters",
        "pre-workout warm-up routine",
    ],
    "recovery_and_sleep": [
        "importance of sleep for muscle recovery",
        "active recovery days",
        "signs of overtraining",
    ],
    "macronutrients": [
        "recommended daily protein intake",
        "carbs before and after training",
        "healthy fats in the diet",
    ],
    "hydration": [
        "water needs during exercise",
        "electrolytes and athletic performance",
    ],
    "supplements": [
        "creatine: benefits and safety",
        "whey protein vs. plant-based protein",
        "caffeine as an ergogenic aid",
    ],
    "weight_management": [
        "healthy caloric deficit",
        "weight loss vs. fat loss",
        "long-term weight maintenance",
    ],
    "injury_prevention": [
        "common running injuries",
        "training-related lower back pain",
        "when to stop a workout due to pain",
    ],
    "special_populations": [
        "exercise during pregnancy",
        "strength training for older adults",
        "fitness for overweight beginners",
    ],
}

PROMPT_TEMPLATE = """You are an expert in exercise science and nutrition.
Generate {n} question-and-answer pairs in English about the subtopic
"{subtopic}" (within the general topic "{topic}").

Rules:
- Questions should sound like real questions from users of a fitness app.
- Answers should be 3 to 6 sentences long, accurate, based on general
  evidence, and free of personalized medical advice (add a disclaimer when relevant).
- Do not repeat similar questions.
- Return ONLY a valid JSON array, with no extra text or Markdown, in this format:
[
  {{"question": "...", "answer": "..."}},
  ...
]
"""


def call_ollama(prompt: str) -> str:
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"temperature": 0.7}},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def extract_json_array(text: str):
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in response:\n{text[:300]}")
    return json.loads(match.group(0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/dataset.json")
    parser.add_argument("--per-subtopic", type=int, default=8)
    args = parser.parse_args()

    records = []
    for topic, subtopics in TOPICS.items():
        for subtopic in subtopics:
            prompt = PROMPT_TEMPLATE.format(n=args.per_subtopic, subtopic=subtopic, topic=topic)
            print(f"[generating] {topic} / {subtopic}")
            for attempt in range(3):
                try:
                    raw = call_ollama(prompt)
                    qa_pairs = extract_json_array(raw)
                    break
                except Exception as e:
                    print(f"  attempt {attempt + 1} failed: {e}")
                    time.sleep(2)
            else:
                print("  skipping subtopic after repeated failures")
                continue

            for pair in qa_pairs:
                if "question" not in pair or "answer" not in pair:
                    continue
                records.append(
                    {
                        "id": str(uuid.uuid4()),
                        "topic": topic,
                        "subtopic": subtopic,
                        "question": pair["question"].strip(),
                        "answer": pair["answer"].strip(),
                    }
                )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()
