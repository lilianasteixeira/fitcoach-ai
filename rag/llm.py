"""
Simple wrapper for calling the chat LLM via Ollama, plus an "LLM-as-a-judge"
helper function used during evaluation (evaluation/llm_eval.py).
"""

import json
import os
import re
import time

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "600"))


def generate(prompt: str, temperature: float = 0.3) -> dict:
    start = time.time()
    resp = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    latency = time.time() - start
    return {
        "answer": data["response"].strip(),
        "latency_seconds": round(latency, 3),
        "model": OLLAMA_MODEL,
    }


JUDGE_PROMPT = """You are an impartial evaluator of answers from a fitness/nutrition assistant.

USER QUESTION: {question}

CONTEXT PROVIDED TO THE ASSISTANT: {context}

ASSISTANT'S ANSWER: {answer}

Rate the answer on two dimensions, each from 1 (poor) to 5 (excellent):
- relevance: does the answer directly address the question?
- faithfulness: does the answer rely only on information present in the context, without inventing facts?

Return ONLY a JSON object, with no extra text:
{{"relevance": <1-5>, "faithfulness": <1-5>, "justification": "<1 sentence>"}}
"""


def judge_answer(question: str, context: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT.format(question=question, context=context, answer=answer)
    result = generate(prompt, temperature=0.0)
    match = re.search(r"\{.*\}", result["answer"], re.DOTALL)
    if not match:
        return {"relevance": None, "faithfulness": None, "justification": "parse_error"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"relevance": None, "faithfulness": None, "justification": "parse_error"}
