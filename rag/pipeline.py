"""
Full RAG pipeline: takes a question, retrieves context, builds the prompt,
and calls the LLM. Used both by the Streamlit app and by the evaluation
scripts.
"""

import os

from rag import prompts
from rag.llm import generate
from rag.search import SEARCH_STRATEGIES

DEFAULT_SEARCH_STRATEGY = os.getenv("SEARCH_STRATEGY", "hybrid")
DEFAULT_PROMPT_TEMPLATE = os.getenv("PROMPT_TEMPLATE", "detailed")
TOP_K = int(os.getenv("RAG_TOP_K", "5"))


def answer_question(
    question: str,
    search_strategy: str = DEFAULT_SEARCH_STRATEGY,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    top_k: int = TOP_K,
) -> dict:
    search_fn = SEARCH_STRATEGIES[search_strategy]
    documents = search_fn(question, top_k=top_k)

    prompt = prompts.build_prompt(question, documents, template_name=prompt_template)
    result = generate(prompt)

    return {
        "question": question,
        "answer": result["answer"],
        "latency_seconds": result["latency_seconds"],
        "model": result["model"],
        "search_strategy": search_strategy,
        "prompt_template": prompt_template,
        "sources": [
            {"topic": d["topic"], "question": d["question"], "score": d["score"]}
            for d in documents
        ],
    }
