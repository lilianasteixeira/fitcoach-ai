"""
Retrieval functions: vector search (Qdrant), text search (BM25), and
hybrid search (fusion of both).

This enables comparing different approaches during retrieval evaluation
(evaluation/retrieval_eval.py) and choosing the best one for the final
application.
"""

import os
import pickle

import requests
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "fitcoach_knowledge")
BM25_INDEX_PATH = os.getenv("BM25_INDEX_PATH", "data/bm25_index.pkl")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

_qdrant_client = None
_bm25_corpus = None
_bm25_index = None


def _get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=QDRANT_URL)
    return _qdrant_client


def _get_bm25():
    global _bm25_corpus, _bm25_index
    if _bm25_index is None:
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_corpus = pickle.load(f)
        tokenized = [doc["text"].lower().split() for doc in _bm25_corpus]
        _bm25_index = BM25Okapi(tokenized)
    return _bm25_corpus, _bm25_index


def embed_query(text: str) -> list:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def vector_search(query: str, top_k: int = 5):
    client = _get_qdrant()
    vector = embed_query(query)
    hits = client.query_points(collection_name=COLLECTION_NAME, query=vector, limit=top_k).points
    return [
        {
            "doc_id": h.payload["doc_id"],
            "question": h.payload["question"],
            "answer": h.payload["answer"],
            "topic": h.payload["topic"],
            "score": h.score,
        }
        for h in hits
    ]


def text_search(query: str, top_k: int = 5):
    corpus, bm25 = _get_bm25()
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(corpus, scores), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "doc_id": doc["doc_id"],
            "question": doc["question"],
            "answer": doc["answer"],
            "topic": doc["topic"],
            "score": float(score),
        }
        for doc, score in ranked
    ]


def hybrid_search(query: str, top_k: int = 5, alpha: float = 0.5):
    """Combines vector_search and text_search via Reciprocal Rank Fusion (RRF).

    alpha is currently unused, kept for potential future weighting.
    """
    v_results = vector_search(query, top_k=20)
    t_results = text_search(query, top_k=20)

    rrf_scores = {}
    doc_lookup = {}
    k = 60  # standard RRF constant
    for rank, doc in enumerate(v_results):
        rrf_scores[doc["doc_id"]] = rrf_scores.get(doc["doc_id"], 0) + 1 / (k + rank + 1)
        doc_lookup[doc["doc_id"]] = doc
    for rank, doc in enumerate(t_results):
        rrf_scores[doc["doc_id"]] = rrf_scores.get(doc["doc_id"], 0) + 1 / (k + rank + 1)
        doc_lookup.setdefault(doc["doc_id"], doc)

    ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {**doc_lookup[doc_id], "score": score} for doc_id, score in ranked_ids
    ]


SEARCH_STRATEGIES = {
    "vector": vector_search,
    "text": text_search,
    "hybrid": hybrid_search,
}
