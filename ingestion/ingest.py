"""
Automated ingestion pipeline:
1. Reads data/dataset.json
2. Generates embeddings for each document via Ollama (nomic-embed-text)
3. Indexes them into Qdrant (vector collection)
4. Also saves a local corpus (pickle) used for BM25 (text search)

Usage:
    python ingestion/ingest.py --dataset data/dataset.json
"""

import argparse
import json
import os
import pickle

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "fitcoach_knowledge")
EMBED_DIM = 768  # nomic-embed-text dimension


def embed_text(text: str) -> list:
    resp = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset.json")
    parser.add_argument("--bm25-out", default="data/bm25_index.pkl")
    args = parser.parse_args()

    with open(args.dataset, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL)

    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    points = []
    corpus_for_bm25 = []
    for i, doc in enumerate(dataset):
        # The indexed text combines question + answer for better recall
        text_to_embed = f"{doc['question']}\n{doc['answer']}"
        print(f"[{i + 1}/{len(dataset)}] generating embedding: {doc['question'][:60]}")
        vector = embed_text(text_to_embed)

        points.append(
            PointStruct(
                id=i,
                vector=vector,
                payload={
                    "doc_id": doc["id"],
                    "topic": doc["topic"],
                    "subtopic": doc["subtopic"],
                    "question": doc["question"],
                    "answer": doc["answer"],
                },
            )
        )
        corpus_for_bm25.append(
            {
                "doc_id": doc["id"],
                "text": text_to_embed,
                "question": doc["question"],
                "answer": doc["answer"],
                "topic": doc["topic"],
            }
        )

    print("Uploading points to Qdrant...")
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Indexed {len(points)} documents into '{COLLECTION_NAME}'.")

    os.makedirs(os.path.dirname(args.bm25_out), exist_ok=True)
    with open(args.bm25_out, "wb") as f:
        pickle.dump(corpus_for_bm25, f)
    print(f"BM25 corpus saved to {args.bm25_out}")


if __name__ == "__main__":
    main()
