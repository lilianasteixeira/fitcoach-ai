.PHONY: up up-with-ollama down venv pull-models pull-models-native generate-data generate-ground-truth ingest eval-retrieval eval-llm logs

# Uses the local virtual environment's Python by default (see `make venv`).
# Override with `make generate-data PYTHON=python3` to use your system Python instead.
PYTHON ?= .venv/bin/python3

# Recommended path: run Ollama natively on the host (required for Metal GPU
# acceleration on macOS — see README "Troubleshooting: Ollama timeouts").
# Starts qdrant, postgres, grafana, and the app; the app talks to your
# host's Ollama at http://host.docker.internal:11434.
up:
	docker compose up -d --build

# Alternative path: also run Ollama inside Docker (slower on macOS, no GPU
# access, but convenient on Linux or if you don't want to install Ollama
# natively). Set OLLAMA_URL=http://ollama:11434 in your .env before using this.
up-with-ollama:
	docker compose --profile with-ollama up -d --build

# Creates a local virtual environment and installs dependencies into it.
# Run this once before generate-data / generate-ground-truth / ingest / eval-*.
venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

down:
	docker compose --profile with-ollama down

logs:
	docker compose logs -f app

# Pulls the required models into your NATIVE Ollama installation (recommended,
# see README). Requires Ollama installed on the host: https://ollama.com/download
pull-models-native:
	ollama pull llama3.2
	ollama pull nomic-embed-text

# Pulls the required models into the CONTAINERIZED Ollama (only if you used
# `make up-with-ollama`).
pull-models:
	docker exec fitcoach-ollama ollama pull llama3.2
	docker exec fitcoach-ollama ollama pull nomic-embed-text

# Generates the synthetic dataset (requires Ollama running, natively or containerized)
generate-data:
	OLLAMA_URL=http://localhost:11434 $(PYTHON) data/generate_dataset.py --out data/dataset.json --per-subtopic 8

generate-ground-truth:
	OLLAMA_URL=http://localhost:11434 $(PYTHON) data/generate_ground_truth.py --dataset data/dataset.json --out data/ground_truth.json

# Automated ingestion of the dataset into Qdrant + BM25 index
ingest:
	OLLAMA_URL=http://localhost:11434 QDRANT_URL=http://localhost:6333 $(PYTHON) ingestion/ingest.py --dataset data/dataset.json

eval-retrieval:
	OLLAMA_URL=http://localhost:11434 QDRANT_URL=http://localhost:6333 PYTHONPATH=. $(PYTHON) evaluation/retrieval_eval.py

eval-llm:
	OLLAMA_URL=http://localhost:11434 QDRANT_URL=http://localhost:6333 PYTHONPATH=. $(PYTHON) evaluation/llm_eval.py
