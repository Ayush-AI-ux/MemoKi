# MemoKi: Episodic Memory for LLMs

## Setup
    python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
    pytest

## Layout
core/ engine (types, counters, retriever interface, later: cues, tree)
baselines/ brute force, FAISS/HNSW, tree-without-cues
eval/ metrics, experiment runner, plots
server/ FastAPI    extension/ Chrome MV3    dashboard/ React+Vite
configs/global.yaml  the single Gap-4 protocol (same data, queries, model, k, seeds)
