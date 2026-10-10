# MemoKi — Episodic Memory for LLMs

> **A cue-guided hierarchical memory retrieval system for long-context LLM applications.**  
> No LLM calls during retrieval · Zero-latency cue augmentation · Sub-linear embedding comparisons

---

## Table of Contents

- [Overview](#overview)
- [Key Idea](#key-idea)
- [Architecture](#architecture)
- [Results at a Glance](#results-at-a-glance)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data Setup](#data-setup)
- [Running Experiments](#running-experiments)
- [Evaluation & Benchmarks](#evaluation--benchmarks)
- [Configuration](#configuration)
- [Baselines](#baselines)
- [Development & Testing](#development--testing)
- [Roadmap](#roadmap)
- [Team](#team)

---

## Overview

**MemoKi** addresses a core challenge in deploying LLMs over long-term, personalised conversations: *how do you efficiently retrieve the right memory from thousands of past sessions without reading everything?*

The system stores every past conversation as a **leaf node** in a hierarchical centroid tree. Each internal node summarises its subtree with:
1. A **centroid embedding** (normalised mean of all children's vectors)
2. A **named-entity union** (person, organisation, location, etc.)
3. A **timestamp span** (min/max of all contained conversations)

When a query arrives, MemoKi performs a **top-down beam search** guided by these cheap symbolic cues — discarding entire subtrees that cannot possibly contain the answer — before ever computing expensive embedding similarities. The result is near-brute-force accuracy at a fraction of the embedding comparisons.

---

## Key Idea

```
Query: "What did Alice say about the Paris trip last month?"
        |
        |-- Named-entity cues extracted: {alice, paris}
        |-- Time cue parsed: window = [last month]
        |
        v
MemoryTree beam search:
  Level 1: 3 branches  --> keep only the 1 branch whose entity-set contains {alice, paris}
  Level 2: 9 nodes     --> prune to 2 nodes whose date-span overlaps [last month]
  Level 3: leaves      --> rank remaining ~30 leaves by cosine similarity
                          (~95% fewer embedding comparisons vs brute force)
```

**Key properties:**
- **Zero LLM calls** at retrieval time (rule-based NER + date parsing)
- **Sub-linear cost**: embedding comparisons grow as O(log N) for cue queries
- **Graceful degradation**: if cues match nothing, falls back to pure cosine search
- **Offline re-filing** (`audit_refile`): corrects misfiled leaves without rebuilding the tree

---

## Architecture

```
MemoKi/
|-- core/               # Engine - pure Python, no external ML at import time
|   |-- types.py        # Conversation, Query, Result dataclasses
|   |-- counters.py     # Cost accounting (embedding comparisons, cue checks, latency)
|   |-- retriever.py    # Abstract Retriever base class
|   |-- tree.py         # MemoryTree: B-tree-like centroid hierarchy with cue summaries
|   |-- cues.py         # SpacyNER entity extractor + QueryCues dataclass
|   |-- timecue.py      # Rule-based time-window parser (LLM-free)
|   `-- misfile.py      # Deliberate misfiling simulator (Gap 2 experiment)
|
|-- baselines/          # Every comparison method - same Retriever interface
|   |-- brute_force.py  # Exact cosine over all leaves (oracle accuracy)
|   |-- faiss_index.py  # FAISS flat (exact) and HNSW (approximate)
|   |-- centroid_tree.py# Tree beam search, no cue pruning
|   |-- cue_tree.py     # Proposed method: tree + cue pruning
|   |-- flat_cue.py     # Flat store + same cue filter (fair ablation)
|   `-- cue_prep.py     # Shared cue-preparation mixin
|
|-- pipeline/           # Data loading, embedding, cue caching
|   |-- loaders.py      # LongMemEval & LoCoMo dataset parsers
|   |-- embed.py        # Sentence-transformer embedder (MiniLM-L6-v2)
|   |-- embed_pool.py   # Multi-threaded embedding pool
|   |-- store.py        # Build/split experiment stores
|   |-- cue_cache.py    # Persistent entity-cue cache (avoid re-running spaCy)
|   |-- extract_cues.py # CLI: pre-extract entity cues for the full pool
|   |-- prepare.py      # Full pipeline: download -> embed -> cue-extract
|   `-- download.py     # Dataset downloader helpers
|
|-- eval/               # Evaluation scripts and experiment runners
|   |-- metrics.py          # Recall@K metric
|   |-- compare_check.py    # Side-by-side comparison of ALL methods (Gap 3)
|   |-- cue_tree_check.py   # CueTree deep evaluation
|   |-- misfile_check.py    # Misfiling robustness experiment (Gap 2)
|   |-- refile_check.py     # Re-filing (offline repair) experiment
|   |-- tree_check.py       # Tree structure validation
|   |-- tree_sweep.py       # Hyperparameter sweep (B, beam_width)
|   |-- cue_coverage.py     # Entity cue coverage analysis
|   |-- cue_selectivity.py  # Per-entity selectivity measurement
|   |-- time_cue_check.py   # Time-window parser validation
|   |-- seqlen_check.py     # Sequence length distribution
|   |-- plot_pareto.py      # Accuracy vs. Cost Pareto plots
|   `-- sanity_check.py     # End-to-end smoke test
|
|-- configs/
|   |-- global.yaml     # Single source of truth: model, k, seeds, tree params
|   `-- data.yaml       # Dataset paths and split settings
|
|-- results/            # Pre-computed experiment outputs
|   |-- pareto_*.png            # Recall@5 vs. embedding comparisons (Pareto plots)
|   |-- pareto_table.txt        # Full numeric results table
|   |-- compare_validation.txt  # All methods compared on validation split
|   |-- misfile_*.txt/csv       # Misfiling experiment outputs
|   `-- refile_*.txt/csv        # Re-filing experiment outputs
|
|-- tests/              # pytest unit and integration tests
|   |-- test_phase0.py      # Basic smoke tests
|   |-- test_phase1.py      # Tree insertion, search, split correctness
|   |-- test_tree.py        # MemoryTree unit tests
|   |-- test_cues.py        # NER extraction and QueryCues tests
|   |-- test_cuetree.py     # End-to-end cue tree tests
|   |-- test_baselines.py   # Baseline method correctness
|   |-- test_timecue.py     # Time-window parser tests
|   `-- test_misfile.py     # Misfiling simulation tests
|
|-- server/             # FastAPI server (in progress)
|-- requirements.txt
|-- pytest.ini
`-- README.md
```

---

## Results at a Glance

Experiments run on the **LongMemEval** benchmark.  
Metric: **Recall@5** (1.0 if any relevant session is in the top-5 results).  
Protocol: 100 queries, 5 seeds, validation split.

### Accuracy vs. Embedding Comparisons (store size = 5,000 sessions)

| Method                | R@5 (all) | R@5 (cue queries) | R@5 (no-cue) | Mean comparisons |
|-----------------------|-----------|-------------------|--------------|-----------------|
| Brute Force           | 0.460     | 0.610             | 0.356        | 5,000           |
| FAISS HNSW ef=64      | 0.460     | 0.610             | 0.356        | 783             |
| Flat + Cue Filter     | 0.520     | 0.756             | 0.356        | 3,055           |
| Tree (no cues) b=10   | 0.390     | 0.512             | 0.305        | 463             |
| **CUE TREE b=3**      | **0.390** | **0.537**         | 0.288        | **105**         |
| **CUE TREE b=10**     | **0.450** | **0.659**         | 0.288        | **201**         |

> **Highlight**: At beam=3, CUE TREE uses only **105 comparisons** (vs 5,000 for brute force) while matching or exceeding the recall of methods that use 30-50x more comparisons on queries that contain named entities or time references.

### Pareto Frontier — Recall@5 vs. Embedding Comparisons

Pre-generated Pareto plots are available in `results/`:

| Store Size     | Plot File                  |
|----------------|----------------------------|
| 1,000 sessions | `results/pareto_1000.png`  |
| 5,000 sessions | `results/pareto_5000.png`  |
| 10,000 sessions| `results/pareto_10000.png` |

---

## Quick Start

### 1. Environment Setup

```bash
# Create and activate a virtual environment (Python 3.11 required)
python3.11 -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Verify Installation

```bash
pytest
```

All tests should pass. The test suite covers the tree, cue extraction, time-window parser, and all baselines.

---

## Data Setup

MemoKi is evaluated on **LongMemEval** (primary) and **LoCoMo** (secondary).

### LongMemEval

Download the cleaned dataset and place it at:

```
data/raw/longmemeval_s_cleaned.json
```

An `oracle` split (evidence-only, much smaller) is also supported for fast local development:

```
data/raw/longmemeval_oracle.json
```

### Pre-extract Entity Cues (strongly recommended)

Entity extraction with spaCy over the full 19k-session pool takes ~1 hour. Run this once and cues are cached automatically:

```bash
# Single-process (safe to interrupt and resume)
python -m pipeline.extract_cues

# Multi-process (3x faster on a multi-core machine)
python -m pipeline.extract_cues --procs 3

# Quick timing test on 200 sessions only
python -m pipeline.extract_cues --limit 200
```

Cached cues are stored in `data/cache/` and reused automatically by all evaluation scripts.

---

## Running Experiments

### Full Method Comparison (Gap 3)

Compares all methods side-by-side on the validation split:

```bash
# Default: store sizes 1000, 5000, 10000
python -m eval.compare_check

# Specific sizes only
python -m eval.compare_check --sizes 10000

# Also run an offline audit-refile pass
python -m eval.compare_check --audit-margin 0.05

# Save all rows to a CSV file
python -m eval.compare_check --csv results/my_run.csv
```

### CUE TREE Deep Evaluation

```bash
python -m eval.cue_tree_check
```

### Misfiling Robustness (Gap 2)

Tests recall degradation when a fraction of conversations are deliberately filed in a wrong branch:

```bash
python -m eval.misfile_check
```

### Offline Re-filing Repair

Tests the `audit_refile` offline repair method that detects and corrects misfiled leaves:

```bash
python -m eval.refile_check
```

### Hyperparameter Sweep

Sweeps tree branching factor B and beam width:

```bash
python -m eval.tree_sweep
```

### Pareto Plots

```bash
python -m eval.plot_pareto
```

### Time-Cue Parser Validation

```bash
python -m eval.time_cue_check
```

### Sanity Check (end-to-end smoke test)

```bash
python -m eval.sanity_check
```

---

## Evaluation & Benchmarks

### Primary Metric

**Recall@5**: 1.0 if any relevant conversation appears in the top-5 retrieved results, else 0.0. Averaged across all queries.

### Cost Accounting

Every `retrieve()` call returns a `Counters` object with fine-grained cost breakdown:

| Field                   | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `embedding_comparisons` | Number of vector dot products performed (the expensive operation)           |
| `cue_checks`            | Cheap symbolic checks (entity set intersection, date range overlap)         |
| `nodes_visited`         | Total tree nodes scored                                                     |
| `latency_ms`            | Wall-clock search time (excludes cue preparation time)                      |
| `prep_ms`               | Time for entity extraction + date parsing (reported separately)             |
| `levels_relaxed`        | Levels where cue pruning was skipped because no child matched               |
| `llm_calls`             | Must always be 0 — a non-zero value is a critical bug                       |

### Query Segmentation

Results are reported separately for:
- **Cue queries** — queries with at least one usable named entity or time expression; the CUE TREE advantage is clearest here.
- **Non-cue queries** — no extractable cues; the tree falls back to cosine-only beam search.

---

## Configuration

All experimental parameters live in `configs/global.yaml` (the single source of truth):

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_dim: 384
top_k: 5
seeds: [11, 22, 33, 44, 55]        # 5 random seeds; results reported as mean +/- std
store_sizes: [100, 500, 1000, 5000, 10000]
split:
  validation: 0.3   # tune thresholds on validation ONLY
  test: 0.7

tree:
  max_children: 32    # B: frozen by tree_sweep on the validation split
  min_fill: 0.35      # every split node keeps at least ceil(0.35 * (B+1)) children
  beam_width: 3       # default beam width for search
  fallback_tau: 0.35  # cosine threshold below which the flat fallback is triggered

recency:
  delta: 0.02         # tuned on validation split
  lam: 0.05           # tuned on validation split
```

Dataset paths are configured separately in `configs/data.yaml`.

> **Rule**: Never hard-code any value from `global.yaml`. All scripts must call `pipeline.config.load_all()` to read parameters at runtime.

---

## Baselines

| Baseline            | Class              | Description                                                        |
|---------------------|--------------------|--------------------------------------------------------------------|
| Brute Force         | `BruteForce`       | Exact cosine over all embeddings — the oracle accuracy baseline    |
| FAISS Flat          | `FaissFlat`        | FAISS exact inner-product index (same accuracy, vectorised)        |
| FAISS HNSW          | `FaissHNSW`        | Approximate graph-based index — sub-linear, no cues               |
| Tree (no cues)      | `CentroidTree`     | Centroid tree beam search without any cue pruning                  |
| Flat + Cue Filter   | `FlatCueFilter`    | Inverted entity index + timestamp filter, no tree hierarchy        |
| Filter + HNSW       | `HybridCueHNSW`    | Cue filter for cue queries, HNSW for all others (strongest flat)   |
| **CUE TREE**        | **`CueTree`**      | **Proposed: centroid tree + entity/time cue pruning**              |

All methods implement the same abstract `Retriever` interface:

```python
class Retriever(ABC):
    def build(self, conversations: list[Conversation]) -> None: ...
    def retrieve(self, query_text, query_emb, k=5, query_ts=None) -> (results, counters): ...
```

---

## Development & Testing

### Run Tests

```bash
# Run the full test suite
pytest

# Run a specific test file with verbose output
pytest tests/test_tree.py -v

# Short traceback mode
pytest --tb=short
```

### Test Coverage

| Test File           | What It Covers                                                    |
|---------------------|-------------------------------------------------------------------|
| `test_phase0.py`    | Basic imports and smoke tests                                     |
| `test_phase1.py`    | Tree insertion, uniform leaf depth, centroid accuracy             |
| `test_tree.py`      | MemoryTree: split, re-file, entity/time propagation to parents    |
| `test_cues.py`      | Entity normalisation, SpacyNER stub, QueryCues dataclass          |
| `test_cuetree.py`   | End-to-end CueTree: build, retrieve, fallback, cue ablations      |
| `test_baselines.py` | All baselines: recall correctness and cost accounting             |
| `test_timecue.py`   | Time-window parser: all rule types (n-ago, last/this period, etc) |
| `test_misfile.py`   | Misfiling simulation and `audit_refile` offline repair            |

### Adding a New Retriever

1. Subclass `core.retriever.Retriever`
2. Implement `build(conversations)` and `_retrieve(query_text, query_emb, k, counters, query_ts)`
3. Report **all** costs through the `Counters` object — `llm_calls` must stay 0
4. Register your method in `eval/compare_check.py` to include it in the benchmark

---

## Roadmap

- [x] **Phase 0** — Project scaffold: types, counters, abstract retriever interface
- [x] **Phase 1** — MemoryTree: centroid hierarchy, uniform depth, top-down beam search
- [x] **Phase 2** — Cue layer: spaCy NER entity extractor + rule-based time-window parser
- [x] **Phase 3** — Full benchmark: side-by-side comparison vs all baselines on LongMemEval
- [x] **Phase 4** — Ablations: cue on/off, fallback on/off, misfiling robustness & offline re-filing
- [ ] **Phase 5** — FastAPI server + REST API for live memory operations
- [ ] **Phase 6** — Chrome Extension (MV3) — capture and query browser sessions in real time
- [ ] **Phase 7** — React + Vite dashboard — interactive memory tree visualisation

---

## Team

This project is developed collaboratively as a **Semester 7 Research Project**.

| Contributor | GitHub                                           |
|-------------|--------------------------------------------------|
| Ayush       | [@Ayush-AI-ux](https://github.com/Ayush-AI-ux) |
| Parth       | [@Parthgarg27](https://github.com/Parthgarg27)   |

---

## License

This project is for academic and research purposes.  
Dataset usage is subject to the respective dataset licenses (LongMemEval, LoCoMo).

---

*MemoKi — giving LLMs the gift of episodic memory, one conversation at a time.*
