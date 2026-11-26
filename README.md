# MemoKi — Episodic Memory for LLMs

> **LLM-free, cue-guided hierarchical memory retrieval for long-term LLM conversations.**
> Zero LLM calls at ingestion and search · Brute-force-level recall at 3–16% of the embedding comparisons · Honest, pre-registered evaluation against strong flat baselines

**Status: development stage.** All headline numbers below are on the **validation split (100 queries, one seed = 11)**. The test split (329 queries) is untouched and reserved for the final run. Treat every number as a development result, not a final claim. See [Validation hygiene](#validation-hygiene-and-limitations).

---

## Table of Contents

- [Overview](#overview)
- [Key Idea](#key-idea)
- [What we can and cannot claim](#what-we-can-and-cannot-claim)
- [Architecture](#architecture)
- [Results (validation, seed 11)](#results-validation-seed-11)
- [Quick Start](#quick-start)
- [Data Setup](#data-setup)
- [Running Experiments](#running-experiments)
- [Evaluation and Cost Accounting](#evaluation-and-cost-accounting)
- [Configuration](#configuration)
- [Methods compared](#methods-compared)
- [Validation hygiene and limitations](#validation-hygiene-and-limitations)
- [Development and Testing](#development-and-testing)
- [Roadmap](#roadmap)
- [Team](#team)

---

## Overview

LLMs keep no memory of earlier conversations. The standard fix, RAG over a flat vector store, compares the query against every stored item, so cost grows with the store and results get noisier as it grows.

**MemoKi** stores each past conversation whole (never chunked) as a leaf in a balanced centroid tree. Every internal node carries cheap summaries of its subtree:

1. a **centroid embedding** (normalised mean of its children),
2. a **named-entity union** (extracted with spaCy),
3. a **date span** (min/max timestamp beneath it).

At query time a **top-down beam search** uses name and date cues from the query to skip branches by set membership and range overlap *before* any vector comparison. No LLM is used at ingestion or at query time.

Only **name and date cues** are implemented. Emotion and conversation-type cues from the original synopsis are not built yet (see [Roadmap](#roadmap)).

---

## Key Idea

Illustrative example (the real tree has up to 32 children per node, height 3 at 10,000 sessions):

```
Query: "What did Alice say about the Paris trip last month?"
        |
        |-- Name cues extracted:  {alice, paris}
        |-- Time cue parsed:      window = last month
        v
Beam search from the root:
  each level: children whose entity set or date span cannot match are
  skipped with a cheap cue check (no vector comparison)
  leaf level: surviving candidates are ranked by cosine similarity
```

Design properties:

- **Zero LLM calls** at ingestion and search (counter is 0 everywhere).
- **Cues are gates with a safety valve, not hard filters.** About 46% of queries contain no usable cue. If no child matches at a level, pruning is skipped for that level. Queries without cues behave exactly like the no-cue tree (verified at every size).
- **Cost grows slowly with store size**: 162 / 303 / 308 comparisons per query at 1k / 5k / 10k sessions. (HNSW also stays flat, so this is not unique to the tree.)
- **Query-time flat fallback is implemented but OFF by default.** It replaced good cue-guided answers with worse flat ones (recall fell, cost rose). It is reported as a negative result.
- **Offline re-filing** (`audit_refile`) repairs misfiled leaves without rebuilding the tree.

---

## What we can and cannot claim

**Defensible now (development evidence):**

- A cue-guided hierarchical memory reaches brute-force recall at about **3% to 16%** of the embedding comparisons, with no LLM calls.
- Cues give a statistically significant gain over the same tree without cues (10 wins, 1 loss, **p = 0.012**).
- On queries that have a cue, recall is far above brute force (0.634 vs 0.488 at 10,000 sessions), though a flat metadata filter gets the same gain.
- Misfiling costs 2 to 4 recall points at 10–20% misfiled; an offline audit recovers 91–100% of lost queries at about +2% query cost.
- **Versus LLM-based memory (Mem0):** 0 LLM calls at ingestion and search against 1 call per session; about 1 minute against about 5 hours of ingestion on the same laptop (extrapolated from a 3-session slice).

**Not defensible (do not claim):**

- "Faster than flat search" or "beats HNSW". Filter + HNSW is equal or slightly better above about 230 comparisons (0.440 at 297 vs 0.420 at 308, p = 0.69), and cue extraction costs about 5 ms per query, so end to end the cue tree is slower than HNSW.
- "The hierarchy is why accuracy improves" (a flat cue filter gets the same gain).
- "O(log N) cost". Not demonstrated; only three store sizes were measured.
- Any final claim about Gap 3, or accuracy versus Mem0/HiMem.
- Anything stated as final before the test-split run.

**Where the tree does win:** very low budgets. At 10,000 sessions the cue tree reaches recall 0.350 with 108 comparisons, where plain HNSW needs about 320.

---

## Architecture

```
MemoKi/
|-- core/               # Engine - pure Python, no external ML at import time
|   |-- types.py        # Conversation, Query, Result dataclasses
|   |-- counters.py     # Cost accounting (embedding comparisons, cue checks, latency, LLM calls)
|   |-- retriever.py    # Abstract Retriever base class (shared timing and counters)
|   |-- tree.py         # MemoryTree: balanced centroid hierarchy with cue summaries
|   |-- cues.py         # spaCy entity extractor + QueryCues dataclass
|   |-- timecue.py      # Rule-based time-window parser (LLM-free)
|   `-- misfile.py      # Deliberate misfiling simulator (Gap 2)
|
|-- baselines/          # Every method behind the same Retriever interface
|   |-- brute_force.py  # Exact cosine over all leaves
|   |-- faiss_index.py  # FAISS flat (exact) and HNSW (approximate)
|   |-- centroid_tree.py# Tree beam search, no cue pruning
|   |-- cue_tree.py     # Proposed method: tree + cue gate
|   |-- flat_cue.py     # Flat store + same cue filter (fair ablation)
|   `-- cue_prep.py     # Shared cue-preparation mixin
|
|-- pipeline/           # Data loading, embedding, cue caching
|   |-- loaders.py      # LongMemEval and LoCoMo parsers
|   |-- embed.py / embed_pool.py   # MiniLM-L6-v2 embedding (cached, multi-threaded)
|   |-- store.py        # Nested stores, seeded distractors, validation/test split
|   |-- cue_cache.py    # Resumable entity-cue cache (survives crashes)
|   |-- extract_cues.py # CLI: pre-extract entity cues for the full pool
|   |-- prepare.py      # download -> embed -> cue-extract
|   `-- download.py     # Dataset downloader helpers
|
|-- eval/               # Experiment runners
|   |-- metrics.py          # Recall@K
|   |-- compare_check.py    # All methods side by side (Gap 1 / Gap 4)
|   |-- cue_tree_check.py   # Cue tree vs no cues, fallback thresholds
|   |-- misfile_check.py    # Misfiling damage + query-time recovery tiers (Gap 2)
|   |-- refile_check.py     # Offline audit / re-filing (Gap 2)
|   |-- recency_check.py    # Recency rule pilot on update pairs (Gap 3)
|   |-- locomo_ours.py      # Our method on LoCoMo slice (Gap 4)
|   |-- label_sample.py / score_labels.py  # Hand-labelling sheet for cue accuracy
|   |-- tree_check.py / tree_sweep.py      # Tree validation and structure sweep
|   |-- cue_coverage.py / cue_selectivity.py / time_cue_check.py  # Cue quality checks
|   |-- seqlen_check.py, plot_pareto.py, sanity_check.py
|
|-- spike/
|   `-- mem0_spike.py   # Mem0 run with an LLM-call meter (separate venv, .venv-mem)
|
|-- configs/            # global.yaml (single source of truth), data.yaml
|-- results/            # Saved experiment outputs (tables, CSVs, Pareto plots)
|-- tests/              # pytest suite (81 tests)
|-- server/             # FastAPI server (in progress)
|-- requirements.txt
|-- pytest.ini
`-- README.md
```

---

## Results (validation, seed 11)

Data: LongMemEval-S (19,195 sessions, 470 answerable questions; split 141 validation / 329 test). Queries: 100 validation queries, identical at every store size. Metric: Recall@5. Cost: mean embedding comparisons per query (cue checks counted separately). Embeddings: all-MiniLM-L6-v2, 256 tokens, computed once and shared by every method. Hardware: CPU laptop, 8 cores, 7.8 GB RAM.

### Gap 1: accuracy and cost against store size (tree beam = 10)

Recall@5 / mean comparisons:

| Method | 1,000 sessions | 5,000 sessions | 10,000 sessions |
|---|---|---|---|
| Brute force | 0.630 / 1,000 | 0.460 / 5,000 | 0.360 / 10,000 |
| HNSW (ef 16) | 0.640 / 255 | 0.450 / 316 | 0.350 / 320 |
| Filter + HNSW (ef 16) | 0.710 / 162 | 0.520 / 243 | 0.440 / 297 |
| Tree, no cues | 0.600 / 254 | 0.390 / 463 | 0.330 / 457 |
| **Cue tree** | **0.680 / 162** | **0.470 / 303** | **0.420 / 308** |

The cue tree uses 16% / 6.1% / 3.1% of brute-force comparisons. At beam 3 it uses 76 / 105 / 108 comparisons and scores 0.660 / 0.390 / 0.350 (beam 3 is too narrow at 5,000: 7 points below brute force).

**Queries with a usable cue (41) vs without (59), 10,000 sessions, beam 10:**

| Group | Brute force | Tree, no cues | Cue tree |
|---|---|---|---|
| With a cue (41) | 0.488 | 0.415 | 0.634 |
| Without a cue (59) | 0.271 | 0.271 | 0.271 |

**Ablation (no cues / names / names + time):** 0.600 / 0.670 / 0.680 at 1k; 0.390 / 0.460 / 0.470 at 5k; 0.330 / 0.410 / 0.420 at 10k. Time adds about 1 point.

**Tree structure** (frozen by a rule written before the sweep; share of brute-force hits kept at 10,000 sessions): 8 children → 61% (height 7); 32 children, min fill 0.35 → 86% (height 3, chosen).

**Cue quality:** names appear in 36–37% of queries; recall 97% (36/37) on validation (95% on the earlier test-query check); one name check removes about 99% of distractors (selectivity 0.9%). Dates apply to only 8% of queries; recall 11/11.

**Latency:** search is about 0.27 ms vs 0.98 ms for brute force at 10,000 sessions, but name extraction adds about 5 ms per query, so **no latency claim is made**. Counts are the primary cost measure.

**Verdict: partly supported.** Strong against brute force; a tie or slight loss against filter + HNSW.

### Gap 2: misfiled conversations

Damage at 5,000 sessions (beam 10, 5 repeats, clean recall 0.470):

| Misfiled | Recall@5 | Lost queries | Cue / no-cue |
|---|---|---|---|
| 5% | 0.468 | 4 | 0 / 4 |
| 10% | 0.450 | 12 | 4 / 8 |
| 20% | 0.432 | 25 | 6 / 19 |

**Query-time recovery fails the synopsis criterion** (over 80% recovery with under 25% extra cost): entity lookup recovers 4/12 at +5.7%; wider beam ×8 recovers 10/12 at +397%; full flat recovers 12/12 at +1,642% but breaks 26 correct queries. The "weak similarity" trigger does not separate lost queries from normal ones.

**Offline audit** (margin frozen at 0.10 by a pre-written rule): recovers 91–100% of lost queries at all three sizes, query cost within about 2%, and meets the criterion.

| Size | Misfiled | Lost | Recovered | Harmed | Recall: misfiled → audited |
|---|---|---|---|---|---|
| 1,000 | 10% | 5 | 5 | 3 | 0.676 → 0.690 |
| 5,000 | 10% | 12 | 11 | 4 | 0.450 → 0.484 |
| 10,000 | 10% | 11 | 10 | 17 | 0.398 → 0.400 |

Side effects: the audit re-files 10–15% of healthy leaves, harms some other queries, and its net recall effect ranges from -0.8 to +4.2 points (about zero at 10,000 sessions). Offline cost is roughly quadratic in store size (4.75 million comparisons at 10,000). It does **not** reliably improve a healthy tree (+1, +2, -1 points; not significant). Our misfiling simulation is easier than natural misfiling.

**Verdict: largely answered on development data.** Still to do: hand-labelled cue-accuracy sheet and the test split.

### Gap 3: outdated information (pilot)

25 update pairs from LongMemEval knowledge-update queries, 5,000 sessions. The LLM-free recency rule (bonus for the newer conversation among near-tied candidates; two variants, `top` and `local`) is free, adds no comparisons and costs no recall at small settings. The stale rate moved in the right direction (older-above-newer 50% → 25% on the pairs where both were retrieved), but **nothing is significant** (2 pairs of 8, p = 0.5), and for most pairs neither session is retrieved at all (the newer session is in the top 5 for only 20–36% of pairs, because one vector per whole session dilutes single facts). The pairwise analysis and the `local` variant were added after seeing the first run and are reported as post-hoc. Frozen for the test split: δ = 0.01, λ = 0.01, both variants.

**Verdict: pilot, inconclusive but harmless.**

### Gap 4: fair comparison and Mem0

One config, same data, queries, embedding model, seeds and split for every method; shared counters; rules written before results; paired exact tests; partial ablation (no cues / names / names + time / fallback).

| Item | Result |
|---|---|
| Mem0 2.2.1, 3 LoCoMo sessions (llama3.2:3b, CPU) | 3 LLM calls (1 per add), 3.2 min; 0 LLM calls per search |
| Mem0 extrapolated to all of LoCoMo | about 304 calls, about 5 hours |
| MemoKi, same slice | 0 LLM calls, 0.60 s ingestion (about 61 s extrapolated) |
| Evidence hit rate (top 1 / 2 / 3 sessions) | 72% / 96% / 100% (3-session slice, so top 3 is trivial) |
| Tokens retrieved | MemoKi about 612 per session; Mem0 about 285 for 14.5 memories (we expect to lose on this) |

Open issues with the Mem0 numbers: possible context truncation at 2,048 tokens, `limit=5` apparently ignored, reduced mode with a 3B local model. **Accuracy is not compared with Mem0 or HiMem; HiMem has not been run.**

---

## Quick Start

```bash
# Python 3.11 required
python3.11 -m venv .venv
source .venv/bin/activate          # Linux / macOS
# .venv\Scripts\activate           # Windows (PowerShell)

pip install -r requirements.txt
python -m spacy download en_core_web_sm

pytest
```

---

## Data Setup

MemoKi is evaluated on **LongMemEval-S** (primary) and **LoCoMo** (Mem0/HiMem comparison).

```
data/raw/longmemeval_s_cleaned.json     # full LongMemEval-S (264.5 MB)
data/raw/longmemeval_oracle.json        # small oracle file for fast development
```

### Pre-extract entity cues (strongly recommended)

Entity extraction over the full 19,195-session pool takes about 40 minutes per 10,000 sessions with 3 processes. The cache is resumable, so it is safe to interrupt.

```bash
python -m pipeline.extract_cues                 # single process
python -m pipeline.extract_cues --procs 3       # faster on multi-core
python -m pipeline.extract_cues --limit 200     # quick timing test
```

Embeddings and cues are cached in `data/cache/` and reused by every evaluation script.

---

## Running Experiments

| Result | Command | Saved in `results/` |
|---|---|---|
| Tests | `pytest` | |
| Every method side by side (Gap 1 / 4) | `python -m eval.compare_check --sizes 1000 5000 10000 --efs 5 8 16 32 64 128 --beams 1 2 3 5 10 20 --audit-margin 0.0 --csv results/pareto_audit.csv` | `pareto_audit_table.txt`, `pareto_audit.csv`, `pareto_*.png` |
| Tree structure | `python -m eval.tree_sweep` | console output |
| Cue tree vs no cues, fallback | `python -m eval.cue_tree_check` (and `--taus 0.30 0.40`) | `cue_tree_validation.txt`, `cue_tree_fallback_validation.txt` |
| Misfiling and query-time recovery (Gap 2) | `python -m eval.misfile_check --sizes 5000` | `misfile_5000.txt` |
| Offline audit repair (Gap 2) | `python -m eval.refile_check --sizes 1000 5000 10000` | `refile_5000.txt`, `refile_1000_10000.txt` |
| Date cue | `python -m eval.time_cue_check` | console output |
| Entity cue quality | `python -m eval.cue_selectivity` | console output |
| Recency pilot (Gap 3) | `python -m eval.recency_check --csv results/recency_5000.csv` | `recency_5000.txt` |
| Our method on LoCoMo | `python -m eval.locomo_ours --sessions 3` | `locomo_ours_3.txt` |
| Mem0 spike (inside `.venv-mem`) | `python -m spike.mem0_spike --sessions 3 --granularity session` | `mem0_spike.json` |
| Hand-label cue accuracy | `python -m eval.label_sample`, then `python -m eval.score_labels` | |
| Pareto plots | `python -m eval.plot_pareto` | `pareto_*.png` |
| End-to-end smoke test | `python -m eval.sanity_check` | |

---

## Evaluation and Cost Accounting

**Primary metric:** Recall@5, a query is a hit if any relevant conversation is in the top 5, averaged over queries. Abstention questions are excluded (no evidence conversation to score).

Every `retrieve()` returns a `Counters` object. Timing and counter reset live in shared code, so no method can time itself flatteringly.

| Field | Description |
|---|---|
| `embedding_comparisons` | Vector dot products (the expensive operation); the **primary cost measure** |
| `cue_checks` | Cheap symbolic checks (entity set membership, date overlap), counted separately |
| `nodes_visited` | Tree nodes scored |
| `latency_ms` | Wall-clock search time (excludes cue preparation) |
| `prep_ms` | Entity extraction + date parsing, reported separately (about 5 ms) |
| `levels_relaxed` | Levels where cue pruning was skipped because no child matched |
| `llm_calls` | Must always be 0; non-zero is a critical bug |

Results are reported separately for **cue queries** (at least one usable name or date) and **no-cue queries** (the tree behaves like the plain centroid tree).

---

## Configuration

`configs/global.yaml` is the single source of truth. Scripts must read it via `pipeline.config.load_all()`; never hard-code values.

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
embedding_dim: 384
top_k: 5
seeds: [11, 22, 33, 44, 55]   # all development results so far use seed 11 only;
                              # multi-seed runs are planned for the final test-split run
store_sizes: [100, 500, 1000, 5000, 10000]
split:
  validation: 0.3             # 141 queries; design decisions use validation ONLY
  test: 0.7                   # 329 queries; untouched until the final run

tree:
  max_children: 32            # frozen by tree_sweep (rule written before the results)
  min_fill: 0.35
  beam_width: 10              # beam used for the headline results (3 = low-budget mode)
  fallback_tau: null          # flat fallback implemented but OFF by default (negative result)
  audit_margin: 0.10          # offline re-filing margin, frozen by the amended rule

recency:
  delta: 0.01                 # frozen for the test split
  lam: 0.01
```

> The values for `beam_width`, `fallback_tau`, `audit_margin` and `recency` above reflect the frozen settings in the project reports. If your local `global.yaml` still has older values (for example `beam_width: 3`, `fallback_tau: 0.35`, `delta: 0.02`, `lam: 0.05`), update it to match before the final run.

---

## Methods compared

| Method | Class | Description |
|---|---|---|
| Brute force | `BruteForce` | Exact cosine over every conversation |
| FAISS flat | `FaissFlat` | Exact inner-product index |
| HNSW (ef 5 to 128) | `FaissHNSW` | Approximate graph index, no cues |
| Flat + cue filter | `FlatCueFilter` | Same name and date filter via an inverted index, no tree |
| Filter + HNSW | `HybridCueHNSW` | Cue filter when cues exist, HNSW otherwise (**strongest flat opponent**) |
| Tree, no cues | `CentroidTree` | Centroid routing only (32 children, height 3 at 10,000) |
| **Cue tree (proposed)** | **`CueTree`** | Tree + name/date summaries + cue gate before vector comparison |
| Cue tree + audit | `CueTree` + `audit_refile` | Cue tree after one offline re-filing pass |

External systems: **Mem0** (flat, LLM-built memory; spike in progress) and **HiMem** (hierarchical, LLM-guided; feasibility not yet checked).

All methods implement one interface:

```python
class Retriever(ABC):
    def build(self, conversations: list[Conversation]) -> None: ...
    def retrieve(self, query_text, query_emb, k=5, query_ts=None) -> (results, counters): ...
```

---

## Validation hygiene and limitations

**What protects the results:** fixed, saved validation/test split with no design decisions on test; rules written before results (tree structure, fallback threshold, audit margin, recency selection); paired exact tests; counts rather than milliseconds as the cost measure; strong opponents (HNSW, filter + HNSW); bugs found and fixed with regression tests (entity-cache loss after a crash, a date-phrase crash, a stale cue cache across ablation switches, negative search time, model-load inflation of prep time, CSV columns, a flawed first Gap 3 rule).

**What weakens the results (disclose in the final report):**

1. 100 queries and one seed: one query is one point, and many differences are 1 to 4 queries.
2. Small denominators (Gap 2 lost counts 4, 12, 25; Gap 3 pairs 8 to 25); repeats reuse the same queries, so they are correlated.
3. Early checks used test queries (first baseline table, embedding-model comparison, first entity check). They only informed defaults that were kept (MiniLM, entity cue); the entity check has since been replicated on validation.
4. Two post-hoc changes: the audit-margin rule was tightened after seeing data, and Gap 3 gained a second analysis and rule variant.
5. Mem0 numbers come from a 3-session slice, linear extrapolation, a 3B local model in reduced mode, and an unresolved context-truncation question.
6. The misfiling simulation is easier than natural misfiling.
7. Sub-millisecond timings are noisy on a laptop; name extraction (about 5 ms) outweighs search time.
8. One vector per whole conversation dilutes single facts; this likely limits absolute recall and has not been proven.

---

## Development and Testing

```bash
pytest                         # full suite (81 tests)
pytest tests/test_tree.py -v   # one file, verbose
pytest --tb=short
```

| Test file | What it covers |
|---|---|
| `test_phase0.py` | Config loads, brute force finds a known item and counts correctly, deterministic ranking |
| `test_phase1.py` | Tree insertion, uniform leaf depth, centroid accuracy |
| `test_tree.py` | Split, re-file, entity/time propagation to parents |
| `test_cues.py` | Entity normalisation, extractor, `QueryCues` |
| `test_cuetree.py` | End-to-end cue tree: build, retrieve, fallback, ablation switches |
| `test_baselines.py` | Baseline correctness and cost accounting |
| `test_timecue.py` | Time-window parser rules |
| `test_misfile.py` | Misfiling simulation and `audit_refile` |

### Adding a new retriever

1. Subclass `core.retriever.Retriever`.
2. Implement `build(conversations)` and `_retrieve(query_text, query_emb, k, counters, query_ts)`.
3. Report all costs through `Counters`; `llm_calls` must stay 0 for LLM-free methods.
4. Register the method in `eval/compare_check.py`.

---

## Roadmap

**Done (development data):**

- [x] Foundation: config, shared counters, retriever interface, brute-force baseline, fixed split, resumable caches
- [x] Data and benchmark: five nested store sizes, embedding cache, embedding-length and model decisions
- [x] Cue layer: spaCy name extraction, rule-based date parser
- [x] Balanced tree with cue-guided beam search; structure frozen by a pre-written rule
- [x] Strong opponents: HNSW, flat + cue filter, filter + HNSW, paired tests
- [x] Gap 1 (cost vs scale): partly supported
- [x] Gap 2 (misfiling): damage measured, offline audit works, query-time fallback fails
- [x] Gap 3 pilot: recency rule is harmless, not significant (25 pairs)
- [x] Gap 4 foundation and first Mem0 slice

**Remaining, in order:**

1. Re-run the 1-session Mem0 diagnostic (context truncation); re-measure at 5 memories
2. Run our method on all 19 sessions of LoCoMo conv-26 (`eval.locomo_ours --sessions 19`)
3. HiMem feasibility check (may be impractical on a CPU laptop)
4. Hand-label 100 conversations to finish Gap 2 cue accuracy
5. Decide on type and emotion cues (test quickly, or drop with stated reasons)
6. **Final test-split run:** 3 to 5 seeds, all five store sizes, more queries, confidence intervals, all frozen settings
7. Optional: 50k to 100k stress test; cheaper name extractor (to address the 5 ms prep cost)
8. FastAPI server, Chrome extension (MV3) for capture and insertion, minimal React + Vite dashboard
9. Final write-up

---

## Team

Department of Computer Engineering and Applications, GLA University, Mathura, UP.

| Name | University Roll Number |
|---|---|
| Ayush Mittal | 2315800020 |
| Madhav Gupta | 2315800048 |
| Parth Garg | 2315800060 |
| Ronit Singh | 2315800070 |

Project mentor: Dr. Prasenjit Dhar

---

## License

This project is for academic and research purposes. Dataset usage is subject to the respective dataset licenses (LongMemEval, LoCoMo).

---

*MemoKi: giving LLMs episodic memory without paying an LLM for it.*
