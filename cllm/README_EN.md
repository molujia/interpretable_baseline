# CLLM v8 — Interpretable and Explainable Root Cause Localization

> **CLLM** combines a fully offline CF-CBN causal engine (**Interpretability Layer**)
> with an LLM-powered Explainer agent (**Explainability Layer**) for microservice RCL.

---

## Quick Start

```bash
pip install -r requirements.txt

# Pure CF-CBN (no LLM)
python main.py --dataset ds25

# With LLM (Workflow A + B, conditional trigger)
python main.py --dataset ds25 --use-llm

# With RAG-based adaptive alpha
python main.py --dataset ds25 --use-llm --alpha-strategy rag

# Single-case analysis
python main.py --single 42 --use-llm --llm-mode always --verbose
```

---

## Architecture

```
              Explainability Layer  (LLM — off the RCL critical path)
  Alert ──►  EdgeFilter  ·  MetricClassifier  ·  Anonymizer Barrier
             Explainer WF-A (pre SRE confirm)  ·  WF-B (post SRE confirm)
                              │
                     (cleaned evidence)
                              │
              Interpretability Layer  (zero LLM, fully auditable)
             CF Engine  ──►  CBN Accumulator (α-tuner)  ──►  Case Store
                              │
                        Top-K Ranking + English Fault Ticket
```

### Three LLM-containing modules

| Module | Invocation time | Role |
|---|---|---|
| **EdgeFilter** | System init (before any fault case) | Step-0: script removes prior-knowledge spurious edges; Step-2: LLM applies collider-elimination rules |
| **MetricClassifier** | System init | LLM groups semantically equivalent metrics; console prints merged classes |
| **Explainer** | After CF-CBN ranking | WF-A: semantic interpretation; WF-B: post-mortem diagnosis |

> **Why EdgeFilter and MetricClassifier use LLM even without `--use-llm`:**
> These two modules are initialised at system startup, independent of the
> per-case `--use-llm` flag.  The `--use-llm` flag only controls whether
> the **Explainer** (WF-A/WF-B) is triggered.  EdgeFilter and
> MetricClassifier always call the LLM during initialisation because:
> (1) prior knowledge cannot be fully codified as scripts;
> (2) domain rules change over time without a code release;
> (3) rapid adaptation (e.g., canary deployments) requires runtime updates.
> The LLM call happens once at startup, not on the critical path.

---

## Adaptive Alpha Strategies

`score(s) = α·CF(s) + (1-α)·CBN(s)`

| Strategy | Flag | Mechanism |
|---|---|---|
| **Adaptive** (default) | `adaptive` | Bisection LOO search every 30 cases on a 60-case rolling window |
| **RAG (BoW)** | `rag` | `α = 1 − cosine_sim(new_evidence, best_match)` — local BoW encoder, no API |
| **RAG (API)** | `rag_api` | Same as `rag` but vectors from APIYI text-embedding-3-small (1536-dim) |
| **Jaccard** | `jaccard` | `α = 1 − max_weighted_jaccard(new_features, history_features)` |
| **Entropy** | `entropy` | `α = α_min + (α_max−α_min)·H_norm(CBN_posterior)` |

### Switching the RAG embedding backend

Open `cfcbn/rag_case_store.py` and change the toggle at the top:

```python
# Local BoW encoder (default, no API key needed)
USE_OPENAI_EMBED = False

# OpenAI text-embedding-3-small (requires key)
USE_OPENAI_EMBED = True
OPENAI_API_KEY   = "sk-your-key-here"   # fill in your key
```

After switching, delete `case_store/rag_index.faiss`, `rag_vocab.pkl`, and
`rag_meta.pkl` so the index is rebuilt with the new backend.

---

## Fault Ticket Format (fully English)

```
======================================================================
  CLLM v5 -- Fault Analysis Ticket
  Fault category : jvm fault  |  Fault type: jvm gc
======================================================================

[ 1 ]  CF-CBN Root-Cause Ranking  (Counterfactual + Bayesian fusion)
  #1  adservice                       fused_score=1.0000
       [alpha=1.000 [adaptive]]

[ 2 ]  Anomaly Evidence  (raw metrics that drove the CF score -- judgement basis)
  Top candidates:
    adservice     ['pod_cpu_usage', 'rrt', 'server_error']  (CF_raw=0.3821)
    frontend      ['client_error', 'rrt']  (CF_raw=0.2104)

[ 3 ]  LLM Workflow A -- Semantic Interpretation
  Trigger reason : C1: low confidence gap ...
  Pattern        : ...
  Candidate analysis:
    #1 adservice   [ROOT CAUSE]  Strong CPU + error signal ...

[ 4 ]  Similar Historical Cases
  [similarity=0.92]  root_cause=adservice  fault_type=jvm gc  CF-CBN correct

----------------------------------------------------------------------
  ACTION  Confirm or correct Top-1 using the SRE console.
======================================================================
```

---

## Ablation Groups

```bash
python ablation/ablation_run.py --dataset ds25 --group <GROUP>
```

| Group | Content |
|---|---|
| `A` | Alpha strategy comparison (9 curves): fixed α=1.0/0.5/0.7, v4 exp-decay, v5 adaptive, RAG(BoW), RAG(API), Jaccard, Entropy |
| `B` | eval_window sensitivity |
| `C` | EdgeFilter topology-noise ablation |
| `D` | MetricClassifier metric-noise ablation |
| `CD` | **Combined** EdgeFilter + MetricClassifier (4 configurations) |
| `all` | All groups |

### Group CD configurations

| Config | EdgeFilter | MetricClassifier |
|---|---|---|
| `CD-full` | ✓ | ✓ |
| `CD-no-ef` | ✗ | ✓ |
| `CD-no-mc` | ✓ | ✗ |
| `CD-neither` | ✗ | ✗ |

> **Note:** Running `--group CD` also calls `EdgeFilterAgent.run()` (Steps 0–3)
> just like `--group C`, so the LLM-assisted collider-elimination step is exercised.

---

## Console Output

### EdgeFilter
```
[EdgeFilter] Step 0: Removed frontend->emailservice | prior knowledge
[EdgeFilter] Step 0 summary: 4 edge(s) removed: frontend->emailservice, ...
[EdgeFilter] Step 1: Found 4 collider nodes ...
[EdgeFilter] Step 3: No collider edges removed (no applicable rules).
[EdgeFilter] Run complete: 27 input edges -> 23 after removing 4 spurious edge(s).
```

### MetricClassifier
```
[MetricCLS] Initialized: 12 classes from 18 metrics
[MetricCLS] Deduplicated metric groups (each counted once):
  Class 2: ['rrt', 'rrt_max', 'rrt_p99']
[MetricCLS] Merged: 'rrt_p99' -> class 2 (with existing: ['rrt', 'rrt_max'])
```

---

## LLM Backend Configuration

Edit `utils/llm_adapter.py`:

```python
PLATFORM = "mock"     # no LLM (safe default for testing)
PLATFORM = "apiyi"    # APIYI relay (recommended)
PLATFORM = "openai"   # OpenAI direct
PLATFORM = "volc"     # Volcengine Ark
```

---

## File Structure

```
cllm_v8/
├── main.py
├── pipeline.py
├── datasets.py  /  evaluate.py  /  util.py
├── cfcbn/
│   ├── cbn_accumulator.py      CBN + AdaptiveAlphaTuner
│   ├── crfd_cbn_engine.py      CF-CBN engine
│   ├── rag_case_store.py       RAG store (BoW or OpenAI backend)   [NEW]
│   └── alpha_strategies.py     RAG / Jaccard / Entropy strategies   [NEW]
├── agents/
│   ├── collider_topology.py    EdgeFilter (Step-0 prior + Step-2 LLM)
│   ├── llm_workflow.py         WF-A + WF-B + English ticket
│   ├── case_store.py           Persistent case archive
│   ├── metric_classifier.py    Online metric deduplication (LLM)
│   └── knowledge/
│       ├── edge_prior_knowledge.json
│       └── collider_prior_knowledge.txt
├── ablation/
│   ├── ablation_run.py         Groups A/B/C/D/CD   [CD added]
│   └── ablation_plot.py
├── data/
│   ├── ds25_faults.json  /  ds25_topo_s2s.json
│   └── tt_faults.json    /  tt_topo_s2s.json
├── utils/  (llm_adapter.py, anonymizer.py)
├── requirements.txt
├── README_EN.md    (this file)
└── README_ZH.md    (Chinese version)
```

---

## What Changed in v8

| # | Change | File(s) |
|---|---|---|
| 1 | Fault tickets fully English + anomaly metrics per candidate (judgement basis) | `agents/llm_workflow.py` |
| 2 | EdgeFilter / MetricClassifier detailed console output | `agents/collider_topology.py`, `agents/metric_classifier.py` |
| 3 | Group CD combined ablation (4 configs, with EdgeFilterAgent LLM call) | `ablation/ablation_run.py` |
| 4 | RAG-based adaptive alpha (local BoW default + OpenAI backend option) | `cfcbn/rag_case_store.py` |
| 5 | Three alpha strategies: RAG, Jaccard, Entropy | `cfcbn/alpha_strategies.py` |
| 6 | `--alpha-strategy` CLI flag | `main.py`, `pipeline.py` |
| 7 | Zero Chinese characters/symbols across all source files | all `.py` files |
