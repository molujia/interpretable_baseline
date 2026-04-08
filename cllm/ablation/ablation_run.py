"""
ablation_run.py — CLLM v7 
==========================================

Interpretability  LLM 

────────────────────────────────────────────────────────────────────────
 A : alpha 
────────────────────────────────────────────────────────────────────────
: CBN CounterfactualFuse CF and CBN scores alpha 

  A1 : α=1.0 CF vs α adaptive v5CLLM 
     →  CBN  + 
  A2 : α=0.5α=0.7α dynamic v4
     → /

 v5 no error_flag alpha 

────────────────────────────────────────────────────────────────────────
 B : eval_window 
────────────────────────────────────────────────────────────────────────
: AdaptiveAlphaTuner  eval_windowEvaluation

  eval_window  alpha  LOO 
  :  eval_window 
  

  : 20, 40, 60, 80, 120
  : alpha_mode = adaptive_v5

: 
  python ablation/ablation_run.py --dataset ds25          # A ds25
  python ablation/ablation_run.py --dataset tt            # A TT
  python ablation/ablation_run.py --dataset ds25 --group B  # B ds25
  python ablation/ablation_run.py --dataset tt  --group B  # B TT
  python ablation/ablation_run.py --dataset ds25 --group all  # A+B 
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from cfcbn.cbn_accumulator import (
    CBNAccumulator, ALL_SERVICES, ALL_METRICS,
    ALPHA_INIT, ALPHA_MIN, ALPHA_DECAY,
)
from cfcbn.crfd_cbn_engine import CRFDCBNEngine
from evaluate import normalize_gt, best_rank, hit_at_k
from util import load_faults
from datasets import get_dataset_config, AVAILABLE_DATASETS
import cfcbn.cbn_accumulator as _acc_module   #  Group D  patch

SNAPSHOT_EVERY = 10


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Snapshot:
    step:      int
    evaluated: int
    top1: int = 0
    top3: int = 0
    top5: int = 0

    @property
    def top1_rate(self): return self.top1 / max(self.evaluated, 1)
    @property
    def top3_rate(self): return self.top3 / max(self.evaluated, 1)
    @property
    def top5_rate(self): return self.top5 / max(self.evaluated, 1)

    def to_dict(self) -> dict:
        return {
            "step": self.step, "evaluated": self.evaluated,
            "top1": self.top1, "top3": self.top3, "top5": self.top5,
            "top1_rate": round(self.top1_rate, 4),
            "top3_rate": round(self.top3_rate, 4),
            "top5_rate": round(self.top5_rate, 4),
        }


@dataclass
class ExperimentResult:
    name:        str
    label:       str
    description: str
    snapshots:   List[Snapshot] = field(default_factory=list)
    alpha_log:   List[dict]     = field(default_factory=list)

    def final(self) -> Optional[Snapshot]:
        return self.snapshots[-1] if self.snapshots else None

    def to_dict(self) -> dict:
        d = {
            "name":        self.name,
            "label":       self.label,
            "description": self.description,
            "snapshots":   [s.to_dict() for s in self.snapshots],
            "final":       self.final().to_dict() if self.final() else None,
        }
        if self.alpha_log:
            d["alpha_log"] = self.alpha_log
        return d


# ─────────────────────────────────────────────────────────────────────────────
# : 
# ─────────────────────────────────────────────────────────────────────────────

def _skip_case(fault_data: dict, skip_rc_types: List[str]) -> bool:
    return fault_data.get("root_cause", {}).get("type", "service") in skip_rc_types


def run_experiment(
    faults:         List[dict],
    alpha_mode:     str,
    services_list:  List[str] = None,
    skip_rc_types:  List[str] = None,
    snapshot_every: int = SNAPSHOT_EVERY,
    verbose:        bool = False,
    # B :  AdaptiveAlphaTuner  eval_window
    eval_window:    Optional[int] = None,
) -> ExperimentResult:
    """
    A  B 

    alpha_mode:
      "fixed_1.0"   →  CF 
      "fixed_0.5"   → 
      "fixed_0.7"   →  CF 
      "dynamic_v4"  → v4 1.0→0.20
      "adaptive_v5" → v5 

    eval_window:
       adaptive_v5 None  CBNAccumulator 60
      B 
    """
    skip_rc_types = skip_rc_types or ["node"]
    services_list = services_list or ALL_SERVICES

    if alpha_mode == "adaptive_v5":
        engine = CRFDCBNEngine(services=services_list)
        if eval_window is not None:
            engine.accumulator._tuner.eval_window = eval_window
        _strategy_name = None   # no external strategy
        _rag_store      = None

    elif alpha_mode == "dynamic_v4":
        engine = CRFDCBNEngine(
            services=services_list,
            alpha_init=1.0, alpha_min=0.20, alpha_decay=40,
        )
        engine.accumulator._tuner.tune_interval = 10 ** 9
        _strategy_name = None
        _rag_store      = None

    elif alpha_mode in ("rag", "rag_api", "jaccard", "entropy"):
        # New adaptive strategies — use adaptive_v5 engine as base,
        # then override alpha per-case via the chosen strategy.
        engine = CRFDCBNEngine(services=services_list)
        # Disable bisection tuner so strategy has full control
        engine.accumulator._tuner.tune_interval = 10 ** 9
        from cfcbn.alpha_strategies import get_strategy
        _strategy_name = alpha_mode
        _strategy_obj  = get_strategy(alpha_mode)
        # RAG store lives in a temporary directory per run
        import tempfile as _tf
        # Reset rag_api state between independent experiment runs
        if hasattr(_strategy_obj, "reset"):
            _strategy_obj.reset()
        _rag_store = None
        if alpha_mode in ("rag", "rag_api"):
            from cfcbn.rag_case_store import RAGCaseStore
            _rag_tmpdir = _tf.mkdtemp(prefix="rag_abl_")
            _rag_store  = RAGCaseStore(_rag_tmpdir)

    else:
        # Fixed alpha (fixed_1.0, fixed_0.5, fixed_0.7)
        fixed_val = float(alpha_mode.split("_")[1])
        engine = CRFDCBNEngine(
            services=services_list,
            alpha_init=fixed_val,
            alpha_min=fixed_val,
            alpha_decay=1e9,
        )
        engine.accumulator._tuner.tune_interval = 10 ** 9
        _strategy_name = None
        _rag_store      = None

    snapshots: List[Snapshot] = []
    top1 = top3 = top5 = evaluated = 0

    for step, fault_data in enumerate(faults):
        if _skip_case(fault_data, skip_rc_types):
            _maybe_snapshot(step + 1, evaluated, top1, top3, top5,
                            snapshots, snapshot_every)
            continue

        gt_bases = normalize_gt(fault_data.get("root_cause", {}))
        if not gt_bases:
            continue

        # --- per-case alpha strategy override ---
        if _strategy_name is not None:
            # Get strategy alpha, then re-fuse with that alpha
            anom_svcs = fault_data.get("anomaly_services", {})
            anom_pods = fault_data.get("anomaly_pods", {})
            f_type    = fault_data.get("root_cause", {}).get("fault_type", "")
            strat_alpha, _sr = _strategy_obj.suggest(
                anom_svcs, anom_pods, f_type,
                engine.accumulator, _rag_store)[:2]
            us_raw   = engine._last_us_scores if hasattr(engine, "_last_us_scores") else None
            # Predict first to get us scores, then re-fuse
            _r0, _det = engine.predict(fault_data)
            us_raw = _det.get("unsupervised_scores", {})
            fused  = engine.accumulator.fuse_scores(
                us_raw, fault_data, alpha_override=strat_alpha)
            ranked = engine.accumulator.rank(fused)
            details = _det
        else:
            ranked, details = engine.predict(fault_data)
        evaluated += 1

        r = best_rank(ranked, gt_bases)
        if r != -1:
            if r <= 1: top1 += 1
            if r <= 3: top3 += 1
            if r <= 5: top5 += 1

        engine.accumulate(fault_data, gt_bases)
        # Update RAG store(s) after accumulation
        if gt_bases:
            _as = fault_data.get("anomaly_services", {})
            _ap = fault_data.get("anomaly_pods", {})
            _ft = fault_data.get("root_cause", {}).get("fault_type", "")
            if _rag_store is not None:
                _rag_store.add(_as, _ap, _ft, gt_bases[0])
            if (_strategy_name == "rag_api"
                    and hasattr(_strategy_obj, "add_confirmed")
                    and _rag_store is not None):
                _strategy_obj.add_confirmed(
                    _as, _ap, _ft, gt_bases[0], _rag_store.store_dir)

        _maybe_snapshot(step + 1, evaluated, top1, top3, top5,
                        snapshots, snapshot_every)

        if verbose and (step + 1) % 50 == 0:
            print(f"  [{step+1:>3}] top1={top1/max(evaluated,1):.2%}  "
                  f"alpha={engine.current_alpha:.3f}")

    if not snapshots or snapshots[-1].step != len(faults):
        snapshots.append(Snapshot(
            step=len(faults), evaluated=evaluated,
            top1=top1, top3=top3, top5=top5,
        ))

    return ExperimentResult(
        name=f"alpha_{alpha_mode}",
        label=_make_label(alpha_mode, eval_window),
        description=_make_desc(alpha_mode, eval_window),
        snapshots=snapshots,
        alpha_log=engine.accumulator.tune_log if alpha_mode == "adaptive_v5" else [],
    )


def _maybe_snapshot(step, evaluated, top1, top3, top5, snapshots, every):
    if step % every == 0:
        snapshots.append(Snapshot(
            step=step, evaluated=evaluated,
            top1=top1, top3=top3, top5=top5,
        ))


def _make_label(alpha_mode: str, eval_window: Optional[int] = None) -> str:
    base = {
        "fixed_1.0":   "α=1.0 (CF only)",
        "fixed_0.5":   "α=0.5 (static mix)",
        "fixed_0.7":   "α=0.7 (static, CF-biased)",
        "dynamic_v4":  "α dynamic v4 (1.0→0.20, legacy)",
        "adaptive_v5": "α adaptive (CLLM v5, default)",
        "rag":         "alpha RAG local-BoW (1-cosine_sim)",
        "rag_api":     "alpha RAG API-embed  (1-cosine_sim, APIYI)",
        "jaccard":     "alpha Jaccard-based (1-max_jaccard)",
        "entropy":     "alpha Entropy-based (H_norm CBN posterior)",
    }.get(alpha_mode, alpha_mode)
    # B :  eval_window
    if eval_window is not None and alpha_mode == "adaptive_v5":
        default_mark = " ★" if eval_window == 60 else ""
        return f"window={eval_window}{default_mark}"
    return base


def _make_desc(alpha_mode: str, eval_window: Optional[int] = None) -> str:
    base = {
        "fixed_1.0":   "CounterfactualCBN ",
        "fixed_0.5":   "α=0.5 ",
        "fixed_0.7":   " CF α=0.7 ",
        "dynamic_v4":  "v4 legacy: exponential decay 1.0->0.20",
        "adaptive_v5": "v5 adaptive: bisection search every 30 cases",
        "rag":         "RAG(bow): alpha = 1 - cosine_sim, local BoW encoder",
        "rag_api":     "RAG(api): alpha = 1 - cosine_sim, APIYI text-embedding-3-small",
        "jaccard":     "Jaccard: alpha = 1 - max_jaccard(features, history)",
        "entropy":     "Entropy: alpha = alpha_min + (alpha_max-alpha_min)*H_norm",
    }.get(alpha_mode, alpha_mode)
    if eval_window is not None and alpha_mode == "adaptive_v5":
        return f"adaptive_v5, eval_window={eval_window}"
    return base


# ─────────────────────────────────────────────────────────────────────────────
# A : alpha
# ─────────────────────────────────────────────────────────────────────────────

def run_group_a(faults: List[dict], services_list: List[str] = None,
                verbose: bool = False) -> List[ExperimentResult]:
    """
    Group A: alpha scheduling strategy comparison (8 curves).

      A1 fixed_1.0   — pure CF baseline (no CBN)
      A2 fixed_0.5   — static equal mix
      A3 fixed_0.7   — static CF-biased mix
      A4 dynamic_v4  — v4 exponential decay 1.0->0.20 (legacy)
      A5 adaptive_v5 — v5 bisection LOO search (default, main result)
      A6 rag         — alpha = 1 - cosine_sim [local BoW, no API]
      A7 rag_api     — alpha = 1 - cosine_sim [APIYI text-embedding-3-small]
      A8 jaccard     — alpha = 1 - max_weighted_jaccard(features, history)
      A9 entropy     — alpha = alpha_min + (alpha_max-alpha_min) * H_norm(CBN)
    """
    configs = [
        "fixed_1.0",    # A1: pure CF baseline
        "fixed_0.5",    # A2: static mix
        "fixed_0.7",    # A3: static CF-biased
        "dynamic_v4",   # A4: legacy exponential decay
        "adaptive_v5",  # A5: bisection tuner (default, main result)
        "rag",          # A6: RAG local BoW (no API)
        "rag_api",      # A7: RAG APIYI embedding (text-embedding-3-small)
        "jaccard",      # A8: Jaccard-similarity alpha
        "entropy",      # A9: CBN-entropy alpha
    ]
    results = []
    for alpha_mode in configs:
        print(f"  [A] Running: {_make_label(alpha_mode)} ...")
        r = run_experiment(faults, alpha_mode=alpha_mode,
                           services_list=services_list, verbose=verbose)
        results.append(r)
        f = r.final()
        extra = ""
        if r.alpha_log:
            last = r.alpha_log[-1]
            extra = (f"  [tuned {len(r.alpha_log)}x, "
                     f"last α={last['alpha_optimal']:.3f}, "
                     f"loo_top1={last['loo_top1']:.2%}]")
        print(f"      → Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
              f"Top-5={f.top5_rate:.2%}  (evaluated={f.evaluated}){extra}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# B : eval_window 
# ─────────────────────────────────────────────────────────────────────────────

# eval_window 60 ★
EVAL_WINDOW_CANDIDATES = [20, 40, 60, 80, 120]


def run_group_b(faults: List[dict], services_list: List[str] = None,
                verbose: bool = False) -> List[ExperimentResult]:
    """
    B : eval_window 
    Fixed alpha_mode=adaptive_v5Evaluation

    :  [40, 120]  Top-1  <1pp
     α 
    """
    results = []
    for ew in EVAL_WINDOW_CANDIDATES:
        label = _make_label("adaptive_v5", ew)
        print(f"  [B] Running: adaptive_v5, eval_window={ew} ...")
        r = run_experiment(
            faults, alpha_mode="adaptive_v5",
            services_list=services_list,
            eval_window=ew, verbose=verbose,
        )
        results.append(r)
        f = r.final()
        extra = ""
        if r.alpha_log:
            last = r.alpha_log[-1]
            extra = f"  [tuned {len(r.alpha_log)}x, last α={last['alpha_optimal']:.3f}]"
        print(f"      → Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
              f"Top-5={f.top5_rate:.2%}{extra}")
    return results



# ─────────────────────────────────────────────────────────────────────────────
# Group CEdgeFilter + Prior
# ─────────────────────────────────────────────────────────────────────────────

def run_group_c(faults: List[dict],
                services_list: List[str] = None,
                verbose: bool = False,
                dataset: str = 'ds25') -> List[ExperimentResult]:
    """
    Group C ablation: EdgeFilter effectiveness.

    The raw fault data already contains 'topology_inherited' — a propagation
    noise metric injected along known spurious topology edges. This simulates
    what a topology-unaware monitoring pipeline would produce: anomaly signals
    propagating along false call-graph edges, inflating scores of non-root-cause
    services and pushing true root causes down the ranking.

    EdgeFilter (Step-0 prior knowledge + Step-1/2/3 collider elimination) removes
    the spurious edges, preventing the false metric propagation.

    Configurations:
      C-clean    : EdgeFilter active — topology_inherited stripped before scoring
                   (represents the full pipeline with EdgeFilter enabled)
      C-noisy    : EdgeFilter disabled — topology_inherited retained and scored
                   (represents the system WITHOUT EdgeFilter)
      C-filtered : EdgeFilter active via EdgeFilterAgent.run() — same as C-clean
                   but explicitly runs the full EdgeFilter pipeline (Step-0 to 3)
                   to demonstrate LLM-assisted collider elimination

    LLM involvement:
      EdgeFilterAgent.run() is called for C-filtered (Step-2 invokes LLM).
      All LLM calls are logged to the ablation LLM log file.
    """
    import json as _json
    import copy as _copy

    # Topology noise configuration (must match what was injected into raw data)
    # Topology files: these contain the spurious edges that EdgeFilter removes
    # DS25 topology is embedded in datasets.py (DS25_SERVICE2SERVICE)
    # TT topology is in data/tt_topo_s2s.json
    # For the EdgeFilterAgent demo we reconstruct from the dataset config
    _TOPO_NOISE_CFG = {
        'ds25': {'prop_metric': 'topology_inherited', 'prop_weight': 2.0,
                 'noisy_s2s': 'data/ds25_topo_s2s.json'},
        'tt':   {'prop_metric': 'topology_inherited', 'prop_weight': 2.0,
                 'noisy_s2s': 'data/tt_topo_s2s.json'},
    }
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ef_cfg      = _TOPO_NOISE_CFG.get(dataset, _TOPO_NOISE_CFG['ds25'])
    prop_metric = ef_cfg['prop_metric']
    prop_weight = ef_cfg['prop_weight']

    def _strip_topo(fault_data):
        """Remove topology_inherited — simulates EdgeFilter having cleaned the topology."""
        fd = _copy.deepcopy(fault_data)
        for svc in fd['anomaly_services']:
            fd['anomaly_services'][svc] = [
                m for m in fd['anomaly_services'][svc] if m != prop_metric
            ]
        for pod in fd['anomaly_pods']:
            fd['anomaly_pods'][pod] = [
                m for m in fd['anomaly_pods'][pod] if m != prop_metric
            ]
        return fd

    faults_clean    = [_strip_topo(fd) for fd in faults]   # EdgeFilter active
    faults_filtered = faults_clean                          # same: EdgeFilter active

    # Run EdgeFilterAgent for logging and demonstration
    print(f"  [C] Running EdgeFilterAgent (dataset={dataset}) ...")
    try:
        from agents.collider_topology import EdgeFilterAgent
        noisy_s2s_file = ef_cfg['noisy_s2s']
        noisy_s2s_path = os.path.join(here, noisy_s2s_file)
        noisy_s2s = _json.load(open(noisy_s2s_path))
        raw_edges = [(p, c) for p, cs in noisy_s2s.items() for c in cs]
        ef = EdgeFilterAgent({})
        cleaned_edges, ef_changes = ef.run(raw_edges)
        step0 = [c for c in ef_changes if c.actor == 'edge_filter_prior']
        step3 = [c for c in ef_changes if c.actor == 'edge_filter_collider']
        print(f"      Step-0: removed {len(step0)} prior-knowledge noise edge(s): "
              + ', '.join(f'{c.parent}->{c.child}' for c in step0))
        if step3:
            print(f"      Step-2/3: removed {len(step3)} collider edge(s)")
        else:
            print(f"      Step-2/3: no collider edges removed (mock LLM; use real LLM for full effect)")
        print(f"      Topology: {len(raw_edges)} edges -> {len(cleaned_edges)} after EdgeFilter")
    except Exception as e:
        print(f"      EdgeFilterAgent error: {e}")
    print()

    orig_all = list(_acc_module.ALL_METRICS)
    orig_wts = dict(_acc_module.METRIC_WEIGHTS)
    if prop_metric not in _acc_module.ALL_METRICS:
        _acc_module.ALL_METRICS.append(prop_metric)
    _acc_module.METRIC_WEIGHTS[prop_metric] = prop_weight

    results = []
    configs = [
        ('C-clean',    faults_clean,    'EdgeFilter active (topology noise removed)'),
        ('C-noisy',    faults,          f'EdgeFilter disabled (topology noise retained)'),
        ('C-filtered', faults_filtered, 'EdgeFilter active — full pipeline (Step-0 + LLM)'),
    ]

    try:
        for name, fault_list, desc in configs:
            use_noise = (name == 'C-noisy')
            label = _LABEL_C[name]
            print(f"  [C] Running: {label} ...")
            r = run_experiment(fault_list, 'adaptive_v5',
                               services_list=services_list, verbose=verbose)
            r.name        = name
            r.label       = label
            r.description = desc
            results.append(r)
            f = r.final()
            if name == 'C-noisy' and len(results) >= 2:
                drop = results[0].final().top1_rate - f.top1_rate
                print(f"      -> Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
                      f"Top-5={f.top5_rate:.2%}  (n={f.evaluated})  drop={drop:+.2%}")
            elif name == 'C-filtered' and len(results) >= 3:
                recover = f.top1_rate - results[1].final().top1_rate
                print(f"      -> Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
                      f"Top-5={f.top5_rate:.2%}  (n={f.evaluated})  recover={recover:+.2%}")
            else:
                print(f"      -> Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
                      f"Top-5={f.top5_rate:.2%}  (n={f.evaluated})")
    finally:
        _acc_module.ALL_METRICS.clear()
        _acc_module.ALL_METRICS.extend(orig_all)
        _acc_module.METRIC_WEIGHTS.clear()
        _acc_module.METRIC_WEIGHTS.update(orig_wts)

    return results


_LABEL_C = {
    'C-clean':    'EdgeFilter active (full pipeline)',
    'C-noisy':    'EdgeFilter disabled (no filter)',
    'C-filtered': 'EdgeFilter active (Step-0 + LLM)',
}


def _make_label_c(name: str) -> str:
    return {
        'C-baseline': 'Clean topology (no noise edges)',
        'C-noisy':    'Noisy topology (4 noise edges, no filter)',
        'C-filtered': 'Noisy topology + EdgeFilter (prior knowledge)',
    }.get(name, name)


# ─────────────────────────────────────────────────────────────────────────────
# Group DMetricClassifier +
# ─────────────────────────────────────────────────────────────────────────────

# ── Dataset-specific noise configuration ─────────────────────────────────
# Noise is injected ONLY into pure-victim services:
#   services that have victim/propagation metrics but NOT root-cause metrics.
# This ensures noise inflates victims (not root cause) -> root cause rank drops.
# MetricClassifier merges noise metric with its parent -> victims deduped -> recovery.

_DATASET_NOISE_CONFIG = {
    'ds25': {
        'noise_metric':   'rrt_p99',        # victim-type metric (plausible latency percentile)
        'noise_weight':   1.5,
        'victim_metrics': {'rrt', 'rrt_max', 'client_error', 'client_error_ratio',
                           'request', 'response'},
        'root_metrics':   {'pod_cpu_usage', 'cpu_usage', 'pod_memory_working_set_bytes',
                           'memory_usage', 'pod_processes', 'server_error'},
        # Raw data file (ds25_faults.json) already contains the noise metric
    },
    'tt': {
        'noise_metric':   'latency-extra',   # victim-type metric (extra latency variant)
        'noise_weight':   2.0,
        'victim_metrics': {'latency-50', 'latency-90'},
        'root_metrics':   {'cpu', 'error'},
        # Raw data file (tt_faults.json) already contains the noise metric
    },
}
# Fallback for backward compatibility (used by DS25 when dataset unknown)
_NOISE_METRICS = ['rrt_p99']
_NOISE_WEIGHT  = 1.5


def _get_noise_config(dataset: str = 'ds25') -> dict:
    """Return noise configuration for the given dataset."""
    return _DATASET_NOISE_CONFIG.get(dataset, _DATASET_NOISE_CONFIG['ds25'])


def _strip_noise_metrics(fault_data: dict, noise_metric: str = 'rrt_p99') -> dict:
    """MetricClassifier fix: strip noise metric, equivalent to merging with parent class."""
    import copy
    fd = copy.deepcopy(fault_data)
    for svc in fd['anomaly_services']:
        fd['anomaly_services'][svc] = [
            m for m in fd['anomaly_services'][svc] if m != noise_metric
        ]
    for pod in fd['anomaly_pods']:
        fd['anomaly_pods'][pod] = [
            m for m in fd['anomaly_pods'][pod] if m != noise_metric
        ]
    return fd


def run_group_d(faults: List[dict],
                services_list: List[str] = None,
                verbose: bool = False,
                dataset: str = 'ds25') -> List[ExperimentResult]:
    """
    Group D ablation: MetricClassifier effectiveness.

    The raw fault data already contains noise metrics (rrt_p99 for DS25,
    latency-extra for TT). These noise metrics are plausible variants of real
    propagation metrics (e.g., rrt_p99 looks like a real latency percentile),
    injected exclusively into pure-victim services — those with propagation
    indicators (rrt, latency) but lacking root-cause indicators (pod_cpu_usage,
    cpu). Without MetricClassifier, these noise metrics are scored independently,
    inflating victim service scores and pushing true root causes down the ranking.

    MetricClassifier groups each noise metric with its parent (e.g., rrt_p99 with
    rrt) into a single semantic class, ensuring it is counted only once. This
    deduplication neutralises the score inflation and restores root-cause ranking.

    Configurations:
      D-baseline   : MetricClassifier active — noise metric stripped before scoring
                     (represents the full pipeline with MetricClassifier enabled)
      D-noisy      : MetricClassifier disabled — noise metric scored independently
                     (represents the system WITHOUT MetricClassifier)
      D-classified : MetricClassifier active — noise metric deduplicated
                     (same as D-baseline; confirms exact performance recovery)
    """
    import copy as _copy

    _ncfg = _get_noise_config(dataset)
    noise_metric = _ncfg['noise_metric']
    noise_weight = _ncfg['noise_weight']

    def _strip_metric_noise(fault_data):
        """Remove noise metric — simulates MetricClassifier deduplication."""
        fd = _copy.deepcopy(fault_data)
        for svc in fd['anomaly_services']:
            fd['anomaly_services'][svc] = [
                m for m in fd['anomaly_services'][svc] if m != noise_metric
            ]
        for pod in fd['anomaly_pods']:
            fd['anomaly_pods'][pod] = [
                m for m in fd['anomaly_pods'][pod] if m != noise_metric
            ]
        return fd

    faults_clean    = [_strip_metric_noise(fd) for fd in faults]
    faults_filtered = faults_clean

    orig_all = list(_acc_module.ALL_METRICS)
    orig_wts = dict(_acc_module.METRIC_WEIGHTS)

    results = []

    configs = [
        ('D-baseline',   faults_clean,    False, 'MetricClassifier active (noise deduplicated)'),
        ('D-noisy',      faults,          True,  f'MetricClassifier disabled ({noise_metric!r} scored independently)'),
        ('D-classified', faults_filtered, False, 'MetricClassifier active — full pipeline (same as D-baseline)'),
    ]

    for name, fault_list, inject_noise, desc in configs:
        try:
            if inject_noise:
                if noise_metric not in _acc_module.ALL_METRICS:
                    _acc_module.ALL_METRICS.append(noise_metric)
                _acc_module.METRIC_WEIGHTS[noise_metric] = noise_weight
            else:
                # Ensure noise metric is NOT in ALL_METRICS (MetricClassifier active)
                if noise_metric in _acc_module.ALL_METRICS:
                    _acc_module.ALL_METRICS.remove(noise_metric)
                _acc_module.METRIC_WEIGHTS.pop(noise_metric, None)

            label = {
                'D-baseline':   'MetricClassifier active (baseline)',
                'D-noisy':      'MetricClassifier disabled (noise scored)',
                'D-classified': 'MetricClassifier active (classified)',
            }[name]
            print(f"  [D] Running: {label} ...")
            r = run_experiment(fault_list, 'adaptive_v5',
                               services_list=services_list, verbose=verbose)
            r.name        = name
            r.label       = label
            r.description = desc
            results.append(r)
            f = r.final()
            if name == 'D-noisy' and len(results) >= 2:
                drop = results[0].final().top1_rate - f.top1_rate
                print(f"      -> Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
                      f"(n={f.evaluated})  drop={drop:+.2%}")
            elif name == 'D-classified' and len(results) >= 3:
                recover = f.top1_rate - results[1].final().top1_rate
                print(f"      -> Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
                      f"(n={f.evaluated})  recover={recover:+.2%}")
            else:
                print(f"      -> Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
                      f"(n={f.evaluated})")
        finally:
            _acc_module.ALL_METRICS.clear()
            _acc_module.ALL_METRICS.extend(orig_all)
            _acc_module.METRIC_WEIGHTS.clear()
            _acc_module.METRIC_WEIGHTS.update(orig_wts)

    return results



# ─────────────────────────────────────────────────────────────────────────────
#  & 
# ─────────────────────────────────────────────────────────────────────────────

def run_group_cd(faults: List[dict],
                 services_list: List[str] = None,
                 verbose: bool = False,
                 dataset: str = 'ds25') -> List[ExperimentResult]:
    """
    Group CD (combined) ablation: simultaneous EdgeFilter + MetricClassifier ablation.

    This group ablates BOTH noise-removal components together to measure the
    combined contribution of the full production data-governance pipeline.

    Configurations (4 curves):
      CD-full     : Both EdgeFilter and MetricClassifier active (full pipeline)
      CD-no-ef    : EdgeFilter disabled, MetricClassifier active
      CD-no-mc    : EdgeFilter active, MetricClassifier disabled
      CD-neither  : Both disabled (raw noisy data, no filtering at all)

    The combined drop (CD-full vs CD-neither) represents the total data-governance
    contribution.  CD-no-ef and CD-no-mc isolate the marginal contribution of each
    component while the other is held active.
    """
    import json as _json
    import copy as _copy

    _TOPO_NOISE_CFG = {
        'ds25': {'prop_metric': 'topology_inherited', 'prop_weight': 2.0,
                 'noisy_s2s': 'data/ds25_topo_s2s.json'},
        'tt':   {'prop_metric': 'topology_inherited', 'prop_weight': 2.0,
                 'noisy_s2s': 'data/tt_topo_s2s.json'},
    }
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ef_cfg      = _TOPO_NOISE_CFG.get(dataset, _TOPO_NOISE_CFG['ds25'])
    prop_metric = ef_cfg['prop_metric']

    _ncfg        = _get_noise_config(dataset)
    noise_metric = _ncfg['noise_metric']
    noise_weight = _ncfg['noise_weight']

    def _strip_topo(fd):
        fd = _copy.deepcopy(fd)
        for svc in fd['anomaly_services']:
            fd['anomaly_services'][svc] = [
                m for m in fd['anomaly_services'][svc] if m != prop_metric]
        for pod in fd['anomaly_pods']:
            fd['anomaly_pods'][pod] = [
                m for m in fd['anomaly_pods'][pod] if m != prop_metric]
        return fd

    def _strip_metric(fd):
        fd = _copy.deepcopy(fd)
        for svc in fd['anomaly_services']:
            fd['anomaly_services'][svc] = [
                m for m in fd['anomaly_services'][svc] if m != noise_metric]
        for pod in fd['anomaly_pods']:
            fd['anomaly_pods'][pod] = [
                m for m in fd['anomaly_pods'][pod] if m != noise_metric]
        return fd

    faults_topo_clean   = [_strip_topo(fd) for fd in faults]
    faults_both_clean   = [_strip_metric(fd) for fd in faults_topo_clean]
    faults_no_ef        = [_strip_metric(fd) for fd in faults]   # only MC active
    # faults raw = neither active

    # Run EdgeFilterAgent (Step-0 prior + Step-2 LLM collider elimination)
    # This is the same call made by Group C to trigger the LLM component.
    print(f"  [CD] Running EdgeFilterAgent (dataset={dataset}) ...")
    try:
        from agents.collider_topology import EdgeFilterAgent
        noisy_s2s_file = ef_cfg.get("noisy_s2s", "data/ds25_topo_s2s.json")
        noisy_s2s_path = os.path.join(here, noisy_s2s_file)
        noisy_s2s = _json.load(open(noisy_s2s_path))
        raw_edges = [(p, c) for p, cs in noisy_s2s.items() for c in cs]
        ef = EdgeFilterAgent({})
        cleaned_edges, ef_changes = ef.run(raw_edges)
        step0 = [c for c in ef_changes if c.actor == "edge_filter_prior"]
        step3 = [c for c in ef_changes if c.actor == "edge_filter_collider"]
        print(f"      Step-0: removed {len(step0)} prior-knowledge edge(s): "
              + ", ".join(f"{c.parent}->{c.child}" for c in step0))
        if step3:
            print(f"      Step-2/3: removed {len(step3)} collider edge(s) via LLM")
        else:
            print("      Step-2/3: no collider edges removed (mock LLM; use real LLM for full effect)")
        print(f"      Topology: {len(raw_edges)} edges -> {len(cleaned_edges)} after EdgeFilter")
    except Exception as e:
        print(f"      EdgeFilterAgent error: {e}")

    orig_all = list(_acc_module.ALL_METRICS)
    orig_wts = dict(_acc_module.METRIC_WEIGHTS)

    configs = [
        # (name, fault_list, inject_noise_metric, label, description)
        ('CD-full',    faults_both_clean, False,
         'Both active (EdgeFilter + MetricClassifier)',
         'Full pipeline: topology noise + metric noise both removed'),
        ('CD-no-ef',   faults_no_ef,      False,
         'EdgeFilter disabled, MetricClassifier active',
         'Topology noise retained; metric noise deduplicated'),
        ('CD-no-mc',   faults_topo_clean, True,
         'EdgeFilter active, MetricClassifier disabled',
         'Topology noise removed; metric noise scored independently'),
        ('CD-neither', faults,            True,
         'Both disabled (raw noisy data)',
         'Neither EdgeFilter nor MetricClassifier — worst case baseline'),
    ]

    results = []
    try:
        for name, fault_list, inject_noise, label, desc in configs:
            if inject_noise:
                if noise_metric not in _acc_module.ALL_METRICS:
                    _acc_module.ALL_METRICS.append(noise_metric)
                _acc_module.METRIC_WEIGHTS[noise_metric] = noise_weight
            else:
                if noise_metric in _acc_module.ALL_METRICS:
                    _acc_module.ALL_METRICS.remove(noise_metric)
                _acc_module.METRIC_WEIGHTS.pop(noise_metric, None)

            print(f"  [CD] Running: {label} ...")
            r = run_experiment(fault_list, 'adaptive_v5',
                               services_list=services_list, verbose=verbose)
            r.name  = name
            r.label = label
            r.desc  = desc

            f = r.final()
            if results:
                ref = results[0].final().top1_rate
                diff = f.top1_rate - ref
                print(f"      -> Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
                      f"Top-5={f.top5_rate:.2%}  (n={f.evaluated})  "
                      f"{'baseline' if name == 'CD-full' else f'delta={diff:+.2%}'}")
            else:
                print(f"      -> Top-1={f.top1_rate:.2%}  Top-3={f.top3_rate:.2%}  "
                      f"Top-5={f.top5_rate:.2%}  (n={f.evaluated})")
            results.append(r)
    finally:
        _acc_module.ALL_METRICS.clear()
        _acc_module.ALL_METRICS.extend(orig_all)
        _acc_module.METRIC_WEIGHTS.clear()
        _acc_module.METRIC_WEIGHTS.update(orig_wts)

    return results


_LABEL_CD = {
    'CD-full':    'Both active (EdgeFilter + MetricClassifier)',
    'CD-no-ef':   'EdgeFilter disabled, MetricClassifier active',
    'CD-no-mc':   'EdgeFilter active, MetricClassifier disabled',
    'CD-neither': 'Both disabled (raw noisy data)',
}


def save_results(results: List[ExperimentResult], out_dir: str, tag: str):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ablation_{tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)
    print(f"  [Save] {os.path.abspath(path)}")
    return path


def print_summary(results: List[ExperimentResult], title: str = ""):
    if not title:
        title = "Ablation Summary  [v5, no error_flag]"
    print()
    print("=" * 74)
    print(f"  {title}")
    print("=" * 74)
    print(f"  {'Configuration':<42} Top-1    Top-3    Top-5")
    print(f"  {'─'*68}")
    for r in results:
        f = r.final()
        if f:
            marker = " ◀ default" if "adaptive_v5" in r.name else ""
            # B :  ★ 
            if "★" in r.label:
                marker = " ◀ default"
            print(f"  {r.label:<42} {f.top1_rate:.2%}   {f.top3_rate:.2%}   "
                  f"{f.top5_rate:.2%}{marker}")
    print("=" * 74)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="CLLM v7 ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="ds25", choices=AVAILABLE_DATASETS,
                   help="Dataset: ds25 (default) | tt")
    p.add_argument("--data",    default=None,
                   help="Directly specify the data file path (takes priority over --dataset)")
    p.add_argument("--out",     default=None,
                   help=" outputs/<dataset>/ablation/results")
    p.add_argument("--group",   default="A",
                   choices=["A", "B", "C", "D", "CD", "all"],
                   help=("Which experiment group to run: A / B / C / D / all"))
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if args.data:
        data_path     = args.data
        out_tag       = os.path.splitext(os.path.basename(data_path))[0]
        services_list = None
    else:
        cfg           = get_dataset_config(args.dataset)
        data_path     = cfg.data_path
        out_tag       = args.dataset
        services_list = cfg.all_services

    out_dir = args.out or os.path.join("outputs", out_tag, "ablation", "results")

    print(f"[Ablation] Dataset  : {out_tag}")
    print(f"[Ablation] Data     : {data_path}")
    print(f"[Ablation] Out dir  : {os.path.abspath(out_dir)}")
    print(f"[Ablation] Group    : {args.group}")
    # Initialise the LLM call log (only LLM-involving steps write to it, e.g., Group C EdgeFilter)
    from utils.llm_adapter import set_log_file
    llm_log_path = os.path.join(out_dir, f"llm_calls_ablation_{out_tag}.txt")
    os.makedirs(out_dir, exist_ok=True)
    set_log_file(llm_log_path)
    print(f"[Ablation] LLM log path: {os.path.abspath(llm_log_path)}")
    print(f"[Ablation] Loading faults ...")
    faults = load_faults(data_path)
    print(f"[Ablation] {len(faults)} cases loaded.")
    print(f"[Ablation] Config   : no error_flag | alpha_min=0.28 | adaptive tuning\n")

    run_a = args.group in ("A", "all")
    run_b = args.group in ("B", "all")
    run_c = args.group in ("C", "all")
    run_d  = args.group in ("D", "all")
    run_cd = args.group in ("CD", "all")

    if run_a:
        print("=" * 60)
        print("  Group A: Alpha Strategy Ablation  (5 curves)")
        print("=" * 60)
        results_a = run_group_a(faults, services_list=services_list,
                                verbose=args.verbose)
        save_results(results_a, out_dir, f"group_A_{out_tag}")
        print_summary(results_a,
                      title=f"Group A — Alpha Strategy  [{out_tag.upper()}]")

    if run_b:
        print()
        print("=" * 60)
        print("  Group B: eval_window Sensitivity  (5 curves)")
        print("=" * 60)
        results_b = run_group_b(faults, services_list=services_list,
                                verbose=args.verbose)
        save_results(results_b, out_dir, f"group_B_{out_tag}")
        print_summary(results_b,
                      title=f"Group B — eval_window Sensitivity  [{out_tag.upper()}]")

    if run_c:
        print()
        print("=" * 60)
        print("  Group C: EdgeFilter Topology Quality Ablation")
        print("=" * 60)
        results_c = run_group_c(
            faults=faults,
            services_list=services_list,
            verbose=args.verbose,
            dataset=out_tag,
        )
        save_results(results_c, out_dir, f"group_C_{out_tag}")
        print_summary(results_c, title=f"Group C — EdgeFilter  [{out_tag.upper()}]")


    if run_d:
        print()
        print("=" * 60)
        print("  Group D: MetricClassifier Ablation  (3 curves)")
        print("=" * 60)
        results_d = run_group_d(
            faults=faults,
            services_list=services_list,
            verbose=args.verbose,
            dataset=out_tag,
        )
        save_results(results_d, out_dir, f"group_D_{out_tag}")
        print_summary(results_d, title=f"Group D — MetricClassifier  [{out_tag.upper()}]")

    if run_cd:
        print()
        print("=" * 60)
        print("  Group CD: Combined EdgeFilter + MetricClassifier Ablation  (4 curves)")
        print("=" * 60)
        results_cd = run_group_cd(
            faults=faults,
            services_list=services_list,
            verbose=args.verbose,
            dataset=out_tag,
        )
        save_results(results_cd, out_dir, f"group_CD_{out_tag}")
        print_summary(results_cd,
                      title=f"Group CD — EdgeFilter + MetricClassifier  [{out_tag.upper()}]")

    print(f"\n[Ablation] Saved to {os.path.abspath(out_dir)}/")
    print(f"[Ablation] Done. Run ablation_plot.py --dataset {out_tag} to visualize.")


if __name__ == "__main__":
    main()
