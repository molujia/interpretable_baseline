"""
crfd_engine.py — Simplified CRFD (Counterfactual Reasoning for Fault Diagnosis) engine.

Implements a deterministic, formula-based approximation of the GNN-based counterfactual
approach from Liu et al. "CRFD: Counterfactual Reasoning for Fault Diagnosis", adapted to
run directly on the case dict format without requiring trained neural-network weights or
raw trace data.

=== Algorithm (GNN counterfactual + topology propagation) ===

1.  Build weighted metric matrix  X  (shape: services × metrics).
    X[s, m] = METRIC_WEIGHTS[m]  if metric m is anomalous in service s, else 0.

2.  For each service s compute the **counterfactual score**:
      cf[s]  = ||X|| − ||X_do_s||
    where X_do_s is X with row s zeroed (simulates "what if service s returns to normal?").
    Also compute the direct anomaly magnitude:
      direct[s] = Σ_m X[s, m]

3.  Add an **upstream propagation bonus**:
    If service B calls service A (topology edge B → A, meaning B depends on A), then
    A's failure propagates to B — B becomes anomalous because of A.
    Concretely, for each service A:
      propagation[A] = 0.25 × Σ_{B that calls A} direct[B]
    This captures the GNN message-passing intuition: a node deep in the call chain that
    causes anomalies in its callers accumulates those callers' scores.

4.  Final score:
      score[s] = cf[s]  +  0.3 × direct[s]  +  propagation[s]

Differences from CLLM CF-CBN
------------------------------
  CLLM CF-CBN : cf + (1−α) × CBN_posterior   (alpha-fused with online Bayesian history)
  CRFD-sim    : cf + 0.3×direct + propagation  (graph-structural, stateless)

The propagation term is the key differentiator: it rewards services that are "deep"
call-chain dependencies whose failure cascades to many callers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from typing import Dict, List, Tuple

from cfcbn.cbn_accumulator import METRIC_WEIGHTS, base_name

_METRICS_LIST: List[str] = list(METRIC_WEIGHTS.keys())  # fixed column order
_M = len(_METRICS_LIST)
_METRIC_IDX: Dict[str, int] = {m: i for i, m in enumerate(_METRICS_LIST)}


class CRFDEngine:
    """
    Simplified CRFD engine operating on case dicts.

    Stateless: no online learning; topology is fixed at construction time.

    Parameters
    ----------
    services : List[str]
        Full service list (short base names).
    service2service : Dict[str, List[str]], optional
        Caller → [callees] adjacency.  E.g. {"frontend": ["cartservice", ...], ...}
        means "frontend calls cartservice".
    """

    def __init__(
        self,
        services: List[str],
        service2service: Dict[str, List[str]] = None,
    ):
        self.services = list(services)
        self._svc_idx: Dict[str, int] = {s: i for i, s in enumerate(self.services)}

        # Reverse topology: callers[A] = list of services B such that B calls A.
        # If A fails, all of A's callers may show propagated anomalies.
        self.callers: Dict[str, List[str]] = {s: [] for s in self.services}
        if service2service:
            for caller, callees in service2service.items():
                for callee in callees:
                    if callee in self.callers:
                        self.callers[callee].append(caller)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_service_metrics(self, case: dict) -> Dict[str, List[str]]:
        svc_m: Dict[str, List[str]] = {s: [] for s in self.services}
        for raw, mets in case.get("anomaly_services", {}).items():
            b = base_name(raw)
            if b in svc_m:
                svc_m[b].extend(mets)
        for raw, mets in case.get("anomaly_pods", {}).items():
            b = base_name(raw)
            if b in svc_m:
                svc_m[b].extend(mets)
        for s in svc_m:
            svc_m[s] = list(dict.fromkeys(svc_m[s]))
        return svc_m

    def _build_matrix(self, svc_m: Dict[str, List[str]]) -> np.ndarray:
        """Build the weighted metric matrix X (services × metrics)."""
        X = np.zeros((len(self.services), _M), dtype=np.float64)
        for si, s in enumerate(self.services):
            for m in svc_m[s]:
                mi = _METRIC_IDX.get(m)
                if mi is not None:
                    X[si, mi] = METRIC_WEIGHTS[m]
        return X

    # ── public API ─────────────────────────────────────────────────────────────

    def predict(self, case: dict) -> Tuple[List[str], Dict]:
        """
        Predict root-cause ranking for a single case.

        Returns
        -------
        ranked_list : List[str]
        details : dict
            'scores'       : Dict[str, float]
            'cf_scores'    : Dict[str, float]
            'direct_scores': Dict[str, float]
            'propagation'  : Dict[str, float]
            'svc_metrics'  : Dict[str, list]
        """
        svc_m = self._get_service_metrics(case)
        X = self._build_matrix(svc_m)
        total_norm = float(np.linalg.norm(X))

        # Step 1 — CF score and direct score
        cf_scores: Dict[str, float] = {}
        direct_scores: Dict[str, float] = {}
        for si, s in enumerate(self.services):
            X_do = X.copy()
            X_do[si, :] = 0.0
            cf_scores[s] = total_norm - float(np.linalg.norm(X_do))
            direct_scores[s] = float(np.sum(X[si, :]))

        # Step 2 — upstream propagation bonus
        propagation: Dict[str, float] = {}
        for s in self.services:
            bonus = sum(direct_scores.get(c, 0.0) * 0.25 for c in self.callers.get(s, []))
            propagation[s] = bonus

        # Step 3 — combine
        scores: Dict[str, float] = {
            s: cf_scores[s] + 0.3 * direct_scores[s] + propagation[s]
            for s in self.services
        }

        ranked = sorted(self.services, key=lambda s: scores[s], reverse=True)

        return ranked, {
            "scores": scores,
            "cf_scores": cf_scores,
            "direct_scores": direct_scores,
            "propagation": propagation,
            "svc_metrics": svc_m,
        }
