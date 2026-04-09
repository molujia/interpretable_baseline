"""
rcd_engine.py — Simplified RCD (Root Cause Discovery) engine for interpretability evaluation.

Implements a deterministic, formula-based approximation of the PC-algorithm causal discovery
approach from Wang et al. "RCD: Causal Discovery for Root Cause Analysis", adapted to run
directly on the case dict format used by this baseline.

=== Algorithm (PC / conditional-independence inspired) ===

For each service s:
  score[s] = Σ_{m ∈ anomalous_metrics(s)}  weight[m] / prevalence[m]

where:
  prevalence[m] = number of distinct services that list metric m as anomalous.

Intuition
---------
In the PC algorithm, conditional independence (CI) tests determine whether an edge
between a metric variable and the F-node (failure node) should be kept.  When a metric
m appears anomalous in MANY services simultaneously it can be "explained away" by the
shared upstream failure — its p-value rises and the edge to F-node weakens.  Conversely,
a metric that is anomalous in ONLY ONE service provides a strong, unique signal (low
p-value) and points directly to that service as root cause.

Dividing by prevalence captures this: prevalence=1 → full weight; prevalence=k → weight/k.

Differences from CLLM CF-CBN
------------------------------
  CLLM : CF-delta score + alpha × CBN-posterior  (has online memory)
  RCD  : Σ weight/prevalence                     (stateless, each case independent)
"""

import re
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Tuple

from cfcbn.cbn_accumulator import METRIC_WEIGHTS, base_name


class RCDEngine:
    """
    Simplified RCD engine operating on case dicts.

    Stateless: no online learning; every case is scored independently.

    Parameters
    ----------
    services : List[str]
        Full service list for the dataset (short base names).
    """

    def __init__(self, services: List[str]):
        self.services = list(services)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_service_metrics(self, case: dict) -> Dict[str, List[str]]:
        """Return {service_base_name: [anomalous_metric, ...]} for the current case."""
        svc_m: Dict[str, List[str]] = {s: [] for s in self.services}

        for raw, mets in case.get("anomaly_services", {}).items():
            b = base_name(raw)
            if b in svc_m:
                svc_m[b].extend(mets)

        for raw, mets in case.get("anomaly_pods", {}).items():
            b = base_name(raw)
            if b in svc_m:
                svc_m[b].extend(mets)

        # Deduplicate while preserving order
        for s in svc_m:
            svc_m[s] = list(dict.fromkeys(svc_m[s]))

        return svc_m

    # ── public API ─────────────────────────────────────────────────────────────

    def predict(self, case: dict) -> Tuple[List[str], Dict]:
        """
        Predict root-cause ranking for a single case.

        Returns
        -------
        ranked_list : List[str]
            Services ordered by RCD score, highest first.
        details : dict
            'scores'            : Dict[str, float]  — per-service score
            'metric_prevalence' : Dict[str, int]    — how many services share each metric
            'svc_metrics'       : Dict[str, list]   — resolved service→metrics mapping
        """
        svc_m = self._get_service_metrics(case)

        # Step 1 — metric prevalence
        metric_prevalence: Dict[str, int] = {}
        for mets in svc_m.values():
            for m in mets:
                metric_prevalence[m] = metric_prevalence.get(m, 0) + 1

        # Step 2 — score per service
        scores: Dict[str, float] = {}
        for s in self.services:
            mets = svc_m[s]
            if not mets:
                scores[s] = 0.0
                continue
            score = 0.0
            for m in mets:
                w = METRIC_WEIGHTS.get(m, 1.0)
                p = metric_prevalence.get(m, 1)
                score += w / p
            scores[s] = score

        ranked = sorted(self.services, key=lambda s: scores[s], reverse=True)

        return ranked, {
            "scores": scores,
            "metric_prevalence": metric_prevalence,
            "svc_metrics": svc_m,
        }
