"""
crfd_cbn_engine.py — CRFD-CBN Fuse CF and CBN scores
"""
import re
import numpy as np
from typing import Dict, List, Tuple, Any

from cfcbn.cbn_accumulator import (
    CBNAccumulator, ALL_SERVICES, ALL_METRICS,
    METRIC_WEIGHTS, build_svc_metrics, base_name,
    ALPHA_INIT, ALPHA_MIN, ALPHA_DECAY,
)

ANOMALY_SIGNAL = 3.0
NORMAL_SIGNAL  = 0.0


def _build_matrix(case: Dict, services: List[str]) -> np.ndarray:
    """Build service × metric weighted feature matrix"""
    svc_m = build_svc_metrics(case, services)
    X = np.zeros((len(services), len(ALL_METRICS)), dtype=np.float32)
    for si, svc in enumerate(services):
        for mi, metric in enumerate(ALL_METRICS):
            if metric in svc_m[svc]:
                X[si, mi] = ANOMALY_SIGNAL * METRIC_WEIGHTS.get(metric, 1.0)
    return X


def unsupervised_rcl(case: Dict, services: List[str]) -> Dict[str, float]:
    """
    Counterfactual root-cause score: for each service, apply do-intervention (zero out its row)
    and measure the resulting reduction in total anomaly magnitude.
    Also fuses a direct anomaly weight score (sum of metric weights) as an auxiliary signal.
    """
    X = _build_matrix(case, services)
    total_norm = float(np.linalg.norm(X))

    svc_m = build_svc_metrics(case, services)
    scores = {}
    for si, svc in enumerate(services):
        # Counterfactual reduction
        X_do = X.copy()
        X_do[si, :] = NORMAL_SIGNAL
        cf_score = float(np.linalg.norm(X) - np.linalg.norm(X_do))

        # Direct anomaly weight score (auxiliary)
        direct = sum(METRIC_WEIGHTS.get(m, 1.0) for m in svc_m[svc])

        # Fuse: counterfactual dominates; direct score provides non-zero baseline
        scores[svc] = cf_score + 0.3 * direct

    return scores


class CRFDCBNEngine:
    def __init__(self,
                 services    = None,
                 alpha_init  = ALPHA_INIT,
                 alpha_min   = ALPHA_MIN,
                 alpha_decay = ALPHA_DECAY):
        self.services    = services or ALL_SERVICES
        self.accumulator = CBNAccumulator(
            services=self.services,
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_decay=alpha_decay,
        )

    @property
    def n_accumulated(self): return self.accumulator.total
    @property
    def current_alpha(self):  return self.accumulator.alpha

    def predict(self, case: Dict) -> Tuple[List[str], Dict]:
        us   = unsupervised_rcl(case, self.services)
        fuse = self.accumulator.fuse_scores(us, case)
        rank = self.accumulator.rank(fuse)
        return rank, {
            'alpha': self.current_alpha,
            'n_accumulated': self.n_accumulated,
            'unsupervised_scores': us,
            'fused_scores': fuse,
        }

    def accumulate(self, case: Dict, root_causes: List[str]):
        # us_scoresAdaptive alpha LOO Evaluation
        us = unsupervised_rcl(case, self.services)
        self.accumulator.add_case(case, root_causes, us_scores=us)

    def predict_then_accumulate(self, case: Dict, root_causes: List[str]):
        ranked, det = self.predict(case)
        self.accumulate(case, root_causes)
        return ranked, det
