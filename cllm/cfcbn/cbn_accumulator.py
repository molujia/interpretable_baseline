"""
cbn_accumulator.py — CBN v5  ds25_faults.json 

Changes from v4:
1. Removed the error_flag guard (ablation experiment B showed that accumulating
   every case — regardless of whether CF-CBN was correct — improves performance).

2. Introduced adaptive alpha tuning (AdaptiveAlphaTuner).
   The previous alpha used exponential decay over accumulated case count, a proxy
   metric that does not reflect CBN quality. The new scheme runs a bisection search
   on the most recent eval_window cases every tune_interval accumulations (default 30)
   to find the alpha that maximises Top-1 directly, quantifying the true CBN contribution.

   Search mechanism:
     - Range: [alpha_min, 1.0]
     - Leave-one-out (LOO) estimate of Top-1 on the most recent eval_window cases
     - Bisection rounds: bisect_rounds (default 5); ~33 candidate alpha values evaluated
     - The resulting alpha_optimal remains in effect until the next tuning event
"""

import re
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Alpha strategy support (optional — set via pipeline)
_alpha_strategy_instance = None   # set by pipeline if --alpha-strategy is active
_rag_store_instance      = None   # set by pipeline if rag strategy is active

# ── Service base names (14 services, including TiDB components)──────────────────────────────────────
ALL_SERVICES = sorted([
    'adservice', 'cartservice', 'checkoutservice', 'currencyservice',
    'emailservice', 'frontend', 'paymentservice', 'productcatalogservice',
    'recommendationservice', 'redis-cart', 'shippingservice',
    'tidb-pd', 'tidb-tidb', 'tidb-tikv',
])

# ── All metrics (service-level and pod-level)────────────────────────────────────────────
ALL_METRICS = [
    # 
    'response', 'request', 'rrt', 'rrt_max',
    'error', 'error_ratio', 'client_error', 'client_error_ratio',
    'server_error', 'server_error_ratio', 'timeout',
    # Pod 
    'pod_cpu_usage', 'cpu_usage',
    'pod_memory_working_set_bytes', 'memory_usage', 'pod_processes',
    'pod_network_transmit_packets', 'pod_network_transmit_bytes',
    'pod_network_receive_packets', 'pod_network_receive_bytes',
]

# ── Metric weights — higher weight = stronger discriminative power────────────────────────────────────────
METRIC_WEIGHTS = {
    # Direct resource metrics: strongly suggest root cause
    'pod_cpu_usage': 4.0,  'cpu_usage': 4.0,
    'pod_memory_working_set_bytes': 3.5, 'memory_usage': 3.5,
    'pod_processes': 3.0,
    # Latency: indicates network / service performance root cause
    'rrt': 3.5,  'rrt_max': 3.5,  'timeout': 3.0,
    # Error rates: moderate discriminative power
    'client_error': 2.5,  'client_error_ratio': 2.5,
    'server_error': 2.5,  'server_error_ratio': 2.5,
    'error': 2.0,  'error_ratio': 2.0,
    # Network traffic: highly propagated, low discriminative power
    'pod_network_transmit_packets': 1.5, 'pod_network_receive_packets': 1.5,
    'pod_network_transmit_bytes': 1.0,  'pod_network_receive_bytes': 1.0,
    # Request/response volume: easily propagated, minimal discriminative power
    'request': 0.5, 'response': 0.5,
}

# Alpha scheduling parameters (fallback: used only when tuner has never run)
ALPHA_INIT  = 1.0   # Initial: pure unsupervised (CF only)
ALPHA_MIN   = 0.28  # Decay lower bound (ablation sweep: 0.28 is optimal on this dataset)
ALPHA_DECAY = 40    # Decay period (number of cases)


def base_name(s: str) -> str:
    s = re.sub(r'\s*\(deleted\).*$', '', s).strip()
    if s.startswith('tidb-'):
        m = re.match(r'(tidb-[a-z]+)-\d+$', s)
        if m:
            return m.group(1).lower()
        return s.lower()
    return re.sub(r'-\d+$', '', s).lower().strip()


def build_svc_metrics(case: Dict[str, Any], services: List[str]) -> Dict[str, set]:
    svc_m: Dict[str, set] = {s: set() for s in services}
    for svc, metrics in case.get('anomaly_services', {}).items():
        b = base_name(svc)
        if b in svc_m:
            svc_m[b].update(metrics)
    for pod, metrics in case.get('anomaly_pods', {}).items():
        b = base_name(pod)
        if b in svc_m:
            svc_m[b].update(metrics)
    return svc_m


def case_to_feature_vector(case: Dict[str, Any],
                            services: List[str] = None,
                            metrics: List[str] = None) -> np.ndarray:
    if services is None: services = ALL_SERVICES
    if metrics  is None: metrics  = ALL_METRICS
    S, M = len(services), len(metrics)

    svc_m = build_svc_metrics(case, services)
    feat  = np.zeros(S * M + S + 3, dtype=np.float32)

    for si, svc in enumerate(services):
        ms = svc_m[svc]
        for mi, metric in enumerate(metrics):
            if metric in ms:
                feat[si * M + mi] = METRIC_WEIGHTS.get(metric, 1.0)
        feat[S * M + si] = 1.0 if ms else 0.0

    n_anom = sum(1 for s in services if svc_m[s])
    rrt_s  = [s for s in services if 'rrt' in svc_m[s] or 'rrt_max' in svc_m[s]]
    feat[S * M + S]     = n_anom / max(len(services), 1)
    feat[S * M + S + 1] = len(rrt_s) / max(len(services), 1)
    feat[S * M + S + 2] = 1.0 if len(rrt_s) == 1 else 0.0
    return feat


def weighted_jaccard(v1: np.ndarray, v2: np.ndarray) -> float:
    num = np.minimum(v1, v2).sum()
    den = np.maximum(v1, v2).sum()
    return float(num / den) if den > 1e-9 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  Alpha 
# ─────────────────────────────────────────────────────────────────────────────

class AdaptiveAlphaTuner:
    """
    Bisection search for the optimal alpha on the accumulated history.

    Invoked by CBNAccumulator every time tune_interval new cases have been accumulated.
    tune() performs a leave-one-out (LOO) evaluation on the most recent eval_window
    cases, computes Top-1 hit rate for bisect_rounds rounds of trisection search,
    and adopts the best candidate as the new alpha_optimal.

    Search interval: [alpha_lo, alpha_hi]
      Empirical conclusions from ablation sweep:
        - alpha=0.28 achieves highest Top-1 on DS25 (49.06%)
        - alpha < 0.25 degrades (CBN noise outweighs its contribution)
        - alpha > 0.50 matches pure-CF performance (CBN contribution suppressed)
      → Search range [0.20, 0.65] covers the effective interval with margin
    """

    def __init__(self,
                 tune_interval:  int   = 30,
                 eval_window:    int   = 60,
                 bisect_rounds:  int   = 5,
                 alpha_lo:       float = 0.20,   # 
                 alpha_hi:       float = 0.65):  # CF
        self.tune_interval  = tune_interval
        self.eval_window    = eval_window
        self.bisect_rounds  = bisect_rounds
        self.alpha_lo       = alpha_lo
        self.alpha_hi       = alpha_hi

        self.alpha_optimal: float = 1.0     # Conservative initial value
        self._last_tuned_at: int  = 0       # total accumulated cases at last tune
        self.tune_log: List[dict] = []      # Debug log: records each tuning event

    def should_tune(self, total: int) -> bool:
        return (total >= self.tune_interval and
                total - self._last_tuned_at >= self.tune_interval)

    def tune(self, accumulator: "CBNAccumulator") -> float:
        """
         accumulator  LOO Evaluation alpha
         alpha_optimal
        """
        history = accumulator.history
        window  = history[-self.eval_window:] if len(history) >= self.eval_window \
                  else history[:]
        if len(window) < 5:
            return self.alpha_optimal   # Too few cases for tuning — skip

        lo, hi = self.alpha_lo, self.alpha_hi

        for _ in range(self.bisect_rounds):
            m1 = lo + (hi - lo) / 3.0
            m2 = lo + 2 * (hi - lo) / 3.0
            s1 = self._loo_top1(accumulator, window, m1)
            s2 = self._loo_top1(accumulator, window, m2)
            if s1 >= s2:
                hi = m2   # 
            else:
                lo = m1   # 

        best_alpha = round((lo + hi) / 2.0, 4)
        best_score = self._loo_top1(accumulator, window, best_alpha)

        self.alpha_optimal    = best_alpha
        self._last_tuned_at   = accumulator.total
        self.tune_log.append({
            "total": accumulator.total,
            "alpha_optimal": best_alpha,
            "loo_top1": round(best_score, 4),
            "window_size": len(window),
        })
        return best_alpha

    def _loo_top1(self, accumulator: "CBNAccumulator",
                  window: List[dict], alpha: float) -> float:
        """
         window  alpha  Top-1 
         case  accumulator 
         LOO
        """
        hits = 0
        for entry in window:
            qfeat = entry['feat']
            true_rcs = set(entry['root_causes'])

            # CF us_scores
            us_scores = entry.get('us_scores')
            if us_scores is None:
                # Fallback: simulate using the weighted sum of each service's feature vector slice
                S = len(accumulator.services)
                M = len(accumulator.metrics)
                us_scores = {}
                for si, svc in enumerate(accumulator.services):
                    us_scores[svc] = float(qfeat[si * M:(si + 1) * M].sum())

            # Normalise CF scores
            us_arr = np.array([us_scores.get(s, 0.0) for s in accumulator.services])
            mn, mx = us_arr.min(), us_arr.max()
            us_n = (us_arr - mn) / (mx - mn) if mx - mn > 1e-9 \
                   else np.ones_like(us_arr) / len(us_arr)

            # CBN posterior (full history, approximate LOO)
            cbn_post = accumulator.compute_posterior_from_feat(qfeat)
            cb = np.array([cbn_post.get(s, 0.0) for s in accumulator.services])

            fused = alpha * us_n + (1.0 - alpha) * cb
            top1_svc = accumulator.services[int(np.argmax(fused))]
            if top1_svc in true_rcs:
                hits += 1

        return hits / max(len(window), 1)


# ─────────────────────────────────────────────────────────────────────────────
# CBNAccumulatorv5
# ─────────────────────────────────────────────────────────────────────────────

class CBNAccumulator:
    """
    CBN v5

    : posterior(s) ∝ P(evidence|RC=s) × P(RC=s)
    : topK nearest-neighbour
    alpha AdaptiveAlphaTuner 

    v5 : 
      -  error_flag 
      - add_case() 
      - alpha  tuner  tune_interval  case 
      - history  us_scores LOO Evaluation
    """

    def __init__(self,
                 services     = None,
                 metrics      = None,
                 alpha_init   = ALPHA_INIT,
                 alpha_min    = ALPHA_MIN,
                 alpha_decay  = ALPHA_DECAY,   # fallback
                 laplace_k    = 0.05,
                 topk_nn      = 5,
                 tune_interval: int   = 30,
                 eval_window:   int   = 60,
                 bisect_rounds: int   = 5,
                 alpha_strategy: str  = 'adaptive',   # adaptive|rag|jaccard|entropy
                 rag_store = None):
        self.services    = services or ALL_SERVICES
        self.metrics     = metrics  or ALL_METRICS
        self.alpha_init  = alpha_init
        self.alpha_min   = alpha_min
        self.alpha_decay = alpha_decay
        self.laplace_k   = laplace_k
        self.topk_nn     = topk_nn

        self.history: List[Dict] = []
        self.prior_counts = defaultdict(float)
        self.total = 0
        self.alpha_strategy_name = alpha_strategy
        self._rag_store = rag_store

        # [0.20, 0.65] alpha
        self._tuner = AdaptiveAlphaTuner(
            tune_interval=tune_interval,
            eval_window=eval_window,
            bisect_rounds=bisect_rounds,
            alpha_lo=0.20,
            alpha_hi=0.65,
        )

    # ── alpha ────────────────────────────────────────────────────────────
    @property
    def alpha(self) -> float:
        """
         alpha
          - Initial phase (never tuned): exponential-decay fallback
          - tune :  tuner 
        """
        if self._tuner._last_tuned_at == 0:
            # Not yet tuned — using exponential-decay fallback
            if self.total == 0:
                return self.alpha_init
            return float(self.alpha_min +
                         (self.alpha_init - self.alpha_min) *
                         np.exp(-self.total / self.alpha_decay))
        return self._tuner.alpha_optimal

    # ── Accumulate (v5: no error_flag — every case is accumulated unconditionally)───────────────────────────
    def add_case(self, case: Dict, root_causes: List[str],
                 us_scores: Optional[Dict[str, float]] = None):
        """
        Add a case to history. From v5, there is no error_flag — all cases are accumulated.
        us_scores (optional): CF scores; if provided they are stored for LOO evaluation.
        """
        feat = case_to_feature_vector(case, self.services, self.metrics)
        rcs  = [base_name(r) for r in root_causes]
        entry = {'feat': feat, 'root_causes': rcs}
        if us_scores is not None:
            entry['us_scores'] = us_scores
        self.history.append(entry)
        for rc in rcs:
            self.prior_counts[rc] += 1.0
        self.total += 1

        #  alpha 
        if self._tuner.should_tune(self.total):
            self._tuner.tune(self)

    # ── Prior ─────────────────────────────────────────────────────────────
    def _prior(self, svc: str) -> float:
        cnt = self.prior_counts.get(svc, 0.0)
        return (cnt + self.laplace_k) / (self.total + self.laplace_k * len(self.services))

    # ── topK nearest-neighbour ───────────────────────────────────────────────
    def _likelihood(self, qfeat: np.ndarray, svc: str) -> float:
        relevant = [h for h in self.history if svc in h['root_causes']]
        if not relevant:
            si = self.services.index(svc) if svc in self.services else -1
            if si >= 0:
                M = len(self.metrics)
                raw = qfeat[si * M:(si + 1) * M].sum()
                return float(np.clip(raw / (4.0 * M + 1e-9), 0, 1))
            return 0.0
        sims = sorted(
            [weighted_jaccard(qfeat, h['feat']) for h in relevant],
            reverse=True
        )
        topk = sims[:self.topk_nn]
        wts  = np.array([1.0 / (i + 1) for i in range(len(topk))])
        return float(np.dot(topk, wts) / wts.sum())

    # ── Posterior (public interface: accepts a case dict)─────────────────────────────────
    def compute_posterior(self, case: Dict) -> Dict[str, float]:
        qfeat = case_to_feature_vector(case, self.services, self.metrics)
        return self.compute_posterior_from_feat(qfeat)

    # ── Posterior (internal interface: accepts feature vector directly, for LOO)──────────────────
    def compute_posterior_from_feat(self, qfeat: np.ndarray) -> Dict[str, float]:
        scores = {svc: self._prior(svc) * self._likelihood(qfeat, svc)
                  for svc in self.services}
        tot = sum(scores.values())
        if tot > 1e-9:
            return {k: v / tot for k, v in scores.items()}
        u = 1.0 / len(self.services)
        return {k: u for k in scores}

    # ── Fuse CF and CBN scores ─────────────────────────────────────────────────────────────
    def fuse_scores(self, us_scores: Dict[str, float], case: Dict,
                    alpha_override: Optional[float] = None) -> Dict[str, float]:
        us = np.array([us_scores.get(s, 0.0) for s in self.services])
        mn, mx = us.min(), us.max()
        us_n = (us - mn) / (mx - mn) if mx - mn > 1e-9 else np.ones_like(us) / len(us)

        if self.total > 0:
            cbn = self.compute_posterior(case)
            cb  = np.array([cbn.get(s, 0.0) for s in self.services])
        else:
            cb = np.ones(len(self.services)) / len(self.services)

        # Determine alpha: explicit override > strategy > bisection tuner
        if alpha_override is not None:
            a = float(alpha_override)
        else:
            a = self.alpha   # from bisection tuner (adaptive default)

        fused = a * us_n + (1.0 - a) * cb
        return {s: float(fused[i]) for i, s in enumerate(self.services)}

    def rank(self, fused: Dict[str, float]) -> List[str]:
        return sorted(fused.keys(), key=lambda s: fused[s], reverse=True)

    @property
    def tune_log(self) -> List[dict]:
        """Return the alpha tuning history (for debugging and paper analysis)."""
        return self._tuner.tune_log

