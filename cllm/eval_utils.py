"""
eval_utils.py — Shared utilities for CF-CBN interpretability evaluation

Implements the Human-grounded Evaluation framework from Doshi-Velez & Kim (2017) §3.2:
  - Forward Simulation: LLM agent predicts model output given algorithm explanation
  - Counterfactual Simulation: LLM agent identifies input changes to flip prediction

Usage: imported by eval_forward_simulation.py and eval_counterfactual_simulation.py
"""

import os
import re
import sys
import json
import math
import copy
import time
import requests
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# ============================================================
# LLM Configuration — edit here or set env var EVAL_LLM_API_KEY
# ============================================================
EVAL_PLATFORM = "apiyi"    # "apiyi" | "openai" | "volc"
EVAL_MODEL    = "gpt-4o"   # strong reasoning model for arithmetic chain-of-thought
EVAL_API_KEY  = os.environ.get("EVAL_LLM_API_KEY", "sk-your-apiyi-key-here")

_PLATFORM_URLS = {
    "apiyi":  "https://api.apiyi.com/v1",
    "openai": "https://api.openai.com/v1",
    "volc":   "https://ark.cn-beijing.volces.com/api/v3",
}

_MAX_RETRIES = 3
_RETRY_BASE  = 10   # seconds; doubled each retry


# ============================================================
# Standalone LLM client (independent of main llm_adapter.py)
# ============================================================

class EvalLLMClient:
    """
    OpenAI-compatible LLM client for interpretability evaluation.
    Supports multi-turn conversation history and retries.
    Independent of cllm/utils/llm_adapter.py configuration.
    """

    def __init__(self,
                 platform: str = None,
                 model:    str = None,
                 api_key:  str = None):
        self.platform = platform or EVAL_PLATFORM
        self.model    = model    or EVAL_MODEL
        self.api_key  = api_key  or EVAL_API_KEY
        if self.platform not in _PLATFORM_URLS:
            raise ValueError(f"Unknown platform '{self.platform}'. "
                             f"Valid: {list(_PLATFORM_URLS)}")
        self.base_url = _PLATFORM_URLS[self.platform]
        print(f"[EvalLLM] platform={self.platform}  model={self.model}  "
              f"base_url={self.base_url}")

    def invoke(self,
               prompt:  str,
               system:  Optional[str] = None,
               history: Optional[List[dict]] = None) -> str:
        """
        Call LLM. Supports multi-turn conversation via `history`.

        history: list of {"role": "user"/"assistant", "content": "..."} dicts
                 (acts as single-agent memory across cases)
        """
        messages: List[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type":  "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {"model": self.model, "messages": messages}

        last_exc = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=180)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except Exception as exc:
                last_exc = exc
                wait = _RETRY_BASE * (2 ** (attempt - 1))
                print(f"[EvalLLM] Attempt {attempt}/{_MAX_RETRIES} failed: {exc}. "
                      f"Retrying in {wait}s...")
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(f"LLM failed after {_MAX_RETRIES} retries: {last_exc}")

    def invoke_json(self,
                    prompt:  str,
                    system:  Optional[str] = None,
                    history: Optional[List[dict]] = None) -> dict:
        """Call LLM and parse JSON from the response."""
        raw = self.invoke(prompt, system, history)
        return _extract_json(raw)


# ============================================================
# JSON extraction helper
# ============================================================

def _extract_json(text: str) -> dict:
    """Find and parse the last complete JSON object in text."""
    # Find last closing brace
    end = text.rfind("}")
    if end < 0:
        return {"_parse_error": True, "raw": text[:300]}

    # Walk backwards to find matching opening brace
    depth = 0
    for i in range(end, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                candidate = text[i:end + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    break

    # Fallback: try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"_parse_error": True, "raw": text[:300]}


# ============================================================
# Teaching prompt for Forward Simulation
# ============================================================

SYSTEM_FORWARD_SIM = """\
You are simulating the CF-CBN (Counterfactual Causal Bayesian Network) fault diagnosis model.
Given anomaly data from a microservice system, predict which service is the root cause.

CRITICAL RULES:
1. Do NOT write or execute any code. Use step-by-step mental arithmetic only.
2. Show every computation step clearly before the final JSON.
3. Output ONE JSON object at the very end — nothing after it.

=== SERVICE NAME NORMALIZATION ===
When reading anomaly_services and anomaly_pods keys:
  • Strip trailing pod-index suffix: "auth-0", "auth-1" → "auth"
  • Also strip "ts-" prefix AND "-service"/"-mongo"/"-other-service" suffix to match base names
    Example: "ts-auth-service" → strip "ts-" prefix and "-service" suffix → "auth"
    Example: "ts-auth-mongo"   → strip "ts-" and "-mongo" → "auth"
  • Only keep services whose base name appears in the MODEL SERVICE LIST provided.
  • Merge pod metrics into the same base service.

=== METRIC WEIGHT TABLE ===
pod_cpu_usage / cpu_usage                         : 4.0
pod_memory_working_set_bytes / memory_usage       : 3.5
rrt / rrt_max                                     : 3.5
pod_processes                                     : 3.0
timeout                                           : 3.0
client_error / client_error_ratio                 : 2.5
server_error / server_error_ratio                 : 2.5
error / error_ratio                               : 2.0
pod_network_transmit_packets                      : 1.5
pod_network_receive_packets                       : 1.5
pod_network_transmit_bytes                        : 1.0
pod_network_receive_bytes                         : 1.0
request / response                                : 0.5
[any other metric, e.g. "cpu","mem","latency-*"]  : 1.0

=== CF-CBN ALGORITHM (10 steps) ===

STEP 1: Normalize each anomaly_services/anomaly_pods key to its base name.
        Collect the union of metrics for each recognized service.

STEP 2: For each recognized service s with ≥1 anomalous metric, compute row_norm:
          row_s = 3.0 × sqrt( sum of weight_m² over all anomalous metrics m )
        Services with no recognized metrics: row_s = 0.

STEP 3: Compute total_norm:
          total = sqrt( sum of row_s² over ALL services in the model list )

STEP 4: Compute CF score for each service s:
          CF[s] = total − sqrt( total² − row_s² )
        (Services with row_s = 0 have CF[s] = 0)

STEP 5: Compute direct score:
          direct[s] = sum of weight_m over each anomalous metric m of service s

STEP 6: Compute unsupervised combined score:
          unsup[s] = CF[s] + 0.3 × direct[s]

STEP 7: Normalize unsup scores to [0, 1]:
          min_u = min(unsup values); max_u = max(unsup values)
          If max_u > min_u: unsup_norm[s] = (unsup[s] − min_u) / (max_u − min_u)
          Else (all equal, e.g. all zero): unsup_norm[s] = 1.0 / n_services

STEP 8: Apply CBN prior (if n_history > 0):
          prior[s] = (historical_count[s] + 0.05) / (n_history + 0.05 × n_services)
          fused[s] = alpha × unsup_norm[s] + (1 − alpha) × prior[s]
        If n_history = 0: fused[s] = unsup_norm[s]

STEP 9: Rank all services by fused[s] descending.

STEP 10: Output the JSON below. Include ALL services with anomalous metrics (row_s > 0).

=== OUTPUT FORMAT (JSON only — nothing after this block) ===
{
  "total_norm": <float, 2 decimal places>,
  "alpha_used": <float>,
  "n_history": <int>,
  "rankings": [
    {
      "service":    "<base_name>",
      "row_norm":   <float, 4 dp>,
      "cf_score":   <float, 4 dp>,
      "direct":     <float, 4 dp>,
      "unsup":      <float, 4 dp>,
      "unsup_norm": <float, 4 dp>,
      "prior":      <float, 4 dp>,
      "fused_score":<float, 4 dp>
    }
  ],
  "top1_prediction": "<service_name>"
}
Sort rankings by fused_score descending.\
"""


# ============================================================
# Prompt for Counterfactual Simulation
# ============================================================

SYSTEM_COUNTERFACTUAL = """\
You are analyzing a counterfactual scenario for the CF-CBN fault diagnosis model.
The model has made a prediction. Your job: suggest what changes to the anomaly input
would cause the model to predict a DIFFERENT service as the root cause.

RULES:
1. Do NOT write or run code. Reason step-by-step.
2. Output ONE JSON object at the very end — nothing after it.

=== HOW CF-CBN SCORES SERVICES ===
Each service s gets a combined score:
  • CF score: measures how much total anomaly magnitude drops when service s's metrics
    are zeroed out. Services with MORE high-weight metrics have BIGGER CF scores.
  • Direct score: sum of metric weights for service s (multiplied by 0.3 as bonus).

HIGH-WEIGHT metrics (most impactful to remove/add):
  pod_cpu_usage=4.0, pod_memory_working_set_bytes=3.5, rrt/rrt_max=3.5,
  pod_processes=3.0, timeout=3.0, error metrics=2.0–2.5
LOW-WEIGHT: network traffic=1.0–1.5, request/response=0.5
UNKNOWN metrics (e.g. "cpu","mem","latency-*"): weight=1.0

The service with the HIGHEST combined score is predicted as root cause.

=== HOW TO CHANGE THE PREDICTION FROM SERVICE X ===
Strategy A: Reduce X's score — remove its high-weight anomaly metrics
Strategy B: Boost another service — add high-weight metrics to it
Strategy C: Combine A+B, or simply remove service X entirely

=== OUTPUT FORMAT (JSON only — nothing after this block) ===
{
  "current_top1":   "<service X that is currently predicted>",
  "target_service": "<service you want to become top-1 instead>",
  "reasoning":      "<step-by-step explanation of why changes will shift the scores>",
  "changes": [
    {"action": "remove_metrics", "service": "<name>", "metrics": ["<m1>", "<m2>"]},
    {"action": "add_metrics",    "service": "<name>", "metrics": ["<m1>", "<m2>"]},
    {"action": "remove_service", "service": "<name>"}
  ]
}
• Use exact service names as they appear in the anomaly_services / anomaly_pods data.
• For add_metrics: prefer high-weight standard names: pod_cpu_usage, rrt, pod_memory_working_set_bytes, error.\
"""


# ============================================================
# Prompt builders
# ============================================================

def build_forward_prompt(
    case:         dict,
    services:     List[str],
    alpha:        float,
    prior_counts: dict,
    n_history:    int,
    memory_ctx:   str = "",
) -> str:
    """Build user-turn prompt for the forward simulation task."""
    n_svc = len(services)
    lines = []

    # Memory context (last-N case outcomes as agent "memory")
    if memory_ctx:
        lines.append("=== MEMORY: Recent case outcomes ===")
        lines.append(memory_ctx)
        lines.append("")

    lines.append("=== CURRENT CASE TO PREDICT ===")
    lines.append("")
    lines.append(f"MODEL SERVICE LIST ({n_svc} recognized base names):")
    lines.append(", ".join(services))
    lines.append("")

    # Anomaly services (show only those with metrics)
    anom_svc  = case.get("anomaly_services", {})
    anom_pods = case.get("anomaly_pods", {})
    nonempty_svc  = {k: v for k, v in anom_svc.items()  if v}
    nonempty_pods = {k: v for k, v in anom_pods.items() if v}

    lines.append("anomaly_services (service → [anomalous metrics]):")
    if nonempty_svc:
        for svc, metrics in nonempty_svc.items():
            lines.append(f"  {svc}: {metrics}")
    else:
        lines.append("  (none with metrics)")

    lines.append("anomaly_pods (pod → [anomalous metrics]):")
    if nonempty_pods:
        for pod, metrics in nonempty_pods.items():
            lines.append(f"  {pod}: {metrics}")
    else:
        lines.append("  (none with metrics)")

    lines.append("")
    lines.append(f"CBN STATE: n_history={n_history}, alpha={alpha:.4f}")

    if n_history > 0 and prior_counts:
        lines.append("Prior root-cause counts (service → count so far):")
        top10 = sorted(prior_counts.items(), key=lambda x: -x[1])[:10]
        for svc, cnt in top10:
            lines.append(f"  {svc}: {int(cnt)}")
        if len(prior_counts) > 10:
            remaining = len(prior_counts) - 10
            lines.append(f"  ... ({remaining} more services with count ≤ {int(top10[-1][1])})")
    lines.append("")
    lines.append(
        "Now apply the CF-CBN algorithm step-by-step to this case. "
        "At the very end output ONE JSON object (nothing after it)."
    )
    return "\n".join(lines)


def build_counterfactual_prompt(
    case:         dict,
    top1_service: str,
    model_scores: Dict[str, float],
    memory_ctx:   str = "",
) -> str:
    """Build user-turn prompt for the counterfactual simulation task."""
    lines = []

    if memory_ctx:
        lines.append("=== MEMORY: Recent case outcomes ===")
        lines.append(memory_ctx)
        lines.append("")

    lines.append("=== COUNTERFACTUAL TASK ===")
    lines.append("")
    lines.append(f"CURRENT MODEL PREDICTION: {top1_service}")
    lines.append("")

    # Show top-3 scores as context
    top3 = sorted(model_scores.items(), key=lambda x: -x[1])[:3]
    lines.append("Current top-3 model scores:")
    for svc, sc in top3:
        lines.append(f"  {svc}: {sc:.4f}")
    lines.append("")

    # Show anomaly data
    anom_svc  = case.get("anomaly_services", {})
    anom_pods = case.get("anomaly_pods",     {})
    nonempty_svc  = {k: v for k, v in anom_svc.items()  if v}
    nonempty_pods = {k: v for k, v in anom_pods.items() if v}

    lines.append("anomaly_services:")
    if nonempty_svc:
        for svc, metrics in nonempty_svc.items():
            lines.append(f"  {svc}: {metrics}")
    else:
        lines.append("  (none with metrics)")

    lines.append("anomaly_pods:")
    if nonempty_pods:
        for pod, metrics in nonempty_pods.items():
            lines.append(f"  {pod}: {metrics}")
    else:
        lines.append("  (none with metrics)")

    lines.append("")
    lines.append(
        f"Your task: suggest changes to the anomaly data so that the model predicts "
        f"a service OTHER THAN '{top1_service}'. "
        "Output ONE JSON object at the very end (nothing after it)."
    )
    return "\n".join(lines)


# ============================================================
# Response parsers
# ============================================================

def parse_forward_response(raw: str, services: List[str]) -> Tuple[str, Dict[str, float]]:
    """
    Parse the agent's forward-simulation response.
    Returns (top1_prediction, {service: fused_score}) or ("unknown", {}) on failure.
    """
    data = _extract_json(raw)
    if data.get("_parse_error"):
        return ("unknown", {})

    top1   = data.get("top1_prediction", "unknown")
    scores: Dict[str, float] = {}

    for item in data.get("rankings", []):
        svc = item.get("service", "")
        sc  = item.get("fused_score", 0.0)
        if svc:
            try:
                scores[svc] = float(sc)
            except (TypeError, ValueError):
                pass

    return (top1, scores)


def parse_counterfactual_response(raw: str) -> Tuple[List[dict], str, str]:
    """
    Parse the agent's counterfactual response.
    Returns (changes_list, target_service, reasoning).
    """
    data = _extract_json(raw)
    if data.get("_parse_error"):
        return ([], "unknown", "")

    changes   = data.get("changes", [])
    target    = data.get("target_service", "unknown")
    reasoning = data.get("reasoning", "")

    # Validate change dicts
    valid_changes = []
    for ch in changes:
        if isinstance(ch, dict) and "action" in ch and "service" in ch:
            valid_changes.append(ch)

    return (valid_changes, target, reasoning)


# ============================================================
# Change application (counterfactual)
# ============================================================

def _norm_name(s: str) -> str:
    """Normalize a service/pod name for fuzzy matching."""
    s = s.lower().strip()
    s = re.sub(r"-\d+$", "", s)          # strip pod index
    if s.startswith("ts-"):
        s = s[3:]                         # strip ts- prefix
    for suffix in ("-service", "-mongo", "-other-service", "-other"):
        if s.endswith(suffix):
            s = s[:-len(suffix)]
            break
    return s


def _find_key(target: str, d: dict) -> Optional[str]:
    """Find key in dict matching target by exact or normalized name."""
    if target in d:
        return target
    norm_t = _norm_name(target)
    for key in d:
        if _norm_name(key) == norm_t:
            return key
    return None


def apply_changes(case_orig: dict, changes: List[dict]) -> dict:
    """
    Apply counterfactual changes to a deep copy of case_orig.
    Modifies anomaly_services and anomaly_pods in place.
    """
    case     = copy.deepcopy(case_orig)
    anom_svc = case.setdefault("anomaly_services", {})
    anom_pod = case.setdefault("anomaly_pods",     {})

    for ch in changes:
        action  = ch.get("action",  "")
        target  = ch.get("service", "")
        metrics = ch.get("metrics", [])

        if action == "remove_service":
            key = _find_key(target, anom_svc)
            if key:
                del anom_svc[key]
            # Also remove matching pods
            for pod_key in list(anom_pod.keys()):
                if _norm_name(pod_key) == _norm_name(target):
                    del anom_pod[pod_key]

        elif action == "remove_metrics":
            key = _find_key(target, anom_svc)
            if key:
                anom_svc[key] = [m for m in anom_svc[key] if m not in metrics]
            for pod_key in list(anom_pod.keys()):
                if _norm_name(pod_key) == _norm_name(target):
                    anom_pod[pod_key] = [m for m in anom_pod[pod_key] if m not in metrics]

        elif action == "add_metrics":
            key = _find_key(target, anom_svc)
            if key:
                existing = set(anom_svc[key])
                anom_svc[key] = list(existing | set(metrics))
            else:
                # Create new entry using the target name as provided
                anom_svc[target] = list(metrics)

    return case


# ============================================================
# Distribution and KL divergence
# ============================================================

def to_distribution(
    score_dict: Dict[str, float],
    services:   List[str],
    eps:        float = 0.01,
) -> np.ndarray:
    """
    Convert a score dict to a probability distribution over `services`.
    Services not in score_dict get 0 before eps smoothing.
    Returns a normalized array (sum=1).
    """
    arr = np.array(
        [max(score_dict.get(s, 0.0), 0.0) for s in services],
        dtype=np.float64,
    )
    # If all zeros, return uniform
    if arr.sum() < 1e-12:
        return np.ones(len(services), dtype=np.float64) / len(services)
    # Add eps smoothing and normalize
    arr = arr + eps
    return arr / arr.sum()


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL(P ∥ Q) = Σ P_i log(P_i / Q_i).
    Clips both arrays to [eps, 1] before computing.
    """
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


# ============================================================
# Progress tracking (crash-safe append-only JSONL)
# ============================================================

def load_progress(out_dir: str) -> int:
    """Return last completed case index, or -1 if none."""
    path = os.path.join(out_dir, "progress.json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("last_done", -1)
        except Exception:
            pass
    return -1


def save_progress(out_dir: str, last_done: int):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "progress.json"), "w", encoding="utf-8") as f:
        json.dump({"last_done": last_done}, f)


def append_record(out_dir: str, record: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "records.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(out_dir: str) -> List[dict]:
    path = os.path.join(out_dir, "records.jsonl")
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ============================================================
# Early stopping and memory
# ============================================================

def early_stop_check(
    eval_records: List[dict],
    mode:         str,
    n_check:      int   = 20,
    threshold:    float = 0.05,
) -> bool:
    """
    Return True if the first n_check evaluated cases all show poor performance.

    mode="forward":        checks top1_match rate
    mode="counterfactual": checks success rate

    Triggers only once: when len(eval_records) == n_check exactly.
    """
    if len(eval_records) != n_check:
        return False

    first_n = eval_records[:n_check]
    if mode == "forward":
        score = sum(1 for r in first_n if r.get("top1_match")) / n_check
    else:
        score = sum(1 for r in first_n if r.get("success")) / n_check

    return score < threshold


def build_memory_ctx(records: List[dict], n: int = 3) -> str:
    """
    Build a memory context string from the last n non-skipped records.
    This is passed to the LLM prompt so the agent can learn from recent cases.
    """
    recent = [r for r in records if not r.get("skipped")][-n:]
    if not recent:
        return ""
    lines = []
    for r in recent:
        if "model_top1" in r:          # forward sim
            lines.append(
                f"Case #{r['index']}: model→{r.get('model_top1','?')} | "
                f"agent→{r.get('agent_top1','?')} | match={r.get('top1_match','?')} | "
                f"KL={r.get('kl_div', '?')}"
            )
        elif "original_top1" in r:     # counterfactual
            lines.append(
                f"Case #{r['index']}: original→{r.get('original_top1','?')} | "
                f"new→{r.get('new_top1','?')} | "
                f"success={r.get('success','?')}"
            )
    return "\n".join(lines)


# ============================================================
# Summary report
# ============================================================

def write_summary(out_dir: str, all_records: List[dict], mode: str) -> str:
    """Write a human-readable summary.txt and return its text."""
    eval_recs = [r for r in all_records if not r.get("skipped")]
    skipped   = sum(1 for r in all_records if r.get("skipped"))
    n         = max(len(eval_recs), 1)
    W         = 65

    lines = [
        "=" * W,
        f"  CF-CBN Interpretability Evaluation — {mode.replace('_', ' ').title()}",
        f"  Ref: Doshi-Velez & Kim (2017) §3.2 Human-grounded Evaluation",
        "=" * W,
        "",
        f"  Total cases   : {len(all_records)}",
        f"  Evaluated     : {len(eval_recs)}",
        f"  Skipped       : {skipped}",
        "",
    ]

    if mode == "forward":
        top1_n = sum(1 for r in eval_recs if r.get("top1_match"))
        top3_n = sum(1 for r in eval_recs if r.get("top3_match"))
        kl_vals = [r["kl_div"] for r in eval_recs
                   if "kl_div" in r and not math.isnan(r["kl_div"]) and r["kl_div"] < 9]
        avg_kl  = sum(kl_vals) / max(len(kl_vals), 1)

        lines += [
            "─" * W,
            "  Forward Simulation Metrics",
            "─" * W,
            f"  Top-1 match rate : {top1_n}/{n} = {top1_n/n:.2%}",
            f"  Top-3 match rate : {top3_n}/{n} = {top3_n/n:.2%}",
            f"  Avg KL divergence: {avg_kl:.4f}  (lower = better alignment)",
            "",
            "─" * W,
            "  Per-Case Detail",
            "─" * W,
            f"  {'#':>4}  {'Match':>5}  {'KL':>7}  {'α':>5}  "
            f"{'model_top1':<22}  {'agent_top1':<22}",
            "  " + "─" * (W - 2),
        ]
        for r in eval_recs:
            m_sym = "T" if r.get("top1_match") else "F"
            kl    = r.get("kl_div", float("nan"))
            kl_s  = f"{kl:.3f}" if not math.isnan(kl) else "  NaN"
            lines.append(
                f"  {r['index']:>4}  {m_sym:>5}  {kl_s:>7}  "
                f"{r.get('alpha', 0):>5.3f}  "
                f"{str(r.get('model_top1','?')):<22}  "
                f"{str(r.get('agent_top1','?')):<22}"
            )

    elif mode == "counterfactual":
        succ_n = sum(1 for r in eval_recs if r.get("success"))
        lines += [
            "─" * W,
            "  Counterfactual Simulation Metrics",
            "─" * W,
            f"  Success rate : {succ_n}/{n} = {succ_n/n:.2%}",
            f"  (success = agent changes caused model to predict a different service)",
            "",
            "─" * W,
            "  Per-Case Detail",
            "─" * W,
            f"  {'#':>4}  {'OK':>4}  {'original':<22}  {'new':<22}  {'target':<22}",
            "  " + "─" * (W - 2),
        ]
        for r in eval_recs:
            sym = "OK" if r.get("success") else "--"
            lines.append(
                f"  {r['index']:>4}  {sym:>4}  "
                f"{str(r.get('original_top1','?')):<22}  "
                f"{str(r.get('new_top1','?')):<22}  "
                f"{str(r.get('agent_target','?')):<22}"
            )

    lines += ["", "=" * W, ""]
    text = "\n".join(lines)

    path = os.path.join(out_dir, "summary.txt")
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"\n[Summary] Written → {os.path.abspath(path)}")
    # Print header lines to console
    for ln in lines[:20]:
        print(ln)
    return text
