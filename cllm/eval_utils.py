"""
eval_utils.py — Shared utility module for CF-CBN interpretability evaluation scripts.

Provides:
  - EvalLLMClient: OpenAI-compatible API client with retry logic (apiyi/openai/volc platforms)
  - SYSTEM_FORWARD_SIM / SYSTEM_COUNTERFACTUAL: system prompt constants
  - build_forward_prompt / build_counterfactual_prompt: prompt builders
  - parse_forward_response / parse_counterfactual_response: structured JSON parsers
  - apply_changes: apply counterfactual edits to a case dict
  - to_distribution / kl_divergence: metric helpers
  - load_progress / save_progress / append_record / load_records: progress tracking
  - early_stop_check: abort evaluation if performance is below threshold
  - build_memory_ctx: build LLM memory context from recent records
  - write_summary: write summary.txt for forward or counterfactual evaluation

Imported by eval_forward_simulation.py and eval_counterfactual_simulation.py.
"""

import os
import sys
import json
import math
import re
import copy
import time

import requests
import numpy as np

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

EVAL_PLATFORM = "apiyi"
EVAL_MODEL = "gpt-4o"
EVAL_API_KEY = os.environ.get("EVAL_LLM_API_KEY", "sk-your-apiyi-key-here")

_PLATFORM_BASE_URLS = {
    "apiyi":  "https://api.apiyi.com/v1",
    "openai": "https://api.openai.com/v1",
    "volc":   "https://ark.cn-beijing.volces.com/api/v3",
}


# ---------------------------------------------------------------------------
# EvalLLMClient
# ---------------------------------------------------------------------------


class EvalLLMClient:
    """Standalone OpenAI-compatible LLM client for interpretability evaluation.

    Supports apiyi, openai, and volc platforms with automatic retry logic.
    """

    def __init__(self, platform: str = "apiyi", model: str = "gpt-4o", api_key: str = None):
        self.platform = platform.lower()
        self.model = model
        self.api_key = api_key or EVAL_API_KEY
        base_url = _PLATFORM_BASE_URLS.get(self.platform)
        if base_url is None:
            raise ValueError(
                f"Unknown platform '{platform}'. "
                f"Choose from: {list(_PLATFORM_BASE_URLS.keys())}"
            )
        self.chat_url = base_url.rstrip("/") + "/chat/completions"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, prompt: str, system: str = None, history: list = None) -> list:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            for turn in history:
                messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _call_api(self, messages: list) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model, "messages": messages}
        response = requests.post(self.chat_url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _retry_call(self, messages: list, retries: int = 3) -> str:
        last_exc = None
        for attempt in range(retries):
            try:
                return self._call_api(messages)
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt  # 1s, 2s, 4s
                print(
                    f"[EvalLLMClient] Attempt {attempt + 1}/{retries} failed: {exc}. "
                    f"Retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
        raise RuntimeError(
            f"All {retries} retries failed. Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(self, prompt: str, system: str = None, history: list = None) -> str:
        """Send a prompt and return the raw string response.

        Args:
            prompt:  User-turn content.
            system:  Optional system message (prepended).
            history: Optional list of prior turns as {"role": ..., "content": ...} dicts.

        Returns:
            Assistant response as a plain string.
        """
        messages = self._build_messages(prompt, system=system, history=history)
        return self._retry_call(messages)

    def invoke_json(self, prompt: str, system: str = None, history: list = None) -> dict:
        """Send a prompt and parse the response as JSON.

        Handles Markdown code fences and finds the last top-level {...} block.

        Returns:
            Parsed dict.  Raises ValueError if JSON cannot be extracted.
        """
        raw = self.invoke(prompt, system=system, history=history)
        return _extract_last_json(raw)


# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------


def _extract_last_json(raw: str) -> dict:
    """Extract the last top-level JSON object from a raw string.

    Handles Markdown code fences (```json ... ```) and finds the outermost
    ``{...}`` block that ends at the last ``}`` in the text.

    Strategy: start from the last ``}`` and scan left counting braces until
    the matching ``{`` is found, then try to parse that substring.

    Raises:
        ValueError: if no valid JSON object can be found.
    """
    # Strip Markdown code fences
    text = re.sub(r"```(?:json)?\s*", "", raw)
    text = text.replace("```", "")

    # Find the last '}'
    last_close = text.rfind("}")
    if last_close == -1:
        raise ValueError("No JSON object found in response (no closing brace).")

    # Walk backwards from last_close to find the matching opening brace
    depth = 0
    start = -1
    for i in range(last_close, -1, -1):
        if text[i] == "}":
            depth += 1
        elif text[i] == "{":
            depth -= 1
            if depth == 0:
                start = i
                break

    if start == -1:
        raise ValueError("No JSON object found in response (unmatched brace).")

    candidate = text[start : last_close + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse JSON: {exc}\nCandidate text:\n{candidate}"
        ) from exc


# ---------------------------------------------------------------------------
# System prompt constants
# ---------------------------------------------------------------------------

SYSTEM_FORWARD_SIM = (
    "You are simulating the CF-CBN (Counterfactual Causal Bayesian Network) fault diagnosis model.\n"
    "Given anomaly data from a microservice system, predict which service is the root cause.\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. Do NOT write or execute any code. Use step-by-step mental arithmetic only.\n"
    "2. Show computation steps clearly before the final JSON.\n"
    "3. Output ONE JSON object at the very end \u2014 nothing after it.\n"
    "\n"
    "=== SERVICE NAME NORMALIZATION ===\n"
    "When reading anomaly_services and anomaly_pods keys:\n"
    '- Strip trailing pod-index suffix: "auth-0", "auth-1" \u2192 "auth"\n'
    '- Also strip "ts-" prefix AND "-service"/"-mongo"/"-other-service" suffix to match base names\n'
    '  Example: "ts-auth-service" \u2192 strip "ts-" and "-service" \u2192 "auth"\n'
    '  Example: "ts-auth-mongo" \u2192 "auth"\n'
    "- Only keep services whose base name appears in the MODEL SERVICE LIST provided.\n"
    '- Merge pod metrics into the same service (e.g., "auth-0" and "auth-1" \u2192 "auth")\n'
    "\n"
    "=== METRIC WEIGHT TABLE ===\n"
    "pod_cpu_usage / cpu_usage                        : 4.0\n"
    "pod_memory_working_set_bytes / memory_usage      : 3.5\n"
    "rrt / rrt_max                                    : 3.5\n"
    "pod_processes                                    : 3.0\n"
    "timeout                                          : 3.0\n"
    "client_error / client_error_ratio                : 2.5\n"
    "server_error / server_error_ratio                : 2.5\n"
    "error / error_ratio                              : 2.0\n"
    "pod_network_transmit_packets                     : 1.5\n"
    "pod_network_receive_packets                      : 1.5\n"
    "pod_network_transmit_bytes                       : 1.0\n"
    "pod_network_receive_bytes                        : 1.0\n"
    "request / response                               : 0.5\n"
    '[any other metric, e.g. "cpu","mem","latency-*"] : 1.0\n'
    "\n"
    "=== CF-CBN ALGORITHM (10 steps) ===\n"
    "\n"
    "STEP 1: Map each anomaly entry to its base service name (see normalization above).\n"
    "Collect the union of metrics for each recognized service from both anomaly_services and anomaly_pods.\n"
    "\n"
    "STEP 2: For each recognized service s with at least one metric, compute row_norm:\n"
    "  row_s = sqrt( sum over each anomalous metric m of: (3.0 \u00d7 weight_m)^2 )\n"
    "        = 3.0 \u00d7 sqrt( sum of weight_m^2 )\n"
    "Services with no recognized metrics: row_s = 0.\n"
    "\n"
    "STEP 3: Compute total_norm:\n"
    "  total = sqrt( sum over ALL services s of: row_s^2 )\n"
    "(Include all services in the model service list, most have row_s=0)\n"
    "\n"
    "STEP 4: Compute CF score for each service s:\n"
    "  CF[s] = total \u2212 sqrt( total^2 \u2212 row_s^2 )\n"
    "(Services with row_s=0: CF[s]=0)\n"
    "\n"
    "STEP 5: Compute direct score:\n"
    "  direct[s] = sum of weight_m for each anomalous metric m of service s\n"
    "\n"
    "STEP 6: Compute unsupervised score:\n"
    "  unsup[s] = CF[s] + 0.3 \u00d7 direct[s]\n"
    "\n"
    "STEP 7: Normalize to [0,1]:\n"
    "  min_u = min(unsup values over all services)\n"
    "  max_u = max(unsup values over all services)\n"
    "  If max_u > min_u: unsup_norm[s] = (unsup[s] \u2212 min_u) / (max_u \u2212 min_u)\n"
    "  Else (all equal): unsup_norm[s] = 1.0 / n_services\n"
    "\n"
    "STEP 8: Apply CBN prior (if n_history > 0):\n"
    "  prior[s] = (historical_count[s] + 0.05) / (n_history + 0.05 \u00d7 n_services)\n"
    "  fused[s] = alpha \u00d7 unsup_norm[s] + (1\u2212alpha) \u00d7 prior[s]\n"
    "  If n_history = 0: fused[s] = unsup_norm[s]\n"
    "\n"
    "STEP 9: Rank all services by fused[s] descending.\n"
    "\n"
    "STEP 10: Output the JSON below. Include ALL services with anomalous metrics (those with row_s > 0).\n"
    "\n"
    "=== OUTPUT FORMAT (JSON only \u2014 nothing after this block) ===\n"
    "{\n"
    '  "total_norm": <float, 2 decimal places>,\n'
    '  "alpha_used": <float>,\n'
    '  "n_history": <int>,\n'
    '  "rankings": [\n'
    "    {\n"
    '      "service": "<base_name>",\n'
    '      "row_norm": <float, 4 decimal places>,\n'
    '      "cf_score": <float, 4 decimal places>,\n'
    '      "direct": <float, 4 decimal places>,\n'
    '      "unsup": <float, 4 decimal places>,\n'
    '      "unsup_norm": <float, 4 decimal places>,\n'
    '      "prior": <float, 4 decimal places>,\n'
    '      "fused_score": <float, 4 decimal places>\n'
    "    }\n"
    "  ],\n"
    '  "top1_prediction": "<service_name>"\n'
    "}\n"
    "Sort rankings by fused_score descending."
)

SYSTEM_COUNTERFACTUAL = (
    "You are analyzing a counterfactual scenario for the CF-CBN fault diagnosis model.\n"
    "The model has made a prediction. Your job: suggest what changes to the anomaly input\n"
    "would cause the model to predict a DIFFERENT service as the root cause.\n"
    "\n"
    "RULES:\n"
    "1. Do NOT write or run code. Reason step-by-step.\n"
    "2. Output ONE JSON object at the very end.\n"
    "\n"
    "=== HOW CF-CBN SCORES SERVICES ===\n"
    "Each service s gets a score based on:\n"
    "  1. CF score: how much total anomaly magnitude drops if service s\u2019s metrics are zeroed out\n"
    "     \u2192 Services with more high-weight metrics have bigger CF scores\n"
    "  2. Direct score: sum of metric weights for service s (\u00d70.3 bonus)\n"
    "\n"
    "HIGH-WEIGHT METRICS (most impactful): pod_cpu_usage=4.0, pod_memory=3.5, rrt/rrt_max=3.5\n"
    "MEDIUM: pod_processes=3.0, timeout=3.0, error metrics=2.0-2.5\n"
    'LOW: network=1.0-1.5, request/response=0.5\n'
    'UNKNOWN metrics (e.g. "cpu","mem","latency-*"): weight=1.0\n'
    "\n"
    "The service with the HIGHEST combined score is predicted as root cause.\n"
    "\n"
    "=== TO CHANGE THE PREDICTION FROM SERVICE X ===\n"
    "Strategy A: Reduce X\u2019s score \u2014 remove its high-weight metrics (especially cpu/memory/rrt)\n"
    "Strategy B: Boost another service\u2019s score \u2014 add high-weight metrics to it\n"
    "Strategy C: Combine A+B or simply remove service X entirely\n"
    "\n"
    "=== OUTPUT FORMAT ===\n"
    "{\n"
    '  "current_top1": "<service X>",\n'
    '  "target_service": "<service you want to become top-1>",\n'
    '  "reasoning": "<why these changes reduce X\'s score / boost target\'s score>",\n'
    '  "changes": [\n'
    '    {"action": "remove_metrics", "service": "<name>", "metrics": ["<m1>","<m2>"]},\n'
    '    {"action": "add_metrics",    "service": "<name>", "metrics": ["<m1>","<m2>"]},\n'
    '    {"action": "remove_service", "service": "<name>"}\n'
    "  ]\n"
    "}\n"
    "Use exact service names as they appear in the anomaly data.\n"
    "For add_metrics: use standard metric names like pod_cpu_usage, rrt, error, pod_memory_working_set_bytes."
)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_forward_prompt(
    case: dict,
    services: list,
    alpha: float,
    prior_counts: dict,
    n_history: int,
    memory_ctx: str = "",
) -> str:
    """Build the user-turn prompt for forward simulation.

    Args:
        case:         A single anomaly case dict with keys anomaly_services, anomaly_pods, etc.
        services:     Full list of service names in the model.
        alpha:        Fusion weight (0-1) for CBN prior.
        prior_counts: Dict mapping service name to historical fault count.
        n_history:    Number of historical cases seen so far.
        memory_ctx:   Optional memory context string prepended to the prompt.

    Returns:
        Formatted prompt string.
    """
    lines = []

    if memory_ctx:
        lines.append("=== MEMORY CONTEXT (recent examples) ===")
        lines.append(memory_ctx)
        lines.append("")

    lines.append("=== MODEL SERVICE LIST ===")
    lines.append(", ".join(services))
    lines.append("")

    lines.append("=== ANOMALY DATA ===")

    anomaly_services = case.get("anomaly_services", {})
    anomaly_pods = case.get("anomaly_pods", {})
    svc_entries = {k: v for k, v in anomaly_services.items() if v}
    pod_entries = {k: v for k, v in anomaly_pods.items() if v}

    lines.append("anomaly_services:")
    if svc_entries:
        for svc, metrics in svc_entries.items():
            lines.append(f"  {svc}: {json.dumps(metrics)}")
    else:
        lines.append("  (none)")

    lines.append("anomaly_pods:")
    if pod_entries:
        for pod, metrics in pod_entries.items():
            lines.append(f"  {pod}: {json.dumps(metrics)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append("=== CBN STATE ===")
    lines.append(f"n_history : {n_history}")
    lines.append(f"alpha     : {alpha}")

    if n_history > 0 and prior_counts:
        sorted_counts = sorted(prior_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        lines.append("prior_counts (top-10):")
        for svc, cnt in sorted_counts:
            lines.append(f"  {svc}: {cnt}")

    lines.append("")
    lines.append(
        "Apply the CF-CBN algorithm (all 10 steps) to the anomaly data above "
        "and output the result JSON."
    )
    return "\n".join(lines)


def build_counterfactual_prompt(
    case: dict,
    top1_service: str,
    model_scores: dict,
    memory_ctx: str = "",
) -> str:
    """Build the user-turn prompt for counterfactual analysis.

    Args:
        case:          The anomaly case dict.
        top1_service:  The service currently predicted as root cause.
        model_scores:  Dict mapping service name to fused_score.
        memory_ctx:    Optional memory context string prepended to the prompt.

    Returns:
        Formatted prompt string.
    """
    lines = []

    if memory_ctx:
        lines.append("=== MEMORY CONTEXT ===")
        lines.append(memory_ctx)
        lines.append("")

    lines.append("=== CURRENT PREDICTION ===")
    lines.append(f"top1_prediction: {top1_service}")
    lines.append("")

    sorted_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    lines.append("=== TOP-3 SCORES (for context) ===")
    for rank, (svc, score) in enumerate(sorted_scores, 1):
        lines.append(f"  #{rank}  {svc}: {score:.4f}")
    lines.append("")

    lines.append("=== FULL ANOMALY DATA ===")
    anomaly_services = case.get("anomaly_services", {})
    anomaly_pods = case.get("anomaly_pods", {})
    svc_entries = {k: v for k, v in anomaly_services.items() if v}
    pod_entries = {k: v for k, v in anomaly_pods.items() if v}

    lines.append("anomaly_services:")
    if svc_entries:
        for svc, metrics in svc_entries.items():
            lines.append(f"  {svc}: {json.dumps(metrics)}")
    else:
        lines.append("  (none)")

    lines.append("anomaly_pods:")
    if pod_entries:
        for pod, metrics in pod_entries.items():
            lines.append(f"  {pod}: {json.dumps(metrics)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(
        "Task: Suggest the minimal set of changes to the anomaly_services / anomaly_pods data "
        "that would cause CF-CBN to predict a DIFFERENT service as the root cause. "
        "Output the result JSON."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------


def parse_forward_response(raw: str, services: list) -> tuple:
    """Parse the LLM forward-simulation response.

    Args:
        raw:      Raw LLM response string.
        services: Full list of model service names.

    Returns:
        Tuple of (top1: str, scores: dict).
        On any parse failure returns ("unknown", {}).
    """
    try:
        data = _extract_last_json(raw)
        top1 = str(data.get("top1_prediction", "unknown")).strip()
        scores = {}
        for entry in data.get("rankings", []):
            svc = str(entry.get("service", "")).strip()
            fused = float(entry.get("fused_score", 0.0))
            if svc:
                scores[svc] = fused
        return top1, scores
    except Exception as exc:
        print(f"[parse_forward_response] Parse failure: {exc}", file=sys.stderr)
        return "unknown", {}


def parse_counterfactual_response(raw: str) -> list:
    """Parse the LLM counterfactual response.

    Args:
        raw: Raw LLM response string.

    Returns:
        List of change dicts.  Returns [] on any parse failure.
    """
    try:
        data = _extract_last_json(raw)
        changes = data.get("changes", [])
        if not isinstance(changes, list):
            return []
        return changes
    except Exception as exc:
        print(f"[parse_counterfactual_response] Parse failure: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Case mutation helpers (counterfactual)
# ---------------------------------------------------------------------------


def _normalize_name(s: str) -> str:
    """Normalize a service/pod name for fuzzy matching.

    Steps:
      1. Strip leading "ts-" prefix.
      2. Strip trailing "-service", "-mongo", "-other-service" suffixes.
      3. Strip trailing "-<digit(s)>" pod-index suffix.
    """
    s = s.strip()
    if s.startswith("ts-"):
        s = s[3:]
    for suffix in ("-other-service", "-service", "-mongo"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    s = re.sub(r"-\d+$", "", s)
    return s


def _find_key(target: str, svc_dict: dict):
    """Find the actual key in svc_dict that matches target.

    Tries exact match first, then normalized match.

    Returns:
        The matching key string, or None if not found.
    """
    if target in svc_dict:
        return target
    norm_target = _normalize_name(target)
    for key in svc_dict:
        if _normalize_name(key) == norm_target:
            return key
    return None


def apply_changes(case_orig: dict, changes: list) -> dict:
    """Apply a list of counterfactual changes to a case dict.

    Args:
        case_orig: Original case dict (will NOT be mutated).
        changes:   List of change dicts as returned by parse_counterfactual_response.

    Returns:
        A deep copy of the case with the changes applied.
    """
    case = copy.deepcopy(case_orig)
    anomaly_services = case.setdefault("anomaly_services", {})
    anomaly_pods = case.setdefault("anomaly_pods", {})

    for change in changes:
        action = change.get("action", "")
        svc_name = change.get("service", "")

        if action == "remove_metrics":
            metrics_to_remove = change.get("metrics", [])
            key = _find_key(svc_name, anomaly_services)
            if key is not None:
                if isinstance(anomaly_services[key], list):
                    anomaly_services[key] = [
                        m for m in anomaly_services[key] if m not in metrics_to_remove
                    ]
                elif isinstance(anomaly_services[key], dict):
                    for m in metrics_to_remove:
                        anomaly_services[key].pop(m, None)
            norm_target = _normalize_name(svc_name)
            for pod_key in list(anomaly_pods.keys()):
                if _normalize_name(pod_key) == norm_target:
                    if isinstance(anomaly_pods[pod_key], list):
                        anomaly_pods[pod_key] = [
                            m for m in anomaly_pods[pod_key] if m not in metrics_to_remove
                        ]
                    elif isinstance(anomaly_pods[pod_key], dict):
                        for m in metrics_to_remove:
                            anomaly_pods[pod_key].pop(m, None)

        elif action == "add_metrics":
            metrics_to_add = change.get("metrics", [])
            key = _find_key(svc_name, anomaly_services)
            if key is None:
                anomaly_services[svc_name] = list(metrics_to_add)
            else:
                existing = anomaly_services[key]
                if isinstance(existing, list):
                    for m in metrics_to_add:
                        if m not in existing:
                            existing.append(m)
                elif isinstance(existing, dict):
                    for m in metrics_to_add:
                        if m not in existing:
                            existing[m] = 1

        elif action == "remove_service":
            key = _find_key(svc_name, anomaly_services)
            if key is not None:
                del anomaly_services[key]
            norm_target = _normalize_name(svc_name)
            for pod_key in list(anomaly_pods.keys()):
                if _normalize_name(pod_key) == norm_target:
                    del anomaly_pods[pod_key]

    return case


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def to_distribution(score_dict: dict, services: list, eps: float = 0.01) -> np.ndarray:
    """Convert a score dict to a probability distribution over all services.

    Non-listed services get 0 before eps addition.

    Args:
        score_dict: Dict mapping service name to numeric score.
        services:   Ordered list of all services (defines the array axis).
        eps:        Small constant added to every element before normalization.

    Returns:
        1-D numpy array of length len(services) summing to 1.
        If all zeros after eps addition: returns uniform.
    """
    arr = np.array([score_dict.get(svc, 0.0) for svc in services], dtype=float)
    arr = arr + eps
    total = arr.sum()
    if total <= 0.0:
        return np.ones(len(services), dtype=float) / len(services)
    return arr / total


def kl_divergence(p_arr: np.ndarray, q_arr: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL divergence KL(P || Q).

    Args:
        p_arr: Reference distribution (numpy array).
        q_arr: Approximate distribution (numpy array).
        eps:   Clipping floor to avoid log(0).

    Returns:
        Scalar KL divergence value.
    """
    p = np.clip(np.asarray(p_arr, dtype=float), eps, 1.0)
    q = np.clip(np.asarray(q_arr, dtype=float), eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

_PROGRESS_FILE = "progress.json"
_RECORDS_FILE = "records.jsonl"


def load_progress(out_dir: str) -> int:
    """Load the index of the last completed case.

    Returns:
        Last completed case index, or -1 if no progress file exists.
    """
    path = os.path.join(out_dir, _PROGRESS_FILE)
    if not os.path.isfile(path):
        return -1
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("last_done", -1))
    except Exception:
        return -1


def save_progress(out_dir: str, last_done: int) -> None:
    """Write the last completed case index to out_dir/progress.json."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _PROGRESS_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"last_done": last_done}, f)


def append_record(out_dir: str, record: dict) -> None:
    """Append a result record as a JSONL line to out_dir/records.jsonl."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, _RECORDS_FILE)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_records(out_dir: str) -> list:
    """Load all records from out_dir/records.jsonl.

    Returns:
        List of record dicts.  Returns [] if the file does not exist.
    """
    path = os.path.join(out_dir, _RECORDS_FILE)
    if not os.path.isfile(path):
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


# ---------------------------------------------------------------------------
# Early-stop check
# ---------------------------------------------------------------------------


def early_stop_check(
    eval_records: list,
    mode: str,
    n_check: int = 20,
    threshold: float = 0.05,
) -> bool:
    """Decide whether to abort an evaluation run early.

    Args:
        eval_records: List of result dicts accumulated so far.
        mode:         "forward" or "counterfactual".
        n_check:      Minimum number of records needed to trigger a check.
        threshold:    Abort if the measured rate falls below this value.

    Returns:
        True if early stopping is warranted (score < threshold AND
        len(eval_records) >= n_check), False otherwise.
    """
    if len(eval_records) < n_check:
        return False

    subset = eval_records[:n_check]

    if mode == "forward":
        matches = sum(1 for r in subset if r.get("top1_match", False))
        rate = matches / n_check
    elif mode == "counterfactual":
        successes = sum(1 for r in subset if r.get("success", False))
        rate = successes / n_check
    else:
        return False

    return rate < threshold


# ---------------------------------------------------------------------------
# Memory context builder
# ---------------------------------------------------------------------------


def build_memory_ctx(records: list, n: int = 3) -> str:
    """Build a memory context string from the last n non-skipped records.

    Args:
        records: Full list of eval records.
        n:       Number of recent non-skipped records to include.

    Returns:
        Multi-line string suitable for prepending to a prompt.
        Returns "" if no suitable records exist.
    """
    valid = [r for r in records if not r.get("skipped", False)]
    recent = valid[-n:] if len(valid) >= n else valid

    if not recent:
        return ""

    lines = []
    for rec in recent:
        case_id = rec.get("case_id", "?")
        gt = rec.get("ground_truth", "?")
        pred = rec.get("top1_prediction", rec.get("predicted", "?"))
        match = rec.get("top1_match", None)
        match_str = ""
        if match is not None:
            match_str = " [CORRECT]" if match else " [WRONG]"
        lines.append(f"Case {case_id}: ground_truth={gt}, predicted={pred}{match_str}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------


def write_summary(out_dir: str, all_records: list, mode: str) -> None:
    """Write a summary.txt to out_dir.

    Args:
        out_dir:     Output directory (created if it does not exist).
        all_records: Full list of result records.
        mode:        "forward" or "counterfactual".
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== CF-CBN LLM Evaluation Summary ===\n")
        f.write(f"Mode   : {mode}\n")
        f.write(f"Total  : {len(all_records)} records\n\n")

        if mode == "forward":
            _write_forward_summary(f, all_records)
        elif mode == "counterfactual":
            _write_counterfactual_summary(f, all_records)
        else:
            f.write("(Unknown mode - no per-mode statistics computed.)\n")

    print(f"[write_summary] Summary written to {summary_path}")


def _get_rec(r: dict, *keys, default=None):
    """Return the first non-None value found among the given keys."""
    for k in keys:
        v = r.get(k)
        if v is not None:
            return v
    return default


def _write_forward_summary(f, records: list) -> None:
    """Write forward-simulation statistics.

    Accepts records from both eval_forward_simulation.py (CLLM, legacy field names)
    and eval_rcd/crfd_forward_simulation.py (canonical field names):
      index / case_id
      gt / gt_bases / ground_truth
      agent_top1 / top1_prediction
      kl_div / kl_divergence
    """
    valid = [r for r in records if not r.get("skipped", False)]
    n = len(valid)
    if n == 0:
        f.write("No valid records to summarize.\n")
        return

    def _kl(r):
        v = _get_rec(r, "kl_divergence", "kl_div")
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    kl_vals = [v for r in valid for v in [_kl(r)] if v is not None and not (v != v)]  # exclude NaN
    top1_matches = [r for r in valid if r.get("top1_match", False)]
    top3_matches = [r for r in valid if r.get("top3_match", False)]

    avg_kl = sum(kl_vals) / len(kl_vals) if kl_vals else float("nan")
    top1_rate = len(top1_matches) / n
    top3_rate = len(top3_matches) / n

    f.write(f"Average KL divergence : {avg_kl:.4f}\n")
    f.write(f"Top-1 match rate      : {top1_rate:.4f}  ({len(top1_matches)}/{n})\n")
    f.write(f"Top-3 match rate      : {top3_rate:.4f}  ({len(top3_matches)}/{n})\n\n")

    f.write(
        f"{'Case':<12} {'Ground Truth':<20} {'Predicted':<20} "
        f"{'Top1':>6} {'Top3':>6} {'KL':>10}\n"
    )
    f.write("-" * 80 + "\n")
    for r in valid:
        case_id = str(_get_rec(r, "case_id", "index", default="?"))
        # ground truth: prefer scalar string; fall back to list → take first element
        gt_raw = _get_rec(r, "ground_truth", "gt", "gt_bases", default="?")
        gt = gt_raw[0] if isinstance(gt_raw, list) else str(gt_raw)
        pred = str(_get_rec(r, "top1_prediction", "agent_top1", default="?"))
        t1 = "Y" if r.get("top1_match", False) else "N"
        t3 = "Y" if r.get("top3_match", False) else "N"
        kl_v = _kl(r)
        kl_str = f"{kl_v:>10.4f}" if kl_v is not None and not (kl_v != kl_v) else f"{'nan':>10}"
        f.write(f"{case_id:<12} {gt:<20} {pred:<20} {t1:>6} {t3:>6} {kl_str}\n")


def _write_counterfactual_summary(f, records: list) -> None:
    """Write counterfactual analysis statistics."""
    valid = [r for r in records if not r.get("skipped", False)]
    n = len(valid)
    if n == 0:
        f.write("No valid records to summarize.\n")
        return

    successes = [r for r in valid if r.get("success", False)]
    success_rate = len(successes) / n

    f.write(f"Success rate : {success_rate:.4f}  ({len(successes)}/{n})\n\n")

    f.write(
        f"{'Case':<12} {'Original Top1':<22} {'Target Service':<22} {'Success':>8}\n"
    )
    f.write("-" * 70 + "\n")
    for r in valid:
        case_id = str(r.get("case_id", "?"))
        orig = str(r.get("original_top1", r.get("top1_prediction", "?")))
        target = str(r.get("target_service", "?"))
        success = "Y" if r.get("success", False) else "N"
        f.write(f"{case_id:<12} {orig:<22} {target:<22} {success:>8}\n")


# ===========================================================================
# RCD (Root Cause Discovery) — system prompts and prompt builders
# ===========================================================================

SYSTEM_FORWARD_SIM_RCD = (
    "You are simulating the RCD (Root Cause Discovery) fault diagnosis algorithm.\n"
    "RCD uses a PC-algorithm (causal graph discovery) to find the root cause by identifying\n"
    "metrics that are most DIRECTLY and UNIQUELY correlated with the failure.\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. Do NOT write or execute any code. Use step-by-step mental arithmetic only.\n"
    "2. Show each computation step before the final JSON.\n"
    "3. Output ONE JSON object at the very end — nothing after it.\n"
    "\n"
    "=== SERVICE NAME NORMALIZATION ===\n"
    "- Strip trailing pod-index: \"auth-0\" → \"auth\"\n"
    "- Strip \"ts-\" prefix and \"-service\" / \"-mongo\" / \"-other-service\" suffix\n"
    "  Example: \"ts-auth-service\" → \"auth\"\n"
    "- Merge pod metrics into the same service base name.\n"
    "- Only keep services whose base name appears in the MODEL SERVICE LIST.\n"
    "\n"
    "=== METRIC WEIGHT TABLE ===\n"
    "pod_cpu_usage / cpu_usage                        : 4.0\n"
    "pod_memory_working_set_bytes / memory_usage      : 3.5\n"
    "rrt / rrt_max                                    : 3.5\n"
    "pod_processes                                    : 3.0\n"
    "timeout                                          : 3.0\n"
    "client_error / client_error_ratio                : 2.5\n"
    "server_error / server_error_ratio                : 2.5\n"
    "error / error_ratio                              : 2.0\n"
    "pod_network_transmit_packets                     : 1.5\n"
    "pod_network_receive_packets                      : 1.5\n"
    "pod_network_transmit_bytes                       : 1.0\n"
    "pod_network_receive_bytes                        : 1.0\n"
    "request / response                               : 0.5\n"
    "[any other metric]                               : 1.0\n"
    "\n"
    "=== RCD ALGORITHM (6 steps) ===\n"
    "\n"
    "STEP 1: Normalize service names. Collect the union of metrics for each recognized\n"
    "  service from both anomaly_services and anomaly_pods (deduplicate per service).\n"
    "\n"
    "STEP 2: For each metric m that appears as anomalous in ANY service, compute:\n"
    "  prevalence[m] = number of distinct recognized services that list metric m.\n"
    "\n"
    "STEP 3: For each service s with at least one anomalous metric, compute:\n"
    "  score[s] = SUM over each metric m in service_s: weight[m] / prevalence[m]\n"
    "  Services with no anomalous metrics: score[s] = 0.\n"
    "\n"
    "STEP 4: Rank services by score descending. Top-1 = root cause.\n"
    "\n"
    "STEP 5 (tie-break): If two services have equal score, rank the one with more\n"
    "  total anomalous metrics higher. If still tied, use alphabetical order.\n"
    "\n"
    "STEP 6: Assign scores. All non-anomalous services receive score 0.\n"
    "\n"
    "=== RCD INTUITION ===\n"
    "A metric that appears in ONLY ONE service (prevalence=1) is highly specific to\n"
    "that service and strongly rejects independence with the failure node — it receives\n"
    "FULL weight. A metric shared across MANY services (e.g., latency spikes seen\n"
    "everywhere) is less diagnostic because it can be explained by upstream propagation\n"
    "— it receives divided weight.\n"
    "\n"
    "=== OUTPUT FORMAT ===\n"
    "Output a single JSON: {\"top1\": \"service_name\", \"scores\": {\"svc1\": 2.5, ...}}\n"
    "Include all services; give 0.0 to services with no anomalous metrics.\n"
)

SYSTEM_COUNTERFACTUAL_RCD = (
    "You are an expert at analyzing the RCD root-cause diagnosis algorithm.\n"
    "Given a case, the current RCD prediction, and the scoring details, suggest\n"
    "what metric changes would flip the top-1 prediction to a DIFFERENT service.\n"
    "\n"
    "RCD scores each service as: score[s] = SUM_m weight[m] / prevalence[m]\n"
    "To flip the prediction you must either:\n"
    "  a) INCREASE another service's score above the current top-1, OR\n"
    "  b) DECREASE the current top-1's score so it falls below another service.\n"
    "\n"
    "Effective strategies:\n"
    "  - Add a HIGH-WEIGHT metric UNIQUE to the target service (prevalence 1 → full weight).\n"
    "  - Remove the current top-1's highest-weight unique metrics.\n"
    "  - Add the same metric to many services (raises prevalence → reduces each score).\n"
    "\n"
    "Output ONE JSON object:\n"
    "{\"target_service\": \"svc\", \"changes\": [{\"action\": \"add_metrics\","
    " \"service\": \"svc\", \"metrics\": [\"pod_cpu_usage\"]}], \"reasoning\": \"...\"}\n"
    "Allowed actions: add_metrics, remove_metrics, add_service, remove_service.\n"
)


def build_rcd_forward_prompt(
    case: dict,
    services: list,
    memory_ctx: str = "",
) -> str:
    """Build the user-turn prompt for RCD forward simulation."""
    lines = []

    if memory_ctx:
        lines.append("=== RECENT CASE MEMORY ===")
        lines.append(memory_ctx)
        lines.append("")

    lines.append("=== MODEL SERVICE LIST ===")
    lines.append(", ".join(services))
    lines.append("")

    # Anomaly data
    anom_svcs = {k: v for k, v in case.get("anomaly_services", {}).items() if v}
    anom_pods = {k: v for k, v in case.get("anomaly_pods", {}).items() if v}

    lines.append("=== ANOMALY DATA ===")
    if anom_svcs:
        lines.append("anomaly_services:")
        for svc, mets in anom_svcs.items():
            lines.append(f"  {svc}: {mets}")
    else:
        lines.append("anomaly_services: (empty)")

    if anom_pods:
        lines.append("anomaly_pods:")
        for pod, mets in anom_pods.items():
            lines.append(f"  {pod}: {mets}")
    else:
        lines.append("anomaly_pods: (empty)")

    lines.append("")
    lines.append(
        "Apply the RCD algorithm (6 steps) to this data. Show each step's arithmetic,\n"
        "then output ONE JSON: {\"top1\": \"service_name\", \"scores\": {...}}"
    )
    return "\n".join(lines)


def build_rcd_counterfactual_prompt(
    case: dict,
    original_top1: str,
    scores: dict,
    services: list,
    memory_ctx: str = "",
) -> str:
    """Build the user-turn prompt for RCD counterfactual simulation."""
    lines = []

    if memory_ctx:
        lines.append("=== RECENT CASE MEMORY ===")
        lines.append(memory_ctx)
        lines.append("")

    lines.append("=== MODEL SERVICE LIST ===")
    lines.append(", ".join(services))
    lines.append("")

    anom_svcs = {k: v for k, v in case.get("anomaly_services", {}).items() if v}
    anom_pods = {k: v for k, v in case.get("anomaly_pods", {}).items() if v}

    lines.append("=== ANOMALY DATA ===")
    if anom_svcs:
        lines.append("anomaly_services:")
        for svc, mets in anom_svcs.items():
            lines.append(f"  {svc}: {mets}")
    if anom_pods:
        lines.append("anomaly_pods:")
        for pod, mets in anom_pods.items():
            lines.append(f"  {pod}: {mets}")

    lines.append("")
    lines.append(f"=== CURRENT RCD PREDICTION ===")
    lines.append(f"Top-1 predicted root cause: {original_top1}")
    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    lines.append("Top-5 scores: " + ", ".join(f"{s}={v:.3f}" for s, v in top5))

    lines.append("")
    lines.append(
        "Suggest metric changes to flip the RCD top-1 prediction away from "
        f"'{original_top1}' to a DIFFERENT service.\n"
        "Output ONE JSON: {\"target_service\": \"svc\", \"changes\": [...], \"reasoning\": \"...\"}"
    )
    return "\n".join(lines)


# ===========================================================================
# CRFD (Counterfactual Reasoning for Fault Diagnosis) — system prompts and builders
# ===========================================================================

SYSTEM_FORWARD_SIM_CRFD = (
    "You are simulating the CRFD (Counterfactual Reasoning for Fault Diagnosis) algorithm.\n"
    "CRFD uses a GNN-based counterfactual approach: for each service, simulate what happens\n"
    "if it returns to normal (do-intervention), and measure total anomaly reduction.\n"
    "Services that cause the greatest reduction when normalized are ranked as root cause.\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. Do NOT write or execute any code. Use step-by-step mental arithmetic only.\n"
    "2. Show each computation step clearly before the final JSON.\n"
    "3. Output ONE JSON object at the very end — nothing after it.\n"
    "\n"
    "=== SERVICE NAME NORMALIZATION ===\n"
    "- Strip trailing pod-index: \"auth-0\" → \"auth\"\n"
    "- Strip \"ts-\" prefix and \"-service\" / \"-mongo\" / \"-other-service\" suffix.\n"
    "- Merge pod metrics into the same service base name.\n"
    "- Only keep services whose base name appears in the MODEL SERVICE LIST.\n"
    "\n"
    "=== METRIC WEIGHT TABLE ===\n"
    "pod_cpu_usage / cpu_usage                        : 4.0\n"
    "pod_memory_working_set_bytes / memory_usage      : 3.5\n"
    "rrt / rrt_max                                    : 3.5\n"
    "pod_processes                                    : 3.0\n"
    "timeout                                          : 3.0\n"
    "client_error / client_error_ratio                : 2.5\n"
    "server_error / server_error_ratio                : 2.5\n"
    "error / error_ratio                              : 2.0\n"
    "pod_network_transmit_packets                     : 1.5\n"
    "pod_network_receive_packets                      : 1.5\n"
    "pod_network_transmit_bytes                       : 1.0\n"
    "pod_network_receive_bytes                        : 1.0\n"
    "request / response                               : 0.5\n"
    "[any other metric]                               : 1.0\n"
    "\n"
    "=== CRFD ALGORITHM (7 steps) ===\n"
    "\n"
    "STEP 1: Normalize service names. Collect the union of metrics per service.\n"
    "\n"
    "STEP 2: For each recognized service s with anomalous metrics, compute:\n"
    "  row_s = 3.0 × sqrt( sum of weight_m^2 for each metric m of service s )\n"
    "  (Services with no metrics: row_s = 0)\n"
    "\n"
    "STEP 3: Compute total_norm:\n"
    "  total = sqrt( sum over ALL services: row_s^2 )\n"
    "\n"
    "STEP 4: Counterfactual (CF) score — how much total norm decreases if s is zeroed:\n"
    "  CF[s] = total − sqrt( total^2 − row_s^2 )\n"
    "  (Services with row_s=0: CF[s] = 0)\n"
    "\n"
    "STEP 5: Direct score:\n"
    "  direct[s] = sum of weight_m for each metric m of service s\n"
    "\n"
    "STEP 6: Upstream propagation bonus (topology).\n"
    "  If service A is listed as a CALLER of service s (A depends on s), and A also\n"
    "  has anomalous metrics, then s's failure may have caused A's anomaly.\n"
    "  propagation[s] = 0.25 × sum of direct[A] for all callers A of s that are anomalous.\n"
    "  (If no topology or no anomalous callers: propagation[s] = 0)\n"
    "\n"
    "STEP 7: Final score and ranking:\n"
    "  score[s] = CF[s] + 0.3 × direct[s] + propagation[s]\n"
    "  Rank services by score descending. Top-1 = root cause.\n"
    "\n"
    "=== CRFD INTUITION ===\n"
    "CRFD emulates a GNN where each service node is connected to its callers.\n"
    "Zeroing service s (do-intervention) removes not just s's own anomaly but also\n"
    "reduces the propagated anomaly signal visible in s's callers.\n"
    "A deep dependency whose failure cascades up the call graph gets the highest score.\n"
    "\n"
    "=== OUTPUT FORMAT ===\n"
    "Output a single JSON: {\"top1\": \"service_name\", \"scores\": {\"svc1\": 2.5, ...}}\n"
    "Include all services; give 0.0 to services with no anomalous metrics.\n"
)

SYSTEM_COUNTERFACTUAL_CRFD = (
    "You are an expert at analyzing the CRFD root-cause diagnosis algorithm.\n"
    "CRFD scores: score[s] = CF[s] + 0.3×direct[s] + propagation[s]\n"
    "where CF[s] = how much total anomaly norm decreases if service s is zeroed,\n"
    "direct[s] = weighted metric count of s, and\n"
    "propagation[s] = 0.25 × sum of direct[A] for callers A of s.\n"
    "\n"
    "To flip the prediction away from the current top-1:\n"
    "  a) Increase another service's score (add high-weight metrics; it should have callers).\n"
    "  b) Decrease current top-1's score (remove its metrics, break its propagation chain).\n"
    "\n"
    "Output ONE JSON:\n"
    "{\"target_service\": \"svc\", \"changes\": [{\"action\": \"add_metrics\","
    " \"service\": \"svc\", \"metrics\": [\"pod_cpu_usage\"]}], \"reasoning\": \"...\"}\n"
    "Allowed actions: add_metrics, remove_metrics, add_service, remove_service.\n"
)


def build_crfd_forward_prompt(
    case: dict,
    services: list,
    topology: dict = None,
    memory_ctx: str = "",
) -> str:
    """Build the user-turn prompt for CRFD forward simulation."""
    lines = []

    if memory_ctx:
        lines.append("=== RECENT CASE MEMORY ===")
        lines.append(memory_ctx)
        lines.append("")

    lines.append("=== MODEL SERVICE LIST ===")
    lines.append(", ".join(services))
    lines.append("")

    # Topology (caller → callees)
    if topology:
        lines.append("=== SERVICE TOPOLOGY (caller → callees) ===")
        for caller, callees in sorted(topology.items()):
            if callees:
                lines.append(f"  {caller} → {', '.join(callees)}")
        lines.append("")

    anom_svcs = {k: v for k, v in case.get("anomaly_services", {}).items() if v}
    anom_pods = {k: v for k, v in case.get("anomaly_pods", {}).items() if v}

    lines.append("=== ANOMALY DATA ===")
    if anom_svcs:
        lines.append("anomaly_services:")
        for svc, mets in anom_svcs.items():
            lines.append(f"  {svc}: {mets}")
    else:
        lines.append("anomaly_services: (empty)")

    if anom_pods:
        lines.append("anomaly_pods:")
        for pod, mets in anom_pods.items():
            lines.append(f"  {pod}: {mets}")
    else:
        lines.append("anomaly_pods: (empty)")

    lines.append("")
    lines.append(
        "Apply the CRFD algorithm (7 steps) to this data. Show each step's arithmetic,\n"
        "then output ONE JSON: {\"top1\": \"service_name\", \"scores\": {...}}"
    )
    return "\n".join(lines)


def build_crfd_counterfactual_prompt(
    case: dict,
    original_top1: str,
    scores: dict,
    services: list,
    topology: dict = None,
    memory_ctx: str = "",
) -> str:
    """Build the user-turn prompt for CRFD counterfactual simulation."""
    lines = []

    if memory_ctx:
        lines.append("=== RECENT CASE MEMORY ===")
        lines.append(memory_ctx)
        lines.append("")

    lines.append("=== MODEL SERVICE LIST ===")
    lines.append(", ".join(services))
    lines.append("")

    if topology:
        lines.append("=== SERVICE TOPOLOGY (caller → callees) ===")
        for caller, callees in sorted(topology.items()):
            if callees:
                lines.append(f"  {caller} → {', '.join(callees)}")
        lines.append("")

    anom_svcs = {k: v for k, v in case.get("anomaly_services", {}).items() if v}
    anom_pods = {k: v for k, v in case.get("anomaly_pods", {}).items() if v}

    lines.append("=== ANOMALY DATA ===")
    if anom_svcs:
        lines.append("anomaly_services:")
        for svc, mets in anom_svcs.items():
            lines.append(f"  {svc}: {mets}")
    if anom_pods:
        lines.append("anomaly_pods:")
        for pod, mets in anom_pods.items():
            lines.append(f"  {pod}: {mets}")

    lines.append("")
    lines.append("=== CURRENT CRFD PREDICTION ===")
    lines.append(f"Top-1 predicted root cause: {original_top1}")
    top5 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
    lines.append("Top-5 scores: " + ", ".join(f"{s}={v:.3f}" for s, v in top5))

    lines.append("")
    lines.append(
        "Suggest metric changes to flip the CRFD top-1 prediction away from "
        f"'{original_top1}' to a DIFFERENT service.\n"
        "Output ONE JSON: {\"target_service\": \"svc\", \"changes\": [...], \"reasoning\": \"...\"}"
    )
    return "\n".join(lines)
