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
import datetime

import requests
import numpy as np

# ---------------------------------------------------------------------------
# Module-level configuration
# ---------------------------------------------------------------------------

EVAL_PLATFORM = "apiyi"
EVAL_MODEL = "gpt-4o"
EVAL_API_KEY = os.environ.get("EVAL_LLM_API_KEY", "KEY")

_PLATFORM_BASE_URLS = {
    "apiyi":  "https://api.apiyi.com/v1",
    "openai": "https://api.openai.com/v1",
    "volc":   "https://ark.cn-beijing.volces.com/api/v3",
}

# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------

_EVAL_LOG_FILE = None      # path to llm_calls.log
_EVAL_FAILURES_DIR = None  # path to failures/ directory


def set_eval_log_file(log_path: str, failures_dir: str) -> None:
    """Initialise per-experiment LLM call logging.

    Call once at the start of each eval run, after out_dir is known.
    Creates the log file directory and failures directory if needed.
    """
    global _EVAL_LOG_FILE, _EVAL_FAILURES_DIR
    _EVAL_LOG_FILE = log_path
    _EVAL_FAILURES_DIR = failures_dir
    os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
    os.makedirs(failures_dir, exist_ok=True)


def _eval_log(text: str) -> None:
    if _EVAL_LOG_FILE is None:
        return
    try:
        with open(_EVAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


def log_failure(case_idx, reason: str, prompt: str, system: str,
                raw_response: str, error_msg=None) -> None:
    """Append a failure record to failures/failures.jsonl.

    Args:
        case_idx:     Case index from the main eval loop.
        reason:       'api_failure' | 'parse_failure' | 'llm_output'
        prompt:       The prompt sent to the LLM.
        system:       The system prompt used.
        raw_response: Raw LLM response text (empty string if invoke() raised).
        error_msg:    Exception message, if any.
    """
    if _EVAL_FAILURES_DIR is None:
        return
    record = {
        "ts":           datetime.datetime.now().isoformat(),
        "case_idx":     case_idx,
        "reason":       reason,
        "prompt":       prompt,
        "system":       system or "",
        "raw_response": raw_response,
        "error":        str(error_msg) if error_msg is not None else None,
    }
    path = os.path.join(_EVAL_FAILURES_DIR, "failures.jsonl")
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# API error type
# ---------------------------------------------------------------------------

class EvalLLMAPIError(RuntimeError):
    """Raised when all per-call retries for an LLM API request fail."""
    pass


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
                _eval_log(f"[retry {attempt + 1}/{retries}] {type(exc).__name__}: {exc}  "
                          f"(waiting {wait}s)")
                print(
                    f"[EvalLLMClient] Attempt {attempt + 1}/{retries} failed: {exc}. "
                    f"Retrying in {wait}s...",
                    file=sys.stderr,
                )
                time.sleep(wait)
        raise EvalLLMAPIError(
            f"All {retries} retries failed. Last error: {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def invoke(self, prompt: str, system: str = None, history: list = None,
               case_idx=None) -> str:
        """Send a prompt and return the raw string response.

        Args:
            prompt:    User-turn content.
            system:    Optional system message (prepended).
            history:   Optional list of prior turns as {"role": ..., "content": ...} dicts.
            case_idx:  Case index for log annotation (optional).

        Returns:
            Assistant response as a plain string.

        Raises:
            EvalLLMAPIError: if all per-call retries are exhausted.
        """
        messages = self._build_messages(prompt, system=system, history=history)
        ts = datetime.datetime.now().isoformat()
        _eval_log(f"\n{'='*20} CASE {case_idx}  {ts} {'='*20}")
        _eval_log(f"[model] {self.model}  [platform] {self.platform}")
        if system:
            _eval_log(f"[system]\n{system}")
        _eval_log(f"[prompt]\n{prompt}")
        result = self._retry_call(messages)   # raises EvalLLMAPIError on total failure
        _eval_log(f"[response]\n{result}")
        return result

    def invoke_json(self, prompt: str, system: str = None, history: list = None,
                    case_idx=None) -> dict:
        """Send a prompt and parse the response as JSON.

        Handles Markdown code fences and finds the last top-level {...} block.

        Returns:
            Parsed dict.  Raises ValueError if JSON cannot be extracted.
        """
        raw = self.invoke(prompt, system=system, history=history, case_idx=case_idx)
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
    "1. Do NOT write or execute any code. Use step-by-step arithmetic only.\n"
    "2. Write your full computation as free text first.\n"
    "3. At the very end, output ONE clean JSON block — nothing after it.\n"
    "4. In the JSON, output ONLY real computed numbers. NEVER use placeholders like <value> or '...'.\n"
    "   If arithmetic is complex, use reasonable approximations — but every field must be an actual number.\n"
    "\n"
    "=== SERVICE NAME NORMALIZATION ===\n"
    "When reading anomaly_services keys:\n"
    '- Strip trailing pod-index suffix: "auth-0", "auth-1" → "auth"\n'
    '- Also strip "ts-" prefix AND "-service"/"-mongo"/"-other-service" suffix\n'
    '  Example: "ts-auth-service" → "auth",  "ts-auth-mongo" → "auth"\n'
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
    '[any other metric, e.g. "cpu","mem","latency-*"] : 1.0\n'
    "\n"
    "=== CF-CBN ALGORITHM (7 steps) ===\n"
    "\n"
    "STEP 1: Map each anomaly_services entry to its base service name.\n"
    "Collect all metrics for each recognized service.\n"
    "\n"
    "STEP 2: For each service s with at least one metric, compute row_norm:\n"
    "  row_s = 3.0 × sqrt( sum of weight_m^2  for all metrics m of service s )\n"
    "Services with no metrics: row_s = 0.\n"
    "\n"
    "STEP 3: Compute total_norm:\n"
    "  total = sqrt( sum over ALL services s of: row_s^2 )\n"
    "\n"
    "STEP 4: Compute CF score for each service s:\n"
    "  CF[s] = total − sqrt( total^2 − row_s^2 )\n"
    "(Services with row_s=0: CF[s]=0)\n"
    "\n"
    "STEP 5: Compute direct score:\n"
    "  direct[s] = sum of weight_m for each metric m of service s\n"
    "\n"
    "STEP 6: Compute combined score and normalize to [0,1]:\n"
    "  unsup[s] = CF[s] + 0.3 × direct[s]\n"
    "  max_u = max( unsup[s] ) over all anomalous services\n"
    "  min_u = min( unsup[s] ) over all anomalous services\n"
    "  score[s] = (unsup[s] − min_u) / (max_u − min_u)   if max_u > min_u\n"
    "             1.0 / n_anomalous_services               otherwise\n"
    "\n"
    "STEP 7: Rank all anomalous services by score[s] descending; identify top-1.\n"
    "\n"
    "=== OUTPUT FORMAT ===\n"
    "Write your computation steps as free text above (any format is fine).\n"
    "Then end your response with exactly this JSON block:\n"
    "\n"
    "```json\n"
    "{\n"
    '  "top1_prediction": "<service_name>",\n'
    '  "rankings": [\n'
    '    {"service": "<base_name>", "score": <float, 4 decimal places>},\n'
    "    ...\n"
    "  ]\n"
    "}\n"
    "```\n"
    "\n"
    "Include ALL anomalous services (row_s > 0), sorted by score descending.\n"
    "Every 'score' value must be a real number you computed — never a placeholder."
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
    memory_ctx: str = "",
    # Deprecated parameters kept for call-site compatibility (ignored)
    alpha: float = 1.0,
    prior_counts: dict = None,
    n_history: int = 0,
) -> str:
    """Build the user-turn prompt for forward simulation.

    Args:
        case:       A single anomaly case dict with at least an anomaly_services key.
        services:   Full list of service names in the model.
        memory_ctx: Optional memory context string prepended to the prompt.

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
    svc_entries = {k: v for k, v in anomaly_services.items() if v}
    lines.append("anomaly_services:")
    if svc_entries:
        for svc, metrics in svc_entries.items():
            lines.append(f"  {svc}: {json.dumps(metrics)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(
        "Apply the CF-CBN algorithm (all 7 steps) to the anomaly data above "
        "and output the result."
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
    svc_entries = {k: v for k, v in anomaly_services.items() if v}

    lines.append("anomaly_services:")
    if svc_entries:
        for svc, metrics in svc_entries.items():
            lines.append(f"  {svc}: {json.dumps(metrics)}")
    else:
        lines.append("  (none)")

    lines.append("")
    lines.append(
        "Task: Suggest the minimal set of changes to the anomaly_services data "
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
        # Accept both "top1_prediction" (CF-CBN new format) and "top1" (CRFD/RCD legacy)
        top1 = str(data.get("top1_prediction", data.get("top1", "unknown"))).strip()
        scores = {}
        rankings = data.get("rankings", [])
        if rankings:
            # Preferred format: rankings array with per-service score
            for entry in rankings:
                svc = str(entry.get("service", "")).strip()
                # Accept "score" (new), "fused_score" (legacy CF-CBN), "unsup_norm"
                raw_val = entry.get("score",
                          entry.get("fused_score",
                          entry.get("unsup_norm", 0.0)))
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    val = 0.0
                if svc:
                    scores[svc] = val
        else:
            # Fallback: flat {"scores": {"svc": value}} dict (CRFD/RCD legacy format)
            for svc, val in data.get("scores", {}).items():
                try:
                    scores[str(svc)] = float(val)
                except (TypeError, ValueError):
                    scores[str(svc)] = 0.0
            if scores and top1 == "unknown":
                top1 = max(scores, key=lambda s: scores[s])
        # NOTE: return is AFTER the loop so all ranking entries are captured
        return top1, scores
    except Exception as exc:
        print(f"[parse_forward_response] Parse failure: {exc}", file=sys.stderr)
        return "unknown", {}


def parse_counterfactual_response(raw: str) -> tuple:
    """Parse the LLM counterfactual response.

    Args:
        raw: Raw LLM response string.

    Returns:
        Tuple of (changes: list, agent_target: str, reasoning: str).
        On any parse failure returns ([], "unknown", "").
    """
    try:
        data = _extract_last_json(raw)
        changes = data.get("changes", [])
        if not isinstance(changes, list):
            changes = []
        target = str(data.get("target_service", "unknown")).strip()
        reasoning = str(data.get("reasoning", "")).strip()
        return changes, target, reasoning
    except Exception as exc:
        print(f"[parse_counterfactual_response] Parse failure: {exc}", file=sys.stderr)
        return [], "unknown", ""


def classify_unknown_reason(raw: str, key: str = "top1_prediction") -> str:
    """Classify why an agent result came back as 'unknown'.

    Used after parse_forward_response or parse_counterfactual_response returns
    an 'unknown' prediction to decide whether the root cause is an API-level
    failure, a JSON-parsing problem, or a genuine 'unknown' from the LLM.

    Args:
        raw: The raw LLM response string (may be empty if invoke() raised).
        key: JSON field to inspect. Use 'top1_prediction' for forward simulation,
             'target_service' for counterfactual simulation.

    Returns:
        'api_failure'   — raw is empty; the LLM API call itself failed
        'parse_failure' — raw is non-empty but valid JSON could not be extracted,
                          or the expected field is absent/wrong type
        'llm_output'    — the LLM returned valid JSON with the named field == 'unknown'
    """
    if not raw:
        return "api_failure"
    try:
        data = _extract_last_json(raw)
        val = str(data.get(key, "")).strip().lower()
        if val == "unknown":
            return "llm_output"
        return "parse_failure"
    except Exception:
        return "parse_failure"


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


def rank_vector_distance(model_ranked: list, agent_ranked: list,
                         all_services: list) -> float:
    """Euclidean distance between model and agent rank-position vectors.

    Each service is assigned a position 1..N (lower = higher rank).
    Services absent from a ranked list are assigned position N (last place).

    Returns raw Euclidean distance (lower = more similar; 0 = identical ranking).
    """
    N = len(all_services)
    svc_set = set(all_services)

    def to_pos(ranked: list) -> dict:
        pos = {s: N for s in all_services}
        for i, s in enumerate(ranked):
            if s in svc_set:
                pos[s] = i + 1
        return pos

    pm = to_pos(model_ranked)
    pa = to_pos(agent_ranked)
    dist = math.sqrt(sum((pm[s] - pa[s]) ** 2 for s in all_services))
    return round(dist, 4)


def _max_rank_dist(N: int) -> float:
    """Theoretical max Euclidean distance for two opposite rankings of N items."""
    return math.sqrt(sum((N + 1 - 2 * i) ** 2 for i in range(1, N + 1)))


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
        threshold:    For "forward": max acceptable avg rank_dist (abort if exceeded).
                      For "counterfactual": min acceptable success rate (abort if below).

    Returns:
        True if early stopping is warranted, False otherwise.
    """
    if len(eval_records) < n_check:
        return False

    subset = eval_records[:n_check]

    if mode == "forward":
        dist_vals = [r.get("rank_dist", 0.0) for r in subset
                     if not r.get("skipped") and "rank_dist" in r]
        if len(dist_vals) < n_check:
            return False
        avg_d = sum(dist_vals) / len(dist_vals)
        return avg_d > threshold
    elif mode == "counterfactual":
        successes = sum(1 for r in subset if r.get("success", False))
        rate = successes / n_check
        return rate < threshold
    else:
        return False


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


def write_summary(out_dir: str, all_records: list, mode: str,
                  label: str = "CF-CBN") -> None:
    """Write a summary.txt to out_dir.

    Args:
        out_dir:     Output directory (created if it does not exist).
        all_records: Full list of result records.
        mode:        "forward" or "counterfactual".
        label:       Engine label for the title line (e.g. "CF-CBN", "CRFD", "RCD").
    """
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.txt")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"=== {label} LLM Evaluation Summary ===\n")
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
    """Write forward-simulation statistics."""
    valid = [r for r in records if not r.get("skipped", False)]
    n = len(valid)
    if n == 0:
        f.write("No valid records to summarize.\n")
        return

    dist_vals = [float(r["rank_dist"]) for r in valid if "rank_dist" in r]
    avg_dist = sum(dist_vals) / max(len(dist_vals), 1) if dist_vals else float("nan")

    f.write(f"Avg rank-vector dist  : {avg_dist:.4f}  (lower = more similar)\n\n")

    f.write(
        f"{'Case':<12} {'Ground Truth':<20} {'Model Top-1':<22} {'Agent Top-1':<22} {'Dist':>8}\n"
    )
    f.write("-" * 90 + "\n")
    for r in valid:
        case_id = str(_get_rec(r, "case_id", "index", default="?"))
        gt_raw = _get_rec(r, "ground_truth", "gt", "gt_bases", default="?")
        gt = gt_raw[0] if isinstance(gt_raw, list) else str(gt_raw)
        model_t1 = str(_get_rec(r, "model_top1", default="?"))
        agent_t1 = str(_get_rec(r, "agent_top1", "top1_prediction", default="?"))
        dist_v = r.get("rank_dist")
        dist_str = f"{dist_v:>8.4f}" if dist_v is not None else f"{'nan':>8}"
        f.write(f"{case_id:<12} {gt:<20} {model_t1:<22} {agent_t1:<22} {dist_str}\n")


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
# Metric 1 — Blind Ternary Interpretability Ranking
# ===========================================================================

_ALGO_BRIEF = {
    "CLLM": (
        "CF-CBN (7 steps):\n"
        "1. Map service names; collect anomalous metrics per service.\n"
        "2. row_s = 3 × sqrt(Σ weight_m²) for each service's metrics.\n"
        "3. total = sqrt(Σ row_s²) over all services.\n"
        "4. CF[s] = total − sqrt(total² − row_s²)  (counterfactual reduction).\n"
        "5. direct[s] = Σ weight_m for service s.\n"
        "6. unsup[s] = CF[s] + 0.3 × direct[s]; min-max normalise to score[s].\n"
        "7. Rank services by score[s] descending.\n"
        "Metric weights: pod_cpu_usage=4.0, memory=3.5, rrt=3.5, "
        "pod_processes=3.0, timeout=3.0, errors=2.0-2.5, network=1.0-1.5, "
        "request/response=0.5."
    ),
    "RCD": (
        "RCD (5 steps):\n"
        "1. Map service names; collect anomalous metrics per service.\n"
        "2. prevalence[m] = number of distinct services sharing metric m.\n"
        "3. score[s] = Σ_{m in s} weight_m / prevalence[m].\n"
        "4. Services not anomalous: score = 0.\n"
        "5. Rank services by score descending.\n"
        "Metric weights: same table as CF-CBN."
    ),
    "CRFD": (
        "CRFD (6 steps):\n"
        "1. Build weighted metric matrix X (services × metrics).\n"
        "2. CF[s] = ||X|| − ||X with row s zeroed||.\n"
        "3. direct[s] = Σ weight_m for service s.\n"
        "4. propagation[s] = 0.25 × Σ_{B calls s} direct[B]  "
        "(callers of s in the topology).\n"
        "5. score[s] = CF[s] + 0.3 × direct[s] + propagation[s].\n"
        "6. Rank services by score descending.\n"
        "Requires: service call graph topology."
    ),
}

SYSTEM_INTERPRETABILITY_RANKING = (
    "You are a neutral evaluator assessing the interpretability of three "
    "fault-diagnosis algorithms (labeled A, B, C).\n\n"
    "TASK: Given the anomaly data and each method's algorithm description plus "
    "its prediction, rank the three methods from MOST to LEAST interpretable.\n\n"
    "'Interpretable' means: a non-expert engineer, armed only with the algorithm "
    "description and the anomaly data, could independently verify the prediction "
    "step by step using pencil and paper.\n\n"
    "IMPORTANT RULES:\n"
    "1. Do NOT consider which prediction might be correct — you do not know "
    "the true root cause.\n"
    "2. Judge only: how clearly and completely does the algorithm description "
    "allow a human to reproduce the exact scores and ranking?\n"
    "3. Criteria: (a) Are all computation steps explicit and deterministic? "
    "(b) Can every output number be traced to the input data? "
    "(c) Does the algorithm depend on hidden learned weights inaccessible "
    "to the human evaluator?\n\n"
    "Output at the end exactly this JSON:\n"
    '{"ranking": ["X", "Y", "Z"], "reasoning": "<≤100 words>"}\n'
    "where X is most interpretable, Z least; X/Y/Z ∈ {A, B, C}."
)


def build_interpretability_ranking_prompt(
    case: dict,
    label_to_method: dict,
    predictions: dict,
    services: list,
) -> str:
    lines = ["=== ANOMALY DATA ==="]
    for svc, mets in case.get("anomaly_services", {}).items():
        if mets:
            lines.append(f"  {svc}: {mets}")
    lines.append("")

    for label in sorted(label_to_method):
        method = label_to_method[label]
        ranked, scores = predictions[method]
        top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
        lines.append(f"=== Method {label} ===")
        lines.append(_ALGO_BRIEF[method])
        lines.append(f"Top-1 prediction : {ranked[0] if ranked else 'unknown'}")
        lines.append("Top-5 scores     :")
        for svc, sc in top5:
            lines.append(f"  {svc}: {sc:.4f}")
        lines.append("")

    lines.append("Rank methods A, B, C from most to least interpretable.")
    return "\n".join(lines)


def parse_interpretability_ranking_response(raw: str) -> tuple:
    """Returns (ranking: list[str], reasoning: str) or ([], '') on failure."""
    try:
        data = _extract_last_json(raw)
        ranking = data.get("ranking", [])
        if (not isinstance(ranking, list) or len(ranking) != 3
                or not all(x in {"A", "B", "C"} for x in ranking)
                or len(set(ranking)) != 3):
            return [], ""
        return list(ranking), str(data.get("reasoning", "")).strip()
    except Exception:
        return [], ""


def write_interpretability_summary(out_dir: str, records: list) -> None:
    valid = [r for r in records if not r.get("skipped") and r.get("cllm_rank")]
    n = max(len(valid), 1)
    path = os.path.join(out_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== Interpretability Ranking Evaluation Summary ===\n")
        f.write(f"Valid cases : {len(valid)}\n\n")
        for method, key in [("CLLM", "cllm_rank"), ("RCD", "rcd_rank"), ("CRFD", "crfd_rank")]:
            avg_rank = sum(r[key] for r in valid) / n
            rank1 = sum(1 for r in valid if r[key] == 1) / n
            f.write(f"  {method:<6}  avg_rank={avg_rank:.2f}  "
                    f"ranked_#1={rank1:.1%}\n")
    print(f"[write_interpretability_summary] {path}")


# ===========================================================================
# RCD (Root Cause Discovery) — system prompts and prompt builders
# ===========================================================================

SYSTEM_FORWARD_SIM_RCD = (
    "You are simulating the RCD (Root Cause Discovery) fault diagnosis algorithm.\n"
    "RCD uses a PC-algorithm (causal graph discovery) to find the root cause by identifying\n"
    "metrics that are most DIRECTLY and UNIQUELY correlated with the failure.\n"
    "\n"
    "CRITICAL RULES:\n"
    "1. Do NOT write or execute any code. Use step-by-step arithmetic only.\n"
    "2. Write your full computation as free text first.\n"
    "3. At the very end, output ONE clean JSON block — nothing after it.\n"
    "4. In the JSON, output ONLY real computed numbers. NEVER use placeholders like <value>.\n"
    "   If arithmetic is complex, use reasonable approximations — but every field must be an actual number.\n"
    "\n"
    "=== SERVICE NAME NORMALIZATION ===\n"
    "- Strip trailing pod-index: \"auth-0\" → \"auth\"\n"
    "- Strip \"ts-\" prefix and \"-service\" / \"-mongo\" / \"-other-service\" suffix\n"
    "  Example: \"ts-auth-service\" → \"auth\"\n"
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
    "=== RCD ALGORITHM (5 steps) ===\n"
    "\n"
    "STEP 1: Normalize service names. Collect metrics for each recognized service\n"
    "  from anomaly_services (deduplicate per service).\n"
    "\n"
    "STEP 2: For each metric m that appears in ANY anomalous service, compute:\n"
    "  prevalence[m] = number of distinct services that list metric m.\n"
    "\n"
    "STEP 3: For each service s with at least one anomalous metric, compute:\n"
    "  score[s] = SUM over each metric m of service s: weight[m] / prevalence[m]\n"
    "\n"
    "STEP 4: Rank anomalous services by score descending. Top-1 = root cause.\n"
    "  Tie-break: more total metrics wins; still tied → alphabetical.\n"
    "\n"
    "STEP 5: Report top-1 and the score for each anomalous service.\n"
    "\n"
    "=== RCD INTUITION ===\n"
    "A metric unique to one service (prevalence=1) gets FULL weight — highly diagnostic.\n"
    "A metric shared across many services gets divided weight — less diagnostic.\n"
    "\n"
    "=== OUTPUT FORMAT ===\n"
    "Write your computation steps as free text above (any format is fine).\n"
    "Then end your response with exactly this JSON block:\n"
    "\n"
    "```json\n"
    "{\n"
    "  \"top1_prediction\": \"<service_name>\",\n"
    "  \"rankings\": [\n"
    "    {\"service\": \"<base_name>\", \"score\": <float, 4 decimal places>},\n"
    "    ...\n"
    "  ]\n"
    "}\n"
    "```\n"
    "\n"
    "Include ALL anomalous services sorted by score descending.\n"
    "Every 'score' value must be a real number you computed — never a placeholder.\n"
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

    anom_svcs = {k: v for k, v in case.get("anomaly_services", {}).items() if v}

    lines.append("=== ANOMALY DATA ===")
    if anom_svcs:
        lines.append("anomaly_services:")
        for svc, mets in anom_svcs.items():
            lines.append(f"  {svc}: {mets}")
    else:
        lines.append("anomaly_services: (empty)")

    lines.append("")
    lines.append(
        "Apply the RCD algorithm (5 steps) to this data and output the result."
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

    lines.append("=== ANOMALY DATA ===")
    if anom_svcs:
        lines.append("anomaly_services:")
        for svc, mets in anom_svcs.items():
            lines.append(f"  {svc}: {mets}")
    else:
        lines.append("anomaly_services: (empty)")

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
    "1. Do NOT write or execute any code. Use step-by-step arithmetic only.\n"
    "2. Write your full computation as free text first.\n"
    "3. At the very end, output ONE clean JSON block — nothing after it.\n"
    "4. In the JSON, output ONLY real computed numbers. NEVER use placeholders like <value>.\n"
    "   If arithmetic is complex, use reasonable approximations — but every field must be an actual number.\n"
    "\n"
    "=== SERVICE NAME NORMALIZATION ===\n"
    "- Strip trailing pod-index: \"auth-0\" → \"auth\"\n"
    "- Strip \"ts-\" prefix and \"-service\" / \"-mongo\" / \"-other-service\" suffix.\n"
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
    "=== CRFD ALGORITHM (6 steps) ===\n"
    "\n"
    "STEP 1: Normalize service names. Collect metrics for each recognized service\n"
    "  from anomaly_services.\n"
    "\n"
    "STEP 2: For each service s with anomalous metrics, compute:\n"
    "  row_s = 3.0 × sqrt( sum of weight_m^2 for each metric m of service s )\n"
    "  (Services with no metrics: row_s = 0)\n"
    "\n"
    "STEP 3: Compute total_norm:\n"
    "  total = sqrt( sum over ALL services: row_s^2 )\n"
    "\n"
    "STEP 4: CF score — how much total norm decreases if service s is zeroed:\n"
    "  CF[s] = total − sqrt( total^2 − row_s^2 )   (0 when row_s = 0)\n"
    "\n"
    "STEP 5: Direct score and propagation bonus (if topology is given):\n"
    "  direct[s] = sum of weight_m for each metric m of service s\n"
    "  propagation[s] = 0.25 × sum of direct[A] for callers A of s that are anomalous\n"
    "  final_score[s] = CF[s] + 0.3 × direct[s] + propagation[s]\n"
    "\n"
    "STEP 6: Rank anomalous services by final_score descending. Top-1 = root cause.\n"
    "\n"
    "=== CRFD INTUITION ===\n"
    "Zeroing service s removes s's own anomaly AND the propagated signal in s's callers.\n"
    "A deep dependency whose failure cascades up the call graph gets the highest score.\n"
    "\n"
    "=== OUTPUT FORMAT ===\n"
    "Write your computation steps as free text above (any format is fine).\n"
    "Then end your response with exactly this JSON block:\n"
    "\n"
    "```json\n"
    "{\n"
    "  \"top1_prediction\": \"<service_name>\",\n"
    "  \"rankings\": [\n"
    "    {\"service\": \"<base_name>\", \"score\": <float, 4 decimal places>},\n"
    "    ...\n"
    "  ]\n"
    "}\n"
    "```\n"
    "\n"
    "Include ALL anomalous services sorted by score descending.\n"
    "Every 'score' value must be a real number you computed — never a placeholder.\n"
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

    lines.append("=== ANOMALY DATA ===")
    if anom_svcs:
        lines.append("anomaly_services:")
        for svc, mets in anom_svcs.items():
            lines.append(f"  {svc}: {mets}")
    else:
        lines.append("anomaly_services: (empty)")

    lines.append("")
    lines.append(
        "Apply the CRFD algorithm (6 steps) to this data and output the result."
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

    lines.append("=== ANOMALY DATA ===")
    if anom_svcs:
        lines.append("anomaly_services:")
        for svc, mets in anom_svcs.items():
            lines.append(f"  {svc}: {mets}")
    else:
        lines.append("anomaly_services: (empty)")

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
