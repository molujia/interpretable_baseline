"""
llm_workflow.py — CLLM v5 LLM Workflow Entry Points

Motivation (Motivation 2 — post-deployment concerns):
  SREs do not know how the system reaches its conclusions, are reluctant to
  trust its outputs for individual cases, and have no systematic mechanism
  for iterative tuning.

  Proposed solution → two LLM workflows:

  Workflow A (real-time semantic assist; root cause unknown)
    Purpose: help the SRE decide "should I trust this CF-CBN Top-1 output?"
    LLM tasks:
      1. Interpret the anomaly signal pattern (root-cause signal vs victim/propagation signal)
      2. Assess the credibility of each Top-3 candidate
      3. Provide a concrete verification guide (which metric or log to check)
    Trigger: after CF-CBN RCL completes, before the fault ticket is generated.
             The SRE does not yet know the true root cause.

  Workflow B (post-mortem diagnosis; root cause confirmed by SRE)
    Purpose: explain WHY CF-CBN missed and suggest how to improve the system.
    LLM tasks:
      B1 — Failure attribution: case_quality_issue vs method_issue
      B2 — Layered optimisation suggestions (data / algorithm / deployment)
    Trigger: SRE confirms the true root cause and finds CF-CBN Top-1 is wrong.

  Optional WF-B extensions (disabled by default):
    enable_case_review   — retrieve similar historical cases, explain known pattern match
    enable_propagation   — BFS-inferred fault propagation path analysis
"""

import json
from typing import Dict, List, Optional, Tuple

from utils.llm_adapter import get_llm
from utils.anonymizer import get_anonymizer
from agents.case_store import FaultCase


# ─────────────────────────────────────────────────────────────────────────────
# Trigger strategy (decide whether to call Workflow A)
# ─────────────────────────────────────────────────────────────────────────────

def should_trigger_workflow_a(
    cfcbn_scores: Dict[str, float],
    cfcbn_ranked: List[str],
    fault_type: str,
    anomaly_services: Dict[str, List[str]],
    anomaly_pods: Dict[str, List[str]],
    case_store,
    confidence_gap_threshold: float = 0.15,
    high_error_rate_threshold: float = 0.4,
    weak_signal_threshold: int = 2,
) -> Tuple[bool, str]:
    """
    Decide whether to trigger Workflow A (LLM real-time assist). Returns (should_trigger, reason).

    Trigger conditions (any one sufficient):
      C1: CF-CBN Top-1 vs Top-2 score gap < confidence_gap_threshold (model uncertain)
      C2: historical CF-CBN error rate for this fault_type >= high_error_rate_threshold
      C3: total anomalous metrics < weak_signal_threshold (very weak signal)
    """
    if len(cfcbn_ranked) >= 2:
        s1 = cfcbn_scores.get(cfcbn_ranked[0], 0.0)
        s2 = cfcbn_scores.get(cfcbn_ranked[1], 0.0)
        if s1 - s2 < confidence_gap_threshold:
            return True, f"C1: low confidence gap (Top-1={s1:.3f} vs Top-2={s2:.3f}, gap={s1-s2:.3f}<{confidence_gap_threshold})"

    err_rate = case_store.get_fault_type_error_rate(fault_type)
    if err_rate >= high_error_rate_threshold:
        return True, f"C2: high historical error rate for fault_type '{fault_type}' ({err_rate:.1%} >= {high_error_rate_threshold:.0%})"

    total_metrics = sum(len(v) for v in anomaly_services.values()) + \
                    sum(len(v) for v in anomaly_pods.values())
    if total_metrics < weak_signal_threshold:
        return True, f"C3: very weak signal (total_metrics={total_metrics} < {weak_signal_threshold})"

    return False, "LLM assist not required"


# ─────────────────────────────────────────────────────────────────────────────
# Workflow A: real-time semantic assist
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow_a_prompt(
    cfcbn_ranked: List[str],
    cfcbn_scores: Dict[str, float],
    anomaly_services: Dict[str, List[str]],
    anomaly_pods: Dict[str, List[str]],
    fault_type: str,
    fault_category: str,
    similar_cases: List[Tuple[float, FaultCase]],
    trigger_reason: str,
) -> str:
    """
    Build the Workflow A LLM prompt.

    Core objective: help the SRE decide whether to trust CF-CBN's Top-1 output.
    The SRE does not yet know the true root cause. The LLM reasons solely from observable anomaly signals
    to provide a verifiable interpretation of the CF-CBN output.
    """
    # Anomalous metric summary
    svc_summary = [f"  {svc}: {metrics}"
                   for svc, metrics in anomaly_services.items() if metrics]
    pod_summary = [f"  {pod}: {metrics}"
                   for pod, metrics in anomaly_pods.items() if metrics]

    # CF-CBN Top-3 (with raw CF score and fused score)
    top3_lines = []
    scores_list = sorted(cfcbn_scores.items(), key=lambda x: -x[1])
    for i, svc in enumerate(cfcbn_ranked[:3], 1):
        sc = cfcbn_scores.get(svc, 0.0)
        top3_lines.append(f"  #{i}  {svc:<30}  fused_score={sc:.4f}")

    # Top-1 vs Top-2 gap (transparency on CF-CBN uncertainty)
    gap_note = ""
    if len(cfcbn_ranked) >= 2:
        s1 = cfcbn_scores.get(cfcbn_ranked[0], 0.0)
        s2 = cfcbn_scores.get(cfcbn_ranked[1], 0.0)
        gap_note = f"Top-1 vs Top-2 score gap: {s1 - s2:.4f}"

    # Similar historical cases (showing past CF-CBN accuracy on similar faults)
    hist_lines = []
    for sim, c in similar_cases[:3]:
        result_str = "✓ correct" if c.cfcbn_correct else f"✗ incorrect (predicted rank in notes)"
        hist_lines.append(
            f"  [similarity={sim:.2f}]  root_cause={c.root_cause}  "
            f"fault_type={c.fault_type}  CF-CBN_result={result_str}"
        )

    prompt = f"""You are a senior SRE. A fault has occurred in the system. CF-CBN (Counterfactual-Causal Bayesian Network)
has completed automated root-cause localisation and produced a Top-3 candidate list.

Your task: help the SRE decide whether to trust CF-CBN's Top-1 output in this case.
Note: the SRE does not yet know the true root cause, and neither do you. Reason purely from observable anomaly signals.
The CF-CBN ranking is final — do not re-rank. Provide interpretation only.

[Trigger Reason (why CF-CBN triggered LLM assist)]
{trigger_reason}
{gap_note}

[Fault Information]
Fault category: {fault_category}
Fault type: {fault_type}

Anomalous services (metrics per service):
{chr(10).join(svc_summary) if svc_summary else "  (none)"}

Anomalous pods (metrics per pod):
{chr(10).join(pod_summary) if pod_summary else "  (none)"}

CF-CBN Top-3 candidates:
{chr(10).join(top3_lines)}


  Root-cause signals (strong): pod_cpu_usage, memory, server_error, timeout, pod_processes
  Propagation signals (weak): client_error, rrt, rrt_max, request (spread widely, low discriminative power)

Similar historical cases (CF-CBN accuracy on comparable faults):
{chr(10).join(hist_lines) if hist_lines else "  (no similar historical cases found)"}

——

: 
   Top-3 : 
    A. /
    B. /client_error
    C. /

: Top-1 Evaluation
   CF-CBN Top-1 Evaluation: 
    -  Top-1  Top-2/3 : 
    -  Top-1 : CF-CBN 
    - /: 

:  SRE 
   SRE" X  Y / Z  Top-1 Top-2"
  

 JSON  markdown 
{{
  "pattern_interpretation": "<: 1~2>",
  "candidate_analysis": [
    {{
      "rank": 1,
      "service": "<Service name>",
      "signal_type": "root_cause_signal | victim_signal | no_signal | mixed",
      "analysis": "<1~2>"
    }},
    {{
      "rank": 2,
      "service": "<Service name>",
      "signal_type": "root_cause_signal | victim_signal | no_signal | mixed",
      "analysis": "<1~2>"
    }},
    {{
      "rank": 3,
      "service": "<Service name>",
      "signal_type": "root_cause_signal | victim_signal | no_signal | mixed",
      "analysis": "<1~2>"
    }}
  ],
  "confidence_assessment": "< CF-CBN Top-1 Evaluation1~2>",
  "verification_guide": "<SRE 1~2>",
  "recommend_top1": true or false
}}
"""
    return prompt


def run_workflow_a(
    cfcbn_ranked: List[str],
    cfcbn_scores: Dict[str, float],
    anomaly_services: Dict[str, List[str]],
    anomaly_pods: Dict[str, List[str]],
    fault_type: str,
    fault_category: str,
    similar_cases: List[Tuple[float, FaultCase]],
    trigger_reason: str,
) -> dict:
    """
    A LLM dict mock 
    """
    llm  = get_llm()
    anon = get_anonymizer()
    prompt = build_workflow_a_prompt(
        cfcbn_ranked, cfcbn_scores, anomaly_services, anomaly_pods,
        fault_type, fault_category, similar_cases, trigger_reason,
    )
    # anonymised before sending to LLMDe-anonymised
    anon_prompt = anon.anonymize(prompt)
    raw_result  = llm.invoke_json(anon_prompt)
    result_str  = anon.deanonymize(
        json.dumps(raw_result, ensure_ascii=False)
    )
    try:
        result = json.loads(result_str)
    except Exception:
        result = raw_result

    if isinstance(result, dict) and result.get("_mock"):
        top1 = cfcbn_ranked[0] if cfcbn_ranked else "unknown"
        # 
        root_signals   = {"pod_cpu_usage","cpu_usage","pod_memory_working_set_bytes",
                          "server_error","timeout","pod_processes"}
        victim_signals = {"client_error","rrt","rrt_max","request","response"}
        candidate_analysis = []
        for i, svc in enumerate(cfcbn_ranked[:3]):
            mets = set(anomaly_services.get(svc, []))
            has_root   = bool(mets & root_signals)
            has_victim = bool(mets & victim_signals)
            if has_root and not has_victim:
                sig = "root_cause_signal"
                analysis = f"{svc} {', '.join(mets & root_signals)}"
            elif has_victim and not has_root:
                sig = "victim_signal"
                analysis = f"{svc} {', '.join(mets & victim_signals)}"
            elif not mets:
                sig = "no_signal"
                analysis = f"{svc}: no anomalous metrics observed; ranking driven by CBN prior"
            else:
                sig = "mixed"
                analysis = f"{svc} "
            candidate_analysis.append({"rank": i+1, "service": svc,
                                        "signal_type": sig, "analysis": analysis})

        top1_sig = candidate_analysis[0]["signal_type"] if candidate_analysis else "unknown"
        if top1_sig == "root_cause_signal":
            confidence = "Top-1 "
            recommend  = True
            verify     = f" {top1}  CPU//server_error "
        elif top1_sig == "victim_signal":
            confidence = "Top-1 CF-CBN "
            recommend  = False
            verify     = f" {top1}  client_error "
        elif top1_sig == "no_signal":
            confidence = "Top-1 CF-CBN Prior"
            recommend  = False
            verify     = f" {top1} "
        else:
            confidence = ""
            recommend  = True
            verify     = f" {top1} "

        result = {
            "pattern_interpretation": f"[Script Mode] {candidate_analysis[0]['analysis'] if candidate_analysis else ''}",
            "candidate_analysis": candidate_analysis,
            "confidence_assessment": confidence,
            "verification_guide": verify,
            "recommend_top1": recommend,
            "_mock": True,
        }

    result["trigger_reason"] = trigger_reason
    return result


def summarize_wfa_for_b(wfa_result: dict) -> str:
    """
    AB prompt 

    :  B1  B1 
    CF-CBN rankfault_type B1 

    : 
      1. ///
      2. WF-A 
      3. WF-A  Top-1 Evaluation
      4. recommend_top1 Top-1
     150~250  WF-A JSON 800~1200 
    """
    if not wfa_result:
        return ""

    lines = ["[WF-A ]"]

    pattern = wfa_result.get("pattern_interpretation", "")
    if pattern:
        lines.append(f"  : {pattern}")

    for ca in wfa_result.get("candidate_analysis", []):
        sig_map = {
            "root_cause_signal": "",
            "victim_signal":     "",
            "no_signal":         "",
            "mixed":             "",
        }
        sig_label = sig_map.get(ca.get("signal_type", ""), ca.get("signal_type", "?"))
        lines.append(f"  #{ca.get('rank')} {ca.get('service','?')}: [{sig_label}] "
                     f"{ca.get('analysis','')}")

    confidence = wfa_result.get("confidence_assessment", "")
    if confidence:
        lines.append(f"  Evaluation: {confidence}")

    recommend = wfa_result.get("recommend_top1", True)
    lines.append(f"  WF-A: {' Top-1' if recommend else ' Top-1'}")

    return "\n".join(lines)



def build_workflow_b1_prompt(
    cfcbn_top1: str,
    cfcbn_scores: Dict[str, float],
    cfcbn_ranked: List[str],
    true_root_cause: str,
    anomaly_services: Dict[str, List[str]],
    anomaly_pods: Dict[str, List[str]],
    fault_type: str,
    fault_category: str,
    similar_mismatch_cases: List[FaultCase],
    wfa_summary: Optional[str] = None,            # always 
    case_review_section: Optional[str] = None,    # Historical case review
    propagation_section: Optional[str] = None,    # Fault propagation path analysis
) -> str:
    """
    B prompt

    :  SRE ——
      " case /
       //"

    wfa_summaryA always 
      summarize_wfa_for_b() 
    """
    top5_lines = []
    for i, svc in enumerate(cfcbn_ranked[:5], 1):
        sc = cfcbn_scores.get(svc, 0.0)
        tag = ""
        if svc == cfcbn_top1:      tag += "  ← CF-CBN prediction (incorrect)"
        if svc == true_root_cause: tag += "  ← SRE-confirmed true root cause"
        top5_lines.append(f"  #{i}  {svc:<30}  score={sc:.4f}{tag}")

    svc_lines = [f"  {svc}: {m}" for svc, m in anomaly_services.items() if m]
    pod_lines = [f"  {pod}: {m}" for pod, m in anomaly_pods.items() if m]

    # 
    rc_metrics = anomaly_services.get(true_root_cause, []) + \
                 anomaly_pods.get(true_root_cause, [])
    rc_signal_note = (
        f" {true_root_cause} : {rc_metrics}"
        if rc_metrics else
        f" {true_root_cause} "
    )

    # CF-CBN 
    top1_metrics = anomaly_services.get(cfcbn_top1, []) + \
                   anomaly_pods.get(cfcbn_top1, [])

    # 
    hist_lines = []
    for c in similar_mismatch_cases[:5]:
        top1_in_notes = c.notes  # "cfcbn_rank=X" 
        hist_lines.append(
            f"  root_cause={c.root_cause}  fault_type={c.fault_type}  {top1_in_notes}"
        )

    prompt = f""" AIOps CF-CBN 
SRE  SRE 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CF-CBN prediction (incorrect): {cfcbn_top1}
  {cfcbn_top1} : {top1_metrics}

SRE-confirmed true root cause:   {true_root_cause}
  {rc_signal_note}

Fault category: {fault_category}
Fault type: {fault_type}

CF-CBN Top-5 rank
{chr(10).join(top5_lines)}


{chr(10).join(svc_lines) if svc_lines else "  "}

Pod
{chr(10).join(pod_lines) if pod_lines else "  "}

fault_type
{chr(10).join(hist_lines) if hist_lines else "  "}

CF-CBN 
  4.0: pod_cpu_usage
  3.5: memory, rrt, rrt_max
  3.0: timeout
  2.5: client_error, server_error
  0.5: request, response

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{f"Workflow A real-time assist summary (executed during fault, provided for reference){chr(10)}{wfa_summary}{chr(10)}" if wfa_summary else ""}
{f"{case_review_section}{chr(10)}{chr(10)}" if case_review_section else ""}{f"{propagation_section}{chr(10)}{chr(10)}" if propagation_section else ""}
Core judgement (answer this first)


  "case_quality_issue"
     {true_root_cause} 
    
    : Root-cause service
              /
    : 

  "method_issue"
     CF-CBN 
    : Root-cause servicescore
              
              Prior
    : //

  "both"

Further analysis (building on the core judgement)
1. CF-CBN  {cfcbn_top1} score
2.  {true_root_cause} rank///Prior
3. fault_type

 JSON  markdown 
{{
  "failure_attribution": "case_quality_issue | method_issue | both",
  "attribution_reasoning": "<1~2>",
  "failure_summary": "<>",
  "why_top1_ranked_first": "</ {cfcbn_top1} score1~2>",
  "why_true_rc_ranked_low": "< {true_root_cause} score1~2>",
  "dimension_analysis": {{
    "metric_weight": "< null>",
    "propagation_path": "< null>",
    "cbn_accumulation": "< null>",
    "monitoring_coverage": "< null>"
  }},
  "systematic_pattern": "<1>"
}}
"""
    return prompt


def build_workflow_b2_prompt(
    diagnosis: dict,
    cfcbn_top1: str,
    true_root_cause: str,
    fault_type: str,
    anomaly_services: Dict[str, List[str]],
    anomaly_pods: Dict[str, List[str]],
) -> str:
    """
    B prompt

     B1  failure_attribution : 
      - case_quality_issue → 
      - method_issue       → //
      - both               → 
    """
    attribution = diagnosis.get("failure_attribution", "method_issue")
    attribution_reasoning = diagnosis.get("attribution_reasoning", "")
    failure_summary = diagnosis.get("failure_summary", "N/A")
    why_top1   = diagnosis.get("why_top1_ranked_first", "N/A")
    why_rc_low = diagnosis.get("why_true_rc_ranked_low", "N/A")
    dim = diagnosis.get("dimension_analysis", {})
    systematic = diagnosis.get("systematic_pattern", "N/A")

    #  attribution 
    if attribution == "case_quality_issue":
        task_focus = """: case 
: 
:  CF-CBN 
"""

    elif attribution == "method_issue":
        task_focus = """: 
:  CF-CBN 
:  CF-CBN //"""

    else:  # both
        task_focus = """:  + 
: 
  1. /
  2.  CF-CBN 
"""

    prompt = f""" AIOps  CF-CBN  SRE 

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 B1
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Failure attribution: {attribution}
: {attribution_reasoning}
: {failure_summary}
CF-CBN : {why_top1}
: {why_rc_low}
: {systematic}

Dimension analysis:
  : {dim.get('metric_weight', 'N/A')}
  : {dim.get('propagation_path', 'N/A')}
  : {dim.get('cbn_accumulation', 'N/A')}
  : {dim.get('monitoring_coverage', 'N/A')}

Fault type: {fault_type}
CF-CBN : {cfcbn_top1} → True root cause: {true_root_cause}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{task_focus}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CF-CBN 
  1. : METRIC_WEIGHTS fault_type
  2. : Service call topology
  3. CBN : Prior λfault_typePrior
  4. : 

: 
  - ""
  - //
  - " rrt  network_attack  3.5  1.5"
  - 

 JSON  markdown 
{{
  "recommendations": [
    {{
      "layer": " |  | CBN | ",
      "category": "method_improvement | monitoring_improvement",
      "priority": " |  | ",
      "action": "<1~2>",
      "expected_effect": "<1>",
      "implementation_hint": "<>"
    }}
  ],
  "immediate_action": "<SRE >",
  "iteration_note": "< SRE : 1>"
}}
"""
    return prompt


def run_workflow_b(
    cfcbn_top1: str,
    cfcbn_scores: Dict[str, float],
    cfcbn_ranked: List[str],
    true_root_cause: str,
    anomaly_services: Dict[str, List[str]],
    anomaly_pods: Dict[str, List[str]],
    fault_type: str,
    fault_category: str,
    similar_mismatch_cases: List[FaultCase],
    wfa_result: Optional[dict] = None,        # WF-A result passed in always mode
    # ── M2 optional extensions: disabled by default; pass True to enable ──────────────────────────
    enable_case_review: bool = False,          # Historical case review
    similar_all_cases: Optional[List] = None,  # List of similar historical cases
    enable_propagation: bool = False,          # Fault propagation path analysis
    propagation_path: Optional[List[str]] = None,
    topology: Optional[Dict[str, List[str]]] = None,
) -> dict:
    """
    BB1 failure attribution + B2 optimisation suggestions
     {"diagnosis": {...}, "recommendations": {...}}

    wfa_resultA dictB1 prompt 
      via summarize_wfa_for_b A→B 
       WF-A JSON  prompt token
    """
    llm  = get_llm()
    anon = get_anonymizer()

    # Generate compact WF-A summary (only populated in always mode)
    wfa_summary = summarize_wfa_for_b(wfa_result) if wfa_result else None

    # ── Extension A: Historical Case Review ──────────────────────────────────────────────
    case_review_section = ""
    if enable_case_review and similar_all_cases:
        lines = ["Historical Case Review (CBN similarity search results)"]
        lines.append(f"  This fault is similar to the following {len(similar_all_cases[:5])}  historical cases:")
        for i, item in enumerate(similar_all_cases[:5], 1):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                first, second = item
                if hasattr(first, "root_cause"):        # (FaultCase, explanation)
                    case, exp = first, second
                    sim_note = str(exp)[:60]
                elif hasattr(second, "root_cause"):     # (score, FaultCase)
                    case, sim_note = second, f"similarity={first:.2f}"
                else:
                    continue
            else:
                continue
            outcome = "✓ CF-CBN" if getattr(case, "cfcbn_correct", False) else "✗ CF-CBN"
            lines.append(
                f"  #{i} [{case.case_id}] root_cause={case.root_cause}  "
                f"fault_type={case.fault_type}  {outcome}  ({sim_note})"
            )
            anomalous = list(getattr(case, "node_metrics", {}).keys())[:3]
            if anomalous:
                lines.append(f"     Anomalous services: {anomalous}")
        lines.append("")
        lines.append("  Based on the above historical cases, analyse whether this fault matches a known pattern,")
        lines.append("  and what value past resolution experience has for the current root-cause localisation.")
        case_review_section = "\n".join(lines)

    # ── BFault propagation path analysis ────────────────────────────────────────────
    propagation_section = ""
    if enable_propagation:
        if propagation_path and len(propagation_path) > 1:
            path_str = " → ".join(propagation_path)
            node_annotations = []
            for svc in propagation_path:
                mets = anomaly_services.get(svc, []) + anomaly_pods.get(svc, [])
                mets_str = ", ".join(mets[:3]) if mets else "()"
                node_annotations.append(f"  {svc}: {mets_str}")
            lines_p = [
                "Fault propagation path analysisBFS ",
                f"  Inferred propagation chain: {path_str}", "",
                "  Anomalous metrics per node:",
            ]
            lines_p.extend(node_annotations)
            lines_p += [
                "", "  Based on the propagation chain above, analyse:",
                "  1. 1. Which node is the fault origin (strongest signal, no upstream anomaly source)",
                "  2. 2. Which nodes are propagation victims (signal originates from upstream)",
                "  3. 3. Whether the propagation path is consistent with the CF-CBN root-cause ranking",
            ]
            propagation_section = "\n".join(lines_p)
        elif topology and true_root_cause:
            propagation_section = (
                f"Fault propagation path analysis\n"
                f"  Root-cause service {true_root_cause} has no detected downstream anomaly propagation."
            )

    # ── B1anonymised before sending to LLM────────────────────────────────
    p1 = build_workflow_b1_prompt(
        cfcbn_top1, cfcbn_scores, cfcbn_ranked,
        true_root_cause, anomaly_services, anomaly_pods,
        fault_type, fault_category, similar_mismatch_cases,
        wfa_summary=anon.anonymize(wfa_summary) if wfa_summary else None,
        case_review_section=anon.anonymize(case_review_section) if case_review_section else None,
        propagation_section=anon.anonymize(propagation_section) if propagation_section else None,
    )
    raw_diag  = llm.invoke_json(anon.anonymize(p1))
    try:
        diagnosis = json.loads(anon.deanonymize(
            json.dumps(raw_diag, ensure_ascii=False)
        ))
    except Exception:
        diagnosis = raw_diag

    if isinstance(diagnosis, dict) and diagnosis.get("_mock"):
        rc_metrics = anomaly_services.get(true_root_cause, []) + \
                     anomaly_pods.get(true_root_cause, [])
        top1_metrics = anomaly_services.get(cfcbn_top1, [])
        victim_signals = {"client_error", "rrt", "rrt_max", "request", "response"}

        if not rc_metrics:
            attribution = "case_quality_issue"
            attr_reason = f" {true_root_cause} "
        elif all(m in victim_signals for m in top1_metrics):
            attribution = "method_issue"
            attr_reason = f"{cfcbn_top1} CF-CBN "
        else:
            attribution = "method_issue"
            attr_reason = f" CF-CBN Prior"

        diagnosis = {
            "failure_attribution": attribution,
            "attribution_reasoning": attr_reason,
            "failure_summary": f"CF-CBN  {cfcbn_top1}  {true_root_cause}",
            "why_top1_ranked_first": "LLM ",
            "why_true_rc_ranked_low": "LLM ",
            "dimension_analysis": {
                "metric_weight": "",
                "propagation_path": "",
                "cbn_accumulation": None,
                "monitoring_coverage": None if rc_metrics else f"{true_root_cause} ",
            },
            "systematic_pattern": "LLM",
            "_mock": True,
        }

    # ── B2anonymised before sending to LLM──────────────────────────────
    p2 = build_workflow_b2_prompt(
        diagnosis, cfcbn_top1, true_root_cause, fault_type,
        anomaly_services, anomaly_pods,
    )
    raw_rec = llm.invoke_json(anon.anonymize(p2))
    try:
        recommendations = json.loads(anon.deanonymize(
            json.dumps(raw_rec, ensure_ascii=False)
        ))
    except Exception:
        recommendations = raw_rec

    if isinstance(recommendations, dict) and recommendations.get("_mock"):
        attribution = diagnosis.get("failure_attribution", "method_issue")
        if attribution == "case_quality_issue":
            recs = [{
                "layer": "",
                "category": "monitoring_improvement",
                "priority": "",
                "action": f" {true_root_cause} CPU//",
                "expected_effect": " CF-CBN ",
                "implementation_hint": " pod_cpu_usage  server_error ",
            }]
            quick = f" {true_root_cause} "
            note  = " case "
        else:
            recs = [
                {
                    "layer": "",
                    "category": "method_improvement",
                    "priority": "",
                    "action": f" {fault_type} rrt/client_error",
                    "expected_effect": "",
                    "implementation_hint": " METRIC_WEIGHTS fault_type",
                },
                {
                    "layer": "CBN",
                    "category": "method_improvement",
                    "priority": "",
                    "action": " Laplace  λfault_typePrior",
                    "expected_effect": "fault_typePrior",
                    "implementation_hint": None,
                },
            ]
            quick = f" {fault_type}  rrt/client_error "
            note  = " 10+ fault_typeEvaluation"

        recommendations = {
            "recommendations": recs,
            "immediate_action": quick,
            "iteration_note": note,
            "_mock": True,
        }

    return {
        "diagnosis"            : diagnosis,
        "recommendations"      : recommendations,
        "case_review"          : case_review_section if enable_case_review else None,
        "propagation"          : propagation_section if enable_propagation else None,
        "case_review_enabled"  : enable_case_review,
        "propagation_enabled"  : enable_propagation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CF-CBN rank + A
# ─────────────────────────────────────────────────────────────────────────────



def generate_fault_ticket(
    cfcbn_ranked: List[str],
    cfcbn_scores: Dict[str, float],
    anomaly_services: Dict[str, List[str]],
    anomaly_pods: Dict[str, List[str]],
    fault_type: str,
    fault_category: str,
    workflow_a_result: Optional[dict] = None,
    similar_cases: Optional[List[Tuple[float, FaultCase]]] = None,
    cf_raw_scores: Optional[Dict[str, float]] = None,
    alpha_info: Optional[str] = None,
) -> str:
    """
    Generate a structured, fully-English fault ticket for SRE review.

    Sections:
      [1] CF-CBN ranking (Top-5) with alpha info
      [2] Anomaly evidence: per-candidate raw metric names (judgement basis)
      [3] LLM Workflow A semantic interpretation (if triggered)
      [4] Similar historical cases
      [5] Action guidance
    """
    from cfcbn.cbn_accumulator import base_name as _bn

    lines = []
    lines.append("=" * 70)
    lines.append("  CLLM v5 -- Fault Analysis Ticket")
    lines.append(f"  Fault category : {fault_category}   |   Fault type: {fault_type}")
    lines.append("=" * 70)

    # [1] CF-CBN ranking
    lines.append("")
    lines.append("[ 1 ]  CF-CBN Root-Cause Ranking  "
                 "(Counterfactual do-intervention + Bayesian fusion)")
    for i, svc in enumerate(cfcbn_ranked[:5], 1):
        sc = cfcbn_scores.get(svc, 0.0)
        lines.append(f"  #{i}  {svc:<34}  fused_score={sc:.4f}")
    if alpha_info:
        lines.append(f"       [alpha={alpha_info}]")

    # [2] Anomaly evidence per candidate (judgement basis)
    lines.append("")
    lines.append("[ 2 ]  Anomaly Evidence  "
                 "(raw metrics that drove the CF score -- judgement basis)")
    lines.append("  Top candidates:")
    has_evidence = False
    for svc in cfcbn_ranked[:5]:
        svc_metrics = list(anomaly_services.get(svc, []))
        pod_metrics = []
        for pod, mets in anomaly_pods.items():
            if _bn(pod) == svc and mets:
                pod_metrics.extend(mets)
        all_m = sorted(set(svc_metrics + pod_metrics))
        cf_sc = cf_raw_scores.get(svc, None) if cf_raw_scores else None
        cf_str = f"  (CF_raw={cf_sc:.4f})" if cf_sc is not None else ""
        if all_m:
            lines.append(f"    {svc:<34}  {all_m}{cf_str}")
            has_evidence = True
        else:
            lines.append(f"    {svc:<34}  (no anomalous metrics){cf_str}")

    lines.append("  Other anomalous entities:")
    shown = set(cfcbn_ranked[:5])
    for svc, metrics in anomaly_services.items():
        if svc not in shown and metrics:
            lines.append(f"    [service] {svc:<30}  {sorted(metrics)}")
            has_evidence = True
    for pod, metrics in anomaly_pods.items():
        if metrics:
            lines.append(f"    [pod]     {pod:<30}  {sorted(metrics)}")
            has_evidence = True
    if not has_evidence:
        lines.append("    (no anomalous metrics observed; "
                     "ranking driven by CBN prior only)")

    # [3] LLM Workflow A
    if workflow_a_result and not workflow_a_result.get("_mock"):
        lines.append("")
        lines.append("[ 3 ]  LLM Workflow A -- Semantic Interpretation  "
                     "(assists SRE in assessing CF-CBN Top-1)")
        lines.append(
            f"  Trigger reason : {workflow_a_result.get('trigger_reason', 'N/A')}")
        lines.append(
            f"  Pattern        : "
            f"{workflow_a_result.get('pattern_interpretation', '')}")
        lines.append("")
        lines.append("  Candidate signal analysis:")
        for ca in workflow_a_result.get("candidate_analysis", []):
            sig = ca.get('signal_type', '')
            sig_tag = {
                "root_cause_signal": "[ROOT CAUSE]",
                "victim_signal":     "[VICTIM    ]",
                "no_signal":         "[NO SIGNAL ]",
                "mixed":             "[MIXED     ]",
            }.get(sig, f"[{sig:<11}]")
            lines.append(
                f"    #{ca.get('rank')} {ca.get('service','?'):<30} "
                f"{sig_tag}  {ca.get('analysis','')}"
            )
        lines.append(
            f"  Assessment     : "
            f"{workflow_a_result.get('confidence_assessment', '')}")
        verify = workflow_a_result.get("verification_guide", "")
        if verify:
            lines.append(f"  Verification   : {verify}")
        rec = workflow_a_result.get("recommend_top1", True)
        lines.append(
            f"  LLM verdict    : "
            f"{'[OK] Trust Top-1' if rec else '[WARN] Uncertain -- verify before acting on Top-1'}"
        )
    elif workflow_a_result and workflow_a_result.get("_mock"):
        lines.append("")
        lines.append("[ 3 ]  LLM Workflow A  (LLM disabled -- mock mode)")
        lines.append(f"  {workflow_a_result.get('confidence_assessment', '')}")
        verify = workflow_a_result.get("verification_guide", "")
        if verify:
            lines.append(f"  Verification : {verify}")

    # [4] Similar historical cases
    if similar_cases:
        lines.append("")
        lines.append("[ 4 ]  Similar Historical Cases")
        for sim, c in similar_cases[:3]:
            ok = "CF-CBN correct" if c.cfcbn_correct else "CF-CBN missed"
            lines.append(
                f"  [similarity={sim:.2f}]  root_cause={c.root_cause}  "
                f"fault_type={c.fault_type}  {ok}"
            )

    # [5] Action guidance
    lines.append("")
    lines.append("-" * 70)
    lines.append("  ACTION  Confirm or correct Top-1 using the SRE console.")
    lines.append("  If CF-CBN is wrong, trigger Workflow B for post-mortem diagnosis.")
    lines.append("=" * 70)

    return "\n".join(lines)

