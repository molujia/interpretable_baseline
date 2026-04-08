"""
workflow_b_postmortem.py — Workflow B: Post-Mortem Diagnosis

Corresponds to Motivation 2 (post-deployment concern): SREs believe the
system cannot be tuned continuously and are reluctant to rely on it.

Trigger condition:
  The SRE has confirmed the true root cause and found that CF-CBN Top-1
  is incorrect (a miss case).

What the LLM does:
  B1 — Failure Attribution:
       Diagnoses WHY CF-CBN missed, choosing between:
         (a) case_quality_issue  — the fault data itself lacks discriminative signal
                                    (monitoring blind-spot, weak anomaly indicators)
         (b) method_issue        — the algorithm has room to improve
                                    (weight calibration, topology, accumulation strategy)

  B2 — Layered Optimisation Suggestions:
       Proposes concrete, prioritised improvements across three layers:
         - Data / monitoring layer    (improve metric coverage)
         - Algorithm / weight layer   (tune METRIC_WEIGHTS, alpha, topology)
         - System / deployment layer  (expand monitoring, revisit topology)

Optional extensions (disabled by default; enable via pipeline flags):
  enable_case_review   — retrieve historically similar cases from the case store
                         and explain whether this fault matches a known pattern
  enable_propagation   — BFS-inferred fault propagation path analysis,
                         helping the SRE understand how the anomaly spread

Design principles:
  - Runs AFTER the SRE confirms the true root cause (post-mortem context).
  - Strictly separates data-quality issues from algorithmic issues to give
    actionable, targeted recommendations.
  - Structured JSON output for reliable downstream consumption.
"""

from agents.llm_workflow import run_workflow_b

__all__ = ['run_workflow_b']
