"""
workflow_a_realtime.py — Workflow A: Real-Time Semantic Assist

Corresponds to Motivation 2 (post-deployment concern): SREs do not trust
system outputs and cannot make effective use of them in production.

Trigger conditions (any one sufficient):
  1. CF-CBN Top-1 vs Top-2 score gap is below the confidence threshold
     (the model itself is uncertain)
  2. The fault type belongs to a historically high-error-rate category
     (e.g., pod failure, DNS error)
  3. The anomaly signal is extremely weak (fewer than 2 anomalous metrics —
     the system is essentially guessing from priors)

What the LLM does:
  1. Interpret the current anomaly signal pattern:
       "root-cause signal" vs "victim/propagation signal"
  2. Provide a per-candidate credibility assessment for the Top-3
  3. Tell the SRE explicitly:
       - What evidence supports Top-1
       - How confident the assessment is
       - How to quickly verify or falsify it (which metric or log to check)

Design principles:
  - Runs BEFORE the SRE knows the true root cause.
    The LLM must reason purely from observable anomaly metrics.
  - Keeps the SRE in the loop: always ends with a concrete verification guide.
  - Output is structured JSON for reliable downstream consumption.
"""

from agents.llm_workflow import run_workflow_a

__all__ = ['run_workflow_a']
