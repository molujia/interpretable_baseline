"""
explainer_v4.py — fault ticket generation and historical case retrieval.

Responsibilities in v4:
  1. Generate structured fault tickets (integrating CF-CBN ranking + Workflow A/B output).
  2. Retrieve similar historical cases (script pre-filter + optional LLM re-ranking).
  3. Infer the fault propagation path (BFS over the service topology).

LLM-based RCL analysis has been moved to Workflow A/B.
"""

import json
from typing import Dict, List, Optional, Tuple, Any

from agents.case_store import CaseReviewer, FaultCase
from utils.llm_adapter import get_llm


class ExplainerV4:

    def __init__(self, case_reviewer: CaseReviewer, metric_classifier=None):
        self.case_reviewer = case_reviewer
        self.metric_classifier = metric_classifier
        self._llm = get_llm()

    # ── BFSService call topology ──────────────────────────

    def infer_propagation(
        self,
        root: str,
        anomaly_summary: Dict[str, List[str]],
        topology: Dict[str, List[str]],
    ) -> List[str]:
        """
        BFS from the root cause along directed topology edges, tracing the anomaly propagation chain.
        anomaly_summary: {service: [metrics]} — presence of an entry indicates the service is anomalous.
        """
        path = [root]
        visited = {root}
        queue = [root]
        anomalous_svcs = set(anomaly_summary.keys())
        while queue:
            cur = queue.pop(0)
            for child in topology.get(cur, []):
                if child not in visited and child in anomalous_svcs:
                    path.append(child)
                    visited.add(child)
                    queue.append(child)
        return path

    # ──  ─────────────────────────────────────────────

    def retrieve_similar_cases(
        self,
        fault_data: dict,
        anomaly_summary: Dict[str, List[str]],
        fault_info: dict,
        top_k: int = 5,
        use_llm: bool = True,
    ) -> List[Tuple[FaultCase, str]]:
        """
        Retrieve similar historical cases.

        Returns:
            [(FaultCase, similarity_explanation), ...]
        """
        if not self.case_reviewer.cases:
            return []

        # Step 1: script pre-filter
        node_metrics = {svc: mets for svc, mets in anomaly_summary.items()}
        candidates = self.case_reviewer.script_filter(
            query_node_metrics=node_metrics,
            metric_classifier=self.metric_classifier,
            top_k=10,
        )

        if not candidates:
            return []

        # Step 2: If candidates > top_k and LLM is enabled, apply LLM re-ranking
        if len(candidates) > top_k and use_llm:
            return self._llm_rerank(candidates, anomaly_summary, fault_info, top_k)

        # Otherwise take the top_k directly from script results
        result = []
        for score, case in candidates[:top_k]:
            exp = self._script_explain(anomaly_summary, case, score)
            result.append((case, exp))
        return result

    def _llm_rerank(
        self,
        candidates: List[Tuple[float, FaultCase]],
        anomaly_summary: Dict[str, List[str]],
        fault_info: dict,
        top_k: int,
    ) -> List[Tuple[FaultCase, str]]:
        summaries = [
            {
                'idx': i,
                'case_id': c.case_id,
                'root_cause': c.root_cause,
                'fault_type': c.fault_type,
                'anomalous_services': c.anomalous_nodes()[:6],
                'script_score': round(sc, 3),
            }
            for i, (sc, c) in enumerate(candidates)
        ]

        prompt = f""" {top_k} 

: 
  Anomalous services: {list(anomaly_summary.keys())}
  : {fault_info.get('fault_type', 'N/A')}
  : {fault_info.get('fault_category', 'N/A')}

:
{json.dumps(summaries, indent=2, ensure_ascii=False)}

 {top_k} 
 JSON  markdown:
[
  {{"idx": <>, "similarity_explanation": "<1>"}},
  ...
]
 {top_k} """

        raw = self._llm.invoke_json(prompt)

        # fallback
        if isinstance(raw, dict) and raw.get('_mock'):
            raw = [
                {'idx': i, 'similarity_explanation': f' {sc:.2f}'}
                for i, (sc, _) in enumerate(candidates[:top_k])
            ]

        if not isinstance(raw, list):
            raw = []

        result = []
        for item in raw[:top_k]:
            idx = item.get('idx', -1)
            if 0 <= idx < len(candidates):
                _, case = candidates[idx]
                exp = item.get('similarity_explanation', '')
                result.append((case, exp))
        return result

    def _script_explain(
        self, anomaly_summary: Dict, case: FaultCase, score: float
    ) -> str:
        q_svcs = set(anomaly_summary.keys())
        c_svcs = set(case.anomalous_nodes())
        shared = q_svcs & c_svcs
        parts = []
        if shared:
            parts.append(f'Anomalous services: {sorted(shared)[:4]}')
        if not parts:
            parts.append(f': {score:.2f}')
        return '; '.join(parts)

    # ──  ──────────────────────────────────────────────────────

    def generate_ticket(
        self,
        fault_data: dict,
        cfcbn_result: dict,
        anomaly_summary: Dict[str, List[str]],
        propagation_path: List[str],
        fault_info: dict,
        similar_cases: List[Tuple[FaultCase, str]],
        workflow_a_result: Optional[Dict] = None,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate the complete fault ticket.

        Returns:
            {
              'ticket_text': str,
              'top_root_cause': str,
              'root_cause_ranking': List[Tuple[str, float]],
              'propagation_path': List[str],
              'similar_cases': [(FaultCase, explanation)],
              'workflow_a_enabled": bool,
            }
        """
        top3 = cfcbn_result.get('top3', [])
        top_root_cause = top3[0][0] if top3 else 'unknown'
        confidence = cfcbn_result.get('confidence', 0.0)
        n_acc = cfcbn_result.get('n_accumulated', 0)
        alpha = cfcbn_result.get('alpha', 1.0)

        lines = [
            '=' * 64,
            '  CLLM v4 — Fault Root Cause Analysis Ticket',
            '=' * 64,
            '',
            f"  Fault Type     : {fault_info.get('fault_type', 'N/A')}",
            f"  Fault Category : {fault_info.get('fault_category', 'N/A')}",
            f"  CF-CBN Status  : {n_acc} cases accumulated | α={alpha:.3f}",
            '',
            '── [A] CF-CBN Root Cause Ranking ──────────────────────',
            f"  Confidence margin: {confidence:.4f}  "
            f"({'LOW → LLM assist triggered' if cfcbn_result.get('needs_llm_assist') else 'HIGH'})",
            '',
        ]
        for i, (svc, score) in enumerate(cfcbn_result.get('top5', []), 1):
            mets = anomaly_summary.get(svc, [])
            met_str = ', '.join(mets[:4]) or '(no metrics)'
            lines.append(f"  #{i}  {svc:<28}  score={score:.4f}  [{met_str}]")

        lines += [
            '',
            '── [B] Fault Propagation Path ──────────────────────────',
            f"  {' → '.join(propagation_path) if propagation_path else 'N/A'}",
            '',
        ]

        # A
        if workflow_a_result and workflow_a_result.get('enabled'):
            lines += [
                '── [C] LLM Semantic Assist (Workflow A) ────────────────',
                f"  Trigger: {workflow_a_result.get('trigger_reason', '')}",
                '',
                f"  {workflow_a_result.get('overall_assessment', '')}",
                '',
                '  Top-3 Candidate Analysis:',
            ]
            for exp in workflow_a_result.get('candidate_explanations', []):
                lines.append(
                    f"  • {exp.get('service',''):<26} [{exp.get('signal_type','')}]"
                    f"  {exp.get('explanation','')}"
                )
            lines += [
                '',
                f"  SRE Guidance : {workflow_a_result.get('sre_guidance', '')}",
                f"  Confidence   : {workflow_a_result.get('llm_confidence_comment', '')}",
                '',
            ]
        else:
            lines += [
                '── [C] LLM Semantic Assist ──────────────────────────────',
                f"  Not triggered (CF-CBN confidence sufficient, margin={confidence:.4f})",
                '',
            ]

        # 
        lines += [
            '── [D] Similar Historical Cases ────────────────────────',
        ]
        if similar_cases:
            for i, (case, exp_text) in enumerate(similar_cases[:5], 1):
                lines.append(
                    f"  #{i}  [{case.case_id}] root={case.root_cause}  "
                    f"type={case.fault_type}  {exp_text}"
                )
        else:
            lines.append('  (no similar cases found)')

        lines += ['', '=' * 64, '']
        ticket_text = '\n'.join(lines)

        return {
            'ticket_text': ticket_text,
            'top_root_cause': top_root_cause,
            'root_cause_ranking': top3,
            'propagation_path': propagation_path,
            'similar_cases': similar_cases,
            'workflow_a_enabled': bool(workflow_a_result and workflow_a_result.get('enabled')),
        }
