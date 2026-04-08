"""
Case Reviewer — pure-script implementation.

Responsibilities:
  1. Maintain a persistent store of historical fault cases and their RCL results.
  2. Update the CBN lower-layer statistics for each new case.
  3. Record the exact CPT key updates and alpha changes produced by each case.
  4. Provide script-based similarity pre-filtering of historical cases for the Explainer.

Persistence strategy:
  - cases.jsonl: append-only, one JSON object per line; crash-safe.
  - cbn_update_log.jsonl: appended after every CBN update, granularity = CPT key.

Startup:
  - Automatically loads all historical cases from cases.jsonl into memory.
  - Does NOT replay CBN updates (the CBN is rebuilt or restored externally).
"""

import json
import math
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Set, Tuple


# ======================================================================
# 
# ======================================================================

@dataclass
class CBNUpdateRecord:
    """Record of a single CPT-key update applied by one case to the CBN."""
    case_id: str
    timestamp: str
    node: str
    parent_cfg: list       # [(parent, state), ...]
    cpt_key: str           # 
    state_updated: int     # 
    alpha_before: list     #  alpha 
    alpha_after: list      #  alpha 
    weight: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FaultCase:
    """Complete record of a single fault case."""
    case_id: str
    timestamp: str
    root_cause: str
    fault_category: str
    fault_type: str
    node_states: Dict[str, int]
    node_metrics: Dict[str, List[str]]
    metric_classes_hit: Dict[str, List[int]]
    rcl_scores: Dict[str, float]
    propagation_path: List[str]
    cbn_updated: bool = False
    cbn_update_keys: List[str] = field(default_factory=list)
    is_novel: bool = False
    notes: str = ""
    cfcbn_correct: bool = False   # CF-CBN Top-1 
    cfcbn_rank: int = -1          # CF-CBN 

    def anomalous_nodes(self) -> List[str]:
        return [n for n, s in self.node_states.items() if s == 1]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FaultCase":
        d.setdefault("cbn_update_keys", [])
        d.setdefault("cbn_updated", False)
        d.setdefault("is_novel", False)
        d.setdefault("notes", "")
        d.setdefault("cfcbn_correct", d.get("cbn_updated", False))
        d.setdefault("cfcbn_rank", -1)
        return cls(**d)


# ======================================================================
# CaseReviewer
# ======================================================================

class CaseReviewer:
    """
    Case Reviewer — 

    Persist to disk: 
      <store_dir>/cases.jsonl            FaultCase
      <store_dir>/cbn_update_log.jsonl   CBNUpdateRecord
    """

    def __init__(self, store_dir: str = "case_store"):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)

        self.cases_path   = os.path.join(store_dir, "cases.jsonl")
        self.cbn_log_path = os.path.join(store_dir, "cbn_update_log.jsonl")

        self.cases: List[FaultCase] = []
        self._id_set: Set[str] = set()
        self._id_counter: int = 0

        # Auto-load on startup
        self._load_cases()
        print(f"[CaseReviewer] Store: {os.path.abspath(store_dir)} | "
              f"Loaded {len(self.cases)} historical cases")

    # ------------------------------------------------------------------
    # 
    # ------------------------------------------------------------------

    def _load_cases(self):
        if not os.path.exists(self.cases_path):
            return
        with open(self.cases_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    case = FaultCase.from_dict(d)
                    if case.case_id not in self._id_set:
                        self.cases.append(case)
                        self._id_set.add(case.case_id)
                        try:
                            num = int(case.case_id.split("-")[-1])
                            self._id_counter = max(self._id_counter, num)
                        except ValueError:
                            pass
                except Exception as e:
                    print(f"[CaseReviewer] Warn: skip bad line: {e}")

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _next_id(self) -> str:
        self._id_counter += 1
        return f"CASE-{self._id_counter:04d}"

    # ------------------------------------------------------------------
    # Core: add case + CBN update + persist
    # ------------------------------------------------------------------

    def add_case(
        self,
        root_cause: str,
        fault_category: str,
        fault_type: str,
        node_states: Dict[str, int],
        node_metrics: Dict[str, List[str]],
        metric_classes_hit: Dict[str, List[int]],
        rcl_scores: Dict[str, float],
        propagation_path: List[str],
        is_novel: bool = False,
        notes: str = "",
        cbn_model=None,
    ) -> "FaultCase":
        case_id   = self._next_id()
        timestamp = datetime.now().isoformat()

        # ── 1. CBN CPT key alpha ──────────────
        cbn_updated     = False
        cbn_update_keys: List[str] = []

        if cbn_model is not None:
            cbn_updated     = True
            cbn_update_keys = self._update_cbn_and_log(
                cbn_model, case_id, timestamp, node_states
            )

        # ── 2.  FaultCase ─────────────────────────────────────────
        case = FaultCase(
            case_id          = case_id,
            timestamp        = timestamp,
            root_cause       = root_cause,
            fault_category   = fault_category,
            fault_type       = fault_type,
            node_states      = node_states,
            node_metrics     = node_metrics,
            metric_classes_hit = metric_classes_hit,
            rcl_scores       = rcl_scores,
            propagation_path = propagation_path,
            cbn_updated      = cbn_updated,
            cbn_update_keys  = cbn_update_keys,
            is_novel         = is_novel,
            notes            = notes,
        )

        # ── 3.  +  ────────────────────────────────────────
        self.cases.append(case)
        self._id_set.add(case_id)
        self._append_case(case)

        print(f"[CaseReviewer] ✓ {case_id} | root={root_cause} | "
              f"novel={is_novel} | cbn={cbn_updated}({len(cbn_update_keys)} keys)")
        return case

    # ------------------------------------------------------------------
    # CBN update + per-key changelog
    # ------------------------------------------------------------------

    def _update_cbn_and_log(
        self,
        cbn_model,
        case_id: str,
        timestamp: str,
        node_states: Dict[str, int],
    ) -> List[str]:
        """
         CBN CPT key  alpha 
         CPT key 
        """
        updated_keys: List[str] = []
        records: List[CBNUpdateRecord] = []

        for node in cbn_model.nodes:
            if node in cbn_model.masked_nodes:
                continue
            if node not in node_states:
                continue

            parents    = cbn_model.get_parents(node)
            parent_cfg = tuple(sorted(
                (p, node_states[p]) for p in parents if p in node_states
            ))
            key     = (node, parent_cfg)
            key_str = f"{node} | parents={dict(parent_cfg)}"

            # Snapshot before update
            alpha_before = list(cbn_model.cpt.alpha[key].copy())

            # Update CPT
            state_val = node_states[node]
            cbn_model.cpt.update(key, state_val, weight=1.0)

            # Snapshot after update
            alpha_after = list(cbn_model.cpt.alpha[key].copy())

            updated_keys.append(key_str)
            records.append(CBNUpdateRecord(
                case_id      = case_id,
                timestamp    = timestamp,
                node         = node,
                parent_cfg   = list(parent_cfg),
                cpt_key      = key_str,
                state_updated= state_val,
                alpha_before = [round(x, 6) for x in alpha_before],
                alpha_after  = [round(x, 6) for x in alpha_after],
                weight       = 1.0,
            ))

        # Sync cbn_model.cases (required by observe_pro)
        cbn_model.cases.append(dict(node_states))

        # Append to log
        self._append_cbn_log(records)
        return updated_keys

    # ------------------------------------------------------------------
    # Append-write (crash-safe)
    # ------------------------------------------------------------------

    def _append_case(self, case: FaultCase):
        with open(self.cases_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    def _append_cbn_log(self, records: List[CBNUpdateRecord]):
        if not records:
            return
        with open(self.cbn_log_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Query CBN update log
    # ------------------------------------------------------------------

    def get_cbn_updates_for_case(self, case_id: str) -> List[dict]:
        """Read all CBN update records for a case from disk."""
        results = []
        if not os.path.exists(self.cbn_log_path):
            return results
        with open(self.cbn_log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("case_id") == case_id:
                        results.append(d)
                except Exception:
                    pass
        return results

    def get_cbn_update_summary(self) -> Dict[str, int]:
        """Count how many CPT keys each case updated."""
        summary: Dict[str, int] = {}
        if not os.path.exists(self.cbn_log_path):
            return summary
        with open(self.cbn_log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    cid = d.get("case_id", "UNKNOWN")
                    summary[cid] = summary.get(cid, 0) + 1
                except Exception:
                    pass
        return summary

    # ------------------------------------------------------------------
    # Script-based pre-filter (called by Explainer)
    # ------------------------------------------------------------------

    @staticmethod
    def _jaccard(a: Set, b: Set) -> float:
        union = a | b
        return len(a & b) / len(union) if union else 0.0

    @staticmethod
    def _cosine(s1: Dict[str, int], s2: Dict[str, int]) -> float:
        nodes = set(s1) | set(s2)
        dot = sum(s1.get(n, 0) * s2.get(n, 0) for n in nodes)
        n1 = math.sqrt(sum(v ** 2 for v in s1.values()))
        n2 = math.sqrt(sum(v ** 2 for v in s2.values()))
        return dot / (n1 * n2) if n1 and n2 else 0.0

    def script_filter(
        self,
        query_node_metrics: Dict[str, List[str]],
        metric_classifier=None,
        exclude_case_id: Optional[str] = None,
        top_k: int = 10,
    ) -> List[Tuple[float, "FaultCase"]]:
        """
        :  metric_enum  top_k 

        per node
           query  case  metric : 
            consistent   = |classes_query ∩ classes_case|    
            inconsistent = |classes_query △ classes_case|    
          node_score = consistent - inconsistent
           = Σ node_score

         metric_classifier 
        """
        if not self.cases:
            return []

        scored: List[Tuple[float, FaultCase]] = []
        for case in self.cases:
            if exclude_case_id and case.case_id == exclude_case_id:
                continue

            if metric_classifier is not None:
                score = metric_classifier.similarity_score(
                    query_node_metrics, case.node_metrics
                )
            else:
                # fallback: 
                q_states = {n: (1 if ms else 0) for n, ms in query_node_metrics.items()}
                score = self._cosine(q_states, case.node_states)

            scored.append((score, case))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    # ------------------------------------------------------------------
    # Additional queries
    # ------------------------------------------------------------------

    def get_case(self, case_id: str) -> Optional[FaultCase]:
        for c in self.cases:
            if c.case_id == case_id:
                return c
        return None

    def recent_cases(self, n: int = 10) -> List[FaultCase]:
        return self.cases[-n:]

    def print_summary(self):
        print(f"\n[CaseReviewer Summary]")
        print(f"  Total cases  : {len(self.cases)}")
        print(f"  Store dir    : {os.path.abspath(self.store_dir)}")
        if self.cases:
            novel   = sum(1 for c in self.cases if c.is_novel)
            updated = sum(1 for c in self.cases if c.cbn_updated)
            print(f"  Novel cases  : {novel}")
            print(f"  CBN updated  : {updated}")
            print(f"  Latest       : {self.cases[-1].case_id} "
                  f"@ {self.cases[-1].timestamp[:19]}")
        cbn_summary = self.get_cbn_update_summary()
        if cbn_summary:
            total_keys = sum(cbn_summary.values())
            print(f"  CPT updates  : {total_keys} key-updates across "
                  f"{len(cbn_summary)} cases")

    def __len__(self) -> int:
        return len(self.cases)

    # ── v4  ──────────────────────────────────────────────────────────

    def get_similar_cases(
        self,
        anomaly_services: dict,
        anomaly_pods: dict,
        fault_type: str,
        top_k: int = 5,
    ):
        """Return similar historical cases as a list of (sim_score, FaultCase) tuples."""
        query_nodes = set(anomaly_services.keys()) | set(anomaly_pods.keys())
        results = []
        for c in self.cases:
            case_nodes = set(c.node_metrics.keys())
            sim = self._jaccard(query_nodes, case_nodes)
            if sim > 0 and c.fault_type == fault_type:
                sim += 0.1   # 
            if sim > 0:
                results.append((sim, c))
        results.sort(key=lambda x: -x[0])
        return results[:top_k]

    def get_mismatch_cases(self, fault_type: str, top_k: int = 5):
        """Return historical miss cases of the same fault type."""
        mismatches = [c for c in self.cases
                      if c.fault_type == fault_type and not c.cbn_updated]
        return mismatches[-top_k:]

    def get_fault_type_error_rate(self, fault_type: str) -> float:
        """Return the historical error rate (0.0–1.0) for this fault type."""
        total = [c for c in self.cases if c.fault_type == fault_type]
        if not total:
            return 0.5    # 
        errors = [c for c in total if not c.cbn_updated]
        return len(errors) / len(total)

    def add_v4_case(
        self,
        case_id: int,
        anomaly_services: dict,
        anomaly_pods: dict,
        fault_type: str,
        fault_category: str,
        root_cause: str,
        cfcbn_top1: str,
        cfcbn_top3: list,
        cfcbn_scores: dict,
        cfcbn_correct: bool,
        cfcbn_rank: int,
        llm_assisted: bool = False,
        llm_workflow_a=None,
        llm_diagnosis=None,
    ) -> "FaultCase":
        """Write a case record from the v4 pipeline (simplified; no CBN model update)."""
        from datetime import datetime
        node_metrics = {**anomaly_services, **anomaly_pods}
        node_states  = {k: 1 for k in node_metrics}
        case = FaultCase(
            case_id   = str(case_id),
            timestamp = datetime.now().isoformat(),
            root_cause    = root_cause,
            fault_category = fault_category,
            fault_type    = fault_type,
            node_states   = node_states,
            node_metrics  = {k: list(v) for k, v in node_metrics.items()},
            metric_classes_hit = {},
            rcl_scores    = cfcbn_scores,
            propagation_path = cfcbn_top3,
            cbn_updated   = cfcbn_correct,
            is_novel      = False,
            notes         = f"cfcbn_rank={cfcbn_rank}",
            cfcbn_correct = cfcbn_correct,
            cfcbn_rank    = cfcbn_rank,
        )
        self.cases.append(case)
        self._append_case(case)
        return case
