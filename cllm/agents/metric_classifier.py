"""
Metric Classifier and State Binner.

Maintains three data structures (JSON-persisted):
  1. metric_set     : set of all metrics seen across all cases
  2. metric_enum    : metric → class ID  {metric: class_id}
  3. class_reasons  : class ID → list of reasons  {class_id: ["high co-occurrence" | "semantic similarity" | ...]}

Unified online processing mode (process_fault_unified):
  For each incoming case:
    1. Add the case's metrics to metric_set
    2. If metric_enum is empty (first case), initialise it using cold_start_mode
    3. Otherwise, dynamically classify unseen metrics via LLM (except in "none" mode)
    4. Return node_states

  There is no distinct "cold-start phase" vs "normal phase" — every case is treated uniformly.

cold_start_mode options:
  "union_find" : script-only, Union-Find clustering based on co-occurrence rate
  "llm"        : pure LLM, semantically classifies the first case's metrics
  "mix"        : default — runs both union_find and llm, fuses via a co-occurrence matrix
  "none"       : disable clustering — each metric forms its own singleton class.
                 No LLM calls. New metrics receive the next sequential class ID.
                 similarity_score falls back to exact metric-level matching (finest granularity).

State Binner:
  Each metric class is counted at most once; a node with ≥1 anomalous metric class → state 1, else 0.
  (In "none" mode, class = metric, so any anomalous metric sets state 1.)

Engineer interaction (forwarded by Manager in fixed format):
  separate / merge / rename_class / show
"""

import json
import math
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Set, Tuple

from utils.llm_adapter import get_llm


# ------------------------------------------------------------------
# 
# ------------------------------------------------------------------

def _cooccurrence_rate(m1: str, m2: str, cases: List[Dict[str, List[str]]]) -> float:
    """m1  m2 : """
    both = either = 0
    for case_metrics in cases:
        flat = {m for ms in case_metrics.values() for m in ms}
        has1, has2 = m1 in flat, m2 in flat
        if has1 or has2:
            either += 1
        if has1 and has2:
            both += 1
    return both / either if either > 0 else 0.0


def _next_class_id(existing: Dict[str, int]) -> int:
    return max(existing.values(), default=-1) + 1


# ------------------------------------------------------------------
# MetricClassifier
# ------------------------------------------------------------------

class MetricClassifier:

    COOCCUR_THRESHOLD = 0.7
    # mix Fuse CF and CBN scores union_find  mix 
    MIX_ALPHA = 0.5

    def __init__(
        self,
        state_file: Optional[str] = None,
        # ──  ──
        cold_start_n: int = 0,
        cold_start_mode: str = "mix",   # "union_find" | "llm" | "mix" | "none"
    ):
        self.state_file      = state_file
        # cold_start_mode "none" 
        self.cold_start_mode = cold_start_mode

        self.metric_set:    List[str]              = []
        self.metric_enum:   Dict[str, int]         = {}
        self.class_reasons: Dict[int, List[str]]   = defaultdict(list)

        self._processed_count: int  = 0
        self._cold_start_done: bool = True   #  True

        self._llm = get_llm()

        if state_file:
            self._load(state_file)

    # ==============================================================
    # Persist to disk
    # ==============================================================

    def _load(self, path: str):
        import os
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            self.metric_set        = s.get("metric_set", [])
            self.metric_enum       = {k: int(v) for k, v in s.get("metric_enum", {}).items()}
            self.class_reasons     = defaultdict(list, {
                int(k): v for k, v in s.get("class_reasons", {}).items()
            })
            self._processed_count  = s.get("processed_count", 0)
            #  _cold_start_done  TruePersist to disk
            self._cold_start_done  = True
            print(f"[MetricCLS] Loaded: {len(self.metric_set)} metrics, "
                  f"{len(set(self.metric_enum.values()))} classes")
        except Exception as e:
            print(f"[MetricCLS] Warn: load failed ({e})")

    def save(self, path: Optional[str] = None):
        p = path or self.state_file
        if not p:
            return
        import os
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "metric_set":      self.metric_set,
                "metric_enum":     self.metric_enum,
                "class_reasons":   {str(k): v for k, v in self.class_reasons.items()},
                "processed_count": self._processed_count,
                "cold_start_done": self._cold_start_done,
            }, f, indent=2, ensure_ascii=False)

    # ==============================================================
    # :  case  case
    # ==============================================================

    def process_fault_unified(
        self,
        node_metrics: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """
         case / case : 

          1.  case  metric_set
          2.  mode=="none" ID LLM
              metric_enum 
                case  cold_start_mode  metric_enum
             : 
                case  LLM 
          3.  +1Persist to disk node_states

        :  0  case case_store 
        LLM ReAct Agent 
         search_similar_cases 
        """
        # Step 1: 
        all_metrics = list({m for ms in node_metrics.values() for m in ms})
        new_to_set = [m for m in all_metrics if m not in self.metric_set]
        for m in new_to_set:
            self.metric_set.append(m)

        # Step 2: 
        if self.cold_start_mode == "none":
            # none :  ID
            truly_new = [m for m in all_metrics if m not in self.metric_enum]
            if truly_new:
                self._assign_individual_ids(truly_new)
        elif not self.metric_enum:
            # :  case 
            print(f"[MetricCLS] First case — initializing metric_enum "
                  f"(mode={self.cold_start_mode}, {len(self.metric_set)} metrics)")
            self._init_enum_from_single_case(node_metrics)
        else:
            # :  LLM 
            truly_new = [m for m in all_metrics if m not in self.metric_enum]
            if truly_new:
                self._dynamic_classify(truly_new)

        self._processed_count += 1
        self.save()
        return self._compute_states(node_metrics)

    def _assign_individual_ids(self, metrics: List[str]):
        """
        none :  ID
         LLM
        """
        for m in metrics:
            cid = _next_class_id(self.metric_enum)
            self.metric_enum[m] = cid
            self.class_reasons[cid] = ["(none)"]
        self.save()

    def _init_enum_from_single_case(self, node_metrics: Dict[str, List[str]]):
        """
        First-case initialisation for non-none modes: build metric_enum from the first case's metric set.
         caseunion_find  1 case 
         union_find  case 
        llm mix Fuse CF and CBN scores
        """
        #  case list of node_metrics
        single_case = [node_metrics]

        mode = self.cold_start_mode
        if mode == "union_find":
            enum = self._build_union_find(single_case)
            self._apply_enum(enum, reason="(UnionFind)")
        elif mode == "llm":
            enum = self._build_llm(single_case)
            self._apply_enum(enum, reason="(LLM)")
        else:  # mix
            enum_uf  = self._build_union_find(single_case)
            enum_llm = self._build_llm(single_case)
            enum = self._fuse_enums(enum_uf, enum_llm, self.metric_set)
            self._apply_enum(enum, reason="(mixFuse CF and CBN scores)")

        print(f"[MetricCLS] Initialized: {len(set(self.metric_enum.values()))} classes "
              f"from {len(self.metric_set)} metrics")
        # Show merged groups (classes with >1 metric)
        _by_cls = {}
        for _m, _cid in self.metric_enum.items():
            _by_cls.setdefault(_cid, []).append(_m)
        _merged = {cid: ms for cid, ms in _by_cls.items() if len(ms) > 1}
        if _merged:
            print("[MetricCLS] Deduplicated metric groups (each counted once):")
            for cid, ms in sorted(_merged.items()):
                print(f"  Class {cid}: {sorted(ms)}")
        self._print_classes()
        self.save()

    # ==============================================================
    #  process_fault_unified
    # ==============================================================

    def run_cold_start(self, cold_cases: List[Dict[str, List[str]]]):
        """
        [] 
         Pipeline 
        """
        print("[MetricCLS] run_cold_start() called (deprecated in unified pipeline — no-op).")

    def process_fault(
        self,
        node_metrics: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """
        []  process_fault_unified
        """
        return self.process_fault_unified(node_metrics)

    def compute_states_only(
        self, node_metrics: Dict[str, List[str]]
    ) -> Dict[str, int]:
        """"""
        return self._compute_states(node_metrics)



    # ==============================================================
    # Union-Find 
    # ==============================================================

    def _build_union_find(
        self, cases: List[Dict[str, List[str]]]
    ) -> Dict[str, int]:
        """ {metric: class_id}"""
        metrics = self.metric_set
        parent = {m: m for m in metrics}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for m1, m2 in combinations(metrics, 2):
            if _cooccurrence_rate(m1, m2, cases) >= self.COOCCUR_THRESHOLD:
                union(m1, m2)

        roots: Dict[str, int] = {}
        result: Dict[str, int] = {}
        cid = 0
        for m in metrics:
            r = find(m)
            if r not in roots:
                roots[r] = cid
                cid += 1
            result[m] = roots[r]
        return result

    # ==============================================================
    # LLM 
    # ==============================================================

    def _build_llm(
        self, cases: List[Dict[str, List[str]]]
    ) -> Dict[str, int]:
        """
         LLM
         {metric: class_id}
        """
        metrics = self.metric_set
        if not metrics:
            return {}

        # 
        freq: Dict[str, int] = defaultdict(int)
        for nm in cases:
            for m in (m for ms in nm.values() for m in ms):
                freq[m] += 1

        metric_info = [{"metric": m, "frequency": freq[m]} for m in metrics]

        prompt = f"""You are a microservice observability expert.
Classify the following anomaly metrics into semantic groups.
Metrics that describe the same failure phenomenon should be in the same class.

Metrics observed across {len(cases)} fault cases:
{json.dumps(metric_info, indent=2)}

Rules:
1. Metrics related to latency/response-time belong together (e.g., rrt, rrt_max, timeout)
2. Metrics related to errors/failures belong together (e.g., error, error_ratio, client_error)
3. Metrics related to throughput/traffic belong together (e.g., request, response)
4. Metrics related to network I/O belong together (e.g., pod_network_*)
5. Metrics related to CPU belong together
6. Metrics related to memory belong together
7. Other metrics that clearly belong together semantically

Respond ONLY with a JSON object mapping each metric to a class_id (integer starting from 0):
{{
  "metric_name": class_id,
  ...
}}
All metrics must be assigned. Use consecutive integers starting from 0.
"""
        result = self._llm.invoke_json(prompt)

        if isinstance(result, dict) and result.get("_mock"):
            # Mock fallback: 
            return self._keyword_classify_all(metrics)

        # 
        enum: Dict[str, int] = {}
        if isinstance(result, dict):
            for m in metrics:
                if m in result and isinstance(result[m], int):
                    enum[m] = result[m]
                else:
                    enum[m] = _next_class_id(enum)
        else:
            enum = self._keyword_classify_all(metrics)

        #  class_id 0 
        old2new: Dict[int, int] = {}
        cid = 0
        normalized: Dict[str, int] = {}
        for m in metrics:
            old = enum.get(m, 0)
            if old not in old2new:
                old2new[old] = cid
                cid += 1
            normalized[m] = old2new[old]
        return normalized

    def _keyword_classify_all(self, metrics: List[str]) -> Dict[str, int]:
        """LLM """
        keyword_groups = [
            (["rrt", "timeout", "latency", "duration"],  0),
            (["error", "fail", "client_error"],           1),
            (["request", "response", "throughput"],       2),
            (["network", "packet", "bytes", "receive", "transmit"], 3),
            (["cpu", "process"],                          4),
            (["memory", "mem", "working_set"],            5),
            (["disk", "io", "written"],                   6),
        ]
        enum: Dict[str, int] = {}
        next_cid = 7
        for m in metrics:
            m_lower = m.lower()
            assigned = False
            for kws, cid in keyword_groups:
                if any(kw in m_lower for kw in kws):
                    enum[m] = cid
                    assigned = True
                    break
            if not assigned:
                enum[m] = next_cid
                next_cid += 1
        return enum

    # ==============================================================
    # Fuse CF and CBN scoresco-association matrix
    # ==============================================================

    def _fuse_enums(
        self,
        enum_a: Dict[str, int],
        enum_b: Dict[str, int],
        metrics: List[str],
        alpha: float = None,
        threshold: float = 0.5,
    ) -> Dict[str, int]:
        """
         union_find(A)  llm(B) Fuse CF and CBN scores

        : 
          1.  C^A, C^BC^X_ij = 1 iff metric i,j  X 
          2. Fuse CF and CBN scores: C_ij = alpha * C^A_ij + (1-alpha) * C^B_ij
          3.  C threshold
             C_ij >= threshold → 

        Returns {metric: class_id}
        """
        alpha = alpha if alpha is not None else self.MIX_ALPHA
        n = len(metrics)
        idx = {m: i for i, m in enumerate(metrics)}

        # 
        ca = [[0.0] * n for _ in range(n)]
        cb = [[0.0] * n for _ in range(n)]
        for i, mi in enumerate(metrics):
            for j, mj in enumerate(metrics):
                ca[i][j] = 1.0 if enum_a.get(mi) == enum_a.get(mj) else 0.0
                cb[i][j] = 1.0 if enum_b.get(mi) == enum_b.get(mj) else 0.0

        # Fuse CF and CBN scores
        c = [[alpha * ca[i][j] + (1 - alpha) * cb[i][j] for j in range(n)]
             for i in range(n)]

        # :  C_ij >= thresholdi  j 
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i in range(n):
            for j in range(i + 1, n):
                if c[i][j] >= threshold:
                    union(i, j)

        roots: Dict[int, int] = {}
        result: Dict[str, int] = {}
        cid = 0
        for i, m in enumerate(metrics):
            r = find(i)
            if r not in roots:
                roots[r] = cid
                cid += 1
            result[m] = roots[r]
        return result

    def _apply_enum(self, enum: Dict[str, int], reason: str):
        """ self.metric_enum / self.class_reasons"""
        self.metric_enum = dict(enum)
        self.class_reasons = defaultdict(list)
        for cid in set(enum.values()):
            self.class_reasons[cid].append(reason)

    # ==============================================================
    # : LLM 
    # ==============================================================

    def _dynamic_classify(self, new_metrics: List[str]):
        """
         LLM New class
        """
        class_summary = self._build_class_summary()

        for metric in new_metrics:
            cid = self._classify_one_llm(metric, class_summary)
            self.metric_enum[metric] = cid
            if cid not in self.class_reasons:
                self.class_reasons[cid] = [""]
            # Show what metrics are already in this class
            by_cls = {m: c for m, c in self.metric_enum.items() if c == cid}
            existing = [m for m, c in self.metric_enum.items() if c == cid and m != metric]
            if existing:
                print(f"[MetricCLS] Merged: '{metric}' → class {cid} " f"(with existing: {existing})")
            else:
                print(f"[MetricCLS] New class {cid}: '{metric}'")

        self.save()

    def _classify_one_llm(self, metric: str, class_summary: dict) -> int:
        prompt = f"""You are classifying an anomaly metric for microservice fault analysis.

Existing metric classes:
{json.dumps(class_summary, indent=2, ensure_ascii=False)}

New metric to classify: "{metric}"

Should this metric join an existing class, or create a new one?

Respond ONLY with JSON:
{{
  "action": "join" | "new",
  "class_id": <int if action=="join", else null>,
  "reasoning": "<one sentence>"
}}
"""
        result = self._llm.invoke_json(prompt)

        if isinstance(result, dict) and not result.get("_mock"):
            action   = result.get("action", "new")
            class_id = result.get("class_id")
            if action == "join" and isinstance(class_id, int) and class_id in self.class_reasons:
                return class_id
            else:
                return _next_class_id(self.metric_enum)

        # Mock / failure: 
        return self._keyword_classify_one(metric)

    def _keyword_classify_one(self, metric: str) -> int:
        """LLM fallback"""
        m = metric.lower()
        if any(k in m for k in ["rrt", "timeout", "latency"]):
            target = 0
        elif any(k in m for k in ["error", "fail"]):
            target = 1
        elif any(k in m for k in ["request", "response"]):
            target = 2
        elif any(k in m for k in ["network", "packet", "bytes", "receive", "transmit"]):
            target = 3
        elif any(k in m for k in ["cpu", "process"]):
            target = 4
        elif any(k in m for k in ["memory", "mem", "working_set"]):
            target = 5
        else:
            return _next_class_id(self.metric_enum)

        #  class_id
        existing_cids = set(self.metric_enum.values())
        if target in existing_cids:
            return target
        return _next_class_id(self.metric_enum)

    # ==============================================================
    # State Binner
    # ==============================================================

    def _compute_states(
        self,
        node_metrics: Dict[str, List[str]],
    ) -> Dict[str, int]:
        """
         ≥1  →  1 0
        """
        states = {}
        for node, metrics in node_metrics.items():
            classes_hit = set()
            for m in metrics:
                if m in self.metric_enum:
                    classes_hit.add(self.metric_enum[m])
                elif m:  # : 
                    classes_hit.add(f"unknown_{m}")
            states[node] = 1 if classes_hit else 0
        return states

    # ==============================================================
    # : metric_enum 
    # ==============================================================

    def similarity_score(
        self,
        nm_a: Dict[str, List[str]],
        nm_b: Dict[str, List[str]],
    ) -> float:
        """
         metric_enum  node_metrics 

        : 
          - 
          - 
         = Σ() - Σ()

        
        """
        nodes = set(nm_a) | set(nm_b)
        total = 0
        for node in nodes:
            classes_a = {self.metric_enum.get(m) for m in nm_a.get(node, [])
                         if m in self.metric_enum}
            classes_b = {self.metric_enum.get(m) for m in nm_b.get(node, [])
                         if m in self.metric_enum}
            consistent    = len(classes_a & classes_b)
            inconsistent  = len(classes_a.symmetric_difference(classes_b))
            total        += consistent - inconsistent
        return float(total)

    # ==============================================================
    # : ""vs""
    # ==============================================================

    def compute_weighted_states(
        self,
        node_metrics: Dict[str, List[str]],
        case_reviewer,          # CaseReviewer 
        min_cases: int = 3,     #  case  0/1
    ) -> Dict[str, float]:
        """
        Compute per-node signal weights in [0, 1] based on historical case
        discriminability. Nodes with no anomaly get weight 0; anomalous nodes
        get weight in [0.1, 1.0] based on how discriminative their metrics are.

        For each anomalous node:
          1. Retrieve rc_cases (node is root cause) and non_cases (node anomalous
             but not root cause) from case_reviewer.
          2. Compute disc(m) = P(m|rc) - P(m|non_rc) for each metric m.
          3. Map mean discriminability to [0, 1] via sigmoid.
          4. If fewer than min_cases available, return 1.0 (no penalisation).

        Returns: {node: weight}  (weight=0 -> no anomaly, weight in [0.1,1.0] -> anomalous)
        """
        import math

        def sigmoid(x, scale=5.0):
            return 1.0 / (1.0 + math.exp(-scale * x))

        #  states state=1 
        base_states = self._compute_states(node_metrics)
        weighted    = {n: float(s) for n, s in base_states.items()}

        if case_reviewer is None:
            return weighted

        cases = getattr(case_reviewer, "cases", [])
        if len(cases) < min_cases:
            return weighted

        for node, state in base_states.items():
            if state == 0:
                continue  # 

            current_metrics = set(node_metrics.get(node, []))
            if not current_metrics:
                continue

            # :  vs 
            rc_metric_lists  = []
            non_metric_lists = []
            for c in cases:
                node_m = c.node_metrics.get(node, [])
                if not node_m:
                    continue
                if c.root_cause == node:
                    rc_metric_lists.append(set(node_m))
                elif c.node_states.get(node, 0) == 1:
                    non_metric_lists.append(set(node_m))

            if len(rc_metric_lists) < 2 or len(non_metric_lists) < 1:
                # 
                continue

            # 
            all_metrics_hist = set(
                m for ms in rc_metric_lists + non_metric_lists for m in ms
            )
            n_rc  = len(rc_metric_lists)
            n_non = len(non_metric_lists)

            disc = {}
            for m in all_metrics_hist:
                p_rc  = sum(1 for ms in rc_metric_lists  if m in ms) / n_rc
                p_non = sum(1 for ms in non_metric_lists if m in ms) / n_non
                disc[m] = p_rc - p_non  #  =  = 

            #  case 
            disc_vals = [disc.get(m, 0.0) for m in current_metrics if m in disc]
            if not disc_vals:
                continue

            mean_disc = sum(disc_vals) / len(disc_vals)
            # sigmoid  [0, 1]mean_disc > 0 → < 0 → 
            w = sigmoid(mean_disc, scale=4.0)
            #  0.1
            weighted[node] = max(w, 0.1)

        return weighted

    # ==============================================================
    #  Manager 
    # ==============================================================

    def handle_instruction(self, instruction: str) -> str:
        instruction = instruction.strip()
        if instruction.lower() == "show":
            return self._show()
        if instruction.lower().startswith("separate:"):
            return self._handle_separate(instruction[len("separate:"):].strip())
        if instruction.lower().startswith("merge:"):
            return self._handle_merge(instruction[len("merge:"):].strip())
        if instruction.lower().startswith("rename_class:"):
            return self._handle_rename(instruction[len("rename_class:"):].strip())
        return f"[MetricCLS] Unknown instruction: {instruction}"

    def _handle_separate(self, args: str) -> str:
        parts      = args.split("|")
        m_list     = [m.strip() for m in parts[0].split(",")]
        reason_str = parts[1].strip() if len(parts) > 1 else ""
        if len(m_list) < 2:
            return "[MetricCLS] separate requires at least 2 metrics."
        ma, mb = m_list[0], m_list[1]
        if ma not in self.metric_enum or mb not in self.metric_enum:
            return f"[MetricCLS] Metrics not found: {ma}, {mb}"
        cid_a, cid_b = self.metric_enum[ma], self.metric_enum[mb]
        if cid_a != cid_b:
            return (f"[MetricCLS] '{ma}'(class {cid_a}) and '{mb}'(class {cid_b}) "
                    f"are already in different classes.")
        classmates = [m for m, c in self.metric_enum.items() if c == cid_a]
        prompt = (
            f'Engineer wants to separate "{ma}" and "{mb}" (reason: "{reason_str}").\n'
            f'They share class {cid_a} with: {classmates}\n'
            f'Which metrics stay in class {cid_a}, and which move to a new class?\n'
            'Respond ONLY with JSON: {"stay_in_original": [...], "move_to_new": [...], "reasoning": "..."}'
        )
        plan = self._llm.invoke_json(prompt)
        if plan.get("_mock"):
            plan = {"stay_in_original": [m for m in classmates if m != mb],
                    "move_to_new": [mb], "reasoning": "mock: moved mb to new class"}
        new_cid = _next_class_id(self.metric_enum)
        for m in plan.get("move_to_new", [mb]):
            if m in self.metric_enum:
                self.metric_enum[m] = new_cid
        self.class_reasons[new_cid] = [f"engineer separated from class {cid_a}: {reason_str}"]
        self.save()
        return (f"[MetricCLS] Separated: {plan.get('move_to_new')} → new class {new_cid}. "
                f"Reasoning: {plan.get('reasoning')}")

    def _handle_merge(self, args: str) -> str:
        parts      = args.split("|")
        m_list     = [m.strip() for m in parts[0].split(",")]
        reason_str = parts[1].strip() if len(parts) > 1 else ""
        if len(m_list) < 2:
            return "[MetricCLS] merge requires at least 2 metrics."
        ma, mb = m_list[0], m_list[1]
        if ma not in self.metric_enum or mb not in self.metric_enum:
            return f"[MetricCLS] Metrics not found: {ma}, {mb}"
        cid_a, cid_b = self.metric_enum[ma], self.metric_enum[mb]
        if cid_a == cid_b:
            return f"[MetricCLS] Already in the same class: {cid_a}"
        moved = [m for m in self.metric_enum if self.metric_enum[m] == cid_b]
        for m in moved:
            self.metric_enum[m] = cid_a
        self.class_reasons[cid_a].append(f"engineer merged class {cid_b}: {reason_str}")
        del self.class_reasons[cid_b]
        self.save()
        return f"[MetricCLS] Merged class {cid_b} → {cid_a}. Moved: {moved}"

    def _handle_rename(self, args: str) -> str:
        parts = [a.strip() for a in args.split(",", 1)]
        if len(parts) < 2:
            return "[MetricCLS] rename_class requires class_id and new_name."
        try:
            cid = int(parts[0])
        except ValueError:
            return f"[MetricCLS] Invalid class_id: {parts[0]}"
        self.class_reasons[cid].append(f"renamed to: {parts[1]}")
        self.save()
        return f"[MetricCLS] Class {cid} renamed to '{parts[1]}'."

    def _show(self) -> str:
        return json.dumps(self._build_class_summary(), indent=2, ensure_ascii=False)

    # ==============================================================
    # 
    # ==============================================================

    def _build_class_summary(self) -> dict:
        by_class: Dict[int, List[str]] = defaultdict(list)
        for m, cid in self.metric_enum.items():
            by_class[cid].append(m)
        return {
            str(cid): {
                "metrics": ms,
                "reasons": self.class_reasons.get(cid, []),
            }
            for cid, ms in sorted(by_class.items())
        }

    def _print_classes(self):
        by_class: Dict[int, List[str]] = defaultdict(list)
        for m, cid in self.metric_enum.items():
            by_class[cid].append(m)
        for cid, ms in sorted(by_class.items()):
            print(f"  Class {cid}: {ms}")

    # ==============================================================
    # :  node_metrics
    # ==============================================================

    @staticmethod
    def extract_node_metrics(
        anomaly_pods:     Dict[str, List[str]],
        anomaly_services: Dict[str, List[str]],
        target_services:  List[str],
    ) -> Dict[str, List[str]]:
        """ pod  + service  {service: [metrics]}"""
        result: Dict[str, List[str]] = {}
        for svc in target_services:
            metrics = set(anomaly_services.get(svc, []))
            for pod_name, pod_m in anomaly_pods.items():
                cleaned = pod_name.replace(" (deleted)", "").strip()
                parts   = cleaned.rsplit("-", 1)
                base    = parts[0] if (len(parts) == 2 and parts[1].isdigit()) else cleaned
                if base == svc:
                    metrics.update(pod_m)
            result[svc] = list(metrics)
        return result
