"""
Edge Filter + Topology Tracker  (formerly Collider Filter)

Two agents:
  ColliderFilterAgent  — four-step automated collider elimination (no direct user interaction)
  TopologyTrackerAgent — accepts engineer instructions (fixed format forwarded by Manager)
                         and maintains the edge addition/removal history

Four-step EdgeFilter pipeline:
  Step 0 (script): read edge_prior_knowledge.json; deterministically remove listed spurious edges
  Step 1 (script): find all collider nodes (in-degree ≥ 2) in the cleaned graph
  Step 2 (LLM):   apply collider_prior_knowledge.txt rules to eliminate colliders
  Step 3 (script): record all changes; unresolved colliders are retained

Prior knowledge files (agents/knowledge/):
  edge_prior_knowledge.json      — explicit spurious-edge list (used directly in Step-0)
  collider_prior_knowledge.txt   — collider elimination rules (injected into LLM prompt in Step-2)

Prior prompt
  1 →  Collider
        A  B  H  A→BA 
        A→H B→H
       : B A  A→B→H 
             A→H  explaining-away 
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from itertools import combinations
from utils.llm_adapter import get_llm


# ===========================================================================
# 
# ===========================================================================

class EdgeChange:
    def __init__(self, action: str, parent: str, child: str,
                 reason: str, actor: str = "system"):
        """
        action: 'add' | 'remove'
        actor:  'collider_filter' | 'topology_tracker' | 'engineer'
        """
        self.action = action
        self.parent = parent
        self.child = child
        self.reason = reason
        self.actor = actor
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return vars(self)

    def __str__(self):
        return (f"[{self.timestamp}] {self.actor.upper()} {self.action.upper()} "
                f"{self.parent} -> {self.child} | {self.reason}")


# ===========================================================================
# ColliderFilterAgent
# ===========================================================================

# collider_prior_knowledge.txt  ColliderFilterAgent 
def _load_collider_prior_knowledge() -> str:
    """ agents/knowledge/collider_prior_knowledge.txt  Collider Prior"""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "knowledge", "collider_prior_knowledge.txt")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    # fallback:
    return ("Rule 1: If A->B->H and A->H both exist (H is host), remove A->H.\n"
            "Rule 2: Remove transitive redundant edges A->C when A->B->C exists.")

PRIOR_KNOWLEDGE_PROMPT = _load_collider_prior_knowledge()


class ColliderFilterAgent:
    """
     Collider 
    """

    def __init__(self,
                 deployment_info: Optional[Dict[str, Dict]] = None,
                 knowledge_dir: Optional[str] = None):
        """
        deployment_info: {
            "host_services": {"host_name": ["svc1", "svc2", ...]},
            "node_types": {"node_name": "service" | "host"}
        }
        knowledge_dir: Prior agents/knowledge/
          : edge_prior_knowledge.json      Step-0
                collider_prior_knowledge.txt   Collider Step-2 LLM prompt
        """
        self.deployment_info = deployment_info or {}
        self.host_services: Dict[str, List[str]] = \
            self.deployment_info.get("host_services", {})
        self.node_types: Dict[str, str] = \
            self.deployment_info.get("node_types", {})

        # service → host 
        self.service_host: Dict[str, str] = {}
        for h, svcs in self.host_services.items():
            for s in svcs:
                self.service_host[s] = h

        # edge_prior_knowledge.jsonStep-0
        here = os.path.dirname(os.path.abspath(__file__))
        kdir = knowledge_dir or os.path.join(here, "knowledge")
        epk_path = os.path.join(kdir, "edge_prior_knowledge.json")
        self._edge_prior_knowledge: List[dict] = []
        if os.path.exists(epk_path):
            try:
                data = json.load(open(epk_path, encoding="utf-8"))
                self._edge_prior_knowledge = data.get("edges_to_remove", [])
            except Exception as e:
                print(f"[EdgeFilter] Warn: could not load edge_prior_knowledge: {e}")

        self._llm = get_llm()

    def run(
        self,
        edges: List[Tuple[str, str]],
    ) -> Tuple[List[Tuple[str, str]], List[EdgeChange]]:
        """
        Execute the four-step EdgeFilter pipeline.

        Args:
            edges: input edge list [(parent, child), ...]

        Returns:
            (cleaned_edges, changes)
        """
        # ── Step 0: Prior ────────────────────────
        working = list(edges)
        step0_changes: List[EdgeChange] = []
        if self._edge_prior_knowledge:
            for rule in self._edge_prior_knowledge:
                src, dst = rule.get("src", ""), rule.get("dst", "")
                reason = rule.get("reason", "edge_prior_knowledge")
                if (src, dst) in working:
                    working.remove((src, dst))
                    chg = EdgeChange("remove", src, dst,
                                     f"[Step-0 edge_prior] {reason}",
                                     "edge_filter_prior")
                    step0_changes.append(chg)
                    print(f"[EdgeFilter] Step 0: Removed {src}→{dst} | prior knowledge")
            if step0_changes:
                _removed = [(c.parent, c.child) for c in step0_changes]
                _edge_list = ", ".join(f"{p}\u2192{c}" for p, c in _removed)
                print(f"[EdgeFilter] Step 0 summary: {len(_removed)} spurious edge(s) "
                      f"removed from topology: {_edge_list}")
            else:
                print("[EdgeFilter] Step 0: No prior-knowledge edges found in graph.")
        else:
            print("[EdgeFilter] Step 0: No edge_prior_knowledge loaded (skipped).")

        # ── Step 1:  Collider ───────────────────────────────
        colliders = self._find_colliders(working)
        if not colliders:
            print("[EdgeFilter] Step 1: No colliders found.")
            return working, step0_changes

        unique_nodes = list(dict.fromkeys(c['node'] for c in colliders))
        print(f"[EdgeFilter] Step 1: Found {len(unique_nodes)} collider nodes "
            f"({len(colliders)} parent-pair combinations): {unique_nodes}")

        # ── Step 2: LLM  Collider ───────────────────────────────────────
        edges_to_remove = self._llm_eliminate(colliders, working)

        # ── Step 3: ──────────────────────────────────────────
        step3_changes: List[EdgeChange] = []
        remaining = list(working)
        for parent, child, reason in edges_to_remove:
            if (parent, child) in remaining:
                remaining.remove((parent, child))
                chg = EdgeChange("remove", parent, child, reason, "edge_filter_collider")
                step3_changes.append(chg)
                print(f"[EdgeFilter] Step 3: Removed {parent}→{child} | {reason}")

        if not step3_changes:
            print("[EdgeFilter] Step 3: No collider edges removed (no applicable rules).")

        all_changes = step0_changes + step3_changes
        # ── EdgeFilter run() summary ───────────────────────────────────
        n_in  = len(edges)
        n_out = len(remaining)
        n_rm  = len(all_changes)
        print(f"[EdgeFilter] Run complete: {n_in} input edges → "
              f"{n_out} after removing {n_rm} spurious edge(s).")
        return remaining, all_changes

    # ------------------------------------------------------------------
    # Step 1:  Collider
    # ------------------------------------------------------------------

    def _find_colliders(self, edges: List[Tuple[str, str]]) -> List[dict]:
        """
        Find true colliders: A -> C <- B where A and B also have a directed edge between them.
        """
        # child -> parents
        in_edges: Dict[str, List[str]] = defaultdict(list)
        edge_set = set(edges)

        for parent, child in edges:
            in_edges[child].append(parent)

        colliders = []

        for node, parents in in_edges.items():
            if len(parents) < 2:
                continue

            #  (A, B)
            for A, B in combinations(parents, 2):
                # 
                if (A, B) in edge_set or (B, A) in edge_set:
                    colliders.append({
                        "node": node,
                        "parents": [A, B],
                        "node_type": self.node_types.get(node, "unknown"),
                        "deployed_services": self.host_services.get(node, []),
                    })

        return colliders

    # ------------------------------------------------------------------
    # Step 2: LLM 
    # ------------------------------------------------------------------

    def _llm_eliminate(
        self,
        colliders: List[dict],
        edges: List[Tuple[str, str]],
    ) -> List[Tuple[str, str, str]]:
        """
         LLMPrior collider 
         [(parent, child, reason), ...]
        """
        #  LLM 
        edge_list = [{"parent": p, "child": c} for p, c in edges]

        prompt = f"""{PRIOR_KNOWLEDGE_PROMPT}

Colliders to analyze:
{json.dumps(colliders, indent=2)}

All current edges:
{json.dumps(edge_list, indent=2)}

Deployment info:
- host_services: {json.dumps(self.host_services, indent=2)}
- node_types: {json.dumps(self.node_types, indent=2)}

For each collider, decide which edges (if any) to remove.

Respond ONLY with a JSON array:
[
  {{
    "collider_node": "<node>",
    "edges_to_remove": [
      {{"parent": "<p>", "child": "<c>", "reason": "<rule applied + explanation>"}}
    ]
  }},
  ...
]
Return an empty edges_to_remove list for colliders where no rule applies.
"""
        result = self._llm.invoke_json(prompt)

        # Mock fallback:  Rule 1 
        if isinstance(result, dict) and result.get("_mock"):
            result = self._script_rule1(colliders, edges)

        removes = []
        if isinstance(result, list):
            for item in result:
                for e in item.get("edges_to_remove", []):
                    removes.append((e["parent"], e["child"], e.get("reason", "prior rule")))
        return removes

    def _script_rule1(
        self,
        colliders: List[dict],
        edges: List[Tuple[str, str]],
    ) -> list:
        """Mock mode: script-based application of Rule 1 (same-host parent-service-to-host edge removal)."""
        edge_set = set(edges)
        result = []
        for col in colliders:
            node = col["node"]
            parents = col["parents"]
            removes = []

            # 
            if self.node_types.get(node) == "host":
                # 
                svcs_on_host = set(self.host_services.get(node, []))
                # 
                for pa in parents:
                    for pb in parents:
                        if pa == pb:
                            continue
                        # pa → pb pa
                        if (pa, pb) in edge_set and pa in svcs_on_host and pb in svcs_on_host:
                            #  pa → host
                            if (pa, node) in edge_set:
                                removes.append({
                                    "parent": pa,
                                    "child": node,
                                    "reason": f"Rule 1: {pa}→{pb} exists and both on host {node}; "
                                              f"remove parent-service→host edge {pa}→{node}"
                                })
            result.append({"collider_node": node, "edges_to_remove": removes})
        return result


# ===========================================================================
# TopologyTrackerAgent
# ===========================================================================

class TopologyTrackerAgent:
    """
    Maintain the causal-graph edge set, process engineer add/remove instructions,
    and record the full change history.
    Accepts fixed-format instructions forwarded by Manager (no intent parsing).

    Fixed instruction format (parsed by Manager before forwarding):
      add: <parent>, <child> [| reason: <reason>]
      remove: <parent>, <child> [| reason: <reason>]
      query: <parent>, <child>        — check whether an edge exists and its history
      history                         — print full change history
    """

    def __init__(
        self,
        collider_filter: Optional[ColliderFilterAgent] = None,
        store_path: Optional[str] = None,
    ):
        self.collider_filter = collider_filter
        self.store_path = store_path   # e.g. "case_store/topology.json"

        self._edges: List[Tuple[str, str]] = []
        self._changes: List[EdgeChange] = []

        # Load from disk on startup (if file exists)
        if store_path:
            self._load()

    # ------------------------------------------------------------------
    # Persist to disk
    # ------------------------------------------------------------------

    def _load(self):
        import os
        if not self.store_path or not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path, encoding="utf-8") as f:
                data = json.load(f)
            self._edges = [tuple(e) for e in data.get("edges", [])]
            for c in data.get("changes", []):
                chg = EdgeChange(
                    action=c["action"], parent=c["parent"], child=c["child"],
                    reason=c["reason"], actor=c.get("actor", "system"),
                )
                chg.timestamp = c["timestamp"]
                self._changes.append(chg)
            print(f"[TopologyTracker] Loaded {len(self._edges)} edges, "
                  f"{len(self._changes)} change records from {self.store_path}")
        except Exception as e:
            print(f"[TopologyTracker] Warn: could not load state: {e}")

    def save(self):
        """Change history store_path"""
        if not self.store_path:
            return
        import os
        os.makedirs(os.path.dirname(os.path.abspath(self.store_path)), exist_ok=True)
        data = {
            "edges": [list(e) for e in self._edges],
            "changes": [c.to_dict() for c in self._changes],
        }
        with open(self.store_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_edges(self, edges: List[Tuple[str, str]]):
        """Initial load of edge set — only used when disk has no history; not recorded in change log."""
        if self._edges:   # Already restored from disk — skipping overwrite
            return
        self._edges = list(edges)

    def get_edges(self) -> List[Tuple[str, str]]:
        return list(self._edges)

    def get_all_nodes(self) -> Set[str]:
        nodes = set()
        for p, c in self._edges:
            nodes.add(p)
            nodes.add(c)
        return nodes

    # ------------------------------------------------------------------
    # Engineer instruction handlingManager
    # ------------------------------------------------------------------

    def handle_instruction(self, instruction: str) -> str:
        """Process a fixed-format instruction forwarded by Manager. Returns a result string."""
        instruction = instruction.strip()

        if instruction.lower() == "history":
            return self._show_history()

        if instruction.lower().startswith("add:"):
            return self._handle_add(instruction[4:].strip())

        if instruction.lower().startswith("remove:"):
            return self._handle_remove(instruction[7:].strip())

        if instruction.lower().startswith("query:"):
            return self._handle_query(instruction[6:].strip())

        return f"[TopologyTracker] Unknown instruction: {instruction}"

    def _parse_edge_args(self, args: str) -> Tuple[Optional[str], Optional[str], str]:
        """Parse 'parent, child [| reason: ...]'"""
        parts = args.split("|")
        edge_part = parts[0].strip()
        reason = parts[1].strip() if len(parts) > 1 else ""
        if reason.lower().startswith("reason:"):
            reason = reason[7:].strip()
        nodes = [n.strip() for n in edge_part.split(",")]
        if len(nodes) < 2:
            return None, None, reason
        return nodes[0], nodes[1], reason

    def _handle_add(self, args: str) -> str:
        parent, child, reason = self._parse_edge_args(args)
        if not parent or not child:
            return "[TopologyTracker] add requires: parent, child"

        if (parent, child) in self._edges:
            return f"[TopologyTracker] Edge {parent}→{child} already exists."

        self._edges.append((parent, child))
        chg = EdgeChange("add", parent, child, reason or "engineer request", "topology_tracker")
        self._changes.append(chg)

        # After adding an edge, re-run EdgeFilter
        extra_changes = []
        if self.collider_filter:
            cleaned, extra = self.collider_filter.run(self._edges)
            if extra:
                self._edges = cleaned
                self._changes.extend(extra)
                extra_changes = extra

        self.save()
        msg = f"[TopologyTracker] Added {parent}→{child}. Reason: {reason or 'engineer request'}"
        if extra_changes:
            msg += f"\n  ColliderFilter removed {len(extra_changes)} edge(s) after re-check."
        return msg

    def _handle_remove(self, args: str) -> str:
        parent, child, reason = self._parse_edge_args(args)
        if not parent or not child:
            return "[TopologyTracker] remove requires: parent, child"

        if (parent, child) not in self._edges:
            return (f"[TopologyTracker] Edge {parent}→{child} not in current graph. "
                    f"Try 'query: {parent}, {child}' to check history.")

        self._edges.remove((parent, child))
        chg = EdgeChange("remove", parent, child, reason or "engineer request", "engineer")
        self._changes.append(chg)
        self.save()
        return f"[TopologyTracker] Removed {parent}→{child}. Reason: {reason or 'engineer request'}"

    def _handle_query(self, args: str) -> str:
        """
        Current status
        : 
        """
        parent, child, _ = self._parse_edge_args(args)
        if not parent or not child:
            return "[TopologyTracker] query requires: parent, child"

        current = (parent, child) in self._edges
        history_for_edge = [
            c for c in self._changes
            if c.parent == parent and c.child == child
        ]

        lines = [f"Edge {parent}→{child}:"]
        lines.append(f"  Current state: {'EXISTS' if current else 'NOT IN GRAPH'}")

        if not history_for_edge:
            lines.append("  History: No recorded changes. Edge may have never existed.")
        else:
            lines.append("  History:")
            for c in history_for_edge:
                lines.append(f"    {c}")

        return "\n".join(lines)

    def _show_history(self) -> str:
        if not self._changes:
            return "[TopologyTracker] No recorded changes."
        return "\n".join(str(c) for c in self._changes)

    def record_collider_filter_changes(self, changes: List[EdgeChange]):
        """Record EdgeFilterAgent initialisation changes and persist to disk."""
        self._changes.extend(changes)
        self.save()


# ===========================================================================
# : TopologyModulePipeline
# ===========================================================================

class TopologyModule:
    """
    Unified entry point combining EdgeFilter and TopologyTracker.
    The pipeline interacts with this class only.
    """

    def __init__(
        self,
        service2service: Dict[str, List[str]],
        deployment_info: Optional[Dict] = None,
        store_path: Optional[str] = None,
    ):
        """
        service2service : predefined service call topology
        deployment_info : {"host_services": {...}, "node_types": {...}}
        store_path      : persistence file path, e.g. "case_store/topology.json"
                          - First run: EdgeFilter cleans the topology and writes the result
                          - Subsequent runs: restored directly from file (cleaning skipped)
        """
        # 
        raw_edges: List[Tuple[str, str]] = []
        for src, dsts in service2service.items():
            for dst in dsts:
                raw_edges.append((src, dst))

        #  Agent
        self.collider_filter = ColliderFilterAgent(deployment_info)
        self.tracker = TopologyTrackerAgent(self.collider_filter, store_path=store_path)

        # If tracker was restored from disk (has history), skip re-running EdgeFilter
        if self.tracker._edges:
            print(f"[TopologyModule] Restored from {store_path}: "
                  f"{len(self.tracker._edges)} edges, "
                  f"{len(self.tracker._changes)} changes")
        else:
            # First run: execute EdgeFilter (four-step pipeline)
            cleaned, changes = self.collider_filter.run(raw_edges)
            self.tracker.load_edges(cleaned)
            self.tracker.record_collider_filter_changes(changes)   # includes save()
            n_removed = len(changes)
            print(f"[TopologyModule] Init: {len(raw_edges)} raw edges → "
                  f"{len(cleaned)} after EdgeFilter "
                  f"({n_removed} removed)")

    def get_edges(self) -> List[Tuple[str, str]]:
        return self.tracker.get_edges()

    def get_topology_dict(self) -> Dict[str, List[str]]:
        """Return adjacency dict {caller: [callee, ...]} for BFS fault propagation inference."""
        adj: Dict[str, List[str]] = {}
        for src, dst in self.get_edges():
            adj.setdefault(src, [])
            if dst not in adj[src]:
                adj[src].append(dst)
        return adj

    def get_all_nodes(self) -> Set[str]:
        return self.tracker.get_all_nodes()

    def handle_instruction(self, instruction: str) -> str:
        """Forward a fixed-format instruction to TopologyTracker."""
        return self.tracker.handle_instruction(instruction)

    def print_summary(self):
        edges = self.get_edges()
        nodes = self.get_all_nodes()
        print(f"[Topology] Nodes={len(nodes)}, Edges={len(edges)}")
        changes = self.tracker._changes
        if changes:
            print(f"  Changes:")
            for c in changes:
                print(f"    {c}")


# ──  ─────────────────────────────────────────────────────────────
# ColliderFilterAgent EdgeFilterAgent
EdgeFilterAgent = ColliderFilterAgent
