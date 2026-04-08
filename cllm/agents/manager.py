"""
Manager Agent



: 
- 
-  LLM 
-  StateBinnerMetricClassifier TopologyTracker
- 

Manager 

 MetricClassifierState Binner
  separate: <metric_a>, <metric_b> | reason: <reason>
  merge: <metric_a>, <metric_b> | reason: <reason>
  rename_class: <class_id>, <name>
  show

 TopologyTracker
  add: <parent>, <child> | reason: <reason>
  remove: <parent>, <child> | reason: <reason>
  query: <parent>, <child>
  history

: 
  Manager 
"""

import json
from typing import Optional

from utils.llm_adapter import get_llm


# LLM
INSTRUCTION_SCHEMA = """
Fixed-format instructions you can output:

FOR MetricClassifier (State Binner):
  separate: <metric_a>, <metric_b> | reason: <reason>
  merge: <metric_a>, <metric_b> | reason: <reason>
  rename_class: <class_id>, <class_name>
  show

FOR TopologyTracker:
  add: <parent_node>, <child_node> | reason: <reason>
  remove: <parent_node>, <child_node> | reason: <reason>
  query: <parent_node>, <child_node>
  history
"""


class Manager:
    """
     + 
    """

    def __init__(
        self,
        metric_classifier=None,
        topology_module=None,
    ):
        self.metric_classifier = metric_classifier
        self.topology_module = topology_module
        self._llm = get_llm()

        # LLM
        self._conversation: list = []

    # ==================================================================
    # 
    # ==================================================================

    def handle(self, engineer_input: str) -> str:
        """
        
        """
        self._conversation.append({"role": "engineer", "content": engineer_input})

        # 1. LLM  → 
        parsed = self._parse_intent(engineer_input)
        target = parsed.get("target")   # "metric_classifier" | "topology_tracker" | "unknown"
        instruction = parsed.get("instruction", "")
        clarification_needed = parsed.get("clarification_needed", False)

        if clarification_needed or target == "unknown":
            msg = parsed.get("clarification_request",
                              "I could not understand your request. Please clarify.")
            self._conversation.append({"role": "manager", "content": msg})
            return f"[Manager] {msg}"

        # 2.  Agent
        result = self._dispatch(target, instruction)
        self._conversation.append({"role": "manager", "content": result})
        return result

    # ==================================================================
    # 
    # ==================================================================

    def _parse_intent(self, engineer_input: str) -> dict:
        """
         LLM 
        """
        context = ""
        if self.metric_classifier:
            enum_summary = {
                str(k): v
                for k, v in list(self.metric_classifier.metric_enum.items())[:20]
            }
            context += f"\nKnown metrics (sample): {json.dumps(enum_summary)}"
        if self.topology_module:
            edges_sample = self.topology_module.get_edges()[:10]
            context += f"\nCurrent edges (sample): {edges_sample}"

        prompt = f"""You are the Manager Agent of a microservice Root Cause Localization system.
Your job: understand the engineer's request and translate it into a FIXED-FORMAT instruction
for one of two sub-agents (MetricClassifier or TopologyTracker).

{INSTRUCTION_SCHEMA}

System context:
{context}

Engineer's request: "{engineer_input}"

Rules:
- If the request is about metric classification (grouping, separating, renaming metrics): target = metric_classifier
- If the request is about graph topology (edges, dependencies): target = topology_tracker
- If you cannot determine the intent: set clarification_needed = true
- Output the instruction in EXACTLY the fixed format shown above (no extra words)

Respond ONLY with a JSON object:
{{
  "target": "metric_classifier" | "topology_tracker" | "unknown",
  "instruction": "<exact fixed-format instruction string>",
  "clarification_needed": false,
  "clarification_request": null,
  "reasoning": "<brief explanation of what you understood>"
}}
"""
        result = self._llm.invoke_json(prompt)

        # fallbackMock LLM
        if isinstance(result, dict) and (result.get("_mock") or "target" not in result):
            result = self._keyword_fallback(engineer_input)

        return result

    def _keyword_fallback(self, text: str) -> dict:
        """ fallback LLM """
        t = text.lower()

        # TopologyTracker 
        if any(w in t for w in ["add edge", "add dependency", "connect"]):
            nodes = self._extract_node_names(text)
            if len(nodes) >= 2:
                return {"target": "topology_tracker",
                        "instruction": f"add: {nodes[0]}, {nodes[1]} | reason: engineer request",
                        "clarification_needed": False}

        if any(w in t for w in ["remove edge", "delete edge", "useless edge", "wrong edge",
                                 "useless", "is wrong", "shouldn't exist", "should not exist",
                                 "edge between", "edge from"]):
            nodes = self._extract_node_names(text)
            if len(nodes) >= 2:
                return {"target": "topology_tracker",
                        "instruction": f"remove: {nodes[0]}, {nodes[1]} | reason: engineer request",
                        "clarification_needed": False}

        if any(w in t for w in ["edge history", "topology history", "change log", "what happened to edge"]):
            return {"target": "topology_tracker",
                    "instruction": "history",
                    "clarification_needed": False}

        if any(w in t for w in ["where is edge", "does edge exist", "check edge"]):
            nodes = self._extract_node_names(text)
            if len(nodes) >= 2:
                return {"target": "topology_tracker",
                        "instruction": f"query: {nodes[0]}, {nodes[1]}",
                        "clarification_needed": False}

        # MetricClassifier separate merge
        if any(w in t for w in ["separate", "split", "not in the same class",
                                 "shouldn't be together", "different class",
                                 "should not be", "should be separated"]):
            metrics = self._extract_metric_names(text)
            if len(metrics) >= 2:
                return {"target": "metric_classifier",
                        "instruction": f"separate: {metrics[0]}, {metrics[1]} | reason: {text}",
                        "clarification_needed": False}

        if any(w in t for w in ["merge", "same class", "should be together", "combine",
                                 "should be in the same"]):
            metrics = self._extract_metric_names(text)
            if len(metrics) >= 2:
                return {"target": "metric_classifier",
                        "instruction": f"merge: {metrics[0]}, {metrics[1]} | reason: {text}",
                        "clarification_needed": False}

        if any(w in t for w in ["show metric", "show class", "list metric", "what metrics"]):
            return {"target": "metric_classifier",
                    "instruction": "show",
                    "clarification_needed": False}

        return {
            "target": "unknown",
            "instruction": "",
            "clarification_needed": True,
            "clarification_request": (
                "I couldn't understand your request. "
                "Please try: 'remove edge A->B', 'separate metric rrt and rrt_max', "
                "'show metric classes', or 'show topology history'."
            ),
        }

    def _extract_node_names(self, text: str) -> list:
        """"""
        import re
        #  service 
        candidates = re.findall(
            r'\b([a-z][a-z0-9_\-]*(?:service|srv|db|tidb|tikv|redis|cart|pd)?)\b',
            text.lower()
        )
        stopwords = {"the", "edge", "between", "and", "or", "is", "are", "this",
                     "think", "that", "not", "add", "remove", "delete", "useless",
                     "wrong", "check", "where", "does", "exist"}
        return [c for c in candidates if c not in stopwords and len(c) > 2]

    def _extract_metric_names(self, text: str) -> list:
        """Metric name"""
        import re
        candidates = re.findall(r'\b([a-z][a-z0-9_]*)\b', text.lower())
        stopwords = {"i", "think", "metric", "metrics", "and", "should", "not",
                     "be", "in", "the", "same", "class", "separate", "merge",
                     "combine", "together", "a", "b"}
        return [c for c in candidates if c not in stopwords and len(c) > 1]

    # ==================================================================
    # 
    # ==================================================================

    def _dispatch(self, target: str, instruction: str) -> str:
        """ Agent"""
        if target == "metric_classifier":
            if self.metric_classifier is None:
                return "[Manager] MetricClassifier not available."
            result = self.metric_classifier.handle_instruction(instruction)
            return f"[Manager → MetricClassifier] {result}"

        elif target == "topology_tracker":
            if self.topology_module is None:
                return "[Manager] TopologyModule not available."
            result = self.topology_module.handle_instruction(instruction)
            return f"[Manager → TopologyTracker] {result}"

        return f"[Manager] Cannot dispatch to unknown target: {target}"

    # ==================================================================
    # 
    # ==================================================================

    def get_conversation_history(self) -> str:
        lines = []
        for turn in self._conversation:
            role = turn["role"].upper()
            lines.append(f"[{role}] {turn['content']}")
        return "\n".join(lines) if lines else "(empty)"
