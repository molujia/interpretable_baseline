"""
utils/anonymizer.py — LLM Anonymised


----
 fault_data Service namePod Metric name
 LLM API


----
Session 

  SVC-001 … SVC-NNN  ← Service name / Pod 
  MET-001 … MET-NNN  ← Metric name
  FT-001  … FT-NNN   ←  / 

: 
  1.  Pipeline  register_*() 
  2.  LLM prompt  anonymize(text) 
  3. LLM  deanonymize(text) De-anonymised
  4. De-anonymisedSRE 


------
cpu, memory, error, timeout …
 LLM 


--------
    from utils.anonymizer import get_anonymizer
    anon = get_anonymizer()
    anon.register_services(["checkoutservice", "paymentservice"])
    anon.register_metrics(["rrt", "pod_cpu_usage", "client_error"])

    prompt   = anon.anonymize(raw_prompt)   # → SVC-001 / MET-001 …
    response = llm.invoke(prompt)
    result   = anon.deanonymize(response)   # → checkoutservice …
"""

import re
from typing import Dict, List, Optional, Set


# 
_WHITELIST: Set[str] = {
    "cpu", "memory", "disk", "network", "io",
    "error", "timeout", "latency", "request", "response",
    "normal", "anomaly", "anomalous", "fault", "failure", "stress",
    "http", "grpc", "tcp", "dns", "pod", "service", "node",
    "usage", "ratio", "bytes", "packets",
}


def _strip_pod_suffix(name: str) -> str:
    """tidb-tikv-0 → tidb-tikvadservice-1 → adservice"""
    name = re.sub(r"\s*\(deleted\).*$", "", name).strip().lower()
    if name.startswith("tidb-"):
        m = re.match(r"(tidb-[a-z]+)-\d+$", name)
        return m.group(1) if m else name
    return re.sub(r"-\d+$", "", name)


def _replace_whole_word(text: str, word: str, replacement: str) -> str:
    """
    : word 
     'cart'  'cartservice' 
    """
    pattern = r"(?<![a-zA-Z0-9_\-])" + re.escape(word) + r"(?![a-zA-Z0-9_\-])"
    return re.sub(pattern, replacement, text, flags=re.IGNORECASE)


class Anonymizer:
    """
    Anonymised
     Pipeline SessionService name
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

        # →
        self._svc: Dict[str, str] = {}
        self._met: Dict[str, str] = {}
        self._ft:  Dict[str, str] = {}

        # →
        self._rev: Dict[str, str] = {}

        # 
        self._svc_n = self._met_n = self._ft_n = 0

        # (see source)
        self._sorted_keys: List[str] = []
        self._dirty = True

    # ──  ──────────────────────────────────────────────────────────

    def register_services(self, names: List[str]):
        for n in names:
            self._add_svc(n)

    def register_metrics(self, names: List[str]):
        for n in names:
            self._add_met(n)

    def register_fault_types(self, fts: List[str]):
        for ft in fts:
            self._add_ft(ft)

    def register_from_case(self, fault_data: dict):
        """ fault_data """
        rc = fault_data.get("root_cause", {})
        for key in ("fault_type", "fault_category"):
            v = rc.get(key, "")
            if v:
                self._add_ft(v)

        names = rc.get("name", "")
        if isinstance(names, str):
            names = [names]
        for n in names:
            self._add_svc(_strip_pod_suffix(n))

        for field in ("anomaly_services", "anomaly_pods"):
            for svc_or_pod, metrics in fault_data.get(field, {}).items():
                self._add_svc(_strip_pod_suffix(svc_or_pod))
                for m in (metrics or []):
                    self._add_met(m)

    # ──  ──────────────────────────────────────────────────────────

    def anonymize(self, text: str) -> str:
        """ text """
        if not self.enabled or not text:
            return text
        self._rebuild()
        result = text
        for real in self._sorted_keys:
            ph = self._lookup(real)
            if ph and real in result:
                result = _replace_whole_word(result, real, ph)
        return result

    def deanonymize(self, text: str) -> str:
        """ text De-anonymised"""
        if not self.enabled or not text:
            return text
        # (see source)
        for ph, real in sorted(self._rev.items(), key=lambda x: -len(x[0])):
            if ph in text:
                text = text.replace(ph, real)
        return text

    def show_mapping(self) -> str:
        """: """
        lines = ["[Anonymizer] Current mapping:"]
        all_fwd = {**self._svc, **self._met, **self._ft}
        for real, ph in sorted(all_fwd.items(), key=lambda x: x[1]):
            lines.append(f"  {ph:<12} ← {real}")
        return "\n".join(lines)

    # ──  ──────────────────────────────────────────────────────────

    def _add_svc(self, name: str) -> str:
        name = name.strip().lower()
        if not name or name in _WHITELIST:
            return name
        if name not in self._svc:
            self._svc_n += 1
            ph = f"SVC-{self._svc_n:03d}"
            self._svc[name] = ph
            self._rev[ph]   = name
            self._dirty = True
        return self._svc[name]

    def _add_met(self, name: str) -> str:
        name = name.strip().lower()
        if not name or name in _WHITELIST:
            return name
        if name not in self._met:
            self._met_n += 1
            ph = f"MET-{self._met_n:03d}"
            self._met[name] = ph
            self._rev[ph]   = name
            self._dirty = True
        return self._met[name]

    def _add_ft(self, name: str) -> str:
        name = name.strip().lower()
        if not name or name in _WHITELIST:
            return name
        if name not in self._ft:
            self._ft_n += 1
            ph = f"FT-{self._ft_n:03d}"
            self._ft[name] = ph
            self._rev[ph]  = name
            self._dirty = True
        return self._ft[name]

    def _lookup(self, real: str) -> Optional[str]:
        real = real.strip().lower()
        return self._svc.get(real) or self._met.get(real) or self._ft.get(real)

    def _rebuild(self):
        if not self._dirty:
            return
        all_keys = list(self._svc) + list(self._met) + list(self._ft)
        self._sorted_keys = sorted(all_keys, key=len, reverse=True)
        self._dirty = False


# ──  ──────────────────────────────────────────────────────────────────

_global: Optional[Anonymizer] = None


def get_anonymizer(enabled: bool = True) -> Anonymizer:
    """ Anonymizer"""
    global _global
    if _global is None:
        _global = Anonymizer(enabled=enabled)
    return _global


def reset_anonymizer():
    """Test/: """
    global _global
    _global = None
