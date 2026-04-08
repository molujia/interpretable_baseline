"""
util.py — CLLM v4 

: 
  - 
  - load_faults / load_topology
  - ProgressTrackerProgress tracking for resumable batch runs
"""

import json
import os
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Default service call topology (Online Boutique microservice example)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SERVICE2SERVICE: Dict[str, List[str]] = {
    "frontend":              ["checkoutservice", "adservice", "recommendationservice",
                              "productcatalogservice", "cartservice", "shippingservice",
                              "currencyservice"],
    "recommendationservice": ["productcatalogservice"],
    "productcatalogservice": ["tidb-tidb", "tidb-tikv", "tidb-pd"],
    "adservice":             ["tidb-tidb", "tidb-tikv", "tidb-pd"],
    "cartservice":           ["redis-cart"],
    "emailservice":          [],
    "paymentservice":        [],
    "checkoutservice":       ["emailservice", "paymentservice"],
    "tidb-tidb":             ["tidb-tikv", "tidb-pd"],
    "shippingservice":       [],
    "tidb-tikv":             ["tidb-tidb", "tidb-pd"],
    "tidb-pd":               ["tidb-tikv", "tidb-tidb"],
    "currencyservice":       [],
    "redis-cart":            [],
}


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

# def load_faults(path: str) -> List[dict]:
# """Load fault cases from JSON file, sorted by integer key"""
#     with open(path, encoding="utf-8") as f:
#         data = json.load(f)
#     return [v for _, v in sorted(data.items(), key=lambda x: int(x[0]))]

def load_faults(path: str, shuffle: bool = True, seed: int = 42) -> List[dict]:
    """Load fault cases from JSON file, sorted by integer key

    Args:
        path:    JSON 
        shuffle: Whether to shuffle, eliminating temporal clustering artefacts in learning curves
        seed:    Shuffle random seed for reproducibility (default: 42)
    """
    import random as _random
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    faults = [v for _, v in sorted(data.items(), key=lambda x: int(x[0]))]
    if shuffle:
        rng = _random.Random(seed)
        rng.shuffle(faults)
    return faults


def load_topology(
    topo_path:   Optional[str] = None,
    deploy_path: Optional[str] = None,
) -> Tuple[Dict[str, List[str]], dict]:
    """
    Load service topology (optional) and deployment info (optional).
    Returns the default topology and empty deployment info if no paths are given.
    """
    s2s    = DEFAULT_SERVICE2SERVICE
    deploy = {}
    if topo_path and os.path.exists(topo_path):
        with open(topo_path, encoding="utf-8") as f:
            s2s = json.load(f)
        print(f"[Util] Topology loaded from {topo_path}")
    if deploy_path and os.path.exists(deploy_path):
        with open(deploy_path, encoding="utf-8") as f:
            deploy = json.load(f)
        print(f"[Util] Deployment info loaded from {deploy_path}")
    return s2s, deploy


# ─────────────────────────────────────────────────────────────────────────────
# ProgressTracker — Progress tracking for resumable batch runs
# ─────────────────────────────────────────────────────────────────────────────

class ProgressTracker:
    """
    Manage batch-run progress and per-node report files.

    Directory layout (progress_dir/data_stem/):
      progress.json            ← index of the last completed case
      records.jsonl            ← evaluation record per case (append-only)
      node_0000_0009.txt       ← statistical report for cases 0-9
      node_0010_0019.txt       ← ...
    """
    NODE_SIZE = 10

    def __init__(self, progress_dir: str, data_stem: str):
        self.base_dir  = os.path.join(progress_dir, data_stem)
        self.data_stem = data_stem
        os.makedirs(self.base_dir, exist_ok=True)
        self._prog_path = os.path.join(self.base_dir, "progress.json")
        self._recs_path = os.path.join(self.base_dir, "records.jsonl")

    # ──  ─────────────────────────────────────────────────────────────

    def load_last_done(self) -> int:
        """Return the index of the last completed case, or -1 if not yet started."""
        if not os.path.exists(self._prog_path):
            return -1
        try:
            with open(self._prog_path) as f:
                return int(json.load(f).get("last_done", -1))
        except Exception:
            return -1

    def save_progress(self, last_done: int):
        with open(self._prog_path, "w") as f:
            json.dump({"last_done": last_done, "data_stem": self.data_stem}, f)

    def reset(self):
        """Clear the progress file (start over)."""
        for p in (self._prog_path, self._recs_path):
            if os.path.exists(p):
                os.remove(p)

    # ──  ─────────────────────────────────────────────────────────────

    def append_record(self, rec: dict):
        """Crash-safe append of a single case record."""
        with open(self._recs_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_records(self) -> List[dict]:
        """Load existing records (to restore statistics when resuming a run)."""
        if not os.path.exists(self._recs_path):
            return []
        records = []
        with open(self._recs_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records

    # ──  ──────────────────────────────────────────────────────────

    def write_node_report(
        self,
        node_start:   int,
        node_end:     int,
        node_records: List[dict],
        all_records:  List[dict],
    ):
        """
        Write a node report file covering cases [node_start, node_end].
        Report content is generated by evaluate.build_node_report().
        """
        from evaluate import build_node_report
        fname = os.path.join(
            self.base_dir, f"node_{node_start:04d}_{node_end:04d}.txt"
        )
        text = build_node_report(node_records, node_start, node_end,
                                 all_records, self.data_stem)
        with open(fname, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Progress] Node report -> {fname}")

    def write_summary(
        self,
        all_records: List[dict],
        use_llm:     bool,
        skip_types:  List[str],
    ):
        """Write the global summary file summary.txt."""
        from evaluate import build_summary_report
        fname = os.path.join(self.base_dir, "summary.txt")
        text  = build_summary_report(all_records, use_llm, skip_types, self.data_stem)
        with open(fname, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"[Progress] Summary -> {fname}")
        print(text)
