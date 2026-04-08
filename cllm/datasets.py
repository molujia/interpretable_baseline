"""
datasets.py — CLLM v5 Dataset configuration centre

Two datasets supported
  ds25  Online Boutique + TiDBdata/ds25_faults.json
  tt    TrainTicket data/tt_faults.json 96/125 cases

Usage:
    from datasets import get_dataset_config
    cfg = get_dataset_config("tt")
    faults = cfg.load_faults()
    s2s    = cfg.service2service

TT dataset cleaning notes:
  29 of the original 125 cases have empty anomaly_services and anomaly_pods
  (no anomaly metrics), providing no signal for any metric-based RCL method.
  These are filtered out, leaving 96 usable cases.
  Filter criterion: sum of all metric list lengths == 0.
"""

import os
import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")


# ── DS25: Online Boutique + TiDB ─────────────────────────────────────────────

DS25_SERVICE2SERVICE: Dict[str, List[str]] = {
    "frontend":              ["checkoutservice", "adservice", "recommendationservice",
                              "productcatalogservice", "cartservice", "shippingservice",
                              "currencyservice",
                              "emailservice", "paymentservice"],  # spurious edges — removed by EdgeFilter (Step-0 prior knowledge)
    "recommendationservice": ["productcatalogservice", "currencyservice"],  # spurious edge — removed by EdgeFilter (Step-0 prior knowledge)
    "productcatalogservice": ["tidb-tidb", "tidb-tikv", "tidb-pd"],
    "adservice":             ["tidb-tidb", "tidb-tikv", "tidb-pd"],
    "cartservice":           ["redis-cart", "paymentservice"],  # spurious edge — removed by EdgeFilter (Step-0 prior knowledge)
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


# ── TT: TrainTicket ──────────────────────────────────────────────────────────

TT_SERVICES = sorted([
    'admin-basic-info', 'admin-order', 'admin-route', 'admin-travel',
    'admin-user', 'assurance', 'auth', 'avatar', 'basic', 'cancel',
    'config', 'consign', 'consign-price', 'contacts', 'delivery',
    'execute', 'food', 'food-delivery', 'frontend', 'gateway',
    'inside-payment', 'mysql', 'nacos', 'news', 'notification',
    'order', 'payment', 'preserve', 'price', 'rabbitmq',
    'rebook', 'route', 'route-plan', 'seat', 'security',
    'sentinel', 'station', 'station-food', 'ticket-office',
    'ticketinfo', 'train', 'train-food', 'travel', 'travel-plan',
    'user', 'verification-code', 'voucher', 'wait-order',
])


# ── DatasetConfig ─────────────────────────────────────────────────────────────

@dataclass
class DatasetConfig:
    name:            str
    data_path:       str
    topo_path:       Optional[str]
    service2service: Dict[str, List[str]]
    all_services:    List[str]        # Full service list (used by CBN accumulator)
    description:     str = ""

    #  outputs/<name>/ 
    store_dir:    str = field(default="")
    progress_dir: str = field(default="")
    ticket_dir:   str = field(default="")
    final_answer: str = field(default="")

    def __post_init__(self):
        base = os.path.join("outputs", self.name)
        # out_root Root output directory for this dataset; main.py creates timestamped subdirectories here
        self.out_root     = base
        if not self.store_dir:
            self.store_dir    = os.path.join(base, "case_store")
        if not self.progress_dir:
            self.progress_dir = os.path.join(base, "progress")
        if not self.ticket_dir:
            self.ticket_dir   = os.path.join(base, "tickets")
        if not self.final_answer:
            self.final_answer = os.path.join(base, "final_answer.txt")

    def load_faults(self, shuffle: bool = True, seed: int = 42) -> List[dict]:
        with open(self.data_path, encoding="utf-8") as f:
            data = json.load(f)
        faults = [v for _, v in sorted(data.items(), key=lambda x: int(x[0]))]
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(faults)
        return faults

    def summary(self) -> str:
        n = len(json.load(open(self.data_path)))
        return (f"Dataset '{self.name}': {n} cases | "
                f"{len(self.all_services)} services | {self.description}")


# ──  ──────────────────────────────────────────────────────────────────

def get_dataset_config(name: str) -> DatasetConfig:
    """
    

    name: "ds25" | "tt"
      ds25 — Online Boutique + TiDB14 servicesds25_faults.json
      tt   — TrainTicket (96/125 cases after filtering, 48 services, tt_faults.json)
    """
    name = name.lower().strip()

    if name == "ds25":
        from cfcbn.cbn_accumulator import ALL_SERVICES
        return DatasetConfig(
            name            = "ds25",
            data_path       = os.path.join(_DATA, "ds25_faults.json"),
            topo_path       = os.path.join(_DATA, "ds25_topo_s2s.json"),
            service2service = DS25_SERVICE2SERVICE,
            all_services    = ALL_SERVICES,
            description     = "Online Boutique + TiDB (14 services, ds25_faults.json)",
        )

    elif name == "tt":
        topo_path = os.path.join(_DATA, "tt_topo_s2s.json")
        s2s = {}
        if os.path.exists(topo_path):
            with open(topo_path) as f:
                s2s = json.load(f)
        return DatasetConfig(
            name            = "tt",
            data_path       = os.path.join(_DATA, "tt_faults.json"),
            topo_path       = topo_path,
            service2service = s2s,
            all_services    = TT_SERVICES,
            description     = "TrainTicket (96/125 cases, zero-signal cases filtered)",
        )

    else:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: 'ds25', 'tt'.\n"
            f"  ds25 — Online Boutique + TiDB  (data/ds25_faults.json)\n"
            f"  tt   — TrainTicket             (data/tt_faults.json)"
        )


AVAILABLE_DATASETS = ["ds25", "tt"]
