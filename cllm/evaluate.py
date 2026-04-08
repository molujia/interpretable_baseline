"""
evaluate.py — CLLM v4 EvaluationReport generation

Multi-root-cause hit rules
  ① pod+service ["tidb-tikv-0", "tidb-tikv"]
     →  ["tidb-tikv"]
     → 

  ② service+service ["checkoutservice", "emailservice"]
     →  Hit@K
     →  rank 

  ③ 
     →  base_name 

: 
  normalize_gt(root_cause)         → List[str]  GT
  best_rank(ranked, gt_bases)      → int        1-indexed-1=
  hit_at_k(ranked, gt_bases, k)    → bool       Top-K 
  best_hit_name(ranked, gt_bases)  → str|None   Top-1 PredictionGTService name
  build_node_report(...)           → str        
  build_summary_report(...)        → str        
  build_final_answer(...)          → str        
"""

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from cfcbn.cbn_accumulator import base_name


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def normalize_gt(root_cause: dict) -> List[str]:
    """
     root_cause 

    Examples
    --------
    {'name': 'emailservice'}              → ['emailservice']
    {'name': ['checkoutservice','shippingservice']} → ['checkoutservice','shippingservice']
    {'name': ['tidb-tikv-0','tidb-tikv']} → ['tidb-tikv']   # pod+service → 
    {'name': ['shippingservice-0']}       → ['shippingservice']
    """
    raw = root_cause.get("name", "")
    if isinstance(raw, str):
        raw = [raw]

    bases = [base_name(n) for n in raw]

    # (see source)
    seen, result = set(), []
    for b in bases:
        if b and b not in seen:
            seen.add(b)
            result.append(b)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def best_rank(ranked: List[str], gt_bases: List[str]) -> int:
    """
    GT 1-indexed
     GT 
     -1
    """
    ranked_bases = [base_name(s) for s in ranked]
    best = -1
    for g in gt_bases:
        try:
            pos = ranked_bases.index(g) + 1   # 1-indexed
            if best == -1 or pos < best:
                best = pos
        except ValueError:
            pass
    return best


def hit_at_k(ranked: List[str], gt_bases: List[str], k: int) -> bool:
    """Top-K  GT """
    top_k = {base_name(s) for s in ranked[:k]}
    return any(g in top_k for g in gt_bases)


def best_hit_name(ranked: List[str], gt_bases: List[str]) -> Optional[str]:
    """
     ranked  GT Service name"Prediction"
     None
    """
    ranked_bases = [base_name(s) for s in ranked]
    best_pos, best_svc = len(ranked_bases) + 1, None
    for g in gt_bases:
        try:
            pos = ranked_bases.index(g)
            if pos < best_pos:
                best_pos, best_svc = pos, g
        except ValueError:
            pass
    return best_svc


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def _sep(char="═", width=72): return char * width
def _rate(h, n): return f"{h/n*100:.1f}%" if n else "N/A"


def build_node_report(
    node_records: List[dict],
    node_start:   int,
    node_end:     int,
    all_records:  List[dict],
    data_stem:    str,
) -> str:
    """ node_start..node_end  case"""
    W = 72
    eval_node = [r for r in node_records if not r.get("skipped")]
    cum_recs  = [r for r in all_records  if not r.get("skipped")]
    n  = max(len(eval_node), 1)
    nc = max(len(cum_recs), 1)

    def h1(recs): return sum(1 for r in recs if 0 < r.get("cfcbn_rank", -1) <= 1)
    def h3(recs): return sum(1 for r in recs if 0 < r.get("cfcbn_rank", -1) <= 3)
    def h5(recs): return sum(1 for r in recs if 0 < r.get("cfcbn_rank", -1) <= 5)
    wa = sum(1 for r in eval_node if r.get("llm_assisted"))
    wb = sum(1 for r in eval_node if r.get("workflow_b_triggered"))

    lines = [
        _sep("=", W),
        f"  CLLM v4 Node Report [{data_stem}]  Cases {node_start}–{node_end}",
        _sep("=", W), "",
        _sep("-", W), "  CF-CBN (this node)", _sep("-", W),
        f"  Top-1 : {h1(eval_node):>4}/{n}  = {_rate(h1(eval_node), n)}",
        f"  Top-3 : {h3(eval_node):>4}/{n}  = {_rate(h3(eval_node), n)}",
        f"  Top-5 : {h5(eval_node):>4}/{n}  = {_rate(h5(eval_node), n)}",
        f"  WF-A triggered : {wa}/{n}",
        f"  WF-B triggered : {wb}/{n}",
        "", _sep("-", W), "  Cumulative (case 0 → present)", _sep("-", W),
        f"  Top-1 : {h1(cum_recs):>4}/{nc}  = {_rate(h1(cum_recs), nc)}",
        f"  Top-3 : {h3(cum_recs):>4}/{nc}  = {_rate(h3(cum_recs), nc)}",
        f"  Top-5 : {h5(cum_recs):>4}/{nc}  = {_rate(h5(cum_recs), nc)}",
        "", _sep("-", W), "  Per-Case Detail", _sep("-", W),
        f"  {'#':>4}  {'Sym':>4}  {'Rank':>5}  {'LLM':>4}  "
        f"{'GT (all roots)':<30}  Pred",
        "  " + "─" * (W - 2),
    ]

    for r in node_records:
        gt_bs  = r.get("gt_bases", [r.get("ground_truth", "?")])
        gt_str = ", ".join(gt_bs)[:30]
        if r.get("skipped"):
            lines.append(f"  {r['index']:>4}  SKIP              {gt_str}")
            continue
        rank = r.get("cfcbn_rank", -1)
        sym  = "OK" if rank == 1 else ("T3" if 1 < rank <= 3 else
               ("T5" if 3 < rank <= 5 else "NO"))
        wa_s = "A" if r.get("llm_assisted") else "-"
        wb_s = "B" if r.get("workflow_b_triggered") else "-"
        pred = str(r.get("cfcbn_top1", "?"))[:22]
        lines.append(
            f"  {r['index']:>4}  [{sym}]  {rank if rank>0 else 'N/A':>5}  "
            f"{wa_s}{wb_s}    {gt_str:<30}  {pred}"
        )

    lines += ["", _sep("=", W), ""]
    return "\n".join(lines)


def build_summary_report(
    all_records: List[dict],
    use_llm:     bool,
    skip_types:  List[str],
    data_stem:   str,
) -> str:
    """"""
    W = 72
    eval_recs = [r for r in all_records if not r.get("skipped")]
    n = max(len(eval_recs), 1)
    skipped = sum(1 for r in all_records if r.get("skipped"))

    h1 = sum(1 for r in eval_recs if 0 < r.get("cfcbn_rank", -1) <= 1)
    h3 = sum(1 for r in eval_recs if 0 < r.get("cfcbn_rank", -1) <= 3)
    h5 = sum(1 for r in eval_recs if 0 < r.get("cfcbn_rank", -1) <= 5)
    wa = sum(1 for r in eval_recs if r.get("llm_assisted"))
    wb = sum(1 for r in eval_recs if r.get("workflow_b_triggered"))
    times = [r.get("total_elapsed", 0.0) for r in eval_recs]
    avg_t = sum(times) / len(times) if times else 0.0

    # 
    ft: Dict[str, dict] = defaultdict(lambda: {"n": 0, "h1": 0, "h3": 0, "h5": 0})
    for r in eval_recs:
        t = r.get("fault_type", "unknown")
        ft[t]["n"]  += 1
        cr = r.get("cfcbn_rank", -1)
        if 0 < cr <= 1: ft[t]["h1"] += 1
        if 0 < cr <= 3: ft[t]["h3"] += 1
        if 0 < cr <= 5: ft[t]["h5"] += 1

    lines = [
        _sep("=", W),
        f"  CLLM v4 — Batch Summary  [{data_stem}]",
        _sep("=", W), "",
        f"  use_llm    : {use_llm}",
        f"  skip_types : {skip_types}",
        f"  Total      : {len(all_records)}   Evaluated: {len(eval_recs)}   Skipped: {skipped}",
        "", _sep("-", W), "  CF-CBN Hit Rate", _sep("-", W),
        f"  Top-1 : {h1:>4}/{n}  = {h1/n*100:.2f}%",
        f"  Top-3 : {h3:>4}/{n}  = {h3/n*100:.2f}%",
        f"  Top-5 : {h5:>4}/{n}  = {h5/n*100:.2f}%",
        "", _sep("-", W), "  LLM Workflows", _sep("-", W),
        f"  WorkflowA (real-time assist) : {wa:>4}/{n} = {wa/n:.2%}",
        f"  WorkflowB (post-mortem diag) : {wb:>4}/{n} = {wb/n:.2%}",
        "", _sep("-", W), "  By Fault Type (Top-1)", _sep("-", W),
    ]
    for fault_t, s in sorted(ft.items(), key=lambda x: -x[1]["n"]):
        fn = max(s["n"], 1)
        lines.append(
            f"  {fault_t:<30}  {s['h1']:>3}/{s['n']:<4}  "
            f"T1={s['h1']/fn:.0%}  T3={s['h3']/fn:.0%}  T5={s['h5']/fn:.0%}"
        )
    lines += [
        "", _sep("-", W), "  Timing", _sep("-", W),
        f"  Avg : {avg_t:.3f} s/case   Total : {sum(times):.2f} s",
        "", _sep("=", W), "",
    ]
    return "\n".join(lines)


def build_final_answer(
    all_records: List[dict],
    output_path: str = "final_answer.txt",
) -> str:
    """-case """
    W = 72
    eval_recs = [r for r in all_records if not r.get("skipped")]
    n = max(len(eval_recs), 1)
    skipped = sum(1 for r in all_records if r.get("skipped"))

    h1 = sum(1 for r in eval_recs if 0 < r.get("cfcbn_rank", -1) <= 1)
    h3 = sum(1 for r in eval_recs if 0 < r.get("cfcbn_rank", -1) <= 3)
    h5 = sum(1 for r in eval_recs if 0 < r.get("cfcbn_rank", -1) <= 5)

    cf_t  = [r.get("cfcbn_elapsed", 0) * 1000 for r in eval_recs]
    llm_t = [r.get("llm_elapsed", 0)           for r in eval_recs]
    tot_t = [r.get("total_elapsed", 0)          for r in eval_recs]
    def avg(lst): return sum(lst) / len(lst) if lst else 0.0
    wa = sum(1 for r in eval_recs if r.get("llm_assisted"))
    wb = sum(1 for r in eval_recs if r.get("workflow_b_triggered"))

    lines = [
        _sep("=", W), "  CLLM v4 — Final Answer", _sep("=", W), "",
        f"  Total: {len(all_records)}   Evaluated: {len(eval_recs)}   Skipped: {skipped}",
        "", _sep("-", W), "  CF-CBN Hit Rate", _sep("-", W),
        f"  Top-1 : {h1:>4}/{n}  = {h1/n*100:.2f}%",
        f"  Top-3 : {h3:>4}/{n}  = {h3/n*100:.2f}%",
        f"  Top-5 : {h5:>4}/{n}  = {h5/n*100:.2f}%",
        "", _sep("-", W), "  LLM Workflows", _sep("-", W),
        f"  WorkflowA : {wa:>4}/{n} = {wa/n:.2%}",
        f"  WorkflowB : {wb:>4}/{n} = {wb/n:.2%}",
        "", _sep("-", W), "  Timing", _sep("-", W),
        f"  CF-CBN  avg : {avg(cf_t):.2f} ms/case   total = {sum(cf_t)/1000:.2f} s",
        f"  LLM     avg : {avg(llm_t):.3f} s/case   total = {sum(llm_t):.2f} s",
        f"  Overall avg : {avg(tot_t):.3f} s/case   total = {sum(tot_t):.2f} s",
        "", _sep("-", W), "  Per-Case Detail", _sep("-", W),
        f"  {'#':>4}  {'Sym':>4}  {'Rank':>5}  {'LLM':>4}  "
        f"{'GT (all roots)':<30}  Pred",
        "  " + "─" * (W - 2),
    ]

    for r in all_records:
        gt_bs  = r.get("gt_bases", [r.get("ground_truth", "?")])
        gt_str = ", ".join(gt_bs)[:30]
        if r.get("skipped"):
            lines.append(f"  {r['index']:>4}  SKIP              {gt_str}")
            continue
        rank = r.get("cfcbn_rank", -1)
        sym  = "OK" if rank == 1 else ("T3" if 1 < rank <= 3 else
               ("T5" if 3 < rank <= 5 else "NO"))
        wa_s = "A" if r.get("llm_assisted") else "-"
        wb_s = "B" if r.get("workflow_b_triggered") else "-"
        pred = str(r.get("cfcbn_top1", "?"))[:22]
        lines.append(
            f"  {r['index']:>4}  [{sym}]  {rank if rank>0 else 'N/A':>5}  "
            f"{wa_s}{wb_s}    {gt_str:<30}  {pred}"
        )

    lines += [
        "", _sep("=", W),
        f"  Top-1={h1/n*100:.2f}%  Top-3={h3/n*100:.2f}%  Top-5={h5/n*100:.2f}%  "
        f"|  CF-CBN avg {avg(cf_t):.1f}ms  LLM avg {avg(llm_t):.2f}s",
        _sep("=", W), "",
    ]

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Evaluate] Final answer -> {os.path.abspath(output_path)}")
    return report
