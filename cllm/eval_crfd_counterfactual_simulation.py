"""
eval_crfd_counterfactual_simulation.py — CRFD Counterfactual Simulation Evaluation
Doshi-Velez & Kim (2017) §3.2 Human-grounded Evaluation: Counterfactual Simulation

An LLM agent receives the current CRFD prediction and must suggest what metric changes
would flip the top-1 prediction to a DIFFERENT service.  Changes are applied to a copy
of the case and the CRFD engine is re-run to verify success.

Metric: success rate — proportion of cases where the agent's changes flip CRFD top-1.

Usage
-----
    cd cllm
    python eval_crfd_counterfactual_simulation.py --dataset tt
    python eval_crfd_counterfactual_simulation.py --dataset ds25 --verbose
    python eval_crfd_counterfactual_simulation.py --dataset tt --reset
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataset_config
from evaluate import normalize_gt
from crfd_engine import CRFDEngine
from eval_utils import (
    EvalLLMClient,
    SYSTEM_COUNTERFACTUAL_CRFD,
    build_crfd_counterfactual_prompt,
    parse_counterfactual_response,
    apply_changes,
    load_progress,
    save_progress,
    append_record,
    load_records,
    early_stop_check,
    build_memory_ctx,
    write_summary,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "CRFD Counterfactual Simulation Evaluation\n"
            "Doshi-Velez & Kim 2017 §3.2: agent identifies changes to flip CRFD prediction"
        )
    )
    p.add_argument("--dataset", default="ds25", choices=["tt", "ds25"])
    p.add_argument("--reset", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


# ── main run logic ─────────────────────────────────────────────────────────────

def run(args):
    cfg = get_dataset_config(args.dataset)
    out_dir = os.path.join("outputs", f"eval_crfd_counterfactual_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    faults = cfg.load_faults(shuffle=True, seed=42)
    engine = CRFDEngine(services=cfg.all_services,
                        service2service=cfg.service2service)
    llm    = EvalLLMClient()

    # ── Reset ─────────────────────────────────────────────────────────────────
    if args.reset:
        for fname in ("progress.json", "records.jsonl"):
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        print(f"[CRFDCf] Progress reset for {args.dataset}.")

    last_done    = load_progress(out_dir)
    eval_records = load_records(out_dir)
    start_idx    = last_done + 1

    if start_idx > 0:
        print(f"[CRFDCf] Resuming from case {start_idx} "
              f"({len(eval_records)} records already saved).")

    # ── Batch loop ────────────────────────────────────────────────────────────
    for idx, fault_data in enumerate(faults):
        if idx < start_idx:
            continue

        rc_info  = fault_data.get("root_cause", {})
        gt_bases = normalize_gt(rc_info)
        rc_type  = rc_info.get("type", "service")

        if rc_type == "node":
            rec = {"index": idx, "skipped": True, "reason": "node-type fault"}
            append_record(out_dir, rec)
            save_progress(out_dir, idx)
            eval_records.append(rec)
            continue

        # ── Original CRFD prediction ──────────────────────────────────────────
        ranked_orig, details_orig = engine.predict(fault_data)
        original_top1    = ranked_orig[0] if ranked_orig else "unknown"
        original_scores  = details_orig["scores"]

        if all(v == 0.0 for v in original_scores.values()):
            rec = {"index": idx, "skipped": True, "reason": "zero-signal case"}
            append_record(out_dir, rec)
            save_progress(out_dir, idx)
            eval_records.append(rec)
            continue

        # ── Build prompt ──────────────────────────────────────────────────────
        memory_ctx = build_memory_ctx(eval_records)
        prompt = build_crfd_counterfactual_prompt(
            case=fault_data,
            original_top1=original_top1,
            scores=original_scores,
            services=cfg.all_services,
            topology=cfg.service2service,
            memory_ctx=memory_ctx,
        )

        # ── LLM call ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            raw = llm.invoke(prompt, system=SYSTEM_COUNTERFACTUAL_CRFD)
        except Exception as e:
            print(f"[CRFDCf] Case {idx}: LLM error — {e}")
            raw = ""
        llm_elapsed = time.perf_counter() - t0

        changes, agent_target, reasoning = parse_counterfactual_response(raw)

        # ── Apply changes and re-run CRFD ─────────────────────────────────────
        n_changes = len(changes)
        if n_changes > 0:
            modified_case  = apply_changes(fault_data, changes)
            ranked_new, _  = engine.predict(modified_case)
            new_top1       = ranked_new[0] if ranked_new else "unknown"
            success        = (new_top1 != original_top1)
        else:
            new_top1 = original_top1
            success  = False

        # ── Record ────────────────────────────────────────────────────────────
        rec = {
            "index":         idx,
            "gt_bases":      gt_bases,
            "fault_type":    rc_info.get("fault_type", "unknown"),
            "original_top1": original_top1,
            "agent_target":  agent_target,
            "new_top1":      new_top1,
            "n_changes":     n_changes,
            "success":       success,
            "reasoning":     reasoning,
            "llm_elapsed":   round(llm_elapsed, 4),
            "case_id":        idx,
            "top1_prediction": original_top1,
            "target_service":  agent_target,
        }
        append_record(out_dir, rec)
        save_progress(out_dir, idx)
        eval_records.append(rec)

        if args.verbose:
            print(f"  [{idx:>4}] orig={original_top1:<22} target={agent_target:<22} "
                  f"new={new_top1:<22} success={'Y' if success else 'N'}  "
                  f"changes={n_changes}  gt={gt_bases}")
        else:
            sym = "OK" if success else "--"
            print(f"  [{idx:>4}] {sym}  orig={original_top1:<22} target={agent_target:<22} "
                  f"new={new_top1:<22}")

        # ── Early stop ────────────────────────────────────────────────────────
        if early_stop_check(eval_records, mode="counterfactual"):
            print(
                f"\n[CRFDCf] EARLY STOP: first {len(eval_records)} cases all show "
                f"success rate < 5%."
            )
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    write_summary(out_dir, eval_records, mode="counterfactual")

    valid        = [r for r in eval_records if not r.get("skipped")]
    n            = max(len(valid), 1)
    success_rate = sum(1 for r in valid if r.get("success")) / n

    print(f"\n{'='*60}")
    print(f"  CRFD Counterfactual Simulation — {args.dataset.upper()}  [{n} cases]")
    print(f"{'='*60}")
    print(f"  Success rate : {success_rate:.2%}  "
          f"({sum(1 for r in valid if r.get('success'))}/{n})")
    print(f"  Results dir  : {os.path.abspath(out_dir)}/")
    print(f"{'='*60}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(_build_parser().parse_args())
