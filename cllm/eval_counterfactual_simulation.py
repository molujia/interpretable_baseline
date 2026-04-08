"""
eval_counterfactual_simulation.py — Counterfactual Simulation Evaluation

Based on: Doshi-Velez & Kim (2017) §3.2 Human-grounded Evaluation
"humans are presented with an explanation, an input, and an output, and are asked
what must be changed to change the method's prediction to a desired output."

The LLM agent acts as a lay human who understands the CF-CBN model. For each
case the agent sees the current prediction and must suggest input changes that
will cause the model to predict a different service. We then apply those changes
and re-run the model to verify.

Usage (from cllm/ directory):
    python eval_counterfactual_simulation.py --dataset tt
    python eval_counterfactual_simulation.py --dataset ds25
    python eval_counterfactual_simulation.py --dataset ds25 --reset

Output: outputs/eval_counterfactual_<dataset>/
    records.jsonl  — append-only per-case results (crash-safe)
    progress.json  — last completed index (for resuming)
    summary.txt    — final statistics

Success metric: proportion of cases where agent's changes actually flip the
model's top-1 prediction to a different service.
"""

import os
import sys
import json
import argparse
from typing import Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataset_config
from cfcbn.crfd_cbn_engine import CRFDCBNEngine
from evaluate import normalize_gt

from eval_utils import (
    EvalLLMClient,
    SYSTEM_COUNTERFACTUAL,
    build_counterfactual_prompt,
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

EARLY_STOP_N         = 20
EARLY_STOP_THRESHOLD = 0.05   # < 5% success rate → needs revision


def _preprocess(fault_data: dict, skip_types=("node",)) -> Optional[List[str]]:
    rc = fault_data.get("root_cause", {})
    if rc.get("type", "service") in skip_types:
        return None
    return normalize_gt(rc)


def run(args):
    cfg     = get_dataset_config(args.dataset)
    out_dir = os.path.join("outputs", f"eval_counterfactual_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Counterfactual Simulation Evaluation  [{args.dataset.upper()}]")
    print(f"  Paper: Doshi-Velez & Kim 2017 §3.2")
    print(f"{'='*60}")
    print(f"  {cfg.summary()}")
    print(f"  Output → {os.path.abspath(out_dir)}/")

    faults = cfg.load_faults(shuffle=True, seed=42)

    engine = CRFDCBNEngine(
        services    = cfg.all_services,
        alpha_init  = 1.0,
        alpha_min   = 0.20,
        alpha_decay = 40.0,
    )

    llm = EvalLLMClient()

    # ── Reset ───────────────────────────────────────────────────────────────
    if args.reset:
        for fname in ("progress.json", "records.jsonl", "summary.txt"):
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        print("[CounterSim] Progress reset — starting from scratch.")

    # ── Resume ──────────────────────────────────────────────────────────────
    last_done = load_progress(out_dir)
    start_idx = last_done + 1

    if start_idx > 0:
        print(f"[CounterSim] Resuming from case {start_idx}. "
              f"Restoring CBN state...")
        for i, fault in enumerate(faults[:start_idx]):
            prep = _preprocess(fault)
            if prep is None:
                continue
            try:
                engine.predict(fault)
                engine.accumulate(fault, prep)
            except Exception:
                pass
        print(f"[CounterSim] CBN restored: {engine.n_accumulated} cases.")

    all_records  = load_records(out_dir)
    eval_records = [r for r in all_records if not r.get("skipped")]
    skipped      = sum(1 for r in all_records if r.get("skipped"))

    print(f"[CounterSim] Evaluating cases {start_idx}–{len(faults)-1}  "
          f"(already done: {len(eval_records)} eval + {skipped} skipped)\n")

    # ── Main loop ────────────────────────────────────────────────────────────
    for idx, fault_data in enumerate(faults):
        if idx < start_idx:
            continue

        prep = _preprocess(fault_data)
        if prep is None:
            rec = {"index": idx, "skipped": True}
            all_records.append(rec)
            append_record(out_dir, rec)
            save_progress(out_dir, idx)
            if args.verbose:
                print(f"  [{idx:>4}] SKIP (rc_type=node)")
            continue

        gt_bases = prep

        # ── Run model on original case ───────────────────────────────────
        try:
            cfcbn_ranked, details = engine.predict(fault_data)
        except Exception as e:
            print(f"  [{idx:>4}] MODEL ERROR: {e}")
            engine.accumulate(fault_data, gt_bases)
            continue

        fused_scores  = details["fused_scores"]
        original_top1 = cfcbn_ranked[0] if cfcbn_ranked else "unknown"

        # ── Build counterfactual prompt ──────────────────────────────────
        memory_ctx = build_memory_ctx(eval_records, n=3)
        prompt = build_counterfactual_prompt(
            case          = fault_data,
            top1_service  = original_top1,
            model_scores  = fused_scores,
            memory_ctx    = memory_ctx,
        )

        # ── Call LLM agent ───────────────────────────────────────────────
        changes = []
        agent_target = "unknown"
        agent_reasoning = ""
        raw_response = ""
        try:
            raw_response    = llm.invoke(prompt, system=SYSTEM_COUNTERFACTUAL)
            changes, agent_target, agent_reasoning = parse_counterfactual_response(raw_response)
        except Exception as e:
            print(f"  [{idx:>4}] LLM ERROR: {e}")

        # ── Apply changes and re-run model ───────────────────────────────
        new_top1 = original_top1
        n_changes_applied = 0
        if changes:
            try:
                modified_case = apply_changes(fault_data, changes)
                new_ranked, _ = engine.predict(modified_case)
                new_top1 = new_ranked[0] if new_ranked else original_top1
                n_changes_applied = len(changes)
            except Exception as e:
                print(f"  [{idx:>4}] APPLY ERROR: {e}")

        # Success: prediction changed AND we actually applied some changes
        success = (new_top1 != original_top1) and n_changes_applied > 0

        # ── Record ───────────────────────────────────────────────────────
        rc_info = fault_data.get("root_cause", {})
        rec = {
            "index":            idx,
            "dataset":          args.dataset,
            "gt":               gt_bases,
            "fault_type":       rc_info.get("fault_type", "unknown"),
            "original_top1":    original_top1,
            "new_top1":         new_top1,
            "success":          success,
            "agent_target":     agent_target,
            "agent_reasoning":  agent_reasoning[:200] if agent_reasoning else "",
            "changes_applied":  changes,
            "n_changes":        n_changes_applied,
        }

        all_records.append(rec)
        eval_records.append(rec)
        append_record(out_dir, rec)

        # Accumulate ORIGINAL case (not modified) for CBN state consistency
        engine.accumulate(fault_data, gt_bases)
        save_progress(out_dir, idx)

        # ── Console line ─────────────────────────────────────────────────
        sym     = "OK" if success else "--"
        n_eval  = len(eval_records)
        s_rate  = sum(1 for r in eval_records if r.get("success")) / max(n_eval, 1)
        print(f"  [{idx:>4}] {sym}  "
              f"orig={original_top1:<22} new={new_top1:<22}  "
              f"n_changes={n_changes_applied}  "
              f"[cum success={s_rate:.1%}]")

        if args.verbose:
            print(f"         gt={gt_bases}  target={agent_target}  "
                  f"changes={[c.get('action') for c in changes]}")

        # ── Early stopping ────────────────────────────────────────────────
        if early_stop_check(eval_records, mode="counterfactual",
                            n_check=EARLY_STOP_N, threshold=EARLY_STOP_THRESHOLD):
            rate = sum(1 for r in eval_records[:EARLY_STOP_N]
                       if r.get("success")) / EARLY_STOP_N
            print(f"\n{'!'*60}")
            print(f"  EARLY STOP: first {EARLY_STOP_N} cases → success={rate:.1%}")
            print(f"  This is below threshold {EARLY_STOP_THRESHOLD:.0%}.")
            print(f"  Possible causes:")
            print(f"    • Agent changes don't parse/apply correctly")
            print(f"    • Model too rigid (uniform scores, naming mismatch)")
            print(f"    • Counterfactual prompt not specific enough")
            print(f"  → Revise SYSTEM_COUNTERFACTUAL in eval_utils.py and retry.")
            print(f"{'!'*60}\n")
            break

    # ── Final summary ────────────────────────────────────────────────────────
    write_summary(out_dir, all_records, mode="counterfactual")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "CF-CBN Counterfactual Simulation Evaluation\n"
            "Doshi-Velez & Kim 2017 §3.2: agent identifies input changes to flip prediction"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="ds25", choices=["tt", "ds25"],
        help="Dataset: tt (TrainTicket) or ds25 (Online Boutique+TiDB)",
    )
    parser.add_argument("--reset",   action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
