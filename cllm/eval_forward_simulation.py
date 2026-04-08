"""
eval_forward_simulation.py — Forward Simulation / Prediction Evaluation

Based on: Doshi-Velez & Kim (2017) §3.2 Human-grounded Evaluation
"humans are presented with an explanation and an input, and must correctly
simulate the model's output (regardless of the true output)."

The LLM agent acts as a lay human who has been taught the CF-CBN algorithm.
For each case, the agent predicts the model's ranking; we measure alignment
via KL divergence and top-1/3 match rate.

Usage (from cllm/ directory):
    python eval_forward_simulation.py --dataset tt
    python eval_forward_simulation.py --dataset ds25
    python eval_forward_simulation.py --dataset ds25 --reset   # restart

Output: outputs/eval_forward_sim_<dataset>/
    records.jsonl  — append-only per-case results (crash-safe)
    progress.json  — last completed index (for resuming)
    summary.txt    — final statistics

LLM config: edit EVAL_PLATFORM / EVAL_MODEL in eval_utils.py
            or set env var EVAL_LLM_API_KEY before running.
"""

import os
import sys
import json
import math
import argparse
from typing import Optional, List, Dict

# Ensure cllm/ is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataset_config
from cfcbn.crfd_cbn_engine import CRFDCBNEngine
from evaluate import normalize_gt

from eval_utils import (
    EvalLLMClient,
    SYSTEM_FORWARD_SIM,
    build_forward_prompt,
    parse_forward_response,
    to_distribution,
    kl_divergence,
    load_progress,
    save_progress,
    append_record,
    load_records,
    early_stop_check,
    build_memory_ctx,
    write_summary,
)

# ─── Early stopping parameters ────────────────────────────────────────────────
EARLY_STOP_N         = 20    # check after this many evaluated cases
EARLY_STOP_THRESHOLD = 0.05  # < 5% top-1 match → prompt needs revision


# ─── Helper ───────────────────────────────────────────────────────────────────

def _preprocess(fault_data: dict, skip_types=("node",)) -> Optional[tuple]:
    """Return (gt_bases, fault_type) or None if this case type is skipped."""
    rc = fault_data.get("root_cause", {})
    if rc.get("type", "service") in skip_types:
        return None
    gt_bases   = normalize_gt(rc)
    fault_type = rc.get("fault_type", "unknown")
    return gt_bases, fault_type


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run(args):
    cfg     = get_dataset_config(args.dataset)
    out_dir = os.path.join("outputs", f"eval_forward_sim_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Forward Simulation Evaluation  [{args.dataset.upper()}]")
    print(f"  Paper: Doshi-Velez & Kim 2017 §3.2")
    print(f"{'='*60}")
    print(f"  {cfg.summary()}")
    print(f"  Output → {os.path.abspath(out_dir)}/")

    # Load faults with same shuffle as main.py for CBN state consistency
    faults = cfg.load_faults(shuffle=True, seed=42)

    # Fresh CF-CBN engine (will be re-accumulated case-by-case)
    engine = CRFDCBNEngine(
        services    = cfg.all_services,
        alpha_init  = 1.0,
        alpha_min   = 0.20,
        alpha_decay = 40.0,
    )

    llm = EvalLLMClient()

    # ── Reset if requested ──────────────────────────────────────────────────
    if args.reset:
        for fname in ("progress.json", "records.jsonl", "summary.txt"):
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        print("[ForwardSim] Progress reset — starting from scratch.")

    # ── Load existing progress ──────────────────────────────────────────────
    last_done = load_progress(out_dir)
    start_idx = last_done + 1

    # ── Restore CBN state by replaying previous cases (no LLM calls) ────────
    if start_idx > 0:
        print(f"[ForwardSim] Resuming from case {start_idx}. "
              f"Restoring CBN state for {start_idx} prior cases...")
        for i, fault in enumerate(faults[:start_idx]):
            prep = _preprocess(fault)
            if prep is None:
                continue
            gt_bases, _ = prep
            try:
                engine.predict(fault)
                engine.accumulate(fault, gt_bases)
            except Exception:
                pass
        print(f"[ForwardSim] CBN restored: {engine.n_accumulated} cases in history.")

    # ── Load existing records ───────────────────────────────────────────────
    all_records  = load_records(out_dir)
    eval_records = [r for r in all_records if not r.get("skipped")]
    skipped      = sum(1 for r in all_records if r.get("skipped"))

    print(f"[ForwardSim] Evaluating cases {start_idx}–{len(faults)-1}  "
          f"(already done: {len(eval_records)} eval + {skipped} skipped)\n")

    # ── Main loop ───────────────────────────────────────────────────────────
    for idx, fault_data in enumerate(faults):
        if idx < start_idx:
            continue

        prep = _preprocess(fault_data)
        if prep is None:
            rec = {
                "index":   idx,
                "skipped": True,
                "gt":      normalize_gt(fault_data.get("root_cause", {})),
            }
            all_records.append(rec)
            append_record(out_dir, rec)
            save_progress(out_dir, idx)
            if args.verbose:
                print(f"  [{idx:>4}] SKIP (rc_type=node)")
            continue

        gt_bases, fault_type = prep

        # ── Run actual CF-CBN model ───────────────────────────────────────
        try:
            cfcbn_ranked, details = engine.predict(fault_data)
        except Exception as e:
            print(f"  [{idx:>4}] MODEL ERROR: {e}")
            engine.accumulate(fault_data, gt_bases)
            continue

        fused_scores  = details["fused_scores"]
        alpha         = details["alpha"]
        n_hist        = details["n_accumulated"]
        prior_counts  = dict(engine.accumulator.prior_counts)
        model_top1    = cfcbn_ranked[0] if cfcbn_ranked else "unknown"
        model_top3    = cfcbn_ranked[:3]

        # ── Build teaching prompt ─────────────────────────────────────────
        memory_ctx = build_memory_ctx(eval_records, n=3)
        prompt = build_forward_prompt(
            case         = fault_data,
            services     = cfg.all_services,
            alpha        = alpha,
            prior_counts = prior_counts,
            n_history    = n_hist,
            memory_ctx   = memory_ctx,
        )

        # ── Call LLM agent (single agent, with memory context) ───────────
        agent_top1, agent_scores = "error", {}
        raw_response = ""
        try:
            raw_response  = llm.invoke(prompt, system=SYSTEM_FORWARD_SIM)
            agent_top1, agent_scores = parse_forward_response(raw_response, cfg.all_services)
        except Exception as e:
            print(f"  [{idx:>4}] LLM ERROR: {e}")

        # ── Compute metrics ───────────────────────────────────────────────
        top1_match = (agent_top1 == model_top1) and agent_top1 != "error"

        agent_ranked = sorted(
            agent_scores.keys(),
            key=lambda s: -agent_scores.get(s, 0.0)
        )
        top3_match = any(s in model_top3 for s in agent_ranked[:3])

        # KL divergence P_model || P_agent (over full service list)
        P_model = to_distribution(fused_scores,  cfg.all_services)
        P_agent = to_distribution(agent_scores,  cfg.all_services)
        kl_div  = kl_divergence(P_model, P_agent)
        if math.isnan(kl_div) or math.isinf(kl_div):
            kl_div = 9.99  # sentinel for parse failures

        # ── Record ────────────────────────────────────────────────────────
        rec = {
            "index":       idx,
            "dataset":     args.dataset,
            "gt":          gt_bases,
            "fault_type":  fault_type,
            "model_top1":  model_top1,
            "model_top3":  model_top3,
            "agent_top1":  agent_top1,
            "agent_top3":  agent_ranked[:3],
            "top1_match":  top1_match,
            "top3_match":  top3_match,
            "kl_div":      round(kl_div, 6),
            "alpha":       round(alpha, 4),
            "n_history":   n_hist,
            # Store top-5 for later analysis
            "model_scores_top5": {
                k: round(v, 5)
                for k, v in sorted(fused_scores.items(), key=lambda x: -x[1])[:5]
            },
            "agent_scores_top5": {
                k: round(v, 4)
                for k, v in sorted(agent_scores.items(), key=lambda x: -x[1])[:5]
            },
        }

        all_records.append(rec)
        eval_records.append(rec)
        append_record(out_dir, rec)

        # ── Accumulate for CBN continuity ────────────────────────────────
        engine.accumulate(fault_data, gt_bases)
        save_progress(out_dir, idx)

        # ── Console line ─────────────────────────────────────────────────
        match_sym = "T" if top1_match else "F"
        n_eval = len(eval_records)
        t1_rate = sum(1 for r in eval_records if r.get("top1_match")) / max(n_eval, 1)
        avg_kl  = sum(r.get("kl_div", 0) for r in eval_records) / max(n_eval, 1)
        print(f"  [{idx:>4}] {match_sym}  "
              f"model={model_top1:<22} agent={agent_top1:<22}  "
              f"KL={kl_div:6.3f}  α={alpha:.3f}  "
              f"[cum T1={t1_rate:.1%} avgKL={avg_kl:.3f}]")

        if args.verbose:
            print(f"         gt={gt_bases}  model_top3={model_top3}  "
                  f"agent_top3={agent_ranked[:3]}")

        # ── Early stopping check ─────────────────────────────────────────
        if early_stop_check(eval_records, mode="forward",
                            n_check=EARLY_STOP_N, threshold=EARLY_STOP_THRESHOLD):
            rate = sum(1 for r in eval_records[:EARLY_STOP_N]
                       if r.get("top1_match")) / EARLY_STOP_N
            print(f"\n{'!'*60}")
            print(f"  EARLY STOP: first {EARLY_STOP_N} cases → top1_match={rate:.1%}")
            print(f"  This is below threshold {EARLY_STOP_THRESHOLD:.0%}.")
            print(f"  Possible causes:")
            print(f"    • Teaching prompt not clear enough for this dataset")
            print(f"    • Model choice too weak (try gpt-4o or better)")
            print(f"    • Dataset has naming mismatch (TT: ts-*-service names)")
            print(f"  → Revise SYSTEM_FORWARD_SIM in eval_utils.py and retry.")
            print(f"{'!'*60}\n")
            break

    # ── Final summary ────────────────────────────────────────────────────────
    write_summary(out_dir, all_records, mode="forward")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "CF-CBN Forward Simulation Evaluation\n"
            "Doshi-Velez & Kim 2017 §3.2: agent simulates model output without code"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", default="ds25", choices=["tt", "ds25"],
        help="Dataset: tt (TrainTicket, 96 cases) or ds25 (Online Boutique+TiDB, 400 cases)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear progress and start from scratch",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print extra detail per case",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
