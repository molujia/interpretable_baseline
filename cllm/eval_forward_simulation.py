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
import argparse
from typing import Optional

# Ensure cllm/ is on path when run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataset_config
from cfcbn.crfd_cbn_engine import CRFDCBNEngine
from evaluate import normalize_gt

from eval_utils import (
    EvalLLMClient,
    EvalLLMAPIError,
    set_eval_log_file,
    log_failure,
    classify_unknown_reason,
    SYSTEM_FORWARD_SIM,
    build_forward_prompt,
    parse_forward_response,
    rank_vector_distance,
    _max_rank_dist,
    load_progress,
    save_progress,
    append_record,
    load_records,
    build_memory_ctx,
    write_summary,
)


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

    faults = cfg.load_faults(shuffle=True, seed=42)

    # Fresh CF-CBN engine — no history accumulation (each case predicted independently)
    engine = CRFDCBNEngine(
        services    = cfg.all_services,
        alpha_init  = 1.0,
        alpha_min   = 1.0,   # keep alpha fixed at 1.0: no CBN history effect
        alpha_decay = 1e9,
    )

    llm = EvalLLMClient()

    # ── Initialise LLM call logging ─────────────────────────────────────────
    set_eval_log_file(
        log_path     = os.path.join(out_dir, "llm_calls.log"),
        failures_dir = os.path.join(out_dir, "failures"),
    )

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

        # ── Run actual CF-CBN model (no history: each case independent) ─────
        try:
            cfcbn_ranked, details = engine.predict(fault_data)
        except Exception as e:
            print(f"  [{idx:>4}] MODEL ERROR: {e}")
            continue

        fused_scores = details["fused_scores"]
        model_top1   = cfcbn_ranked[0] if cfcbn_ranked else "unknown"

        # ── Build teaching prompt ─────────────────────────────────────────
        memory_ctx = build_memory_ctx(eval_records, n=3)
        prompt = build_forward_prompt(
            case       = fault_data,
            services   = cfg.all_services,
            memory_ctx = memory_ctx,
        )

        # ── Call LLM agent (single agent, with memory context) ───────────
        agent_top1, agent_scores = "error", {}
        unknown_reason = None
        raw_response = ""
        for _attempt in range(2):   # initial attempt + 1 case-level retry for API failure
            try:
                raw_response = llm.invoke(
                    prompt, system=SYSTEM_FORWARD_SIM, case_idx=idx)
            except EvalLLMAPIError as e:
                unknown_reason = "api_failure"
                log_failure(idx, "api_failure", prompt, SYSTEM_FORWARD_SIM, "", str(e))
                print(f"  [{idx:>4}] API FAILURE (attempt {_attempt+1}/2): {e}")
                if _attempt == 0:
                    continue   # retry the case
                break
            except Exception as e:
                unknown_reason = "unknown_error"
                log_failure(idx, "unknown_error", prompt, SYSTEM_FORWARD_SIM,
                            raw_response, str(e))
                print(f"  [{idx:>4}] LLM ERROR: {e}")
                break
            agent_top1, agent_scores = parse_forward_response(
                raw_response, cfg.all_services)
            if agent_top1 == "unknown":
                unknown_reason = classify_unknown_reason(raw_response)
                if unknown_reason in ("api_failure", "parse_failure"):
                    log_failure(idx, unknown_reason, prompt, SYSTEM_FORWARD_SIM,
                                raw_response)
                    print(f"  [{idx:>4}] UNKNOWN reason={unknown_reason}"
                          f"  (see failures/ for details)")
                    if unknown_reason == "api_failure" and _attempt == 0:
                        continue   # retry the case
            break

        # ── Compute metrics ───────────────────────────────────────────────
        agent_ranked = sorted(
            agent_scores.keys(),
            key=lambda s: -agent_scores.get(s, 0.0)
        )

        model_ranked_list = sorted(
            [s for s in fused_scores if fused_scores[s] > 0],
            key=lambda s: -fused_scores[s]
        )
        agent_ranked_list = sorted(
            agent_scores.keys(), key=lambda s: -agent_scores.get(s, 0.0)
        )
        rank_dist = rank_vector_distance(model_ranked_list, agent_ranked_list,
                                         cfg.all_services)
        rank_dist_norm = round(rank_dist / _max_rank_dist(len(cfg.all_services)), 4)

        # ── Record ────────────────────────────────────────────────────────
        rec = {
            "index":          idx,
            "case_id":        idx,
            "dataset":        args.dataset,
            "gt":             gt_bases,
            "gt_bases":       gt_bases,
            "ground_truth":   gt_bases[0] if gt_bases else "unknown",
            "fault_type":     fault_type,
            "model_top1":     model_top1,
            "agent_top1":     agent_top1,
            "rank_dist":      rank_dist,
            "rank_dist_norm": rank_dist_norm,
            "model_ranked":   model_ranked_list[:5],
            "agent_ranked":   agent_ranked_list[:5],
            "unknown_reason": unknown_reason,
        }

        all_records.append(rec)
        eval_records.append(rec)
        append_record(out_dir, rec)
        save_progress(out_dir, idx)

        # ── Console line ─────────────────────────────────────────────────
        n_eval = len(eval_records)
        avg_norm = sum(r.get("rank_dist_norm", 0) for r in eval_records) / max(n_eval, 1)
        print(f"  [{idx:>4}]  "
              f"model={model_top1:<22} agent={agent_top1:<22}  "
              f"dist={rank_dist:6.2f} norm={rank_dist_norm:.3f}  "
              f"[cum avg_norm={avg_norm:.3f}]")

        if args.verbose:
            print(f"         gt={gt_bases}  model_top5={model_ranked_list[:5]}  "
                  f"agent_top5={agent_ranked_list[:5]}")

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
