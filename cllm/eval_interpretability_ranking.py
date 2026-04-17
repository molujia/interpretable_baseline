"""
eval_interpretability_ranking.py — Blind Ternary Interpretability Ranking (Metric 1)

Based on: Doshi-Velez & Kim (2017) §3.2 Metric 1 (Forced Choice / Blind Ranking).

An LLM judge sees all three methods' algorithm descriptions + predictions for the
same case (anonymized as A/B/C), and ranks them by interpretability without knowing
the ground truth.

Usage (from cllm/ directory):
    python eval_interpretability_ranking.py --dataset tt
    python eval_interpretability_ranking.py --dataset ds25
    python eval_interpretability_ranking.py --dataset ds25 --reset

Output: outputs/eval_interpretability_ranking_<dataset>/
    records.jsonl  — append-only per-case results (crash-safe)
    progress.json  — last completed index (for resuming)
    summary.txt    — avg_rank and ranked_#1 rate per method
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataset_config
from evaluate import normalize_gt
from cfcbn.crfd_cbn_engine import CRFDCBNEngine
from crfd_engine import CRFDEngine
from rcd_engine import RCDEngine
from eval_utils import (
    EvalLLMClient,
    EvalLLMAPIError,
    set_eval_log_file,
    log_failure,
    SYSTEM_INTERPRETABILITY_RANKING,
    build_interpretability_ranking_prompt,
    parse_interpretability_ranking_response,
    load_progress,
    save_progress,
    append_record,
    load_records,
    write_interpretability_summary,
)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Blind Ternary Interpretability Ranking\n"
            "Doshi-Velez & Kim 2017 §3.2 Metric 1: LLM judge ranks three methods "
            "by interpretability without seeing ground truth."
        )
    )
    p.add_argument("--dataset", default="ds25", choices=["tt", "ds25"],
                   help="Dataset: tt (TrainTicket) or ds25 (Online Boutique+TiDB)")
    p.add_argument("--reset", action="store_true",
                   help="Clear progress and start from scratch")
    p.add_argument("--verbose", action="store_true",
                   help="Print extra detail per case")
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def run(args):
    cfg = get_dataset_config(args.dataset)
    out_dir = os.path.join("outputs", f"eval_interpretability_ranking_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Interpretability Ranking Evaluation  [{args.dataset.upper()}]")
    print(f"  Paper: Doshi-Velez & Kim 2017 §3.2 Metric 1")
    print(f"{'='*60}")
    print(f"  {cfg.summary()}")
    print(f"  Output → {os.path.abspath(out_dir)}/")

    faults = cfg.load_faults(shuffle=True, seed=42)

    # Initialise all three engines
    engine_cllm = CRFDCBNEngine(
        services=cfg.all_services,
        alpha_init=1.0,
        alpha_min=0.20,
        alpha_decay=40.0,
    )
    engine_rcd  = RCDEngine(services=cfg.all_services)
    engine_crfd = CRFDEngine(
        services=cfg.all_services,
        service2service=cfg.service2service,
    )

    llm = EvalLLMClient()

    set_eval_log_file(
        log_path=os.path.join(out_dir, "llm_calls.log"),
        failures_dir=os.path.join(out_dir, "failures"),
    )

    # ── Reset ─────────────────────────────────────────────────────────────────
    if args.reset:
        for fname in ("progress.json", "records.jsonl", "summary.txt"):
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        print("[InterpRank] Progress reset — starting from scratch.")

    # ── Resume ────────────────────────────────────────────────────────────────
    last_done    = load_progress(out_dir)
    start_idx    = last_done + 1
    eval_records = load_records(out_dir)

    # Replay prior cases to restore CLLM CBN state (no LLM calls)
    if start_idx > 0:
        print(f"[InterpRank] Resuming from case {start_idx}. "
              f"Restoring CBN state for {start_idx} prior cases...")
        for i, fault in enumerate(faults[:start_idx]):
            rc = fault.get("root_cause", {})
            if rc.get("type", "service") == "node":
                continue
            gt_bases = normalize_gt(rc)
            try:
                engine_cllm.predict(fault)
                engine_cllm.accumulate(fault, gt_bases)
            except Exception:
                pass
        print(f"[InterpRank] CBN restored: {engine_cllm.n_accumulated} cases.")

    print(f"[InterpRank] Evaluating cases {start_idx}–{len(faults)-1}  "
          f"(already done: {len(eval_records)})\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    for idx, fault_data in enumerate(faults):
        if idx < start_idx:
            continue

        rc_info  = fault_data.get("root_cause", {})
        gt_bases = normalize_gt(rc_info)

        if rc_info.get("type", "service") == "node":
            rec = {"index": idx, "skipped": True, "reason": "node-type fault"}
            append_record(out_dir, rec)
            save_progress(out_dir, idx)
            eval_records.append(rec)
            if args.verbose:
                print(f"  [{idx:>4}] SKIP (node)")
            continue

        # ── Get predictions from all three engines ────────────────────────────
        try:
            cllm_ranked, cllm_det = engine_cllm.predict(fault_data)
            rcd_ranked,  rcd_det  = engine_rcd.predict(fault_data)
            crfd_ranked, crfd_det = engine_crfd.predict(fault_data)
        except Exception as e:
            print(f"  [{idx:>4}] ENGINE ERROR: {e}")
            engine_cllm.accumulate(fault_data, gt_bases)
            continue

        predictions = {
            "CLLM": (cllm_ranked, cllm_det["fused_scores"]),
            "RCD":  (rcd_ranked,  rcd_det["scores"]),
            "CRFD": (crfd_ranked, crfd_det["scores"]),
        }

        # ── Randomize label assignment (reproducible per case) ────────────────
        rng = random.Random(idx)
        methods = ["CLLM", "RCD", "CRFD"]
        rng.shuffle(methods)
        label_to_method = {chr(65 + i): methods[i] for i in range(3)}  # A/B/C

        # ── Build prompt and call LLM ─────────────────────────────────────────
        prompt = build_interpretability_ranking_prompt(
            case=fault_data,
            label_to_method=label_to_method,
            predictions=predictions,
            services=cfg.all_services,
        )

        raw = ""
        ranking, reasoning = [], ""
        unknown_reason = None

        for _attempt in range(2):
            try:
                raw = llm.invoke(prompt, system=SYSTEM_INTERPRETABILITY_RANKING,
                                 case_idx=idx)
            except EvalLLMAPIError as e:
                unknown_reason = "api_failure"
                log_failure(idx, "api_failure", prompt,
                            SYSTEM_INTERPRETABILITY_RANKING, "", str(e))
                print(f"  [{idx:>4}] API FAILURE (attempt {_attempt+1}/2): {e}")
                if _attempt == 0:
                    continue
                break
            except Exception as e:
                unknown_reason = "unknown_error"
                log_failure(idx, "unknown_error", prompt,
                            SYSTEM_INTERPRETABILITY_RANKING, raw, str(e))
                print(f"  [{idx:>4}] LLM ERROR: {e}")
                break

            ranking, reasoning = parse_interpretability_ranking_response(raw)
            if not ranking:
                unknown_reason = "parse_failure"
                log_failure(idx, "parse_failure", prompt,
                            SYSTEM_INTERPRETABILITY_RANKING, raw)
                print(f"  [{idx:>4}] PARSE FAILURE (see failures/)")
            break

        # ── Decode label ranking → method ranking ─────────────────────────────
        if ranking:
            method_ranking = [label_to_method[lbl] for lbl in ranking]
            cllm_rank = method_ranking.index("CLLM") + 1
            rcd_rank  = method_ranking.index("RCD")  + 1
            crfd_rank = method_ranking.index("CRFD") + 1
        else:
            method_ranking = []
            cllm_rank = rcd_rank = crfd_rank = None

        rec = {
            "index":           idx,
            "case_id":         idx,
            "gt_bases":        gt_bases,
            "fault_type":      rc_info.get("fault_type", "unknown"),
            "label_to_method": label_to_method,
            "label_ranking":   ranking,
            "method_ranking":  method_ranking,
            "cllm_rank":       cllm_rank,
            "rcd_rank":        rcd_rank,
            "crfd_rank":       crfd_rank,
            "reasoning":       reasoning[:300] if reasoning else "",
            "unknown_reason":  unknown_reason,
        }
        append_record(out_dir, rec)
        engine_cllm.accumulate(fault_data, gt_bases)
        save_progress(out_dir, idx)
        eval_records.append(rec)

        sym = "OK" if ranking else "??"
        print(f"  [{idx:>4}] {sym}  "
              f"CLLM=#{cllm_rank}  RCD=#{rcd_rank}  CRFD=#{crfd_rank}  "
              f"order={method_ranking}")
        if args.verbose and reasoning:
            print(f"         {reasoning[:120]}")

    # ── Final summary ─────────────────────────────────────────────────────────
    write_interpretability_summary(out_dir, eval_records)

    valid = [r for r in eval_records if not r.get("skipped") and r.get("cllm_rank")]
    n = max(len(valid), 1)
    print(f"\n{'='*60}")
    print(f"  Interpretability Ranking — {args.dataset.upper()}  [{n} valid cases]")
    print(f"{'='*60}")
    for method, key in [("CLLM", "cllm_rank"), ("RCD", "rcd_rank"), ("CRFD", "crfd_rank")]:
        avg_rank = sum(r[key] for r in valid) / n
        rank1    = sum(1 for r in valid if r[key] == 1) / n
        print(f"  {method:<6}  avg_rank={avg_rank:.2f}  ranked_#1={rank1:.1%}")
    print(f"  Results dir : {os.path.abspath(out_dir)}/")
    print(f"{'='*60}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(_build_parser().parse_args())
