"""
eval_crfd_forward_simulation.py — CRFD Forward-Simulation Interpretability Evaluation
Doshi-Velez & Kim (2017) §3.2 Human-grounded Evaluation: Forward Prediction

An LLM agent receives a teaching prompt explaining CRFD's GNN counterfactual scoring
formula (CF + direct + propagation), then predicts the model's top-1 service and score
distribution WITHOUT writing or running any code.

Metric: KL divergence KL(P_model ∥ P_agent) and top-1/top-3 match rate.

Usage
-----
    cd cllm
    python eval_crfd_forward_simulation.py --dataset tt
    python eval_crfd_forward_simulation.py --dataset ds25 --verbose
    python eval_crfd_forward_simulation.py --dataset tt --reset
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataset_config
from evaluate import normalize_gt, hit_at_k
from crfd_engine import CRFDEngine
from eval_utils import (
    EvalLLMClient,
    EvalLLMAPIError,
    set_eval_log_file,
    log_failure,
    classify_unknown_reason,
    SYSTEM_FORWARD_SIM_CRFD,
    build_crfd_forward_prompt,
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


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "CRFD Forward Simulation Evaluation\n"
            "Doshi-Velez & Kim 2017 §3.2: agent simulates CRFD output without code"
        )
    )
    p.add_argument("--dataset", default="ds25", choices=["tt", "ds25"],
                   help="Dataset: tt or ds25")
    p.add_argument("--reset", action="store_true",
                   help="Clear progress and start from scratch")
    p.add_argument("--verbose", action="store_true")
    return p


# ── main run logic ─────────────────────────────────────────────────────────────

def run(args):
    cfg = get_dataset_config(args.dataset)
    out_dir = os.path.join("outputs", f"eval_crfd_forward_sim_{args.dataset}")
    os.makedirs(out_dir, exist_ok=True)

    faults = cfg.load_faults(shuffle=True, seed=42)
    engine = CRFDEngine(services=cfg.all_services,
                        service2service=cfg.service2service)
    llm    = EvalLLMClient()

    # ── Initialise LLM call logging ───────────────────────────────────────────
    set_eval_log_file(
        log_path     = os.path.join(out_dir, "llm_calls.log"),
        failures_dir = os.path.join(out_dir, "failures"),
    )

    # ── Reset ─────────────────────────────────────────────────────────────────
    if args.reset:
        for fname in ("progress.json", "records.jsonl"):
            p = os.path.join(out_dir, fname)
            if os.path.exists(p):
                os.remove(p)
        print(f"[CRFDFwd] Progress reset for {args.dataset}.")

    last_done    = load_progress(out_dir)
    eval_records = load_records(out_dir)
    start_idx    = last_done + 1

    if start_idx > 0:
        print(f"[CRFDFwd] Resuming from case {start_idx} "
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

        # ── CRFD prediction ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        ranked, details = engine.predict(fault_data)
        model_elapsed = time.perf_counter() - t0

        model_top1   = ranked[0] if ranked else "unknown"
        model_top3   = ranked[:3]
        model_scores = details["scores"]

        # ── Build prompt ──────────────────────────────────────────────────────
        memory_ctx = build_memory_ctx(eval_records)
        prompt = build_crfd_forward_prompt(
            case=fault_data,
            services=cfg.all_services,
            topology=cfg.service2service,
            memory_ctx=memory_ctx,
        )

        # ── LLM call ──────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        raw = ""
        unknown_reason = None
        for _attempt in range(2):   # initial attempt + 1 case-level retry for API failure
            try:
                raw = llm.invoke(
                    prompt, system=SYSTEM_FORWARD_SIM_CRFD, case_idx=idx)
                break
            except EvalLLMAPIError as e:
                unknown_reason = "api_failure"
                log_failure(idx, "api_failure", prompt, SYSTEM_FORWARD_SIM_CRFD,
                            "", str(e))
                print(f"[CRFDFwd] Case {idx}: API FAILURE (attempt {_attempt+1}/2): {e}")
                if _attempt == 0:
                    continue   # retry the case
                break
            except Exception as e:
                unknown_reason = "unknown_error"
                log_failure(idx, "unknown_error", prompt, SYSTEM_FORWARD_SIM_CRFD,
                            raw, str(e))
                print(f"[CRFDFwd] Case {idx}: LLM error — {e}")
                break
        llm_elapsed = time.perf_counter() - t1

        agent_top1, agent_scores = parse_forward_response(raw, cfg.all_services)
        if agent_top1 == "unknown" and unknown_reason is None:
            unknown_reason = classify_unknown_reason(raw)
            if unknown_reason in ("api_failure", "parse_failure"):
                log_failure(idx, unknown_reason, prompt, SYSTEM_FORWARD_SIM_CRFD, raw)
                print(f"[CRFDFwd] Case {idx}: UNKNOWN reason={unknown_reason}"
                      f"  (see failures/ for details)")

        # ── KL divergence ─────────────────────────────────────────────────────
        P_model = to_distribution(model_scores, cfg.all_services)
        P_agent = to_distribution(agent_scores, cfg.all_services)
        kl_div  = kl_divergence(P_model, P_agent)

        top1_match = (agent_top1 == model_top1)
        top3_match = (agent_top1 in model_top3)

        # ── Record ────────────────────────────────────────────────────────────
        rec = {
            "index":         idx,
            "gt_bases":      gt_bases,
            "fault_type":    rc_info.get("fault_type", "unknown"),
            "model_top1":    model_top1,
            "model_top3":    model_top3,
            "agent_top1":    agent_top1,
            "top1_match":    top1_match,
            "top3_match":    top3_match,
            "kl_divergence": kl_div,
            "model_elapsed": round(model_elapsed, 4),
            "llm_elapsed":   round(llm_elapsed, 4),
            "case_id":           idx,
            "ground_truth":      gt_bases[0] if gt_bases else "unknown",
            "top1_prediction":   agent_top1,
            "unknown_reason":    unknown_reason,
        }
        append_record(out_dir, rec)
        save_progress(out_dir, idx)
        eval_records.append(rec)

        if args.verbose:
            print(f"  [{idx:>4}] model={model_top1:<22} agent={agent_top1:<22} "
                  f"match={'Y' if top1_match else 'N'}  KL={kl_div:.4f}  "
                  f"gt={gt_bases}")
        else:
            sym = "OK" if top1_match else "--"
            print(f"  [{idx:>4}] {sym}  model={model_top1:<22} agent={agent_top1:<22} "
                  f"KL={kl_div:.4f}")

        # ── Early stop ────────────────────────────────────────────────────────
        if early_stop_check(eval_records, mode="forward"):
            print(
                f"\n[CRFDFwd] EARLY STOP: first {len(eval_records)} cases all show "
                f"top1-match < 5%. The CRFD algorithm explanation may not be "
                f"interpretable enough for the agent to simulate."
            )
            break

    # ── Final summary ─────────────────────────────────────────────────────────
    write_summary(out_dir, eval_records, mode="forward", label="CRFD")

    valid     = [r for r in eval_records if not r.get("skipped")]
    n         = max(len(valid), 1)
    top1_rate = sum(1 for r in valid if r.get("top1_match")) / n
    top3_rate = sum(1 for r in valid if r.get("top3_match")) / n
    kl_vals   = [r["kl_divergence"] for r in valid if "kl_divergence" in r]
    avg_kl    = sum(kl_vals) / len(kl_vals) if kl_vals else float("nan")

    print(f"\n{'='*60}")
    print(f"  CRFD Forward Simulation — {args.dataset.upper()}  [{n} cases]")
    print(f"{'='*60}")
    print(f"  Top-1 match rate : {top1_rate:.2%}  "
          f"({sum(1 for r in valid if r.get('top1_match'))}/{n})")
    print(f"  Top-3 match rate : {top3_rate:.2%}  "
          f"({sum(1 for r in valid if r.get('top3_match'))}/{n})")
    print(f"  Avg KL divergence: {avg_kl:.4f}")
    print(f"  Results dir      : {os.path.abspath(out_dir)}/")
    print(f"{'='*60}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(_build_parser().parse_args())
