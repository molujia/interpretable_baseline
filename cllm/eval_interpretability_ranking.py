"""
eval_interpretability_ranking.py — Blind Ternary Interpretability Ranking (Metric 1)

Based on: Doshi-Velez & Kim (2017) §3.2 Metric 1 (Forced Choice / Blind Ranking).

An LLM judge (or human evaluator) sees all three methods' algorithm descriptions +
predictions for the same case, anonymized as A/B/C, and ranks them by
interpretability without knowing the ground truth.

Modes
-----
Normal (LLM judge):
    python eval_interpretability_ranking.py --dataset tt
    python eval_interpretability_ranking.py --dataset ds25 --reset

Export-only (human judge — no LLM calls):
    python eval_interpretability_ranking.py --dataset tt --export-only
    python eval_interpretability_ranking.py --dataset ds25 --export-only

    Generates one folder per case under cases/ with:
        input.txt   — the anonymized A/B/C content shown to the evaluator
        meta.json   — label→method mapping (for decoding rankings later)
    No LLM is called; a human can read input.txt and record their ranking.

Output root: outputs/eval_interpretability_ranking_<dataset>/
    cases/
        case_0000/
            input.txt   — what the evaluator (LLM or human) sees
            meta.json   — hidden mapping for decoding
    records.jsonl   — per-case LLM results (normal mode only)
    progress.json   — last completed index (normal mode only)
    summary.txt     — avg_rank and ranked_#1 rate per method
"""

import argparse
import json
import os
import random
import sys

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
            "Doshi-Velez & Kim 2017 §3.2 Metric 1: evaluator ranks three methods\n"
            "by interpretability without seeing ground truth.\n\n"
            "Two modes:\n"
            "  (default)      LLM is called to rank each case automatically.\n"
            "  --export-only  Only generate case files for human evaluation;\n"
            "                 no LLM calls are made."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", default="ds25", choices=["tt", "ds25"],
                   help="Dataset: tt (TrainTicket) or ds25 (Online Boutique+TiDB)")
    p.add_argument("--reset", action="store_true",
                   help="Clear LLM progress and start from scratch (normal mode only)")
    p.add_argument("--verbose", action="store_true",
                   help="Print extra detail per case")
    p.add_argument("--export-only", action="store_true",
                   help=(
                       "Export case input files only — no LLM calls. "
                       "Generates cases/case_NNNN/input.txt for human evaluation."
                   ))
    return p


# ── Case file helpers ─────────────────────────────────────────────────────────

def _case_dir(out_dir: str, idx: int) -> str:
    return os.path.join(out_dir, "cases", f"case_{idx:04d}")


def _save_case_files(out_dir: str, idx: int, prompt_text: str,
                     label_to_method: dict, gt_bases: list,
                     fault_type: str) -> None:
    """Write input.txt and meta.json for one case.

    input.txt  — exactly what the evaluator (LLM or human) reads; contains
                 the anonymized A/B/C content without any system-level prompt.
    meta.json  — stores the label→method mapping so that a human ranking can
                 be decoded into per-method ranks after the fact.
    """
    cdir = _case_dir(out_dir, idx)
    os.makedirs(cdir, exist_ok=True)

    # ── input.txt ─────────────────────────────────────────────────────────────
    with open(os.path.join(cdir, "input.txt"), "w", encoding="utf-8") as f:
        f.write(prompt_text)

    # ── meta.json ─────────────────────────────────────────────────────────────
    meta = {
        "case_index":      idx,
        "fault_type":      fault_type,
        "gt_bases":        gt_bases,
        "label_to_method": label_to_method,
        "method_to_label": {v: k for k, v in label_to_method.items()},
        "note": (
            "label_to_method maps A/B/C (as shown in input.txt) to the actual "
            "algorithm name. Use this to decode a ranking like ['B','A','C'] "
            "into per-method positions."
        ),
    }
    with open(os.path.join(cdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


# ── Engine + CBN initialisation shared by both modes ─────────────────────────

def _init_engines(cfg):
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
    return engine_cllm, engine_rcd, engine_crfd


def _restore_cbn_state(engine_cllm, faults, start_idx):
    """Replay prior cases to rebuild CBN history without LLM calls."""
    for fault in faults[:start_idx]:
        rc = fault.get("root_cause", {})
        if rc.get("type", "service") == "node":
            continue
        gt_bases = normalize_gt(rc)
        try:
            engine_cllm.predict(fault)
            engine_cllm.accumulate(fault, gt_bases)
        except Exception:
            pass


# ── Export-only mode ──────────────────────────────────────────────────────────

def export_cases(args):
    """Generate per-case input files for human evaluation. No LLM is called."""
    cfg = get_dataset_config(args.dataset)
    out_dir = os.path.join("outputs", f"eval_interpretability_ranking_{args.dataset}")
    cases_dir = os.path.join(out_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Export-Only Mode — Interpretability Ranking  [{args.dataset.upper()}]")
    print(f"  Generating human-readable case files (no LLM calls)")
    print(f"{'='*60}")
    print(f"  {cfg.summary()}")
    print(f"  Output → {os.path.abspath(cases_dir)}/")
    print()

    faults = cfg.load_faults(shuffle=True, seed=42)
    engine_cllm, engine_rcd, engine_crfd = _init_engines(cfg)

    exported = 0
    skipped  = 0

    for idx, fault_data in enumerate(faults):
        rc_info  = fault_data.get("root_cause", {})
        gt_bases = normalize_gt(rc_info)

        if rc_info.get("type", "service") == "node":
            skipped += 1
            engine_cllm.accumulate(fault_data, gt_bases)
            if args.verbose:
                print(f"  [{idx:>4}] SKIP (node-type fault)")
            continue

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

        rng = random.Random(idx)
        methods = ["CLLM", "RCD", "CRFD"]
        rng.shuffle(methods)
        label_to_method = {chr(65 + i): methods[i] for i in range(3)}

        prompt = build_interpretability_ranking_prompt(
            case=fault_data,
            label_to_method=label_to_method,
            predictions=predictions,
            services=cfg.all_services,
        )

        _save_case_files(
            out_dir=out_dir,
            idx=idx,
            prompt_text=prompt,
            label_to_method=label_to_method,
            gt_bases=gt_bases,
            fault_type=rc_info.get("fault_type", "unknown"),
        )

        engine_cllm.accumulate(fault_data, gt_bases)
        exported += 1

        if args.verbose:
            print(f"  [{idx:>4}] exported → {_case_dir(out_dir, idx)}/")
        elif exported % 10 == 0:
            print(f"  ... {exported} cases exported")

    print(f"\n{'='*60}")
    print(f"  Export complete")
    print(f"  Cases exported : {exported}")
    print(f"  Cases skipped  : {skipped}  (node-type faults)")
    print(f"  Output dir     : {os.path.abspath(cases_dir)}/")
    print(f"{'='*60}")
    print()
    print("  Each case_NNNN/ folder contains:")
    print("    input.txt  — anonymized A/B/C content; give this to the evaluator")
    print("    meta.json  — label→method mapping (for decoding rankings later)")
    print()
    print("  Human evaluation workflow:")
    print("    1. Read input.txt for a case")
    print("    2. Rank methods A, B, C from most to least interpretable")
    print('    3. Record your decision, e.g.: {"ranking":["B","A","C"],"reasoning":"..."}')
    print("    4. Refer to meta.json to decode which algorithm each label maps to")


# ── Normal (LLM judge) mode ───────────────────────────────────────────────────

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
    engine_cllm, engine_rcd, engine_crfd = _init_engines(cfg)
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

    if start_idx > 0:
        print(f"[InterpRank] Resuming from case {start_idx}. "
              f"Restoring CBN state for {start_idx} prior cases...")
        _restore_cbn_state(engine_cllm, faults, start_idx)
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

        # ── Randomize label assignment (same seed as export-only mode) ────────
        rng = random.Random(idx)
        methods = ["CLLM", "RCD", "CRFD"]
        rng.shuffle(methods)
        label_to_method = {chr(65 + i): methods[i] for i in range(3)}

        # ── Build prompt, save case files, call LLM ───────────────────────────
        prompt = build_interpretability_ranking_prompt(
            case=fault_data,
            label_to_method=label_to_method,
            predictions=predictions,
            services=cfg.all_services,
        )

        _save_case_files(
            out_dir=out_dir,
            idx=idx,
            prompt_text=prompt,
            label_to_method=label_to_method,
            gt_bases=gt_bases,
            fault_type=rc_info.get("fault_type", "unknown"),
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
    args = _build_parser().parse_args()
    if args.export_only:
        export_cases(args)
    else:
        run(args)
