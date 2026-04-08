"""
main.py — CLLM v5 Command-line entry point V6  + 


────────
  data/          
    ds25_faults.json     ds25 
    tt_faults.json       TT 96/125 cases
    tt_topo_s2s.json     TT 

  outputs/        + 
    ds25/
      case_store/          CBN Case store
      progress/            resumable-run progress
      tickets/             
      final_answer.txt     Evaluation
      llm_calls.txt        LLM 
      single_0042_20260316_143155/   
        report.txt
        llm_calls.txt
    tt/
      ...

LLM platform configuration
──────────────────────────
   utils/llm_adapter.py  PLATFORM  API_KEY / MODEL
    PLATFORM = "volc"    # 
    PLATFORM = "apiyi"   # APIYI gpt-4.1-mini 
    PLATFORM = "openai"  # OpenAI 
    PLATFORM = "mock"    #  LLM

  Connectivity test
    python utils/llm_adapter.py

Dataset selection
─────────────────
  --dataset ds25   Online Boutique + TiDB14 services
  --dataset tt     TrainTicket48 services 96/125 cases

Run modes
─────────
  Batch evaluation (pure CF-CBN, no LLM calls)
      python main.py
      python main.py --dataset tt

  Batch evaluation (LLM Workflow A/B, conditional trigger)
      python main.py --use-llm
      python main.py --dataset tt --use-llm

  Batch evaluation (LLM always-trigger: every case produces WF-A output)
      python main.py --use-llm --llm-mode always

  Single-case detailed analysis
      python main.py --single 42
      python main.py --dataset tt --single 10 --use-llm --llm-mode always --verbose

  SRE interactive mode
      python main.py --interactive

  Reset progress (restart batch; case store is not cleared)
      python main.py --reset-progress
      python main.py --dataset tt --reset-progress

Ablation experiments
────────────────────
python ablation/ablation_run.py --dataset ds25 --group D
python ablation/ablation_run.py --dataset ds25 --group C
python ablation/ablation_run.py --dataset ds25 --group all  # A+B+C+D

python ablation/ablation_plot.py --dataset ds25 --group D
python ablation/ablation_plot.py --dataset ds25 --group all
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import get_dataset_config, AVAILABLE_DATASETS
from pipeline import CLLMv5Pipeline


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CLLM v5 — CF-CBN + LLM Root Cause Localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    grp_ds = p.add_argument_group("Dataset")
    grp_ds.add_argument(
        "--dataset", default="ds25", choices=AVAILABLE_DATASETS,
        help="Dataset selection (default: ds25): ds25 | tt",
    )
    grp_ds.add_argument("--no-shuffle", action="store_true",
                        help="Disable random shuffle (default seed=42)")
    grp_ds.add_argument("--data", default=None,
                        help="Override data file path (advanced)")
    grp_ds.add_argument("--topo", default=None,
                        help="Override topology file path (advanced)")

    grp_mode = p.add_argument_group("Run Mode")
    grp_mode.add_argument("--use-llm", action="store_true",
                          help="Enable LLM workflows (default: pure CF-CBN)")
    grp_mode.add_argument("--llm-mode", default="conditional",
                          choices=["conditional", "always"],
                          help="LLM trigger mode: conditional (default) | always")
    grp_mode.add_argument("--single", type=int, default=None,
                          help="Single-case analysis: case index (0-based)")
    grp_mode.add_argument("--interactive", action="store_true",
                          help="SRE interactive mode")
    grp_mode.add_argument("--verbose", action="store_true")
    grp_mode.add_argument("--reset-progress", action="store_true",
                          help="Clear batch progress and restart from beginning")

    grp_cf = p.add_argument_group("CF-CBN Params")
    grp_cf.add_argument("--alpha-min",  type=float, default=0.20)
    grp_cf.add_argument("--alpha-decay",type=float, default=40.0)
    grp_cf.add_argument("--conf-gap",   type=float, default=0.15,
                        help="WF-A conditional ")
    grp_cf.add_argument("--err-rate",   type=float, default=0.40,
                        help="WF-A conditional ")
    grp_cf.add_argument("--skip-types", default="node",
                        help="Comma-separated rc_types to skip (default: node)")

    # WF-B --use-llm
    grp_wb = p.add_argument_group("WF-B Extensions")
    grp_wb.add_argument("--wfb-case-review", action="store_true",
                        help=("WF-B :  Case  — "
                              "CBN "))
    grp_wb.add_argument("--wfb-propagation", action="store_true",
                        help=("WF-B :  — "
                              "BFS "))

    grp_alpha = p.add_argument_group("Alpha Strategy")
    grp_alpha.add_argument("--alpha-strategy", default="adaptive",
                          choices=["adaptive", "rag", "rag_api", "jaccard", "entropy"],
                          help=("Alpha scheduling strategy: "
                               "adaptive (bisection tuner, default), "
                               "rag (RAG local BoW, no API), "
                               "rag_api (RAG APIYI embedding, text-embedding-3-small), "
                               "jaccard (direct Jaccard similarity), "
                               "entropy (CBN posterior entropy)"))
    return p


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = build_arg_parser().parse_args()

    # Dataset configuration
    cfg = get_dataset_config(args.dataset)
    print(f"\n[Main] {cfg.summary()}")

    # Ensure output root directory exists
    out_root = os.path.join("outputs", args.dataset)
    os.makedirs(out_root, exist_ok=True)

    data_path  = args.data or cfg.data_path
    data_stem  = os.path.splitext(os.path.basename(data_path))[0]
    skip_types = [t.strip() for t in args.skip_types.split(",") if t.strip()] \
                 if args.skip_types else []

    if args.reset_progress:
        from util import ProgressTracker
        ProgressTracker(cfg.progress_dir, data_stem).reset()
        print(f"[Main] Progress reset: dataset={args.dataset}")

    shuffle = not args.no_shuffle
    faults  = cfg.load_faults(shuffle=shuffle)
    print(f"[Main] Loaded {len(faults)} faults  shuffle={shuffle}")

    pipeline = CLLMv5Pipeline(
        use_llm                  = args.use_llm,
        llm_mode                 = args.llm_mode,
        service2service          = cfg.service2service,
        store_dir                = cfg.store_dir,
        ticket_dir               = cfg.ticket_dir,
        skip_rc_types            = skip_types,
        final_answer_path        = cfg.final_answer,
        alpha_min                = args.alpha_min,
        alpha_decay              = args.alpha_decay,
        alpha_strategy           = args.alpha_strategy,
        confidence_gap_threshold = args.conf_gap,
        high_error_rate_threshold= args.err_rate,
        dataset_config           = cfg,
        wfb_case_review          = args.wfb_case_review,
        wfb_propagation          = args.wfb_propagation,
    )

    if args.interactive:
        _run_interactive(pipeline)

    elif args.single is not None:
        _run_single(pipeline, faults, args.single, args, cfg)

    else:
        _run_batch(pipeline, faults, args, data_stem, cfg)

    pipeline.print_status()


# ── batch ─────────────────────────────────────────────────────────────────────

def _run_batch(pipeline, faults, args, data_stem, cfg):
    from utils.llm_adapter import set_log_file

    log_path = os.path.join("outputs", cfg.name, "llm_calls.txt")
    set_log_file(log_path)

    abs_out = os.path.abspath(os.path.join("outputs", cfg.name))
    print(f"\n  Output dir -> {abs_out}/")
    print(f"    final_answer.txt  Evaluation")
    if args.use_llm:
        print(f"    llm_calls.txt     LLM ")

    pipeline.run_batch(
        faults,
        use_llm      = args.use_llm,
        verbose      = args.verbose,
        data_stem    = data_stem,
        progress_dir = cfg.progress_dir,
    )
    pipeline.case_store.print_summary()
    print(f"\n  Results -> {abs_out}/")


# ── single ────────────────────────────────────────────────────────────────────

def _run_single(pipeline, faults, idx, args, cfg):
    from utils.llm_adapter import set_log_file

    if idx >= len(faults):
        print(f"[Main] Index {idx} out of range (total={len(faults)})")
        return

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("outputs", cfg.name, f"single_{idx:04d}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, "llm_calls.txt")
    set_log_file(log_path)

    print(f"\n  Output dir -> {os.path.abspath(run_dir)}/")

    result = pipeline.run_single(
        faults[idx],
        use_llm    = args.use_llm,
        llm_mode   = args.llm_mode,
        accumulate = False,
        verbose    = args.verbose,
    )
    if result is None:
        print(f"[Main] Case {idx} skipped (rc_type filtered)")
        return

    # Write single-case report
    report_path = os.path.join(run_dir, f"case_{idx:04d}.txt")
    _write_single_report(result, idx, report_path, args.use_llm, args.llm_mode, pipeline)

    # Console summary
    W = 65
    gt = result.get("gt_bases", ["?"])
    print(f"\n{'═'*W}")
    print(f"  CLLM v5 — Case #{idx}  [{cfg.name}]")
    print(f"  GT: {gt}   Fault: {result.get('fault_type','?')}")
    print(f"{'─'*W}")
    print(f"  CF-CBN : rank={result['cfcbn_rank']:>2}  "
          f"hit={'✓' if result['cfcbn_hit1'] else '✗'}  "
          f"top1={result['cfcbn_top1']}")
    print(f"  WF-A   : {'triggered ✓' if result['llm_assisted'] else 'skipped'}")
    print(f"  α      : {result['cfcbn_alpha']:.4f}")
    print(f"{'═'*W}")
    print(f"\n  Report -> {os.path.abspath(report_path)}")
    if args.use_llm:
        print(f"  Log    -> {os.path.abspath(log_path)}")


def _write_single_report(result, idx, path, use_llm, llm_mode, pipeline):
    """"""
    lines = [
        "=" * 65,
        f"  CLLM v5 — Single Case Analysis",
        "=" * 65,
        f"  Case      : {idx}",
        f"  GT roots  : {result['gt_bases']}",
        f"  Pred      : {result['cfcbn_top1']}",
        f"  Rank      : {result['cfcbn_rank']}",
        f"  Hit@1     : {'✓' if result['cfcbn_hit1'] else '✗'}",
        f"  LLM mode  : {llm_mode}",
        f"  WF-A      : {'triggered ✓' if result['llm_assisted'] else 'skipped'}",
        f"  CF-CBN α  : {result['cfcbn_alpha']:.4f}",
        f"  Time      : CF-CBN {result['cfcbn_elapsed']*1000:.1f}ms  "
        f"LLM {result['llm_elapsed']:.2f}s",
    ]

    wfa = result.get("workflow_a_result")
    if wfa:
        lines += ["", "─" * 65, "  [WF-A] ", "─" * 65]
        lines.append(f"   : {wfa.get('trigger_reason', 'N/A')}")
        lines.append(f"   : {wfa.get('pattern_interpretation', '')}")
        lines.append(f"     : {wfa.get('confidence_assessment', '')}")
        lines.append(f"   : {wfa.get('verification_guide', '')}")
        lines.append(f"   : {'✓ ' if wfa.get('recommend_top1') else '✗ '}")

    lines.append("=" * 65)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ── interactive ───────────────────────────────────────────────────────────────

def _run_interactive(pipeline):
    print("\n[Interactive] Enter SRE commands (type quit to exit)")
    while True:
        try:
            cmd = input("\nEngineer> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd.lower() in ("quit", "exit", "q"):
            break
        if not cmd:
            continue
        print(pipeline.engineer(cmd))


if __name__ == "__main__":
    main()
