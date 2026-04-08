"""
pipeline.py — CLLM v5 main pipeline (core class, supporting multiple datasets)

Architecture:
  Lower layer: CF-CBN (CRFDCBNEngine) — all RCL inference; performance improves with accumulated history
  Upper layer: two LLM workflows with two trigger modes:

  ┌─────────────────────────────────────────────────────────────┐
  │  llm_mode = "conditional" (confidence-based trigger, default)              │
  │    Workflow A: triggered on low confidence / high error rate / weak signal  │
  │    Workflow B: triggered after CF-CBN miss → failure attribution + advice   │
  ├─────────────────────────────────────────────────────────────┤
  │  llm_mode = "always"Always-trigger mode                         │
  │    Workflow A: always triggered regardless of CF-CBN confidence            │
  │    Workflow B: triggered after miss (same as conditional)                  │
  │             + automatically injects WF-A summary into WF-B prompt         │
  │             + token-efficient (summary ~200 chars vs full JSON ~1000)     │
  └─────────────────────────────────────────────────────────────┘

  Typical use cases for "always" mode:
    - Single-case debug (--single N --llm-mode always): ensures both A and B produce output
    - Full-pipeline analysis of a specific case (research / paper reproduction)

LLM safety (anonymisation):
  All content sent to the LLM is anonymised via utils.anonymizer.Anonymizer
  (service names → SVC-XXX). Responses are de-anonymised before writing to tickets.
  Anonymisation is handled transparently inside agents/llm_workflow.py.

: 
  util.py      —  /  / ProgressTrackerresumable-run progress
  evaluate.py  — Multi-root-cause hit rules / 
"""

import os
import time
from typing import Dict, List, Optional

from utils.llm_adapter import LLMQuotaExhaustedError
from utils.anonymizer import get_anonymizer
from cfcbn.crfd_cbn_engine import CRFDCBNEngine
from cfcbn.cbn_accumulator import ALL_SERVICES, ALL_METRICS
from cfcbn.rag_case_store import RAGCaseStore
from cfcbn.alpha_strategies import get_strategy, ALPHA_MIN, ALPHA_MAX
from agents.case_store import CaseReviewer as CaseStore
from agents.llm_workflow import (
    should_trigger_workflow_a, run_workflow_a, summarize_wfa_for_b,
    run_workflow_b, generate_fault_ticket,
)
from agents.collider_topology import TopologyModule
from agents.manager import Manager
from evaluate import normalize_gt, best_rank, hit_at_k, build_final_answer
from util import ProgressTracker, DEFAULT_SERVICE2SERVICE
from datasets import DatasetConfig, get_dataset_config


class CLLMv5Pipeline:
    """
    CLLM v5 main pipeline.

    Public methods:
      run_single(fault_data, ...)  → dict | None   single-case inference
      run_batch(fault_list, ...)   → dict           batch inference + evaluation + report writing
      engineer(command)            → str            SRE natural-language interaction
      print_status()                                print current pipeline status

    llm_mode : 
      "conditional"  Confidence-based trigger mode (default)WF-A 
      "always"       Always-trigger modeWF-A WF-B  WF-A 
    """

    LLM_MODES = ("conditional", "always")

    def __init__(
        self,
        use_llm:                    bool  = True,
        llm_mode:                   str   = "conditional",
        service2service:            Optional[Dict]  = None,
        deployment_info:            Optional[dict]  = None,
        store_dir:                  str   = "case_store",
        ticket_dir:                 str   = "tickets",
        skip_rc_types:              Optional[List[str]] = None,
        final_answer_path:          str   = "final_answer.txt",
        alpha_init:                 float = 1.0,
        alpha_min:                  float = 0.20,
        alpha_decay:                float = 40.0,
        confidence_gap_threshold:   float = 0.15,
        high_error_rate_threshold:  float = 0.40,
        weak_signal_threshold:      int   = 2,
        dataset_config:             "DatasetConfig" = None,
        # WF-B optional extensions (disabled by default)
        wfb_case_review:            bool  = False,  #  case 
        wfb_propagation:            bool  = False,  # 
        # Alpha strategy: 'adaptive' (bisection tuner, default) | 'rag' | 'rag_api' | 'jaccard' | 'entropy'
        alpha_strategy:             str   = 'adaptive',
    ):
        if llm_mode not in self.LLM_MODES:
            raise ValueError(f"llm_mode must be one of {self.LLM_MODES}, got '{llm_mode}'")

        self.use_llm           = use_llm
        self.llm_mode          = llm_mode          # "conditional" | "always"
        self.skip_rc_types     = skip_rc_types if skip_rc_types is not None else ["node"]
        self.final_answer_path = final_answer_path
        self.ticket_dir        = ticket_dir
        self.conf_gap          = confidence_gap_threshold
        self.err_rate_thr      = high_error_rate_threshold
        self.weak_sig_thr      = weak_signal_threshold
        self.wfb_case_review   = wfb_case_review
        self.wfb_propagation   = wfb_propagation

        os.makedirs(ticket_dir, exist_ok=True)
        os.makedirs(store_dir,  exist_ok=True)

        # Resolve dataset config (None → ds25 for backward compatibility)
        self.dataset_cfg = dataset_config if dataset_config is not None \
                           else get_dataset_config("ds25")

        print(f"[Pipeline] CLLM v5 initializing  use_llm={use_llm}  llm_mode={llm_mode}  "
              f"dataset={self.dataset_cfg.name}  store={store_dir}")

        # ── Lower layer: CF-CBN engine ────────────────────────────────────────────
        self.engine = CRFDCBNEngine(
            services=self.dataset_cfg.all_services,
            alpha_init=alpha_init,
            alpha_min=alpha_min,
            alpha_decay=alpha_decay,
        )

        # ── Persistent case store ──────────────────────────────────────────────────
        self.case_store = CaseStore(store_dir)

        # ── Alpha strategy ────────────────────────────────────────────────────────
        self.alpha_strategy_name = alpha_strategy
        self._rag_store = None
        if alpha_strategy == 'adaptive':
            print('[Pipeline] Alpha strategy: adaptive (bisection tuner)')
        elif alpha_strategy == 'rag':
            # Local BoW store persisted in store_dir
            self._rag_store = RAGCaseStore(store_dir)
            self._alpha_strat = get_strategy('rag')
            print(f'[Pipeline] Alpha strategy: RAG(bow)  '
                  f'store={store_dir}  vectors={len(self._rag_store)}')
        elif alpha_strategy == 'rag_api':
            # BoW store as anchor (base_dir); API store lives in store_dir_api/
            self._rag_store = RAGCaseStore(store_dir)
            self._alpha_strat = get_strategy('rag_api')
            # Pre-warm the API store so it uses the persistent dir from the start
            self._alpha_strat._get_store(store_dir)
            print(f'[Pipeline] Alpha strategy: RAG(api)  '
                  f'bow_store={store_dir}  api_store={store_dir}_api')
        else:
            self._alpha_strat = get_strategy(alpha_strategy)
            print(f'[Pipeline] Alpha strategy: {alpha_strategy}')

        # ── Topology module (used by Manager)────────────────────────────────────────
        s2s = (service2service
               if service2service is not None
               else self.dataset_cfg.service2service)
        _topo = os.path.join(store_dir, "topology.json")
        self.topo = TopologyModule(s2s, deployment_info, store_path=_topo)

        # ── SRE interaction manager ────────────────────────────────────────────────
        self.manager = Manager(metric_classifier=None, topology_module=self.topo)

        # ── Anonymiser: pre-register known entities; register_from_case() adds more dynamically ──
        anon = get_anonymizer(enabled=use_llm)
        anon.register_services(self.dataset_cfg.all_services)
        anon.register_metrics(ALL_METRICS)

        print(f"[Pipeline] Ready. services={len(self.engine.services)}  "
              f"history={self.engine.n_accumulated}  cases={len(self.case_store)}")

    # ─────────────────────────────────────────────────────────────────────────
    # Preprocessing: parse fault_data and extract multi-root-cause GT information
    # ─────────────────────────────────────────────────────────────────────────

    def _preprocess(self, fault_data: dict):
        """
         fault_data: 
          (gt_bases, rc_type, anomaly_services, anomaly_pods, fault_cat, fault_type)
        gt_bases = Normalised list of root-cause service base names
         rc_type  skip_rc_types  None
        """
        rc_info    = fault_data.get("root_cause", {})
        rc_type    = rc_info.get("type", "service")
        fault_cat  = rc_info.get("fault_category", "unknown")
        fault_type = rc_info.get("fault_type", "unknown")

        if rc_type in self.skip_rc_types:
            return None

        gt_bases = normalize_gt(rc_info)                    # 
        get_anonymizer().register_from_case(fault_data)     # Dynamically register entities from this case

        return (gt_bases, rc_type,
                fault_data.get("anomaly_services", {}),
                fault_data.get("anomaly_pods", {}),
                fault_cat, fault_type)

    # ─────────────────────────────────────────────────────────────────────────
    # 
    # ─────────────────────────────────────────────────────────────────────────

    def run_single(
        self,
        fault_data: dict,
        use_llm:    Optional[bool] = None,
        llm_mode:   Optional[str]  = None,   # None →  self.llm_mode
        accumulate: bool = True,
        verbose:    bool = False,
    ) -> Optional[dict]:
        """
         fault_data 

        llm_mode : 
          None          →  self.llm_mode
          "conditional" → 
          "always"      → Always-trigger modeWF-A 

        Multi-root-cause hit rules evaluate.py
          - service+service Hitrank 
          - pod+service
         None  case  skip_rc_types 
        """
        _use_llm = use_llm if use_llm is not None else self.use_llm
        prep = self._preprocess(fault_data)
        if prep is None:
            return None

        gt_bases, rc_type, anomaly_services, anomaly_pods, fault_cat, fault_type = prep

        # ── CF-CBN inference ───────────────────────────────────────────────────
        t0 = time.perf_counter()
        cfcbn_ranked, details = self.engine.predict(fault_data)
        cfcbn_scores  = details["fused_scores"]
        cfcbn_elapsed = time.perf_counter() - t0

        cfcbn_top1 = cfcbn_ranked[0] if cfcbn_ranked else "unknown"
        cfcbn_top3 = cfcbn_ranked[:3]
        cfcbn_rank = best_rank(cfcbn_ranked, gt_bases)      # Best rank across multiple root causes
        cfcbn_hit1 = hit_at_k(cfcbn_ranked, gt_bases, 1)

        # ── Alpha strategy override ────────────────────────────────────────────
        strat_alpha = None
        strat_reason = ''
        if self.alpha_strategy_name != 'adaptive':
            try:
                if self.alpha_strategy_name == 'rag':
                    from cfcbn.alpha_strategies import RAGAlphaStrategy
                    _s = RAGAlphaStrategy()
                    strat_alpha, strat_reason = _s.suggest(
                        anomaly_services, anomaly_pods, fault_type,
                        self.engine.accumulator, self._rag_store)
                else:
                    strat_alpha, strat_reason = self._alpha_strat.suggest(
                        anomaly_services, anomaly_pods, fault_type,
                        self.engine.accumulator, self._rag_store)
                if verbose:
                    print(f'[Alpha] {strat_reason}')
                # Re-fuse with strategy alpha
                us_raw = details.get('unsupervised_scores', {})
                if us_raw:
                    cfcbn_scores = self.engine.accumulator.fuse_scores(
                        us_raw, fault_data, alpha_override=strat_alpha)
                    cfcbn_ranked = self.engine.accumulator.rank(cfcbn_scores)
                    cfcbn_top1 = cfcbn_ranked[0] if cfcbn_ranked else 'unknown'
                    cfcbn_rank = best_rank(cfcbn_ranked, gt_bases)
                    cfcbn_hit1 = hit_at_k(cfcbn_ranked, gt_bases, 1)
            except Exception as _e:
                if verbose:
                    print(f'[Alpha] strategy error: {_e}; using adaptive alpha')

        if verbose:
            print(f"[CF-CBN] top1={cfcbn_top1}  gt={gt_bases}  rank={cfcbn_rank}  "
                  f"alpha={details['alpha']:.3f}  n={details['n_accumulated']}  "
                  f"t={cfcbn_elapsed*1000:.1f}ms")

        # ── ADecide whether to trigger based on mode ────────────────────────────────
        llm_elapsed, llm_assisted, workflow_a_res, similar_cases = 0.0, False, None, []

        if _use_llm:
            similar_cases = self.case_store.get_similar_cases(
                anomaly_services, anomaly_pods, fault_type, top_k=5
            )

            _mode = llm_mode if llm_mode is not None else self.llm_mode

            if _mode == "always":
                # Always-trigger modeWF-A trigger_reason
                trigger = True
                reason  = "always mode: forced trigger — ensures every case has full LLM interpretation"
            else:
                # Confidence-based trigger mode (conditional, default)
                trigger, reason = should_trigger_workflow_a(
                    cfcbn_scores=cfcbn_scores, cfcbn_ranked=cfcbn_ranked,
                    fault_type=fault_type, anomaly_services=anomaly_services,
                    anomaly_pods=anomaly_pods, case_store=self.case_store,
                    confidence_gap_threshold=self.conf_gap,
                    high_error_rate_threshold=self.err_rate_thr,
                    weak_signal_threshold=self.weak_sig_thr,
                )

            if trigger:
                llm_assisted = True
                t1 = time.perf_counter()
                try:
                    workflow_a_res = run_workflow_a(
                        cfcbn_ranked=cfcbn_ranked, cfcbn_scores=cfcbn_scores,
                        anomaly_services=anomaly_services, anomaly_pods=anomaly_pods,
                        fault_type=fault_type, fault_category=fault_cat,
                        similar_cases=similar_cases, trigger_reason=reason,
                    )
                    if verbose:
                        print(f"[WF-A] triggered (mode={_mode}) → recommend_top1="
                              f"{workflow_a_res.get('recommend_top1')}")
                except LLMQuotaExhaustedError:
                    raise
                except Exception as e:
                    print(f"[WF-A] ERROR: {e}")
                finally:
                    llm_elapsed = time.perf_counter() - t1
            elif verbose:
                print(f"[WF-A] skipped (mode={_mode}): {reason}")

        # ── Generate ticket ──────────────────────────────────────────────────────
        _alpha_display = (f'{strat_alpha:.3f} [{self.alpha_strategy_name}]'
                          if strat_alpha is not None
                          else f'{details["alpha"]:.3f} [adaptive]')
        ticket_text = generate_fault_ticket(
            cfcbn_ranked=cfcbn_ranked, cfcbn_scores=cfcbn_scores,
            anomaly_services=anomaly_services, anomaly_pods=anomaly_pods,
            fault_type=fault_type, fault_category=fault_cat,
            workflow_a_result=workflow_a_res, similar_cases=similar_cases,
            cf_raw_scores=details.get('unsupervised_scores'),
            alpha_info=_alpha_display,
        )
        if verbose:
            print(ticket_text)

        # ── CF-CBN online accumulation + RAG store update ──────────────────────────
        if accumulate and gt_bases:
            self.engine.accumulate(fault_data, gt_bases)
            if self._rag_store is not None:
                # Always update the BoW store (used by 'rag' and as anchor for 'rag_api')
                self._rag_store.add(
                    anomaly_services, anomaly_pods, fault_type,
                    root_cause=cfcbn_top1
                )
            if (self.alpha_strategy_name == 'rag_api'
                    and hasattr(self, '_alpha_strat')
                    and hasattr(self._alpha_strat, 'add_confirmed')
                    and self._rag_store is not None):
                # Also update the API-embedding store
                self._alpha_strat.add_confirmed(
                    anomaly_services, anomaly_pods, fault_type,
                    cfcbn_top1, self._rag_store.store_dir
                )

        return {
            "gt_bases":            gt_bases,
            "rc_type":             rc_type,
            "fault_type":          fault_type,
            "fault_category":      fault_cat,
            "cfcbn_top1":          cfcbn_top1,
            "cfcbn_top3":          cfcbn_top3,
            "cfcbn_scores":        cfcbn_scores,
            "cfcbn_ranked":        cfcbn_ranked[:5],
            "cfcbn_rank":          cfcbn_rank,
            "cfcbn_hit1":          cfcbn_hit1,
            "cfcbn_alpha":         details["alpha"],
            "cfcbn_n_accumulated": details["n_accumulated"],
            "llm_assisted":        llm_assisted,
            "workflow_a_result":   workflow_a_res,
            "ticket_text":         ticket_text,
            "similar_cases_count": len(similar_cases),
            "cfcbn_elapsed":       round(cfcbn_elapsed, 4),
            "llm_elapsed":         round(llm_elapsed, 4),
            "anomaly_services":    anomaly_services,
            "anomaly_pods":        anomaly_pods,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # 
    # ─────────────────────────────────────────────────────────────────────────

    def run_batch(
        self,
        fault_list:   List[dict],
        use_llm:      Optional[bool] = None,
        verbose:      bool  = False,
        data_stem:    str   = "default",
        progress_dir: str   = "progress",
    ) -> dict:
        """
        resumable-run progress
         summary dicttop1_rate / top3_rate / top5_rate …
        """
        _use_llm  = use_llm if use_llm is not None else self.use_llm
        tracker   = ProgressTracker(progress_dir, data_stem)
        last_done = tracker.load_last_done()
        start_idx = last_done + 1

        if start_idx > 0:
            print(f"[Pipeline] Resuming from case {start_idx}")
        print(f"[Pipeline] Batch: {len(fault_list)} cases  use_llm={_use_llm}")

        # Restore existing statistics
        all_records = tracker.load_records()
        ev0 = [r for r in all_records if not r.get("skipped")]
        top1 = sum(1 for r in ev0 if 0 < r.get("cfcbn_rank", -1) <= 1)
        top3 = sum(1 for r in ev0 if 0 < r.get("cfcbn_rank", -1) <= 3)
        top5 = sum(1 for r in ev0 if 0 < r.get("cfcbn_rank", -1) <= 5)
        total   = len(ev0)
        skipped = sum(1 for r in all_records if r.get("skipped"))
        node_buf, node_start = [], start_idx

        for idx, fault_data in enumerate(fault_list):
            if idx < start_idx:
                continue

            # ── Skip handling ──────────────────────────────────────────────
            prep = self._preprocess(fault_data)
            if prep is None:
                rc  = fault_data.get("root_cause", {})
                rec = {"index": idx, "skipped": True,
                       "gt_bases": normalize_gt(rc), "rc_type": rc.get("type", "?")}
                skipped += 1
                all_records.append(rec); node_buf.append(rec)
                tracker.append_record(rec); tracker.save_progress(idx)
                continue

            gt_bases, rc_type, anomaly_services, anomaly_pods, fault_cat, fault_type = prep
            t_total = time.perf_counter()

            try:
                result = self.run_single(fault_data=fault_data, use_llm=_use_llm,
                                         accumulate=True, verbose=verbose)
            except LLMQuotaExhaustedError:
                print(f"\n[Pipeline] Quota exhausted at case {idx}. STOPPING.")
                break
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"[Pipeline] Case {idx} ERROR: {e}")
                continue
            if result is None:
                continue

            total_elapsed = time.perf_counter() - t_total
            total += 1
            cr = result["cfcbn_rank"]
            if cr > 0:
                if cr <= 1: top1 += 1
                if cr <= 3: top3 += 1
                if cr <= 5: top5 += 1

            # ── Workflow B: triggered when Top-1 misses ──────────────────────────
            wb_result, wb_triggered, llm_b_t = None, False, 0.0
            if _use_llm and not result["cfcbn_hit1"]:
                wb_triggered = True
                t_b = time.perf_counter()
                try:
                    similar_mm = self.case_store.get_mismatch_cases(fault_type, top_k=5)
                    similar_all = (
                        self.case_store.get_similar_cases(
                            anomaly_services, anomaly_pods, fault_type, top_k=5
                        ) if self.wfb_case_review else None
                    )
                    prop_path = None
                    if self.wfb_propagation and gt_bases:
                        from agents.explainer_v4 import ExplainerV4
                        exp = ExplainerV4(self.case_store)
                        all_anomaly = {**anomaly_services, **anomaly_pods}
                        prop_path = exp.infer_propagation(
                            gt_bases[0], all_anomaly, self.topo.get_topology_dict()
                        )
                    wb_result = run_workflow_b(
                        cfcbn_top1=result["cfcbn_top1"],
                        cfcbn_scores=result["cfcbn_scores"],
                        cfcbn_ranked=result["cfcbn_ranked"],
                        true_root_cause=gt_bases[0],
                        anomaly_services=anomaly_services,
                        anomaly_pods=anomaly_pods,
                        fault_type=fault_type,
                        fault_category=fault_cat,
                        similar_mismatch_cases=similar_mm,
                        wfa_result=result.get("workflow_a_result"),  # populated in always mode
                        enable_case_review=self.wfb_case_review,
                        similar_all_cases=similar_all,
                        enable_propagation=self.wfb_propagation,
                        propagation_path=prop_path,
                        topology=self.topo.get_topology_dict() if self.wfb_propagation else None,
                    )
                    if verbose:
                        d = wb_result.get("diagnosis", {})
                        print(f"[WF-B] attribution={d.get('failure_attribution','?')}  "
                              f"{d.get('failure_summary','')}")
                except LLMQuotaExhaustedError:
                    raise
                except Exception as e:
                    print(f"[WF-B] ERROR: {e}")
                finally:
                    llm_b_t = time.perf_counter() - t_b
                result["llm_elapsed"] = round(result.get("llm_elapsed", 0) + llm_b_t, 4)

            # ── Write to case store ──────────────────────────────────────────────
            self.case_store.add_v4_case(
                case_id=idx,
                anomaly_services=anomaly_services, anomaly_pods=anomaly_pods,
                fault_type=fault_type, fault_category=fault_cat,
                root_cause=gt_bases[0],
                cfcbn_top1=result["cfcbn_top1"], cfcbn_top3=result["cfcbn_top3"],
                cfcbn_scores={k: round(v, 5) for k, v in result["cfcbn_scores"].items()},
                cfcbn_correct=result["cfcbn_hit1"], cfcbn_rank=cr,
                llm_assisted=result["llm_assisted"],
                llm_workflow_a=result.get("workflow_a_result"),
                llm_diagnosis=wb_result,
            )
            self._save_ticket(idx, result, wb_result)

            # ── Progress record ────────────────────────────────────────────────
            rec = {
                "index":                idx,
                "gt_bases":             gt_bases,
                "rc_type":              rc_type,
                "fault_type":           fault_type,
                "cfcbn_top1":           result["cfcbn_top1"],
                "cfcbn_rank":           cr,
                "cfcbn_hit1":           result["cfcbn_hit1"],
                "llm_assisted":         result["llm_assisted"],
                "workflow_b_triggered": wb_triggered,
                "cfcbn_elapsed":        result["cfcbn_elapsed"],
                "llm_elapsed":          result["llm_elapsed"],
                "total_elapsed":        round(total_elapsed, 4),
            }
            all_records.append(rec); node_buf.append(rec)
            tracker.append_record(rec); tracker.save_progress(idx)

            n   = max(total, 1)
            sym = "OK" if result["cfcbn_hit1"] else "--"
            wa  = "A" if result["llm_assisted"] else " "
            wb  = "B" if wb_triggered           else " "
            print(f"  [{idx:>4}] {sym}  rank={cr:>2}  "
                  f"gt={','.join(gt_bases):<30}  pred={result['cfcbn_top1']:<24}  "
                  f"LLM={wa}{wb}  a={result['cfcbn_alpha']:.3f}  Top1={top1/n:.1%}")

            if len(node_buf) >= ProgressTracker.NODE_SIZE:
                tracker.write_node_report(node_start, idx, node_buf, all_records)
                node_buf, node_start = [], idx + 1

        # Final node report
        if node_buf:
            last_idx = all_records[-1]["index"] if all_records else start_idx
            tracker.write_node_report(node_start, last_idx, node_buf, all_records)

        n = max(total, 1)
        summary = {
            "total":    total + skipped, "evaluated": total, "skipped": skipped,
            "top1":     top1, "top1_rate": top1 / n,
            "top3":     top3, "top3_rate": top3 / n,
            "top5":     top5, "top5_rate": top5 / n,
        }
        build_final_answer(all_records, self.final_answer_path)
        tracker.write_summary(all_records, _use_llm, self.skip_rc_types)
        self._print_summary(summary, all_records)
        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # 
    # ─────────────────────────────────────────────────────────────────────────

    def _save_ticket(self, idx: int, result: dict, workflow_b: Optional[dict] = None):
        """Write the ticket + Workflow B diagnosis to the tickets/ directory."""
        gt_str = "_".join(result.get("gt_bases", ["unknown"]))
        fname  = os.path.join(self.ticket_dir, f"{idx:04d}_{gt_str}.txt")
        lines  = [result.get("ticket_text", "")]
        if workflow_b:
            diag = workflow_b.get("diagnosis", {})
            recs = workflow_b.get("recommendations", {})
            lines += [
                "", "=" * 70,
                "  [WorkflowB] CF-CBN Failure Diagnosis & Optimization",
                "=" * 70,
                f"  Attribution : {diag.get('failure_attribution', '?')}",
                f"  Reasoning   : {diag.get('attribution_reasoning', '')}",
                f"  Summary     : {diag.get('failure_summary', '')}",
                f"  Why Top-1   : {diag.get('why_top1_ranked_first', '')}",
                f"  Why RC Low  : {diag.get('why_true_rc_ranked_low', '')}",
                "", "  Dimension Analysis:",
            ]
            for dim, val in diag.get("dimension_analysis", {}).items():
                if val:
                    lines.append(f"    {dim}: {val}")
            lines += [
                f"\n  Systematic  : {diag.get('systematic_pattern', '')}",
                "", "  Recommendations:", "─" * 70,
                f"  Immediate   : {recs.get('immediate_action', '')}",
                f"  Iteration   : {recs.get('iteration_note', '')}", "",
            ]
            for i, r in enumerate(recs.get("recommendations", []), 1):
                cat = r.get("category", "")
                cat_tag = " []" if "monitoring" in cat else " []"
                lines.append(f"  {i}. [{r.get('layer','?')}][{r.get('priority','?')}]{cat_tag} "
                             f"{r.get('action','')}")
                if r.get("expected_effect"):
                    lines.append(f"     → {r['expected_effect']}")
                if r.get("implementation_hint"):
                    lines.append(f"     ℹ {r['implementation_hint']}")
            lines.append("=" * 70)
        with open(fname, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _print_summary(self, summary: dict, all_records: list):
        ev = [r for r in all_records if not r.get("skipped")]
        n  = max(summary["evaluated"], 1)
        wa = sum(1 for r in ev if r.get("llm_assisted"))
        wb = sum(1 for r in ev if r.get("workflow_b_triggered"))
        print(f"\n{'='*60}")
        print(f"  CLLM v5 — Batch Complete  [llm_mode={self.llm_mode}]")
        print(f"{'='*60}")
        print(f"  CF-CBN Top-1 : {summary['top1']:>4}/{n} = {summary['top1_rate']:.2%}")
        print(f"  CF-CBN Top-3 : {summary['top3']:>4}/{n} = {summary['top3_rate']:.2%}")
        print(f"  CF-CBN Top-5 : {summary['top5']:>4}/{n} = {summary['top5_rate']:.2%}")
        print(f"  WorkflowA    : {wa:>4}/{n} = {wa/n:.2%}  (mode={self.llm_mode})")
        print(f"  WorkflowB    : {wb:>4}/{n} = {wb/n:.2%}")
        print(f"{'='*60}")

    def engineer(self, command: str) -> str:
        return self.manager.handle(command)

    def print_status(self):
        print(f"\n[Pipeline Status]")
        print(f"  dataset        : {self.dataset_cfg.name}")
        print(f"  use_llm        : {self.use_llm}")
        print(f"  llm_mode       : {self.llm_mode}")
        print(f"  skip_rc_types  : {self.skip_rc_types}")
        print(f"  CF-CBN services: {len(self.engine.services)}")
        print(f"  CF-CBN alpha   : {self.engine.current_alpha:.4f}")
        print(f"  Cases accum.   : {self.engine.n_accumulated}")
        print(f"  Cases in store : {len(self.case_store)}")
        print(f"  Topo edges     : {len(self.topo.get_edges())}")
