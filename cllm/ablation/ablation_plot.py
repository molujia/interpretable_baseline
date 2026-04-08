"""
ablation_plot.py — CLLM v7 
==========================================

:  ablation_run.py  JSON 

: outputs/<dataset>/ablation/figures/

  A : group_A.pdf / group_A.png    alpha 
        group_A_summary.txt
  B : group_B.pdf / group_B.png    eval_window 
        group_B_summary.txt

: 
  python ablation/ablation_plot.py --dataset ds25
  python ablation/ablation_plot.py --dataset tt
  python ablation/ablation_plot.py --dataset ds25 --group B
  python ablation/ablation_plot.py --dataset ds25 --group all
"""

import argparse
import json
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets import AVAILABLE_DATASETS

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.ticker
import numpy as np


# ── A  ─────────────────────────────────────────────────────────────
STYLE_MAP_A = {
    "α adaptive (CLLM v5, default)":      {"color": "#E74C3C", "ls": "-",  "lw": 2.6, "zorder": 5},
    "α=1.0 (CF only)":                    {"color": "#222222", "ls": "--", "lw": 1.8, "zorder": 2},
    "α=0.5 (static mix)":                 {"color": "#27AE60", "ls": "-.", "lw": 1.8, "zorder": 3},
    "α=0.7 (static, CF-biased)":          {"color": "#2980B9", "ls": "-.", "lw": 1.8, "zorder": 3},
    "α dynamic v4 (1.0→0.20, legacy)":    {"color": "#E67E22", "ls": "--", "lw": 1.8, "zorder": 4},
}

# ── B window=60 ──────────────────
# 60
STYLE_MAP_B = {
    "window=20":    {"color": "#A569BD", "ls": ":",  "lw": 1.6, "zorder": 2},
    "window=40":    {"color": "#5DADE2", "ls": "--", "lw": 1.8, "zorder": 3},
    "window=60 ★":  {"color": "#E74C3C", "ls": "-",  "lw": 2.6, "zorder": 5},
    "window=80":    {"color": "#27AE60", "ls": "-.", "lw": 1.8, "zorder": 3},
    "window=120":   {"color": "#E67E22", "ls": "--", "lw": 1.8, "zorder": 4},
}

DEFAULT_STYLE = {"color": "#95A5A6", "ls": "-", "lw": 1.5, "zorder": 1}

# ── C EdgeFilter ────────────────────────────────────────────
STYLE_MAP_C = {
    "EdgeFilter active (full pipeline)": {"color": "#27AE60", "ls": "-",  "lw": 2.6, "zorder": 5},
    "EdgeFilter disabled (no filter)":   {"color": "#E74C3C", "ls": "--", "lw": 1.8, "zorder": 3},
    "EdgeFilter active (Step-0 + LLM)":  {"color": "#2980B9", "ls": "-.", "lw": 2.0, "zorder": 4},
}

# ── D MetricClassifier ──────────────────────────────────────
STYLE_MAP_D = {
    "MetricClassifier active (baseline)":    {"color": "#27AE60", "ls": "-",  "lw": 2.6, "zorder": 5},
    "MetricClassifier disabled (noise scored)": {"color": "#E74C3C", "ls": "--", "lw": 1.8, "zorder": 3},
    "MetricClassifier active (classified)":  {"color": "#2980B9", "ls": "-.", "lw": 2.0, "zorder": 4},
}

METRIC_META = {
    "top1_rate": {"title": "Top-1 Accuracy", "ylabel": "Hit Rate @ 1"},
    "top3_rate": {"title": "Top-3 Accuracy", "ylabel": "Hit Rate @ 3"},
    "top5_rate": {"title": "Top-5 Accuracy", "ylabel": "Hit Rate @ 5"},
}


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def load_results(json_path: str):
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def extract_curve(result: dict, metric: str, start_step: int = 0):
    xs, ys = [], []
    for snap in result["snapshots"]:
        if snap["evaluated"] > 0 and snap["step"] >= start_step:
            xs.append(snap["step"])
            ys.append(snap[metric])
    return np.array(xs), np.array(ys)


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def plot_group(results, metrics, fig_title, out_path, style_map=None):
    """
    

    ncol = 
    

    style_map: →None  STYLE_MAP_A
    """
    if style_map is None:
        style_map = STYLE_MAP_A

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5.5 * n_metrics, 4.8))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle(fig_title, fontsize=12, fontweight="bold", y=0.98)

    legend_handles = []
    legend_labels  = []

    # ──  ────────────────────────────────────────────────────────────
    for result in results:
        label = result["label"]
        style = style_map.get(label, DEFAULT_STYLE)

        for ax, metric in zip(axes, metrics):
            xs, ys = extract_curve(result, metric)
            if len(xs) == 0:
                continue
            ax.plot(
                xs, ys,
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                zorder=style["zorder"],
            )

        if label not in legend_labels:
            # Line2D PatchDe-anonymised
            handle = mlines.Line2D(
                [], [],
                color=style["color"],
                linestyle=style["ls"],
                linewidth=style["lw"],
                label=label,
            )
            legend_handles.append(handle)
            legend_labels.append(label)

    # ──  ──────────────────────────────────────────────────────────
    for ax, metric in zip(axes, metrics):
        meta = METRIC_META[metric]
        ax.set_title(meta["title"], fontsize=11, pad=6)
        ax.set_xlabel("Cases Processed", fontsize=10)
        ax.set_ylabel(meta["ylabel"], fontsize=10)

        all_ys = []
        for result in results:
            _, ys = extract_curve(result, metric)
            all_ys.extend(ys.tolist())
        if all_ys:
            y_min, y_max = min(all_ys), max(all_ys)
            pad = max((y_max - y_min) * 0.15, 0.02)
            ax.set_ylim(max(0.0, y_min - pad), min(1.0, y_max + pad))
        else:
            ax.set_ylim(0.0, 1.0)

        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
        )
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=9)

        # 
        for result in results:
            xs, ys = extract_curve(result, metric)
            if len(ys) == 0:
                continue
            style = style_map.get(result["label"], DEFAULT_STYLE)
            ax.annotate(
                f"{ys[-1]:.1%}",
                xy=(xs[-1], ys[-1]),
                xytext=(4, 2), textcoords="offset points",
                fontsize=7.5, color=style["color"], zorder=6,
            )

    # ── ─────────────────────────────────────────────────
    n_cols = len(legend_handles)
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=n_cols,
        fontsize=8.5,
        framealpha=0.92,
        edgecolor="#cccccc",
        handlelength=2.0,
        handleheight=0.9,
        columnspacing=1.2,
        borderpad=0.6,
    )

    fig.subplots_adjust(
        left=0.07, right=0.97,
        top=0.90,  bottom=0.22,
        wspace=0.35,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {os.path.abspath(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# 
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(results, title, out_path):
    W = 74
    lines = [
        "=" * W,
        f"  {title}",
        "=" * W,
        f"  {'Configuration':<42}  {'Top-1':>7}  {'Top-3':>7}  {'Top-5':>7}",
        "  " + "─" * (W - 2),
    ]
    for r in results:
        f = r["final"]
        # /
        is_default = ("adaptive_v5" in r["name"] and "window" not in r["label"]) \
                     or "★" in r.get("label", "")
        marker = " ◀" if is_default else ""
        lines.append(
            f"  {r['label']:<42}  "
            f"{f['top1_rate']:>6.2%}   {f['top3_rate']:>6.2%}   {f['top5_rate']:>6.2%}{marker}"
        )
    lines += ["=" * W, ""]
    text = "\n".join(lines)
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [Table] Saved → {os.path.abspath(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# : 
# ─────────────────────────────────────────────────────────────────────────────

def _find_json(results_dir: str, prefix: str, dataset: str) -> Optional[str]:
    """
     JSON
      1. ablation_<prefix>_<dataset>.json   
      2. ablation_<prefix>.json             
    """
    p1 = os.path.join(results_dir, f"ablation_{prefix}_{dataset}.json")
    if os.path.exists(p1):
        return p1
    p2 = os.path.join(results_dir, f"ablation_{prefix}.json")
    if os.path.exists(p2):
        return p2
    return None


def _dataset_label(dataset: str) -> str:
    return {
        "ds25": "DS25  (Online Boutique + TiDB, 318 cases)",
        "tt":   "TT  (TrainTicket, 96 cases)",
    }.get(dataset, dataset.upper())




# ─────────────────────────────────────────────────────────────────────────────
# B : alpha trajectory tune α
# ─────────────────────────────────────────────────────────────────────────────

def plot_alpha_trajectory(results, fig_title, out_path, style_map=None):
    """
     B  alpha trajectory 

    : tune  accumulated casestotal
    : alpha_optimal LOO  α
     eval_window 

    : 
      " eval_window  α "
      → :  α 
      → :  α  Top-1 
     loo_top1LOO Top-1  CBN 
    """
    if style_map is None:
        style_map = STYLE_MAP_B

    fig, (ax_alpha, ax_loo) = plt.subplots(
        1, 2, figsize=(11, 4.2),
    )
    fig.suptitle(fig_title, fontsize=11, fontweight="bold", y=1.01)

    legend_handles = []
    legend_labels  = []

    for result in results:
        label     = result["label"]
        style     = style_map.get(label, DEFAULT_STYLE)
        alpha_log = result.get("alpha_log", [])
        if not alpha_log:
            continue

        xs      = [e["total"]         for e in alpha_log]
        alphas  = [e["alpha_optimal"]  for e in alpha_log]
        loos    = [e["loo_top1"]       for e in alpha_log]
        ws_list = [e["window_size"]    for e in alpha_log]

        # : alpha_optimal trajectory
        ax_alpha.plot(xs, alphas,
                      color=style["color"], linestyle=style["ls"],
                      linewidth=style["lw"], zorder=style["zorder"],
                      marker="o", markersize=4)

        # : loo_top1 trajectory
        ax_loo.plot(xs, loos,
                    color=style["color"], linestyle=style["ls"],
                    linewidth=style["lw"], zorder=style["zorder"],
                    marker="s", markersize=3.5)

        if label not in legend_labels:
            handle = mlines.Line2D(
                [], [], color=style["color"],
                linestyle=style["ls"], linewidth=style["lw"],
                marker="o", markersize=4, label=label,
            )
            legend_handles.append(handle)
            legend_labels.append(label)

    # ── : alpha trajectory ────────────────────────────────────────
    ax_alpha.set_title("α Trajectory  (per tune event)", fontsize=11, pad=6)
    ax_alpha.set_xlabel("Accumulated Cases at Tune", fontsize=10)
    ax_alpha.set_ylabel("α Optimal  (lower = more CBN)", fontsize=10)
    ax_alpha.set_ylim(0.15, 0.70)
    ax_alpha.axhline(y=0.20, color="#aaaaaa", linestyle=":", linewidth=1.0,
                     label="search_lo=0.20")
    ax_alpha.axhline(y=0.65, color="#aaaaaa", linestyle=":", linewidth=1.0,
                     label="search_hi=0.65")
    ax_alpha.grid(True, alpha=0.3, linestyle=":")
    ax_alpha.tick_params(labelsize=9)
    #  α=1.0  CF 
    ax_alpha.axhline(y=1.0, color="#999999", linestyle="--", linewidth=0.8,
                     alpha=0.5)
    # 
    ax_alpha.fill_between([0, 999], 0.20, 0.65,
                           color="#dddddd", alpha=0.18, label="search range")

    # ── : LOO Top-1 trajectory ───────────────────────────────────
    ax_loo.set_title("LOO Top-1 at Tune  (CBN quality signal)", fontsize=11, pad=6)
    ax_loo.set_xlabel("Accumulated Cases at Tune", fontsize=10)
    ax_loo.set_ylabel("LOO Top-1 Estimate", fontsize=10)
    ax_loo.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )
    ax_loo.grid(True, alpha=0.3, linestyle=":")
    ax_loo.tick_params(labelsize=9)

    # ── ──────────────────────────────────────────────
    fig.legend(
        handles=legend_handles, labels=legend_labels,
        loc="lower center", bbox_to_anchor=(0.5, 0.0),
        ncol=len(legend_handles),
        fontsize=8.5, framealpha=0.92, edgecolor="#cccccc",
        handlelength=2.0, columnspacing=1.2, borderpad=0.6,
    )
    fig.subplots_adjust(
        left=0.08, right=0.97, top=0.88, bottom=0.22, wspace=0.35,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Plot] Saved → {os.path.abspath(out_path)}")


# ─────────────────────────────────────────────────────────────────────────────
# Group C: EdgeFilter topology quality bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_group_c(json_path: str, fig_title: str, out_path: str):
    """
    Bar chart showing topology quality metrics for Group C (EdgeFilter ablation).
    Three groups of bars: Clean / Noisy / Filtered.
    Two sub-plots: (left) Edge counts split by noise/real; (right) Jaccard similarity.
    """
    import json as _json
    data = _json.load(open(json_path))

    labels  = [r['label'] for r in data]
    n_real  = [r['n_edges'] - r['n_noise_edges'] for r in data]
    n_noise = [r['n_noise_edges'] for r in data]
    jaccard = [r['jaccard_to_gt'] for r in data]
    collid  = [r['n_colliders'] for r in data]

    colors_main  = ['#27AE60', '#E74C3C', '#2980B9']  # clean, noisy, filtered
    color_noise  = '#E67E22'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle(fig_title, fontsize=12, fontweight='bold', y=0.98)

    x = np.arange(len(labels))
    w = 0.45

    # ── Left: stacked bar (real edges + noise edges) ──────────────────────
    bars_real  = ax1.bar(x, n_real,  w, label='Real edges',  color=colors_main, alpha=0.85)
    bars_noise = ax1.bar(x, n_noise, w, bottom=n_real,
                         label='Noise edges', color=color_noise, alpha=0.75, hatch='//')

    ax1.set_title('Edge Composition', fontsize=11, pad=6)
    ax1.set_ylabel('Number of Edges', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5, rotation=10, ha='right')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle=':')
    ax1.tick_params(axis='y', labelsize=9)
    # Annotate totals
    for xi, (nr, nn) in enumerate(zip(n_real, n_noise)):
        ax1.text(xi, nr + nn + 0.3, str(nr + nn),
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── Right: Jaccard similarity to ground truth ─────────────────────────
    bars_jac = ax2.bar(x, jaccard, w, color=colors_main, alpha=0.85)
    ax2.set_title('Jaccard Similarity to Ground-Truth Topology', fontsize=11, pad=6)
    ax2.set_ylabel('Jaccard Similarity', fontsize=10)
    ax2.set_ylim(0.80, 1.03)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5, rotation=10, ha='right')
    ax2.axhline(y=1.0, color='#27AE60', linestyle='--', linewidth=1.0, alpha=0.6,
                label='Perfect (=1.0)')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3, linestyle=':')
    ax2.tick_params(axis='y', labelsize=9)
    ax2.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:.3f}')
    )
    for xi, jac in enumerate(jaccard):
        ax2.text(xi, jac + 0.003, f'{jac:.4f}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.18, wspace=0.35)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Plot] Saved → {os.path.abspath(out_path)}")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CLLM v7 ")
    p.add_argument("--dataset", default="ds25", choices=AVAILABLE_DATASETS,
                   help=" ds25")
    p.add_argument("--results", default=None,
                   help="Results JSON directory (default: outputs/<dataset>/ablation/results)")
    p.add_argument("--out",     default=None,
                   help="Figure output directory (default: outputs/<dataset>/ablation/figures)")
    p.add_argument("--group",   default="A",
                   choices=["A", "B", "C", "D", "all"],
                   help="Which experiment group to plot: A (default) | B | C | D | all")
    args = p.parse_args()

    dataset     = args.dataset
    results_dir = args.results or os.path.join("outputs", dataset, "ablation", "results")
    out_dir     = args.out     or os.path.join("outputs", dataset, "ablation", "figures")
    os.makedirs(out_dir, exist_ok=True)
    metrics = ["top1_rate", "top3_rate", "top5_rate"]
    dl = _dataset_label(dataset)

    plot_a = args.group in ("A", "all")
    plot_b = args.group in ("B", "all")
    # C/D handled inline above after B block

    # ── A  ────────────────────────────────────────────────────────────────
    if plot_a:
        path_a = _find_json(results_dir, "group_A", dataset)
        if path_a:
            print(f"\n[Plot] Group A  ·  {dataset}  ·  {path_a}")
            results_a = load_results(path_a)

            # : dynamic_v4 style_map
            target_names_a = {
                "alpha_fixed_1.0", "alpha_fixed_0.5",
                "alpha_fixed_0.7", "alpha_dynamic_v4", "alpha_adaptive_v5",
            }
            results_a_f = [r for r in results_a if r.get("name", "") in target_names_a]
            if not results_a_f:
                results_a_f = results_a

            fig_title = (
                f"Alpha Strategy Comparison — {dl}\n"
                "[CF-CBN · no error_flag · snapshot every 10 cases]"
            )
            for ext in ("pdf", "png"):
                plot_group(
                    results_a_f, metrics=metrics,
                    fig_title=fig_title,
                    out_path=os.path.join(out_dir, f"group_A.{ext}"),
                    style_map=STYLE_MAP_A,
                )
            print_summary_table(
                results_a_f,
                title=f"Group A — Alpha Strategy  [{dataset.upper()}]",
                out_path=os.path.join(out_dir, "group_A_summary.txt"),
            )
        else:
            print(f"[Plot] Group A results not found in {results_dir}")
            print(f"       Run:  python ablation/ablation_run.py --dataset {dataset}")

    # ── B  ────────────────────────────────────────────────────────────────
    if plot_b:
        path_b = _find_json(results_dir, "group_B", dataset)
        if path_b:
            print(f"\n[Plot] Group B  ·  {dataset}  ·  {path_b}")
            results_b = load_results(path_b)

            fig_title_perf = (
                f"eval_window Sensitivity — {dl}\n"
                "[adaptive α · CF-CBN · no error_flag · ★ = default (window=60)]"
            )
            # : Top-1/3/5 
            for ext in ("pdf", "png"):
                plot_group(
                    results_b, metrics=metrics,
                    fig_title=fig_title_perf,
                    out_path=os.path.join(out_dir, f"group_B.{ext}"),
                    style_map=STYLE_MAP_B,
                )

            # : alpha trajectory + LOO Top-1 α vs CBN
            fig_title_traj = (
                f"α Trajectory & LOO Top-1 — {dl}\n"
                "[eval_window sensitivity · ★ = default (window=60)]"
            )
            for ext in ("pdf", "png"):
                plot_alpha_trajectory(
                    results_b,
                    fig_title=fig_title_traj,
                    out_path=os.path.join(out_dir, f"group_B_alpha_trajectory.{ext}"),
                    style_map=STYLE_MAP_B,
                )

            print_summary_table(
                results_b,
                title=f"Group B — eval_window Sensitivity  [{dataset.upper()}]",
                out_path=os.path.join(out_dir, "group_B_summary.txt"),
            )
        else:
            print(f"[Plot] Group B results not found in {results_dir}")
            print(f"       Run:  python ablation/ablation_run.py --dataset {dataset} --group B")
    # ── C EdgeFilter A/B/D ──────────────────────────
    if args.group in ("C", "all"):
        path_c = _find_json(results_dir, "group_C", dataset)
        if path_c:
            print(f"\n[Plot] Group C  ·  {dataset}  ·  {path_c}")
            results_c = load_results(path_c)
            fig_title_c = (
                f"EdgeFilter Ablation — {dl}\n"
                "[topology propagation noise · green = clean · blue = filter applied]"
            )
            for ext in ("pdf", "png"):
                plot_group(
                    results_c, metrics=metrics,
                    fig_title=fig_title_c,
                    out_path=os.path.join(out_dir, f"group_C.{ext}"),
                    style_map=STYLE_MAP_C,
                )
            print_summary_table(
                results_c,
                title=f"Group C — EdgeFilter  [{dataset.upper()}]",
                out_path=os.path.join(out_dir, "group_C_summary.txt"),
            )
        else:
            print(f"[Plot] Group C results not found in {results_dir}")
            print(f"       Run:  python ablation/ablation_run.py --dataset {dataset} --group C")

    # ── D  ────────────────────────────────────────────────────────────────
    if args.group in ("D", "all"):
        path_d = _find_json(results_dir, "group_D", dataset)
        if path_d:
            print(f"\n[Plot] Group D  ·  {dataset}  ·  {path_d}")
            results_d = load_results(path_d)
            fig_title_d = (
                f"MetricClassifier Ablation — {dl}\n"
                "[noise metrics injected · green = clean · blue = classifier applied]"
            )
            for ext in ("pdf", "png"):
                plot_group(
                    results_d, metrics=metrics,
                    fig_title=fig_title_d,
                    out_path=os.path.join(out_dir, f"group_D.{ext}"),
                    style_map=STYLE_MAP_D,
                )
            print_summary_table(
                results_d,
                title=f"Group D — MetricClassifier  [{dataset.upper()}]",
                out_path=os.path.join(out_dir, "group_D_summary.txt"),
            )
        else:
            print(f"[Plot] Group D results not found in {results_dir}")
            print(f"       Run:  python ablation/ablation_run.py --dataset {dataset} --group D")


    print("\n[Plot] All done.")


if __name__ == "__main__":
    main()
