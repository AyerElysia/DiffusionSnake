#!/usr/bin/env python
"""Summarize and plot the controlled grouped per-point FM experiments."""
import csv
import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "v7 uncentered": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v7_grouped_adaptive_adv_noentropy_gpu4/posttrain_rl_v5_geom_action/logs.jsonl",
    "v8 centered": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v8_grouped_centered_noentropy_gpu5/posttrain_rl_v5_geom_action/logs.jsonl",
    "v9 local credit": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v9_grouped_localcredit_gpu7/posttrain_rl_v5_geom_action/logs.jsonl",
    "v10 joint logprob": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v10_jointlogprob_localcredit_gpu7/posttrain_rl_v5_geom_action/logs.jsonl",
    "v11 point marginal": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v11_pointmarginal_localcredit_gpu6/posttrain_rl_v5_geom_action/logs.jsonl",
    "v12 continuous marginal": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v12_pointmarginal_contdist_gpu5/posttrain_rl_v5_geom_action/logs.jsonl",
    "v13 zero-mean local": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v13_zeromean_contdist_gpu6/posttrain_rl_v5_geom_action/logs.jsonl",
    "v14 zero-mean LR1e-4": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v14_zeromean_lr1e4_gpu5/posttrain_rl_v5_geom_action/logs.jsonl",
    "v15 last-two steps": ROOT / "data/outputs/1232_final_v5_perpoint_fmscale_v15_last2_fresh_gpu5/posttrain_rl_v5_geom_action/logs.jsonl",
}
FULL_EVAL_RUNS = {
    "deterministic FM": ROOT / "report/perpoint_final_eval/v3_off",
    "old per-point v3": ROOT / "report/perpoint_final_eval/v3_mean_fixed",
    "delta-NSD RL": ROOT / "report/perpoint_final_eval/delta_nsd_best",
    "v10 deterministic FM": ROOT / "report/perpoint_final_eval/v10_off",
    "v10 per-point": ROOT / "report/perpoint_final_eval/v10_mean",
    "v12 deterministic FM": ROOT / "report/perpoint_final_eval/v12_off",
    "v12 continuous per-point": ROOT / "report/perpoint_final_eval/v12_mean",
    "v15 deterministic FM": ROOT / "report/perpoint_final_eval/v15_step1020_off",
    "v15 last-two per-point": ROOT / "report/perpoint_final_eval/v15_step1020_mean",
}
OUT_DIR = ROOT / "report/perpoint_grouped_comparison"
FIELDS = (
    "step",
    "reward_mean",
    "reward_std_mean",
    "adv_mean",
    "adv_abs_mean",
    "step_quality_std_mean",
    "step_adv_abs_mean",
    "point_quality_std_mean",
    "point_adv_abs_mean",
    "point_quality_abs_mean",
    "point_quality_nonzero_frac",
    "point_marginal_credit",
    "point_marginal_metric",
    "policy_loss",
    "grad_norm",
    "approx_kl",
    "ratio_min",
    "ratio_max",
    "eval_iou",
    "eval_mboundf",
    "diag_group_pref_corr",
    "diag_group_sign_acc",
    "diag_group_pref_n",
    "diag_point_pref_corr",
    "diag_point_centered_pref_corr",
    "diag_point_sign_acc",
    "diag_point_balanced_sign_acc",
    "diag_point_positive_recall",
    "diag_point_negative_recall",
    "diag_point_preference_positive_frac",
    "diag_point_pref_n",
    "diag_policy_gain_mean",
    "diag_policy_gain_positive_frac",
)


def load_rows(path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return sorted(rows, key=lambda row: int(row["step"]))


def write_csv(all_rows):
    path = OUT_DIR / "metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("run",) + FIELDS)
        writer.writeheader()
        for name, rows in all_rows.items():
            for row in rows:
                writer.writerow({"run": name, **{key: row.get(key, "") for key in FIELDS}})
    return path


def values(rows, key):
    selected = [(int(row["step"]), float(row[key])) for row in rows if key in row]
    return ([item[0] for item in selected], [item[1] for item in selected])


def plot_training(all_rows):
    panels = (
        ("reward_std_mean", "Terminal reward std"),
        ("step_quality_std_mean", "Per-step local reward std"),
        ("step_adv_abs_mean", "Per-step |advantage|"),
        ("grad_norm", "Gradient norm"),
        ("policy_loss", "Policy loss"),
        ("approx_kl", "Approximate KL"),
    )
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, panels):
        for name, rows in all_rows.items():
            x, y = values(rows, key)
            if x:
                axis.plot(x, y, label=name, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    path = OUT_DIR / "training_signals.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_point_credit(all_rows):
    panels = (
        ("point_quality_std_mean", "Point credit std"),
        ("point_quality_abs_mean", "Point |credit|"),
        ("point_quality_nonzero_frac", "Nonzero point-credit fraction"),
        ("point_adv_abs_mean", "Point |advantage|"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, panels):
        for name, rows in all_rows.items():
            x, y = values(rows, key)
            if x:
                axis.plot(x, y, marker=".", label=name, alpha=0.85)
        axis.set_title(title)
        axis.set_xlabel("training step")
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    axes[0, 0].axhline(1e-4, color="gray", linestyle="--", linewidth=1,
                       label="advantage std floor")
    axes[0, 0].set_yscale("log")
    path = OUT_DIR / "point_credit_signals.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_eval(all_rows):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = (
        (axes[0, 0], "eval_iou", "Fixed-set IoU"),
        (axes[0, 1], "diag_point_centered_pref_corr", "Centered point preference correlation"),
        (axes[1, 0], "diag_point_balanced_sign_acc", "Balanced point-sign accuracy"),
        (axes[1, 1], "diag_policy_gain_mean", "Deterministic policy gain"),
    )
    for name, rows in all_rows.items():
        for axis, key, title in panels:
            x, y = values(rows, key)
            if x:
                axis.plot(x, y, marker="o", label=name)
            axis.set_title(title)
            axis.set_xlabel("training step")
            axis.grid(alpha=0.25)
    axes[0, 1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    axes[1, 0].axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    axes[1, 1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
    for axis in axes.flat:
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, fontsize=8)
    path = OUT_DIR / "eval_and_local_learning.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def load_full_eval(directory):
    paths = sorted(directory.glob("v3_7_full_test_iou_*.json"))
    if not paths:
        return None
    with paths[-1].open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_full_eval_csv(full_eval):
    fields = (
        "run",
        "evaluated_samples",
        "failed_samples",
        "mean_iou_sample_avg",
        "mean_dice_sample_avg",
        "mean_mboundf_sample_avg",
        "mean_nsd_sample_avg",
        "mean_iou_contour_avg",
        "mean_mboundf_contour_avg",
        "mean_nsd_contour_avg",
        "checkpoint",
        "fm_policy_mode",
    )
    path = OUT_DIR / "full_eval_metrics.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, row in full_eval.items():
            if row is None:
                continue
            writer.writerow({
                "run": name,
                **{key: row.get(key, "") for key in fields[1:-2]},
                "checkpoint": row.get("ckpt", ""),
                "fm_policy_mode": row.get("fm_policy", {}).get("effective_mode", ""),
            })
    return path


def plot_full_eval(full_eval):
    available = [(name, row) for name, row in full_eval.items() if row is not None]
    if not available:
        return None
    names = [item[0] for item in available]
    panels = (
        ("mean_iou_sample_avg", "IoU"),
        ("mean_dice_sample_avg", "Dice"),
        ("mean_mboundf_sample_avg", "mBoundF"),
        ("mean_nsd_sample_avg", "NSD @ 2 px"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, (key, title) in zip(axes.flat, panels):
        vals = [float(row[key]) for _, row in available]
        bars = axis.bar(names, vals)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.set_ylim(max(0.0, min(vals) - 0.01), min(1.0, max(vals) + 0.01))
        for bar, value in zip(bars, vals):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                      ha="center", va="bottom", fontsize=8)
    path = OUT_DIR / "full_eval_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def paired_full_eval_comparisons(full_eval, bootstrap_samples=20000, seed=20260710):
    metric_fields = {
        "iou": "sample_mean_ious",
        "dice": "sample_mean_dices",
        "mboundf": "sample_mean_mboundfs",
        "nsd": "sample_mean_nsds",
    }
    available = [(name, row) for name, row in full_eval.items() if row is not None]
    rng = np.random.default_rng(seed)
    comparisons = []
    for (name_a, row_a), (name_b, row_b) in combinations(available, 2):
        for metric, field in metric_fields.items():
            values_a = np.asarray(row_a.get(field, []), dtype=np.float64)
            values_b = np.asarray(row_b.get(field, []), dtype=np.float64)
            if values_a.size == 0 or values_a.shape != values_b.shape:
                continue
            delta = values_b - values_a
            draw_indices = rng.integers(
                0, delta.size, size=(bootstrap_samples, delta.size), endpoint=False
            )
            bootstrap_means = delta[draw_indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
            comparisons.append({
                "run_a": name_a,
                "run_b": name_b,
                "metric": metric,
                "n": int(delta.size),
                "mean_a": float(values_a.mean()),
                "mean_b": float(values_b.mean()),
                "mean_delta_b_minus_a": float(delta.mean()),
                "bootstrap_ci95_low": float(ci_low),
                "bootstrap_ci95_high": float(ci_high),
                "wins_b": int(np.count_nonzero(delta > 1e-12)),
                "ties": int(np.count_nonzero(np.abs(delta) <= 1e-12)),
                "losses_b": int(np.count_nonzero(delta < -1e-12)),
                "bootstrap_samples": int(bootstrap_samples),
                "bootstrap_seed": int(seed),
            })
    return comparisons


def write_paired_comparisons(full_eval):
    comparisons = paired_full_eval_comparisons(full_eval)
    json_path = OUT_DIR / "paired_full_eval_comparisons.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(comparisons, handle, indent=2, ensure_ascii=False)
    csv_path = OUT_DIR / "paired_full_eval_comparisons.csv"
    if comparisons:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=comparisons[0].keys())
            writer.writeheader()
            writer.writerows(comparisons)
    return csv_path, json_path


def print_summary(all_rows):
    for name, rows in all_rows.items():
        if not rows:
            print(f"{name}: no records")
            continue
        last = rows[-1]
        eval_rows = [row for row in rows if "eval_iou" in row]
        diag_rows = [
            row for row in rows
            if "diag_point_pref_corr" in row or "diag_group_pref_corr" in row
        ]
        message = f"{name}: step={last['step']} n={len(rows)}"
        if "step_quality_std_mean" in last:
            message += f" step_rstd={last['step_quality_std_mean']:.6g} step_|adv|={last['step_adv_abs_mean']:.4f}"
        if eval_rows:
            message += f" eval_iou={eval_rows[-1]['eval_iou']:.4f}"
        if diag_rows:
            diag = diag_rows[-1]
            corr = diag.get("diag_point_pref_corr", diag.get("diag_group_pref_corr"))
            sign = diag.get("diag_point_sign_acc", diag.get("diag_group_sign_acc"))
            message += f" pref_corr={corr:+.4f} sign_acc={sign:.3f}"
            if "diag_policy_gain_mean" in diag:
                message += f" policy_gain={diag['diag_policy_gain_mean']:+.6f}"
        print(message)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = {name: load_rows(path) for name, path in RUNS.items()}
    full_eval = {name: load_full_eval(path) for name, path in FULL_EVAL_RUNS.items()}
    print_summary(all_rows)
    print(f"CSV: {write_csv(all_rows)}")
    print(f"Signals: {plot_training(all_rows)}")
    print(f"Point credit: {plot_point_credit(all_rows)}")
    print(f"Eval: {plot_eval(all_rows)}")
    print(f"Full-eval CSV: {write_full_eval_csv(full_eval)}")
    paired_csv, paired_json = write_paired_comparisons(full_eval)
    print(f"Paired comparisons: {paired_csv}, {paired_json}")
    full_eval_plot = plot_full_eval(full_eval)
    if full_eval_plot is not None:
        print(f"Full-eval plot: {full_eval_plot}")


if __name__ == "__main__":
    main()
