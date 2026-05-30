#!/usr/bin/env python3
"""Cross-session offline racket target predictability diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_body_only_racket_predictability import (  # noqa: E402
    fit_norm,
    load_sequence_samples,
    metric_dict,
    normalize,
    stack_split,
    write_rows,
)


def boolish(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def group_of(sequence: str) -> str:
    return sequence.split("/")[0]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            if (
                boolish(row.get("task_export_passed"))
                and boolish(row.get("integrity_check_passed"))
                and boolish(row.get("dynamic_replay_passed"))
            ):
                row["session_group"] = row.get("session_group") or group_of(row["sequence"])
                rows.append(row)
        return rows


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def deterministic_split(items: list[str], seed: int, train_frac: float = 0.70, val_frac: float = 0.15) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(sorted(items), dtype=object)
    arr = arr[rng.permutation(len(arr))].tolist()
    n = len(arr)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 2) if n >= 3 else max(1, n - 1)
        n_val = 1 if n >= 3 else 0
    return {
        "train": arr[:n_train],
        "validation": arr[n_train:n_train + n_val],
        "test": arr[n_train + n_val:],
    }


def stratified_sequence_split(rows: list[dict[str, str]], seed: int) -> dict[str, list[str]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_group[row["session_group"]].append(row["sequence"])
    split = {"train": [], "validation": [], "test": []}
    for group in sorted(by_group):
        local = deterministic_split(by_group[group], seed + sum(ord(c) for c in group))
        for key in split:
            split[key].extend(local[key])
    return {key: sorted(value) for key, value in split.items()}


def heldout_group_split(rows: list[dict[str, str]], seed: int) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    groups = sorted({row["session_group"] for row in rows})
    group_split = deterministic_split(groups, seed)
    seq_split = {"train": [], "validation": [], "test": []}
    for row in rows:
        for split_name, split_groups in group_split.items():
            if row["session_group"] in split_groups:
                seq_split[split_name].append(row["sequence"])
                break
    return {key: sorted(value) for key, value in seq_split.items()}, group_split


def rows_by_sequence(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["sequence"]: row for row in rows}


def run_models(
    samples: list[dict[str, np.ndarray | str]],
    split: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], np.ndarray, np.ndarray]:
    from sklearn.linear_model import Ridge

    train = set(split["train"])
    test = set(split["test"])
    _, y_train, _ = stack_split(samples, train, "body_features")
    _, y_test, test_seq_labels = stack_split(samples, test, "body_features")
    y_mean = y_train.mean(axis=0)
    predictions = {"constant_mean": np.tile(y_mean[None, :], (len(y_test), 1))}

    for model_name, feature_key in [("body_only_ridge", "body_features"), ("body_plus_racket_pose_ridge", "body_pose_features")]:
        x_train, y_train_model, _ = stack_split(samples, train, feature_key)
        x_test, _, _ = stack_split(samples, test, feature_key)
        x_mean, x_std = fit_norm(x_train)
        y_mean_model, y_std_model = fit_norm(y_train_model)
        model = Ridge(alpha=1.0)
        model.fit(apply_norm(x_train, x_mean, x_std), apply_norm(y_train_model, y_mean_model, y_std_model))
        pred = model.predict(apply_norm(x_test, x_mean, x_std)) * y_std_model + y_mean_model
        pred[:, 3:6] = normalize(pred[:, 3:6])
        predictions[model_name] = pred

    model_rows = []
    per_seq_rows = []
    for model_name, pred in predictions.items():
        model_rows.append({"model": model_name, "split": "test", **metric_dict(y_test, pred)})
        for seq in sorted(set(test_seq_labels.tolist())):
            mask = test_seq_labels == seq
            per_seq_rows.append({"sequence": seq, "model": model_name, **metric_dict(y_test[mask], pred[mask])})
    return model_rows, per_seq_rows, predictions, y_test, test_seq_labels


def improvement(model_rows: list[dict[str, Any]]) -> dict[str, float]:
    body = next(row for row in model_rows if row["model"] == "body_only_ridge")
    pose = next(row for row in model_rows if row["model"] == "body_plus_racket_pose_ridge")
    return {
        "tip_improvement_ratio": float((body["tip_error_mean_m"] - pose["tip_error_mean_m"]) / max(body["tip_error_mean_m"], 1e-8)),
        "axis_improvement_ratio": float((body["long_axis_angle_error_mean_deg"] - pose["long_axis_angle_error_mean_deg"]) / max(body["long_axis_angle_error_mean_deg"], 1e-8)),
    }


def plot_bars(output_dir: Path, prefix: str, model_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [row["model"] for row in model_rows]
    for suffix, key, ylabel in [
        ("tip_error_model_comparison.png", "tip_error_mean_m", "tip mean error (m)"),
        ("axis_error_model_comparison.png", "long_axis_angle_error_mean_deg", "long-axis mean error (deg)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
        ax.bar(labels, [float(row[key]) for row in model_rows], color="#3867d6")
        ax.set_title("offline reference-level diagnostic only\nnot PHC simulated rollout accuracy")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_{suffix}")
        plt.close(fig)


def plot_per_session_tip(output_dir: Path, per_seq_rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in per_seq_rows:
        grouped[(group_of(row["sequence"]), row["model"])].append(float(row["tip_error_mean_m"]))
    groups = sorted({key[0] for key in grouped})
    models = ["body_only_ridge", "body_plus_racket_pose_ridge"]
    x = np.arange(len(groups))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    for i, model in enumerate(models):
        vals = [np.mean(grouped.get((group, model), [np.nan])) for group in groups]
        ax.bar(x + (i - 0.5) * width, vals, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("tip mean error (m)")
    ax.set_title("offline reference-level diagnostic only\nnot PHC simulated rollout accuracy")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "experiment_B_per_session_tip_error.png")
    plt.close(fig)


def plot_representative(output_dir: Path, predictions: dict[str, np.ndarray], y_test: np.ndarray, labels: np.ndarray, prefix: str) -> None:
    sequence = sorted(set(labels.tolist()))[0]
    mask = labels == sequence
    x = np.arange(int(mask.sum()))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=140, sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(x, y_test[mask, i], color="#3867d6", label="reference")
        for model in ["body_only_ridge", "body_plus_racket_pose_ridge"]:
            ax.plot(x, predictions[model][mask, i], label=model, alpha=0.85)
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(["tip x", "tip y", "tip z"][i])
    axes[0].set_title(f"{sequence}\noffline reference-level diagnostic only; not PHC simulated rollout accuracy")
    axes[-1].set_xlabel("sample index within sequence after context crop")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_representative_trajectory_prediction_{sequence.replace('/', '_')}.png")
    plt.close(fig)


def split_counts(split: dict[str, list[str]]) -> dict[str, int]:
    return {key: len(value) for key, value in split.items()}


def group_counts_for_split(split: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    return {key: dict(Counter(group_of(seq) for seq in value)) for key, value in split.items()}


def write_experiment(
    name: str,
    split: dict[str, list[str]],
    group_split: dict[str, list[str]] | None,
    samples: list[dict[str, np.ndarray | str]],
    rows_meta: dict[str, dict[str, str]],
    output_dir: Path,
    plot_dir: Path,
    prefix: str,
) -> dict[str, Any]:
    model_rows, per_seq_rows, predictions, y_test, labels = run_models(samples, split)
    for row in per_seq_rows:
        meta = rows_meta[row["sequence"]]
        row["session_group"] = meta["session_group"]
    imp = improvement(model_rows)
    write_rows(output_dir / f"{prefix}_results.csv", model_rows)
    write_rows(output_dir / f"{prefix}_per_sequence_results.csv", per_seq_rows)
    plot_bars(plot_dir, prefix.replace("experiment_", "experiment_").replace("_session_stratified", "").replace("_heldout_session", ""), model_rows)
    if "B" in prefix:
        plot_per_session_tip(plot_dir, per_seq_rows)
        plot_representative(plot_dir, predictions, y_test, labels, prefix)
    summary = {
        "experiment": name,
        "scope": "offline reference-level diagnostic only; not PHC simulated rollout accuracy",
        "split": split,
        "split_counts": split_counts(split),
        "split_group_counts": group_counts_for_split(split),
        "group_split": group_split,
        "models": model_rows,
        **imp,
    }
    (output_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        f"# {name}",
        "",
        "Scope: offline reference-level diagnostic only; not PHC simulated rollout accuracy.",
        "",
        f"- Split clip counts: `{split_counts(split)}`",
    ]
    if group_split:
        lines.append(f"- Group split: `{group_split}`")
    lines += [
        "",
        "| model | tip mean (m) | tip p90 (m) | axis mean (deg) | axis p90 (deg) | handle mean (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        lines.append(
            f"| `{row['model']}` | {row['tip_error_mean_m']:.6f} | {row['tip_error_p90_m']:.6f} | "
            f"{row['long_axis_angle_error_mean_deg']:.6f} | {row['long_axis_angle_error_p90_deg']:.6f} | "
            f"{row['optional_handle_error_mean_m']:.6f} |"
        )
    lines += [
        "",
        f"- Tip improvement ratio, body+racket_pose vs body-only: `{imp['tip_improvement_ratio']:.6f}`",
        f"- Axis improvement ratio, body+racket_pose vs body-only: `{imp['axis_improvement_ratio']:.6f}`",
    ]
    (output_dir / f"{prefix}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def leave_one_session_out(
    rows: list[dict[str, str]],
    samples: list[dict[str, np.ndarray | str]],
    rows_meta: dict[str, dict[str, str]],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    groups = sorted({row["session_group"] for row in rows})
    result_rows = []
    summaries = []
    for group in groups:
        test = sorted(row["sequence"] for row in rows if row["session_group"] == group)
        rest = sorted(row["sequence"] for row in rows if row["session_group"] != group)
        val_count = max(1, int(round(len(rest) * 0.15)))
        rng = np.random.default_rng(seed + sum(ord(c) for c in group))
        rest_perm = np.asarray(rest, dtype=object)[rng.permutation(len(rest))].tolist()
        split = {
            "validation": sorted(rest_perm[:val_count]),
            "train": sorted(rest_perm[val_count:]),
            "test": test,
        }
        model_rows, per_seq_rows, _, _, _ = run_models(samples, split)
        imp = improvement(model_rows)
        body = next(row for row in model_rows if row["model"] == "body_only_ridge")
        pose = next(row for row in model_rows if row["model"] == "body_plus_racket_pose_ridge")
        result_rows.append(
            {
                "held_out_session_group": group,
                "test_clips": len(test),
                "body_only_tip_mean_error_m": body["tip_error_mean_m"],
                "body_plus_racket_pose_tip_mean_error_m": pose["tip_error_mean_m"],
                "tip_improvement_ratio": imp["tip_improvement_ratio"],
                "body_only_axis_mean_error_deg": body["long_axis_angle_error_mean_deg"],
                "body_plus_racket_pose_axis_mean_error_deg": pose["long_axis_angle_error_mean_deg"],
                "axis_improvement_ratio": imp["axis_improvement_ratio"],
            }
        )
        summaries.append({"held_out_group": group, "split": split, "models": model_rows, **imp})
    write_rows(output_dir / "experiment_C_leave_one_session_out_results.csv", result_rows)
    payload = {"scope": "offline reference-level diagnostic only; not PHC simulated rollout accuracy", "groups": summaries}
    (output_dir / "experiment_C_leave_one_session_out_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Experiment C: Leave-One-Session-Out",
        "",
        "Scope: offline reference-level diagnostic only; not PHC simulated rollout accuracy.",
        "",
        "| held-out group | clips | body tip mean | body+pose tip mean | tip improvement | body axis mean | body+pose axis mean | axis improvement |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result_rows:
        lines.append(
            f"| `{row['held_out_session_group']}` | {row['test_clips']} | {row['body_only_tip_mean_error_m']:.6f} | "
            f"{row['body_plus_racket_pose_tip_mean_error_m']:.6f} | {row['tip_improvement_ratio']:.6f} | "
            f"{row['body_only_axis_mean_error_deg']:.6f} | {row['body_plus_racket_pose_axis_mean_error_deg']:.6f} | "
            f"{row['axis_improvement_ratio']:.6f} |"
        )
    (output_dir / "experiment_C_leave_one_session_out_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--plot_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_radius", type=int, default=2)
    parser.add_argument("--run_leave_one_session_out", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(args.manifest_csv)
    if len({row["session_group"] for row in rows}) < 2:
        raise ValueError("cross-session evaluation requires at least two session groups")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    samples = [load_sequence_samples(Path(row["npz_path"]), args.context_radius) for row in rows]
    meta = rows_by_sequence(rows)

    split_a = stratified_sequence_split(rows, args.seed)
    summary_a = write_experiment(
        "Experiment A: Session-Stratified Sequence Split",
        split_a,
        None,
        samples,
        meta,
        args.output_dir,
        args.plot_dir,
        "experiment_A_session_stratified",
    )

    split_b, group_split_b = heldout_group_split(rows, args.seed)
    summary_b = write_experiment(
        "Experiment B: Held-Out Session Group Split",
        split_b,
        group_split_b,
        samples,
        meta,
        args.output_dir,
        args.plot_dir,
        "experiment_B_heldout_session",
    )

    summary_c = leave_one_session_out(rows, samples, meta, args.output_dir, args.seed) if args.run_leave_one_session_out else None

    lines = [
        "# Cross-Session Predictability Overall Report",
        "",
        "Scope: offline reference-level diagnostic only; not PHC simulated rollout accuracy.",
        "",
        "Fixed passive attachment remains rejected. Dynamic reference replay remains passed. PHC rollout racket accuracy is not computed.",
        "",
        "The raw `racket_pose_parameter` is a source-data time-varying control signal used here to diagnose whether body-only information is insufficient. Future controller goals should prefer explicit root-local racket geometry such as handle, tip, and long-axis targets unless a raw-pose interface is separately justified.",
        "",
        "## Experiment A",
        f"- Tip improvement: `{summary_a['tip_improvement_ratio']:.6f}`",
        f"- Axis improvement: `{summary_a['axis_improvement_ratio']:.6f}`",
        "",
        "## Experiment B",
        f"- Train groups: `{group_split_b['train']}`",
        f"- Validation groups: `{group_split_b['validation']}`",
        f"- Test groups: `{group_split_b['test']}`",
        f"- Tip improvement: `{summary_b['tip_improvement_ratio']:.6f}`",
        f"- Axis improvement: `{summary_b['axis_improvement_ratio']:.6f}`",
    ]
    if summary_c:
        lines += ["", "## Experiment C", f"- Held-out groups evaluated: `{len(summary_c['groups'])}`"]
    lines += [
        "",
        "Interpretation: if body + `racket_pose_parameter` outperforms body-only on held-out sessions, explicit racket-aware information improves cross-session reference-target prediction. This supports moving toward racket-aware controller interface design, but it is not policy success or simulated rollout accuracy.",
    ]
    (args.output_dir / "cross_session_predictability_overall_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"experiment_A": summary_a, "experiment_B": summary_b, "experiment_C_ran": summary_c is not None}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
