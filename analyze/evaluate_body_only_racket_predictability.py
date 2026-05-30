#!/usr/bin/env python3
"""Offline reference-level body-only racket predictability diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            row
            for row in csv.DictReader(f)
            if row.get("task_export_passed") in {"True", "true", "1", True}
            and row.get("integrity_check_passed", "True") in {"True", "true", "1", True}
            and row.get("dynamic_replay_passed", "True") in {"True", "true", "1", True}
        ]


def split_sequences(sequences: list[str], seed: int) -> dict[str, list[str]]:
    rng = np.random.default_rng(seed)
    seqs = np.asarray(sorted(sequences), dtype=object)
    order = rng.permutation(len(seqs))
    seqs = seqs[order].tolist()
    n = len(seqs)
    n_train = max(1, int(round(n * 0.70)))
    n_val = max(1, int(round(n * 0.15))) if n >= 10 else 0
    if n_train + n_val >= n:
        n_train = max(1, n - 2) if n >= 3 else max(1, n - 1)
        n_val = 1 if n >= 3 else 0
    return {
        "train": seqs[:n_train],
        "validation": seqs[n_train:n_train + n_val],
        "test": seqs[n_train + n_val:],
    }


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def root_local_body(data: np.lib.npyio.NpzFile) -> np.ndarray:
    body = np.asarray(data["reference_body_pos"], dtype=np.float64)
    root = np.asarray(data["root_position_phc_world"], dtype=np.float64)
    rot = np.asarray(data["root_rotation_phc_world_matrix"], dtype=np.float64)
    centered = body - root[:, None, :]
    return np.einsum("tij,tkj->tki", np.swapaxes(rot, 1, 2), centered)


def context_stack(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return arr
    chunks = []
    for offset in range(-radius, radius + 1):
        chunks.append(arr[radius + offset: len(arr) - radius + offset])
    return np.concatenate(chunks, axis=1)


def load_sequence_samples(path: Path, context_radius: int) -> dict[str, np.ndarray | str]:
    data = np.load(path, allow_pickle=True)
    sequence = str(data["sequence"].item())
    body_local = root_local_body(data)
    vel = np.zeros_like(body_local)
    vel[1:] = body_local[1:] - body_local[:-1]
    body_base = np.concatenate([body_local.reshape(len(body_local), -1), vel.reshape(len(vel), -1)], axis=1)
    racket_pose = np.asarray(data["racket_pose_parameter"], dtype=np.float64)
    target = np.concatenate(
        [
            np.asarray(data["racket_tip_root_local"], dtype=np.float64),
            np.asarray(data["racket_long_axis_root_local"], dtype=np.float64),
            np.asarray(data["racket_handle_root_local"], dtype=np.float64),
        ],
        axis=1,
    )
    if len(body_base) <= context_radius * 2:
        raise ValueError(f"{sequence} too short for context_radius={context_radius}")
    body_ctx = context_stack(body_base, context_radius)
    pose_ctx = context_stack(racket_pose, context_radius)
    target_mid = target[context_radius: len(target) - context_radius]
    frame_idx = np.asarray(data["source_frame_idx"], dtype=np.int64)[context_radius: len(target) - context_radius]
    return {
        "sequence": sequence,
        "frame_idx": frame_idx,
        "body_features": body_ctx,
        "body_pose_features": np.concatenate([body_ctx, pose_ctx], axis=1),
        "target": target_mid,
    }


def stack_split(samples: list[dict[str, np.ndarray | str]], seqs: set[str], feature_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, seq_labels = [], [], []
    for sample in samples:
        if sample["sequence"] not in seqs:
            continue
        x = np.asarray(sample[feature_key], dtype=np.float64)
        y = np.asarray(sample["target"], dtype=np.float64)
        xs.append(x)
        ys.append(y)
        seq_labels.append(np.asarray([sample["sequence"]] * len(y), dtype=object))
    if not xs:
        return np.empty((0, 0)), np.empty((0, 9)), np.empty((0,), dtype=object)
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(seq_labels)


def fit_norm(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def apply_norm(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tip_err = np.linalg.norm(y_pred[:, :3] - y_true[:, :3], axis=1)
    axis_angle = np.degrees(
        np.arccos(np.clip(np.sum(normalize(y_pred[:, 3:6]) * normalize(y_true[:, 3:6]), axis=1), -1.0, 1.0))
    )
    handle_err = np.linalg.norm(y_pred[:, 6:9] - y_true[:, 6:9], axis=1)
    return {
        "tip_error_mean_m": float(tip_err.mean()),
        "tip_error_p50_m": float(np.percentile(tip_err, 50)),
        "tip_error_p90_m": float(np.percentile(tip_err, 90)),
        "tip_error_max_m": float(tip_err.max()),
        "long_axis_angle_error_mean_deg": float(axis_angle.mean()),
        "long_axis_angle_error_p50_deg": float(np.percentile(axis_angle, 50)),
        "long_axis_angle_error_p90_deg": float(np.percentile(axis_angle, 90)),
        "long_axis_angle_error_max_deg": float(axis_angle.max()),
        "optional_handle_error_mean_m": float(handle_err.mean()),
        "optional_handle_error_p90_m": float(np.percentile(handle_err, 90)),
    }


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_model_comparison(output_dir: Path, rows: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(row["model"]) for row in rows]
    tip = [float(row["tip_error_mean_m"]) for row in rows]
    axis = [float(row["long_axis_angle_error_mean_deg"]) for row in rows]
    for name, values, ylabel in [
        ("tip_error_model_comparison.png", tip, "tip mean error (m)"),
        ("long_axis_error_model_comparison.png", axis, "long-axis mean error (deg)"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 4), dpi=140)
        ax.bar(labels, values, color="#3867d6")
        ax.set_title("offline reference-level diagnostic only\nnot PHC simulated rollout accuracy")
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(output_dir / name)
        plt.close(fig)


def plot_per_sequence_scatter(output_dir: Path, per_seq_rows: list[dict[str, object]]) -> None:
    body = {row["sequence"]: float(row["tip_error_mean_m"]) for row in per_seq_rows if row["model"] == "body_only_ridge"}
    pose = {row["sequence"]: float(row["tip_error_mean_m"]) for row in per_seq_rows if row["model"] == "body_plus_racket_pose_ridge"}
    seqs = sorted(set(body) & set(pose))
    if not seqs:
        return
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    ax.scatter([body[s] for s in seqs], [pose[s] for s in seqs], color="#20bf6b", alpha=0.8)
    lim = max(max(body.values()), max(pose.values())) * 1.08
    ax.plot([0, lim], [0, lim], color="#4b6584", linestyle="--", linewidth=1.0)
    ax.set_xlabel("body-only ridge tip mean error (m)")
    ax.set_ylabel("body + racket_pose ridge tip mean error (m)")
    ax.set_title("offline reference-level diagnostic only\nnot PHC simulated rollout accuracy")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "per_sequence_tip_error_scatter.png")
    plt.close(fig)


def plot_representative_trajectory(
    output_dir: Path,
    sequence: str,
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    mask: np.ndarray,
) -> None:
    x = np.arange(int(mask.sum()))
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), dpi=140, sharex=True)
    labels = ["tip root-local x (m)", "tip root-local y (m)", "tip root-local z (m)"]
    for i, ax in enumerate(axes):
        ax.plot(x, y_true[mask, i], label="reference", color="#3867d6", linewidth=1.5)
        for name, pred in predictions.items():
            if name == "constant_mean":
                continue
            ax.plot(x, pred[mask, i], label=name, linewidth=1.1, alpha=0.85)
        ax.set_ylabel(labels[i])
        ax.grid(True, alpha=0.25)
    axes[0].set_title(f"{sequence}\noffline reference-level diagnostic only; not PHC simulated rollout accuracy")
    axes[-1].set_xlabel("sample index within sequence after context crop")
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"representative_trajectory_prediction_{sequence.replace('/', '_')}.png")
    plt.close(fig)


def write_insufficient_outputs(args: argparse.Namespace, rows: list[dict[str, str]], reason: str) -> None:
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "insufficient_data",
        "reason": reason,
        "sequence_count": len(rows),
        "minimum_required_for_preliminary": 10,
        "seed": args.seed,
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_csv.write_text("model,status,reason\nall,insufficient_data," + reason.replace(",", ";") + "\n", encoding="utf-8")
    args.output_per_sequence_csv.write_text("sequence,status,reason\n", encoding="utf-8")
    args.output_plot_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.bar(["available", "minimum"], [len(rows), 10], color=["#4b7bec", "#eb3b5a"])
    ax.set_title("offline reference-level diagnostic only\nnot PHC simulated rollout accuracy")
    ax.set_ylabel("sequence count")
    fig.tight_layout()
    fig.savefig(args.output_plot_dir / "dataset_size_insufficient.png")
    plt.close(fig)
    report = [
        "# Body-Only Racket Predictability Diagnostic",
        "",
        "Status: insufficient sequence-level task dataset.",
        "",
        f"- Available usable task sequences: `{len(rows)}`",
        "- Minimum for preliminary diagnostic: `10`",
        "",
        "No supervised regression result is reported because sequence-level train/test conclusions would be unreliable.",
        "",
        "This is an offline reference-level diagnostic preparation only, not PHC simulated rollout accuracy.",
    ]
    args.output_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--output_per_sequence_csv", required=True, type=Path)
    parser.add_argument("--output_summary", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--output_plot_dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context_radius", type=int, default=2)
    args = parser.parse_args()

    rows = load_manifest(args.manifest_csv)
    if len(rows) < 10:
        write_insufficient_outputs(
            args,
            rows,
            reason="fewer than 10 sequence-level racket-aware task NPZ files are available",
        )
        return

    from sklearn.linear_model import Ridge

    samples = [load_sequence_samples(Path(row["npz_path"]), args.context_radius) for row in rows]
    sequences = [str(sample["sequence"]) for sample in samples]
    split = split_sequences(sequences, args.seed)
    train, test = set(split["train"]), set(split["test"])
    if not test:
        write_insufficient_outputs(args, rows, reason="sequence split produced no test clips")
        return

    _, y_train, _ = stack_split(samples, train, "body_features")
    _, y_test, test_seq_labels = stack_split(samples, test, "body_features")
    y_mean = y_train.mean(axis=0)
    predictions = {"constant_mean": np.tile(y_mean[None, :], (len(y_test), 1))}
    model_rows = []
    per_seq_rows = []

    for model_name, feature_key in [("body_only_ridge", "body_features"), ("body_plus_racket_pose_ridge", "body_pose_features")]:
        x_train, y_train_model, _ = stack_split(samples, train, feature_key)
        x_test, _, _ = stack_split(samples, test, feature_key)
        x_mean, x_std = fit_norm(x_train)
        y_norm_mean, y_norm_std = fit_norm(y_train_model)
        ridge = Ridge(alpha=1.0)
        ridge.fit(apply_norm(x_train, x_mean, x_std), apply_norm(y_train_model, y_norm_mean, y_norm_std))
        pred = ridge.predict(apply_norm(x_test, x_mean, x_std)) * y_norm_std + y_norm_mean
        pred[:, 3:6] = normalize(pred[:, 3:6])
        predictions[model_name] = pred

    for model_name, pred in predictions.items():
        metrics = metric_dict(y_test, pred)
        model_rows.append({"model": model_name, "split": "test", **metrics})
        for seq in sorted(set(test_seq_labels.tolist())):
            mask = test_seq_labels == seq
            per_seq_rows.append({"sequence": seq, "model": model_name, **metric_dict(y_test[mask], pred[mask])})

    body_metrics = next(row for row in model_rows if row["model"] == "body_only_ridge")
    pose_metrics = next(row for row in model_rows if row["model"] == "body_plus_racket_pose_ridge")
    tip_improvement = (body_metrics["tip_error_mean_m"] - pose_metrics["tip_error_mean_m"]) / max(body_metrics["tip_error_mean_m"], 1e-8)
    axis_improvement = (
        body_metrics["long_axis_angle_error_mean_deg"] - pose_metrics["long_axis_angle_error_mean_deg"]
    ) / max(body_metrics["long_axis_angle_error_mean_deg"], 1e-8)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_csv, model_rows)
    write_rows(args.output_per_sequence_csv, per_seq_rows)
    plot_model_comparison(args.output_plot_dir, model_rows)
    plot_per_sequence_scatter(args.output_plot_dir, per_seq_rows)
    representative = sorted(set(test_seq_labels.tolist()))[0]
    plot_representative_trajectory(
        args.output_plot_dir,
        representative,
        y_test,
        predictions,
        test_seq_labels == representative,
    )
    summary = {
        "status": "completed",
        "sample_size_label": "preliminary" if len(rows) < 30 else "first_version",
        "sequence_count": len(rows),
        "seed": args.seed,
        "context_radius": args.context_radius,
        "split": split,
        "models": model_rows,
        "tip_improvement_ratio_body_plus_pose_vs_body_only": float(tip_improvement),
        "axis_improvement_ratio_body_plus_pose_vs_body_only": float(axis_improvement),
        "scope": "offline reference-level predictability diagnostic only; not PHC simulated rollout accuracy",
    }
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Body-Only Racket Predictability Diagnostic",
        "",
        f"- Sequence count: `{len(rows)}`",
        f"- Split: train `{len(split['train'])}`, validation `{len(split['validation'])}`, test `{len(split['test'])}`",
        f"- Context radius: `{args.context_radius}`",
        "",
        "| model | tip mean (m) | tip p90 (m) | axis mean (deg) | axis p90 (deg) | handle mean (m) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in model_rows:
        report.append(
            f"| `{row['model']}` | {row['tip_error_mean_m']:.6f} | {row['tip_error_p90_m']:.6f} | "
            f"{row['long_axis_angle_error_mean_deg']:.6f} | {row['long_axis_angle_error_p90_deg']:.6f} | "
            f"{row['optional_handle_error_mean_m']:.6f} |"
        )
    report += [
        "",
        f"- Tip improvement ratio, body+racket_pose vs body-only: `{tip_improvement:.6f}`",
        f"- Axis improvement ratio, body+racket_pose vs body-only: `{axis_improvement:.6f}`",
        "",
        "Scope: offline reference-level diagnostic only; not PHC simulated rollout accuracy.",
    ]
    args.output_report.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
