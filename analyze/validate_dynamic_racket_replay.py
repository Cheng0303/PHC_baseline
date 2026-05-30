#!/usr/bin/env python3
"""Validate dynamic kinematic racket replay from reference task targets."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def apply_transform(mats: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.einsum("tij,tj->ti", mats[:, :3, :3], points) + mats[:, :3, 3]


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = normalize(a)
    b_n = normalize(b)
    dots = np.clip(np.sum(a_n * b_n, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def stats(err: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(np.max(err)),
    }


def validate_task(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    sequence = str(data["sequence"].item())
    T_hand = np.asarray(data["source_anatomical_rhand_world"], dtype=np.float64)
    dynamic = np.asarray(data["dynamic_hand_to_racket_transform"], dtype=np.float64)
    T_replayed = np.matmul(T_hand, dynamic)

    handle_local = np.asarray(data["racket_handle_in_racket_frame"], dtype=np.float64)
    tip_local = np.asarray(data["racket_tip_in_racket_frame"], dtype=np.float64)
    head_local = np.asarray(data["racket_head_center_in_racket_frame"], dtype=np.float64)
    handle_ref = np.asarray(data["source_racket_handle_world"], dtype=np.float64)
    tip_ref = np.asarray(data["source_racket_tip_world"], dtype=np.float64)
    head_ref = np.asarray(data["source_racket_head_center_world"], dtype=np.float64)

    handle_replay = apply_transform(T_replayed, handle_local)
    tip_replay = apply_transform(T_replayed, tip_local)
    head_replay = apply_transform(T_replayed, head_local)

    handle_err = np.linalg.norm(handle_replay - handle_ref, axis=1)
    tip_err = np.linalg.norm(tip_replay - tip_ref, axis=1)
    head_err = np.linalg.norm(head_replay - head_ref, axis=1)
    long_ref = tip_ref - handle_ref
    long_replay = tip_replay - handle_replay
    long_axis_angle = angle_between_deg(long_replay, long_ref)
    passed = bool(
        tip_err.mean() < 1e-5
        and tip_err.max() < 1e-4
        and handle_err.max() < 1e-4
        and np.percentile(long_axis_angle, 90) < 1e-3
    )
    row = {
        "sequence": sequence,
        "frame_count": int(len(tip_err)),
        "handle_error_mean_m": stats(handle_err)["mean"],
        "handle_error_p90_m": stats(handle_err)["p90"],
        "handle_error_max_m": stats(handle_err)["max"],
        "tip_error_mean_m": stats(tip_err)["mean"],
        "tip_error_p90_m": stats(tip_err)["p90"],
        "tip_error_max_m": stats(tip_err)["max"],
        "head_center_error_mean_m": stats(head_err)["mean"],
        "head_center_error_p90_m": stats(head_err)["p90"],
        "head_center_error_max_m": stats(head_err)["max"],
        "long_axis_angle_error_mean_deg": stats(long_axis_angle)["mean"],
        "long_axis_angle_error_p90_deg": stats(long_axis_angle)["p90"],
        "passed": passed,
    }
    series = {
        "handle_error": handle_err,
        "tip_error": tip_err,
        "head_center_error": head_err,
        "long_axis_angle_error": long_axis_angle,
    }
    return row, series


def plot_errors(output_dir: Path, sequence: str, series: dict[str, np.ndarray]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(next(iter(series.values()))))
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=140, sharex=True)
    axes[0].plot(x, series["handle_error"], label="handle")
    axes[0].plot(x, series["tip_error"], label="tip")
    axes[0].plot(x, series["head_center_error"], label="head center")
    axes[0].set_ylabel("position error (m)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")
    axes[1].plot(x, series["long_axis_angle_error"], color="purple", label="long-axis angle")
    axes[1].set_ylabel("angle error (deg)")
    axes[1].set_xlabel("frame")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")
    fig.suptitle(sequence)
    fig.tight_layout()
    fig.savefig(output_dir / f"{sequence.replace('/', '_')}_dynamic_replay_errors.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_npz", nargs="+", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--output_plot_dir", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for task in args.task_npz:
        row, series = validate_task(task)
        rows.append(row)
        plot_errors(args.output_plot_dir, str(row["sequence"]), series)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Dynamic Kinematic Racket Replay Validation",
        "",
        "This validates reference-level replay only. It does not evaluate PHC simulated rollout racket accuracy.",
        "",
        "Formula: `T_replayed_racket[t] = T_source_anatomical_rhand[t] @ dynamic_hand_to_racket_transform[t]`.",
        "",
        "| sequence | tip mean (m) | tip max (m) | handle max (m) | long-axis p90 (deg) | passed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['sequence']}` | {row['tip_error_mean_m']:.10f} | {row['tip_error_max_m']:.10f} | "
            f"{row['handle_error_max_m']:.10f} | {row['long_axis_angle_error_p90_deg']:.10f} | {row['passed']} |"
        )
    all_passed = all(bool(row["passed"]) for row in rows)
    lines += [
        "",
        f"All passed: `{all_passed}`.",
        "",
        "Validation threshold: tip mean < 1e-5 m, tip max < 1e-4 m, handle max < 1e-4 m, long-axis p90 < 1e-3 deg.",
        "",
        "Interpretation: dynamic replay proves that the time-varying racket target can reconstruct reference racket geometry at the kinematic reference layer. It does not rescue fixed passive attachment.",
    ]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"all_passed": all_passed, "rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
