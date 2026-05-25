#!/usr/bin/env python3
"""Export per-frame body-only diagnostics from a PHC rollout diagnostic JSON."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


JOINT_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def safe_name(sequence: str) -> str:
    return sequence.replace("/", "_").replace(" ", "_")


def load_metrics(path: Path, sequence: str) -> dict:
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["sequence_name"] == sequence:
                return row
    raise KeyError(f"sequence not found in metrics CSV: {sequence}")


def as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() == "true"


def load_rollout(path: Path, sequence: str) -> tuple[np.ndarray, np.ndarray, list[int], list[bool]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = []
    ref = []
    steps = []
    term = []
    keys_seen = set()
    for rec in payload["records"]:
        keys = rec.get("motion_keys", [])
        if keys:
            keys_seen.update(keys)
        if keys and keys[0] != sequence:
            continue
        if "body_pos" not in rec or "ref_body_pos" not in rec:
            raise KeyError(
                f"{path} does not contain body_pos/ref_body_pos; rerun with PHC_DIAG_FULL_BODY=1"
            )
        body.append(np.asarray(rec["body_pos"][0], dtype=np.float64))
        ref.append(np.asarray(rec["ref_body_pos"][0], dtype=np.float64))
        steps.append(int(rec["step"]))
        term.append(bool(rec["terminate"][0]))

    if not body:
        hint = ", ".join(sorted(keys_seen)[:5])
        raise ValueError(f"no records for {sequence} in {path}; keys seen: {hint}")
    return np.stack(body), np.stack(ref), steps, term


def norm_xy(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec[:2]))


def torso_metrics(joints: np.ndarray) -> tuple[float, float]:
    top_idx = 12 if len(joints) > 12 else min(15, len(joints) - 1)
    torso = joints[top_idx] - joints[0]
    length = float(np.linalg.norm(torso))
    if length < 1e-8:
        return 0.0, 90.0
    score = float(np.clip(torso[2] / length, -1.0, 1.0))
    angle = float(math.degrees(math.acos(np.clip(score, -1.0, 1.0))))
    return score, angle


def trend(values: list[float], frame_idx: int, width: int = 20) -> float:
    start = max(0, frame_idx - width)
    if frame_idx <= start:
        return 0.0
    return float(values[frame_idx] - values[start])


def label_frames(rows: list[dict], completed: bool, termination_frame: int | None) -> None:
    rollout_pelvis = [r["rollout_pelvis_height"] for r in rows]
    reference_pelvis = [r["reference_pelvis_height"] for r in rows]
    root_errors = [r["root_error"] for r in rows]
    mpjpes = [r["frame_mpjpe"] for r in rows]

    for i, row in enumerate(rows):
        label = "normal_tracking"
        if row["root_error"] > 1.0 or row["frame_mpjpe"] > 1.0 or row["max_joint_error"] > 1.0:
            label = "root_divergence_outlier"
        elif completed:
            label = "low_posture_tracking" if row["reference_pelvis_height"] < 0.78 else "normal_tracking"
        elif row["is_pre_termination_window"]:
            rollout_drop = trend(rollout_pelvis, i, 20) < -0.05
            ref_rise = trend(reference_pelvis, i, 30) > 0.03
            root_rise = trend(root_errors, i, 20) > 0.08 or trend(mpjpes, i, 20) > 0.08
            forward_gap = (
                row["rollout_root_forward_relative_to_feet"]
                - row["reference_root_forward_relative_to_feet"]
            )
            lateral_gap = abs(row["rollout_root_lateral_relative_to_feet"]) - abs(
                row["reference_root_lateral_relative_to_feet"]
            )
            rollout_below_ref = row["pelvis_height_error"] < -0.08
            if rollout_drop and root_rise and (ref_rise or rollout_below_ref):
                label = "low_to_upright_recovery_failure"
            elif forward_gap > 0.20 and rollout_drop:
                label = "forward_collapse_candidate"
            elif lateral_gap > 0.20 and rollout_drop:
                label = "lateral_collapse_candidate"
            elif row["reference_pelvis_height"] < 0.78 or row["rollout_pelvis_height"] < 0.78:
                label = "low_posture_tracking"
            else:
                label = "unknown_failure"
        elif row["reference_pelvis_height"] < 0.78:
            label = "low_posture_tracking"
        row["failure_label"] = label


def make_rows(
    sequence: str,
    body: np.ndarray,
    ref: np.ndarray,
    steps: list[int],
    term_flags: list[bool],
    metrics: dict,
) -> list[dict]:
    completed = as_bool(metrics["completed"])
    termination_frame = int(metrics["termination_frame"]) if metrics.get("termination_frame") else None
    rows: list[dict] = []

    for i in range(len(body)):
        joint_errors = np.linalg.norm(body[i] - ref[i], axis=1)
        max_idx = int(np.argmax(joint_errors))
        rollout_upright, rollout_tilt = torso_metrics(body[i])
        ref_upright, ref_tilt = torso_metrics(ref[i])

        rollout_foot_mid = (body[i, 7] + body[i, 8]) / 2.0
        ref_foot_mid = (ref[i, 7] + ref[i, 8]) / 2.0
        rollout_knee_width = norm_xy(body[i, 4] - body[i, 5])
        ref_knee_width = norm_xy(ref[i, 4] - ref[i, 5])
        rollout_foot_width = norm_xy(body[i, 7] - body[i, 8])
        ref_foot_width = norm_xy(ref[i, 7] - ref[i, 8])
        pre_window = (
            termination_frame is not None
            and termination_frame - 60 <= steps[i] <= termination_frame
        )

        row = {
            "sequence": sequence,
            "frame_idx": steps[i],
            "completed": completed,
            "termination_frame": termination_frame,
            "is_pre_termination_window": bool(pre_window),
            "rollout_root_x": float(body[i, 0, 0]),
            "rollout_root_y": float(body[i, 0, 1]),
            "rollout_root_z": float(body[i, 0, 2]),
            "reference_root_x": float(ref[i, 0, 0]),
            "reference_root_y": float(ref[i, 0, 1]),
            "reference_root_z": float(ref[i, 0, 2]),
            "root_error": float(np.linalg.norm(body[i, 0] - ref[i, 0])),
            "frame_mpjpe": float(np.mean(joint_errors)),
            "max_joint_error": float(joint_errors[max_idx]),
            "max_joint_error_joint_name": JOINT_NAMES[max_idx] if max_idx < len(JOINT_NAMES) else str(max_idx),
            "rollout_pelvis_height": float(body[i, 0, 2]),
            "reference_pelvis_height": float(ref[i, 0, 2]),
            "pelvis_height_error": float(body[i, 0, 2] - ref[i, 0, 2]),
            "rollout_head_height": float(body[i, 15, 2]),
            "reference_head_height": float(ref[i, 15, 2]),
            "head_height_error": float(body[i, 15, 2] - ref[i, 15, 2]),
            "rollout_torso_upright_score": rollout_upright,
            "reference_torso_upright_score": ref_upright,
            "torso_upright_error": float(rollout_upright - ref_upright),
            "rollout_body_tilt_angle_deg": rollout_tilt,
            "reference_body_tilt_angle_deg": ref_tilt,
            "rollout_left_ankle_height": float(body[i, 7, 2]),
            "rollout_right_ankle_height": float(body[i, 8, 2]),
            "reference_left_ankle_height": float(ref[i, 7, 2]),
            "reference_right_ankle_height": float(ref[i, 8, 2]),
            "rollout_foot_width": rollout_foot_width,
            "reference_foot_width": ref_foot_width,
            "foot_width_error": float(rollout_foot_width - ref_foot_width),
            "rollout_knee_width": rollout_knee_width,
            "reference_knee_width": ref_knee_width,
            "knee_width_error": float(rollout_knee_width - ref_knee_width),
            "rollout_knee_to_foot_width_ratio": float(rollout_knee_width / max(rollout_foot_width, 1e-6)),
            "reference_knee_to_foot_width_ratio": float(ref_knee_width / max(ref_foot_width, 1e-6)),
            "rollout_pelvis_to_foot_mid_xy_distance_proxy": norm_xy(body[i, 0] - rollout_foot_mid),
            "reference_pelvis_to_foot_mid_xy_distance_proxy": norm_xy(ref[i, 0] - ref_foot_mid),
            "pelvis_to_foot_mid_error_proxy": float(
                norm_xy(body[i, 0] - rollout_foot_mid) - norm_xy(ref[i, 0] - ref_foot_mid)
            ),
            "rollout_root_forward_relative_to_feet": float(body[i, 0, 0] - rollout_foot_mid[0]),
            "reference_root_forward_relative_to_feet": float(ref[i, 0, 0] - ref_foot_mid[0]),
            "rollout_root_lateral_relative_to_feet": float(body[i, 0, 1] - rollout_foot_mid[1]),
            "reference_root_lateral_relative_to_feet": float(ref[i, 0, 1] - ref_foot_mid[1]),
            "phc_terminate_flag": bool(term_flags[i]),
        }
        rows.append(row)

    label_frames(rows, completed, termination_frame)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_series(rows: list[dict], output_dir: Path, termination_frame: int | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = np.asarray([r["frame_idx"] for r in rows])

    def finish(ax, title: str, ylabel: str, filename: str) -> None:
        ax.set_title(title)
        ax.set_xlabel("frame")
        ax.set_ylabel(ylabel)
        if termination_frame is not None:
            ax.axvline(termination_frame, color="black", linestyle="--", linewidth=1.2, label="termination")
            ax.axvspan(max(0, termination_frame - 60), termination_frame, color="red", alpha=0.08, label="pre-failure")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        ax.figure.tight_layout()
        ax.figure.savefig(output_dir / filename, dpi=150)
        plt.close(ax.figure)

    specs = [
        ("frame_mpjpe", None, "Frame MPJPE", "m", "frame_mpjpe.png"),
        ("root_error", None, "Root error", "m", "root_error.png"),
        ("rollout_pelvis_height", "reference_pelvis_height", "Pelvis height", "m", "pelvis_height.png"),
        ("rollout_torso_upright_score", "reference_torso_upright_score", "Torso upright score", "score", "torso_upright.png"),
        ("rollout_knee_width", "reference_knee_width", "Knee width", "m", "knee_width.png"),
        ("rollout_foot_width", "reference_foot_width", "Foot width", "m", "foot_width.png"),
        (
            "rollout_root_forward_relative_to_feet",
            "reference_root_forward_relative_to_feet",
            "Root forward relative to feet proxy",
            "m",
            "root_forward_relative_to_feet_proxy.png",
        ),
    ]
    for a, b, title, ylabel, filename in specs:
        fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
        ax.plot(frames, [r[a] for r in rows], label=a.replace("_", " "))
        if b:
            ax.plot(frames, [r[b] for r in rows], label=b.replace("_", " "))
        finish(ax, title, ylabel, filename)


def set_equal_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=(0, 1))
    maxs = pts.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    radius = max(span * 0.58, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius * 0.6), center[2] + radius * 0.9)


def draw_skeleton(ax, joints: np.ndarray, color: str, label: str, alpha: float) -> None:
    first = True
    for a, b in EDGES:
        ax.plot(
            [joints[a, 0], joints[b, 0]],
            [joints[a, 1], joints[b, 1]],
            [joints[a, 2], joints[b, 2]],
            color=color,
            linewidth=2.0,
            alpha=alpha,
            label=label if first else None,
        )
        first = False
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=10, alpha=alpha)


def render_video(path: Path, sequence: str, body: np.ndarray, ref: np.ndarray, rows: list[dict], fps: int, stride: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = body[::stride]
    ref = ref[::stride]
    rows = rows[::stride]
    pts = np.concatenate([body, ref], axis=1)
    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=fps, bitrate=2800)

    with writer.saving(fig, str(path), dpi=140):
        for i, row in enumerate(rows):
            ax.clear()
            set_equal_axes(ax, pts)
            ax.view_init(elev=16, azim=-68)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            title = (
                f"{sequence}\n"
                f"frame {row['frame_idx']} | PHC rollout=red | reference=blue\n"
                f"MPJPE {row['frame_mpjpe']:.3f} m | root {row['root_error']:.3f} m | "
                f"pelvis {row['rollout_pelvis_height']:.2f}/{row['reference_pelvis_height']:.2f} | "
                f"knee {row['rollout_knee_width']:.2f}/{row['reference_knee_width']:.2f} | "
                f"foot {row['rollout_foot_width']:.2f}/{row['reference_foot_width']:.2f}\n"
                f"label: {row['failure_label']} | termination: {row['termination_frame']}"
            )
            ax.set_title(title, fontsize=9)
            draw_skeleton(ax, ref[i], "#4b7bec", "reference", 0.45)
            draw_skeleton(ax, body[i], "#ff3b30", "PHC rollout", 0.95)
            ax.legend(loc="upper right")
            writer.grab_frame()
    plt.close(fig)


def make_summary(sequence: str, rows: list[dict], metrics: dict, rollout_data: Path, motion_file: Path) -> dict:
    completed = as_bool(metrics["completed"])
    termination_frame = int(metrics["termination_frame"]) if metrics.get("termination_frame") else None
    label_counter = Counter(r["failure_label"] for r in rows)
    pre_rows = [r for r in rows if r["is_pre_termination_window"]]
    pre_counter = Counter(r["failure_label"] for r in pre_rows)
    dominant = "normal_tracking"
    formal_root_outlier = float(metrics["max_root_error"]) > 1.0 or float(metrics["max_mpjpe"]) > 1.0
    if formal_root_outlier:
        dominant = "root_divergence_outlier"
    elif not completed:
        source_counter = pre_counter if pre_counter else label_counter
        known = Counter({k: v for k, v in source_counter.items() if k != "unknown_failure"})
        dominant = (known or source_counter).most_common(1)[0][0]
    return {
        "sequence": sequence,
        "motion_file": str(motion_file),
        "rollout_data": str(rollout_data),
        "frame_count": int(metrics["frame_count"]),
        "diagnostic_frames": len(rows),
        "completed": completed,
        "termination_frame": termination_frame,
        "mean_mpjpe": float(np.mean([r["frame_mpjpe"] for r in rows])),
        "max_mpjpe": float(np.max([r["frame_mpjpe"] for r in rows])),
        "mean_root_error": float(np.mean([r["root_error"] for r in rows])),
        "max_root_error": float(np.max([r["root_error"] for r in rows])),
        "min_rollout_pelvis_height": float(np.min([r["rollout_pelvis_height"] for r in rows])),
        "min_reference_pelvis_height": float(np.min([r["reference_pelvis_height"] for r in rows])),
        "dominant_failure_label_pre_termination": dominant,
        "label_counts": dict(label_counter),
        "pre_termination_label_counts": dict(pre_counter),
        "contact_metrics_available": False,
        "contact_metrics_note": "Current rollout diagnostic contains body positions only; true contact force, true COM, and support polygon metrics were not exported.",
    }


def write_report(path: Path, summary: dict, plot_dir: Path, video_path: Path | None) -> None:
    lines = [
        f"# Body Diagnostic: {summary['sequence']}",
        "",
        f"- Completed: {summary['completed']}",
        f"- Termination frame: {summary['termination_frame']}",
        f"- Diagnostic frames: {summary['diagnostic_frames']} / frame_count {summary['frame_count']}",
        f"- Mean MPJPE: {summary['mean_mpjpe']:.6f} m",
        f"- Max MPJPE: {summary['max_mpjpe']:.6f} m",
        f"- Mean root error: {summary['mean_root_error']:.6f} m",
        f"- Max root error: {summary['max_root_error']:.6f} m",
        f"- Dominant pre-termination label: `{summary['dominant_failure_label_pre_termination']}`",
        "",
        "## Metric Limits",
        "",
        "This diagnostic uses rollout and reference body positions only. Contact force, true COM, foot slip, and support polygon metrics are unavailable in the current exported rollout data, so support-related columns are named as proxies.",
        "",
        "## Outputs",
        "",
        f"- Plots: `{plot_dir}`",
    ]
    if video_path:
        lines.append(f"- Annotated video: `{video_path}`")
    lines.extend([
        "",
        "## Label Counts",
        "",
        "```json",
        json.dumps(summary["label_counts"], indent=2, sort_keys=True),
        "```",
        "",
        "## Pre-Termination Label Counts",
        "",
        "```json",
        json.dumps(summary["pre_termination_label_counts"], indent=2, sort_keys=True),
        "```",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_file", required=True, type=Path)
    parser.add_argument("--metrics_csv", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--rollout_data", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_plot_dir", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--render_overlay_video", type=Path, default=None)
    parser.add_argument("--video_stride", type=int, default=2)
    parser.add_argument("--video_fps", type=int, default=30)
    args = parser.parse_args()

    metrics = load_metrics(args.metrics_csv, args.sequence)
    body, ref, steps, term_flags = load_rollout(args.rollout_data, args.sequence)
    rows = make_rows(args.sequence, body, ref, steps, term_flags, metrics)
    write_csv(args.output_csv, rows)
    plot_series(
        rows,
        args.output_plot_dir,
        int(metrics["termination_frame"]) if metrics.get("termination_frame") else None,
    )

    if args.render_overlay_video:
        render_video(args.render_overlay_video, args.sequence, body, ref, rows, args.video_fps, args.video_stride)

    summary = make_summary(args.sequence, rows, metrics, args.rollout_data, args.motion_file)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_report(args.output_report, summary, args.output_plot_dir, args.render_overlay_video)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
