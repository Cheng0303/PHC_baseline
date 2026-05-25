#!/usr/bin/env python3
"""Convert raw NewRacket racket_tip arrays into candidate PHC-space references."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def safe_name(sequence: str) -> str:
    return sequence.replace("/", "_")


def source_npz(dataset_root: Path, sequence: str) -> Path:
    return dataset_root / f"{sequence}.npz"


def load_ref_body(diagnostic: Path, sequence: str) -> tuple[np.ndarray, list[int]]:
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    ref = []
    steps = []
    for rec in payload["records"]:
        keys = rec.get("motion_keys", [])
        if keys and keys[0] != sequence:
            continue
        ref.append(np.asarray(rec["ref_body_pos"][0], dtype=np.float64))
        steps.append(int(rec["step"]))
    if not ref:
        raise ValueError(f"no reference body records for {sequence} in {diagnostic}")
    return np.stack(ref), steps


def convert_tip(raw_tip: np.ndarray, entry: dict, mode: str) -> tuple[np.ndarray, dict]:
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    trans_orig = np.asarray(entry["trans_orig"], dtype=np.float64)
    ground_fix = entry.get("ground_fix", {})
    ground_offset = float(ground_fix.get("applied_vertical_offset", 0.0))

    if mode == "world_plus_groundfix":
        tip = raw_tip.astype(np.float64).copy()
        tip[:, 2] += ground_offset
        transform = {
            "mode": mode,
            "formula": "tip_phc = raw_racket_tip + [0, 0, groundfix_z]",
            "groundfix_z": ground_offset,
        }
    elif mode == "local_plus_root_trans_offset":
        tip = raw_tip.astype(np.float64) + root[: len(raw_tip)]
        transform = {
            "mode": mode,
            "formula": "tip_phc = raw_racket_tip + converted_root_trans_offset",
            "groundfix_z": ground_offset,
        }
    elif mode == "local_plus_trans_orig_plus_groundfix":
        tip = raw_tip.astype(np.float64) + trans_orig[: len(raw_tip)]
        tip[:, 2] += ground_offset
        transform = {
            "mode": mode,
            "formula": "tip_phc = raw_racket_tip + trans_orig + [0, 0, groundfix_z]",
            "groundfix_z": ground_offset,
        }
    else:
        raise ValueError(f"unknown mode: {mode}")
    return tip, transform


def set_equal_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=(0, 1))
    maxs = pts.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    radius = max(span * 0.58, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius * 0.6), center[2] + radius * 0.9)


def draw_skeleton(ax, joints: np.ndarray) -> None:
    for a, b in EDGES:
        ax.plot(
            [joints[a, 0], joints[b, 0]],
            [joints[a, 1], joints[b, 1]],
            [joints[a, 2], joints[b, 2]],
            color="#4b7bec",
            linewidth=2.0,
            alpha=0.65,
        )
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color="#4b7bec", s=10, alpha=0.65)


def render_reference_video(output: Path, sequence: str, ref_body: np.ndarray, tip: np.ndarray, steps: list[int], stride: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ref_body = ref_body[::stride]
    tip = tip[::stride]
    steps = steps[::stride]
    pts = np.concatenate([ref_body, tip[:, None, :]], axis=1)
    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=30, bitrate=2600)
    trail = []
    with writer.saving(fig, str(output), dpi=140):
        for i in range(len(ref_body)):
            ax.clear()
            set_equal_axes(ax, pts)
            ax.view_init(elev=16, azim=-68)
            draw_skeleton(ax, ref_body[i])
            trail.append(tip[i])
            trail_arr = np.asarray(trail[-45:])
            ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2], color="#00a8ff", linewidth=2.0, label="reference racket tip trail")
            ax.scatter([tip[i, 0]], [tip[i, 1]], [tip[i, 2]], color="#00a8ff", s=42, label="transformed racket tip")
            ax.set_title(f"{sequence}\nmode = reference_only_coordinate_validation | frame {steps[i]}", fontsize=10)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.legend(loc="upper right")
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--converted_motion", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--output_npz", required=True, type=Path)
    parser.add_argument("--output_summary", required=True, type=Path)
    parser.add_argument("--output_video", type=Path, default=None)
    parser.add_argument("--mode", default="local_plus_root_trans_offset", choices=["world_plus_groundfix", "local_plus_root_trans_offset", "local_plus_trans_orig_plus_groundfix"])
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    source = source_npz(args.dataset_root, args.sequence)
    raw = np.load(source, allow_pickle=True)
    motion = joblib.load(args.converted_motion)
    if args.sequence not in motion:
        raise KeyError(args.sequence)
    raw_tip = np.asarray(raw["racket_tip"], dtype=np.float64)
    raw_pose = np.asarray(raw["racket_pose"], dtype=np.float64) if "racket_pose" in raw else None
    tip, transform = convert_tip(raw_tip, motion[args.sequence], args.mode)
    ref_body, steps = load_ref_body(args.diagnostic, args.sequence)
    n = min(len(tip), len(ref_body))
    tip = tip[:n]
    ref_body = ref_body[:n]
    steps = steps[:n]

    right_hand_dist = np.linalg.norm(tip - ref_body[:, 23], axis=1)
    right_wrist_dist = np.linalg.norm(tip - ref_body[:, 21], axis=1)
    validation_pass = bool(np.nanmean(right_hand_dist) < 1.0 and np.nanpercentile(right_hand_dist, 90) < 1.5)
    summary = {
        "sequence": args.sequence,
        "source_npz": str(source),
        "frame_count": int(n),
        "raw_racket_tip_shape": list(raw_tip.shape),
        "raw_racket_pose_shape": list(raw_pose.shape) if raw_pose is not None else None,
        "source_coordinate_assumption": "racket_tip is a 3D point from source NPZ; converter source says racket_tip is an alias of racket_pose after mask/global-rotation processing.",
        "PHC_coordinate_transform_used": transform,
        "groundfix_applied": motion[args.sequence].get("ground_fix", {}),
        "converted_tip_min": tip.min(axis=0).tolist(),
        "converted_tip_max": tip.max(axis=0).tolist(),
        "has_orientation_reference": False,
        "right_hand_distance_mean": float(np.mean(right_hand_dist)),
        "right_hand_distance_p90": float(np.percentile(right_hand_dist, 90)),
        "right_hand_distance_max": float(np.max(right_hand_dist)),
        "right_wrist_distance_mean": float(np.mean(right_wrist_dist)),
        "validation_status": "passed" if validation_pass else "failed",
        "validation_reason": "Mean and p90 distance from transformed racket tip to PHC reference R_Hand are within loose plausibility bounds." if validation_pass else "Transformed racket tip is not consistently near PHC reference R_Hand; cannot claim calibrated coordinates.",
    }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        sequence=np.asarray(args.sequence),
        frame_idx=np.asarray(steps, dtype=np.int32),
        reference_racket_tip_phc_world=tip.astype(np.float32),
        reference_body_pos=ref_body.astype(np.float32),
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_video:
        render_reference_video(args.output_video, args.sequence, ref_body, tip, steps, args.stride)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
