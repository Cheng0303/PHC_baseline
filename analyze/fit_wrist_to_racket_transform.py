#!/usr/bin/env python3
"""Fit a fixed hand-to-racket-tip transform from reference hand states."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_reference(path: Path) -> tuple[str, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    seq = str(data["sequence"].item())
    tip = np.asarray(data["reference_racket_tip_phc_world"], dtype=np.float64)
    body = np.asarray(data["reference_body_pos"], dtype=np.float64)
    return seq, tip, body


def load_hand_state(path: Path, sequence: str) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    seq = str(data["sequence"].item())
    if seq != sequence:
        raise ValueError(f"hand-state sequence mismatch: expected {sequence}, got {seq}")
    hand_pos = np.asarray(data["reference_hand_position_world"], dtype=np.float64)
    quat = np.asarray(data["reference_hand_orientation_world_quat"], dtype=np.float64)
    if quat.size == 0:
        raise ValueError(f"{path} does not contain true reference hand quaternions")
    return hand_pos, quat


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.stack(
        [
            np.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def hand_frames(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hand = body[:, 23]
    wrist = body[:, 21]
    elbow = body[:, 19]
    x_axis = normalize(hand - wrist)
    y_seed = normalize(wrist - elbow)
    z_axis = normalize(np.cross(x_axis, y_seed))
    bad = np.linalg.norm(z_axis, axis=1) < 1e-6
    z_axis[bad] = np.asarray([0.0, 0.0, 1.0])
    y_axis = normalize(np.cross(z_axis, x_axis))
    rot = np.stack([x_axis, y_axis, z_axis], axis=-1)
    return hand, rot


def local_offsets(tips: np.ndarray, body: np.ndarray, hand_state: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if hand_state is None:
        hand, rot = hand_frames(body)
    else:
        hand, quat = hand_state
        rot = quat_xyzw_to_matrix(quat)
        n = min(len(tips), len(hand), len(rot))
        tips, hand, rot = tips[:n], hand[:n], rot[:n]
    return np.einsum("tji,tj->ti", rot, tips - hand)


def reconstruct(body: np.ndarray, local: np.ndarray, hand_state: tuple[np.ndarray, np.ndarray] | None) -> np.ndarray:
    if hand_state is None:
        hand, rot = hand_frames(body)
    else:
        hand, quat = hand_state
        rot = quat_xyzw_to_matrix(quat)
    return hand + np.einsum("tij,j->ti", rot, local)


def error_stats(err: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(err)),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "max": float(np.max(err)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_npz", nargs="+", required=True, type=Path)
    parser.add_argument("--hand_state_npz", nargs="+", default=None, type=Path)
    parser.add_argument("--output_transform", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--pass_mean_threshold", type=float, default=0.30)
    parser.add_argument("--pass_p90_threshold", type=float, default=0.50)
    args = parser.parse_args()

    clips = [load_reference(p) for p in args.reference_npz]
    hand_states = {}
    if args.hand_state_npz:
        for path in args.hand_state_npz:
            data = np.load(path, allow_pickle=True)
            seq = str(data["sequence"].item())
            hand_states[seq] = load_hand_state(path, seq)
    rows = []
    all_pass = True
    for val_idx, (val_seq, val_tip, val_body) in enumerate(clips):
        train_offsets = []
        train_names = []
        for i, (seq, tip, body) in enumerate(clips):
            if i == val_idx:
                continue
            train_offsets.append(local_offsets(tip, body, hand_states.get(seq)))
            train_names.append(seq)
        local = np.concatenate(train_offsets, axis=0).mean(axis=0)
        pred = reconstruct(val_body, local, hand_states.get(val_seq))
        n = min(len(pred), len(val_tip))
        err = np.linalg.norm(pred[:n] - val_tip[:n], axis=1)
        stats = error_stats(err)
        passed = stats["mean"] <= args.pass_mean_threshold and stats["p90"] <= args.pass_p90_threshold
        all_pass = all_pass and passed
        rows.append({
            "validation_clip": val_seq,
            "fitting_clips": ";".join(train_names),
            "attached_body": "R_Hand",
            "local_tip_offset_x": local[0],
            "local_tip_offset_y": local[1],
            "local_tip_offset_z": local[2],
            "mean_tip_error": stats["mean"],
            "median_tip_error": stats["median"],
            "p90_tip_error": stats["p90"],
            "max_tip_error": stats["max"],
            "passed": passed,
        })

    all_offsets = np.concatenate([local_offsets(tip, body, hand_states.get(seq)) for seq, tip, body in clips], axis=0)
    transform = {
        "attached_body": "R_Hand",
        "orientation_source": "true PHC R_Hand quaternion from rollout diagnostic" if hand_states else "geometric frame from R_Elbow, R_Wrist, R_Hand positions; not simulator quaternion",
        "quaternion_convention": "xyzw",
        "local_tip_offset": all_offsets.mean(axis=0).tolist(),
        "local_tip_offset_std": all_offsets.std(axis=0).tolist(),
        "calibration_passed": all_pass,
        "pass_criteria": {
            "mean_tip_error_m": args.pass_mean_threshold,
            "p90_tip_error_m": args.pass_p90_threshold,
        },
        "note": "This transform is valid only if reference-only coordinate validation passed.",
    }

    args.output_transform.parent.mkdir(parents=True, exist_ok=True)
    args.output_transform.write_text(json.dumps(transform, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Wrist-To-Racket Transform Calibration",
        "",
        f"- Attached body: `R_Hand`",
        f"- Orientation source: `{transform['orientation_source']}`.",
        f"- Calibration passed: {all_pass}",
        "",
        "| validation clip | mean | p90 | max | passed |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['validation_clip']}` | {row['mean_tip_error']:.6f} | {row['p90_tip_error']:.6f} | {row['max_tip_error']:.6f} | {row['passed']} |"
        )
    if not all_pass:
        lines += [
            "",
            "Calibration failed. The fixed hand-to-racket transform does not reconstruct the reference racket tip well enough, so rollout racket accuracy should not be claimed.",
        ]
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"calibration_passed": all_pass, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
