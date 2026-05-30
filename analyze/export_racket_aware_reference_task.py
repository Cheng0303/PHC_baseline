#!/usr/bin/env python3
"""Export racket-aware per-sequence reference task targets.

This script only packages validated reference data. It does not simulate PHC
rollouts and does not assume a fixed passive racket attachment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_npz(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def scalar_string(value: np.ndarray) -> str:
    return str(value.item() if hasattr(value, "item") else value)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)


def invert_transform(mats: np.ndarray) -> np.ndarray:
    inv = np.zeros_like(mats)
    rot = mats[:, :3, :3]
    trans = mats[:, :3, 3]
    inv[:, :3, :3] = np.swapaxes(rot, 1, 2)
    inv[:, :3, 3] = -np.einsum("tij,tj->ti", inv[:, :3, :3], trans)
    inv[:, 3, 3] = 1.0
    return inv


def apply_inverse_to_points(mats: np.ndarray, points: np.ndarray) -> np.ndarray:
    rot_t = np.swapaxes(mats[:, :3, :3], 1, 2)
    return np.einsum("tij,tj->ti", rot_t, points - mats[:, :3, 3])


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


def load_ref_root_rotations(diagnostic_json: Path, sequence: str) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(diagnostic_json.read_text(encoding="utf-8"))
    steps = []
    quats = []
    for rec in payload["records"]:
        keys = rec.get("motion_keys", [])
        if keys and keys[0] != sequence:
            continue
        if "ref_body_rot" not in rec:
            raise KeyError(f"{diagnostic_json} record has no ref_body_rot")
        steps.append(int(rec["step"]))
        quats.append(np.asarray(rec["ref_body_rot"][0][0], dtype=np.float64))
    if not steps:
        raise ValueError(f"no records for {sequence} in {diagnostic_json}")
    return np.asarray(steps, dtype=np.int64), np.stack(quats)


def index_positions(frame_idx: np.ndarray) -> dict[int, int]:
    return {int(frame): i for i, frame in enumerate(frame_idx.tolist())}


def align_indices(reference_frames: np.ndarray, source_frames: np.ndarray, extra_frames: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    src_pos = index_positions(source_frames)
    extra_pos = index_positions(extra_frames) if extra_frames is not None else None
    ref_keep = []
    src_keep = []
    extra_keep = []
    out_frames = []
    for i, frame in enumerate(reference_frames.tolist()):
        frame_i = int(frame)
        if frame_i not in src_pos:
            continue
        if extra_pos is not None and frame_i not in extra_pos:
            continue
        ref_keep.append(i)
        src_keep.append(src_pos[frame_i])
        if extra_pos is not None:
            extra_keep.append(extra_pos[frame_i])
        out_frames.append(frame_i)
    if not ref_keep:
        raise ValueError("no overlapping source_frame_idx values across inputs")
    return (
        np.asarray(ref_keep, dtype=np.int64),
        np.asarray(src_keep, dtype=np.int64),
        np.asarray(extra_keep, dtype=np.int64) if extra_pos is not None else None,
        np.asarray(out_frames, dtype=np.int32),
    )


def local_deviation_stats(points: np.ndarray) -> dict[str, float]:
    mean = points.mean(axis=0)
    dev = np.linalg.norm(points - mean[None, :], axis=1)
    return {
        "mean": float(dev.mean()),
        "p90": float(np.percentile(dev, 90)),
        "max": float(dev.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--reference_geometry_npz", required=True, type=Path)
    parser.add_argument("--source_attachment_npz", required=True, type=Path)
    parser.add_argument("--source_geometry_npz", type=Path, default=None)
    parser.add_argument("--diagnostic_json", type=Path, default=None)
    parser.add_argument("--output_npz", required=True, type=Path)
    parser.add_argument("--output_summary_json", required=True, type=Path)
    args = parser.parse_args()

    ref = load_npz(args.reference_geometry_npz)
    attach = load_npz(args.source_attachment_npz)
    source_geom = load_npz(args.source_geometry_npz) if args.source_geometry_npz else {}

    seq_ref = scalar_string(ref["sequence"])
    seq_attach = scalar_string(attach["sequence"])
    if seq_ref != args.sequence or seq_attach != args.sequence:
        raise ValueError(f"sequence mismatch: cli={args.sequence}, ref={seq_ref}, source_attachment={seq_attach}")

    ref_frames = np.asarray(ref["source_frame_idx"], dtype=np.int64)
    attach_frames = np.asarray(attach["source_frame_idx"], dtype=np.int64)
    diag_frames = None
    root_quat = None
    if args.diagnostic_json:
        diag_frames, root_quat_all = load_ref_root_rotations(args.diagnostic_json, args.sequence)
    ref_i, attach_i, diag_i, source_frame_idx = align_indices(ref_frames, attach_frames, diag_frames)

    reference_body_pos = np.asarray(ref["reference_body_pos"], dtype=np.float64)[ref_i]
    root_position = reference_body_pos[:, 0]
    if args.diagnostic_json and diag_i is not None:
        root_quat = root_quat_all[diag_i]
        root_rot = quat_xyzw_to_matrix(root_quat)
        root_rotation_source = "PHC reference ref_body_rot root quaternion from diagnostic JSON, xyzw"
    else:
        root_rot = np.tile(np.eye(3, dtype=np.float64), (len(source_frame_idx), 1, 1))
        root_quat = np.empty((0, 4), dtype=np.float64)
        root_rotation_source = "unavailable; root-local fields use translation-only identity rotation"

    handle_phc = np.asarray(ref["reference_racket_anchor_phc_world"], dtype=np.float64)[ref_i]
    tip_phc = np.asarray(ref["reference_racket_tip_phc_world"], dtype=np.float64)[ref_i]
    head_phc = np.asarray(ref["reference_racket_head_center_phc_world"], dtype=np.float64)[ref_i]
    long_axis_phc = normalize(tip_phc - handle_phc)

    root_rot_t = np.swapaxes(root_rot, 1, 2)
    handle_root_local = np.einsum("tij,tj->ti", root_rot_t, handle_phc - root_position)
    tip_root_local = np.einsum("tij,tj->ti", root_rot_t, tip_phc - root_position)
    long_axis_root_local = np.einsum("tij,tj->ti", root_rot_t, long_axis_phc)
    long_axis_root_local = normalize(long_axis_root_local)

    T_hand = np.asarray(attach["T_source_hand"], dtype=np.float64)[attach_i]
    T_racket = np.asarray(attach["T_source_racket"], dtype=np.float64)[attach_i]
    dynamic = np.asarray(attach["T_hand_to_racket"], dtype=np.float64)[attach_i]
    source_tip = np.asarray(attach["source_racket_tip_world"], dtype=np.float64)[attach_i]
    source_handle = np.asarray(attach["source_racket_anchor_world"], dtype=np.float64)[attach_i]
    source_head = np.asarray(attach["source_racket_head_center_world"], dtype=np.float64)[attach_i]
    racket_pose = np.asarray(attach["racket_pose_parameter"], dtype=np.float64)[attach_i]
    tip_hand = np.asarray(attach["tip_in_hand_frame"], dtype=np.float64)[attach_i]
    handle_hand = np.asarray(attach["anchor_in_hand_frame"], dtype=np.float64)[attach_i]
    head_hand = np.asarray(attach["head_center_in_hand_frame"], dtype=np.float64)[attach_i]

    inv_racket = invert_transform(T_racket)
    handle_racket_frame = apply_inverse_to_points(T_racket, source_handle)
    tip_racket_frame = apply_inverse_to_points(T_racket, source_tip)
    head_racket_frame = apply_inverse_to_points(T_racket, source_head)

    source_mask = np.ones(len(source_frame_idx), dtype=bool)
    if source_geom:
        geom_frames = np.asarray(source_geom["source_frame_idx"], dtype=np.int64)
        geom_pos = index_positions(geom_frames)
        source_mask_all = np.asarray(source_geom.get("source_mask", np.ones(len(geom_frames), dtype=bool)), dtype=bool)
        source_mask = np.asarray([source_mask_all[geom_pos[int(frame)]] if int(frame) in geom_pos else False for frame in source_frame_idx], dtype=bool)

    length = np.linalg.norm(tip_phc - handle_phc, axis=1)
    rel_rot_dev = np.asarray(attach["relative_rotation_angle_deviation_deg"], dtype=np.float64)[attach_i]
    tip_dev_stats = local_deviation_stats(tip_hand)

    fields = {
        "sequence": np.asarray(args.sequence),
        "source_frame_idx": source_frame_idx,
        "valid_mask": source_mask,
        "reference_body_pos": reference_body_pos.astype(np.float32),
        "racket_pose_parameter": racket_pose.astype(np.float32),
        "racket_handle_phc_world": handle_phc.astype(np.float32),
        "racket_tip_phc_world": tip_phc.astype(np.float32),
        "racket_head_center_phc_world": head_phc.astype(np.float32),
        "racket_long_axis_phc_world": long_axis_phc.astype(np.float32),
        "root_position_phc_world": root_position.astype(np.float32),
        "root_rotation_phc_world_matrix": root_rot.astype(np.float32),
        "root_rotation_phc_world_quat_xyzw": root_quat.astype(np.float32),
        "racket_handle_root_local": handle_root_local.astype(np.float32),
        "racket_tip_root_local": tip_root_local.astype(np.float32),
        "racket_long_axis_root_local": long_axis_root_local.astype(np.float32),
        "source_anatomical_rhand_world": T_hand.astype(np.float32),
        "source_racket_transform_world": T_racket.astype(np.float32),
        "dynamic_hand_to_racket_transform": dynamic.astype(np.float32),
        "tip_in_hand_frame": tip_hand.astype(np.float32),
        "handle_in_hand_frame": handle_hand.astype(np.float32),
        "head_center_in_hand_frame": head_hand.astype(np.float32),
        "source_racket_handle_world": source_handle.astype(np.float32),
        "source_racket_tip_world": source_tip.astype(np.float32),
        "source_racket_head_center_world": source_head.astype(np.float32),
        "racket_handle_in_racket_frame": handle_racket_frame.astype(np.float32),
        "racket_tip_in_racket_frame": tip_racket_frame.astype(np.float32),
        "racket_head_center_in_racket_frame": head_racket_frame.astype(np.float32),
    }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **fields)

    warnings = []
    if root_quat.size == 0:
        warnings.append("root_rotation_phc_world unavailable; root-local fields used identity rotation and should be treated as translation-local only.")
    warnings.append("Full racket face orientation is not inferred from handle/tip/head markers; only long-axis targets are marker-derived.")
    warnings.append("dynamic_hand_to_racket_transform is a time-varying diagnostic/reference signal, not a fixed passive attachment.")

    summary = {
        "sequence": args.sequence,
        "frame_count": int(len(source_frame_idx)),
        "source_files_used": {
            "reference_geometry_npz": str(args.reference_geometry_npz),
            "source_attachment_npz": str(args.source_attachment_npz),
            "source_geometry_npz": str(args.source_geometry_npz) if args.source_geometry_npz else None,
            "diagnostic_json": str(args.diagnostic_json) if args.diagnostic_json else None,
        },
        "coordinate_transform_mode": "corrected_source_geometry_to_phc_reference; dynamic attachment remains in custom SMPL source_world",
        "root_rotation_source": root_rotation_source,
        "racket_length_mean": float(length.mean()),
        "racket_length_std": float(length.std()),
        "dynamic_attachment": {
            "tip_local_deviation_mean": tip_dev_stats["mean"],
            "tip_local_deviation_p90": tip_dev_stats["p90"],
            "tip_local_deviation_max": tip_dev_stats["max"],
            "relative_rotation_deviation_mean_deg": float(rel_rot_dev.mean()),
            "relative_rotation_deviation_p90_deg": float(np.percentile(rel_rot_dev, 90)),
            "fixed_passive_attachment_plausible": False,
        },
        "fields_written": sorted(fields.keys()),
        "warnings": warnings,
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
