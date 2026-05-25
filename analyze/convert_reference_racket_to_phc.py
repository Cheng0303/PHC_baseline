#!/usr/bin/env python3
"""Convert reconstructed source racket geometry into PHC reference coordinates.

The official path for calibrated racket validation is
``traced_source_geometry_to_phc``. Older NPZ-racket-tip modes are retained only
as debug candidates because the source NPZ ``racket_tip`` was previously found
to be an alias of ``racket_pose``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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

SOURCE_FRAME_FIELDS = (
    "source_frame_idx",
    "motion_frame_idx",
    "motion_frame",
    "ref_idx",
    "frame_idx",
)


def safe_name(sequence: str) -> str:
    return sequence.replace("/", "_")


def source_npz(dataset_root: Path, sequence: str) -> Path:
    return dataset_root / f"{sequence}.npz"


def load_records(diagnostic: Path, sequence: str) -> list[dict[str, Any]]:
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    records = []
    for rec in payload["records"]:
        keys = rec.get("motion_keys", [])
        if keys and keys[0] != sequence:
            continue
        records.append(rec)
    if not records:
        raise ValueError(f"no diagnostic records for {sequence} in {diagnostic}")
    return records


def extract_source_frame_indices(records: list[dict[str, Any]], allow_step: bool) -> tuple[np.ndarray, str]:
    for field in SOURCE_FRAME_FIELDS:
        if all(field in rec for rec in records):
            return np.asarray([int(rec[field]) for rec in records], dtype=np.int64), field
    if allow_step and all("step" in rec for rec in records):
        return np.asarray([int(rec["step"]) for rec in records], dtype=np.int64), "step_explicitly_allowed_single_motion"
    fields = sorted({key for rec in records[:10] for key in rec.keys()})
    raise ValueError(
        "diagnostic records do not contain a source-frame mapping field. "
        f"Checked {SOURCE_FRAME_FIELDS}; available fields include {fields}. "
        "Re-run rollout export with source_frame_idx/motion_frame_idx, or pass "
        "--allow_step_as_source_idx_for_single_motion only for a known single-motion forced rollout."
    )


def load_ref_body(records: list[dict[str, Any]]) -> np.ndarray:
    return np.stack([np.asarray(rec["ref_body_pos"][0], dtype=np.float64) for rec in records])


def load_debug_raw_tip(dataset_root: Path, sequence: str) -> np.ndarray:
    source = source_npz(dataset_root, sequence)
    raw = np.load(source, allow_pickle=True)
    if "racket_tip" not in raw:
        raise KeyError(f"{source} has no racket_tip key")
    return np.asarray(raw["racket_tip"], dtype=np.float64)


def convert_debug_tip(raw_tip: np.ndarray, entry: dict[str, Any], mode: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    trans_orig = np.asarray(entry["trans_orig"], dtype=np.float64)
    ground_fix = entry.get("ground_fix", {})
    ground_offset = float(ground_fix.get("applied_vertical_offset", 0.0))

    if mode == "debug_world_plus_groundfix":
        tip = raw_tip.astype(np.float64).copy()
        tip[:, 2] += ground_offset
        formula = "tip_phc = raw_racket_tip + [0, 0, groundfix_z]"
    elif mode == "debug_local_plus_root_trans_offset":
        tip = raw_tip.astype(np.float64) + root[: len(raw_tip)]
        formula = "tip_phc = raw_racket_tip + converted_root_trans_offset"
    elif mode == "debug_local_plus_trans_orig_plus_groundfix":
        tip = raw_tip.astype(np.float64) + trans_orig[: len(raw_tip)]
        tip[:, 2] += ground_offset
        formula = "tip_phc = raw_racket_tip + trans_orig + [0, 0, groundfix_z]"
    else:
        raise ValueError(f"unknown debug mode: {mode}")

    transform = {
        "mode": mode,
        "status": "debug_only_not_for_calibrated_accuracy",
        "formula": formula,
        "groundfix_z": ground_offset,
        "warning": "raw NPZ racket_tip is known to have been exported as a racket_pose alias in this dataset.",
    }
    return {"tip": tip, "anchor": np.full_like(tip, np.nan), "head_center": np.full_like(tip, np.nan)}, transform


def convert_source_geometry(source_geometry: Path, entry: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray]:
    source = np.load(source_geometry, allow_pickle=True)
    required = [
        "source_frame_idx",
        "source_racket_tip_world",
        "source_racket_head_center_world",
    ]
    missing = [key for key in required if key not in source]
    if missing:
        raise KeyError(f"{source_geometry} is missing required source geometry arrays: {missing}")

    source_idx = np.asarray(source["source_frame_idx"], dtype=np.int64)
    root = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    trans_orig = np.asarray(entry["trans_orig"], dtype=np.float64)
    n = min(len(root), len(trans_orig))
    root_delta = root[:n] - trans_orig[:n]

    def apply_delta(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if len(points) > len(root_delta):
            raise ValueError(f"source geometry has {len(points)} frames but PHC root delta has {len(root_delta)}")
        return points + root_delta[: len(points)]

    geom = {
        "anchor": apply_delta(source["source_racket_handle_anchor_world"] if "source_racket_handle_anchor_world" in source else source["source_racket_anchor_world"]),
        "joint24": apply_delta(source["source_racket_anchor_world"]) if "source_racket_anchor_world" in source else np.full_like(apply_delta(source["source_racket_tip_world"]), np.nan),
        "tip": apply_delta(source["source_racket_tip_world"]),
        "head_center": apply_delta(source["source_racket_head_center_world"]),
    }
    transform = {
        "mode": "traced_source_geometry_to_phc",
        "formula": "point_phc[t] = source_world_point[t] + (root_trans_offset[t] - trans_orig[t])",
        "root_delta_min": root_delta.min(axis=0).tolist(),
        "root_delta_max": root_delta.max(axis=0).tolist(),
        "groundfix_applied": entry.get("ground_fix", {}),
        "notes": (
            "This applies the same per-frame root translation delta observed in the PHC converted motion. "
            "The traced body converter changes pose rotations for upright_start but does not rotate root translations."
        ),
    }
    return geom, transform, source_idx


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


def render_reference_video(
    output: Path,
    sequence: str,
    ref_body: np.ndarray,
    anchor: np.ndarray,
    tip: np.ndarray,
    head_center: np.ndarray,
    source_idx: np.ndarray,
    stride: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ref_body = ref_body[::stride]
    anchor = anchor[::stride]
    tip = tip[::stride]
    head_center = head_center[::stride]
    source_idx = source_idx[::stride]
    pts = np.concatenate([ref_body, anchor[:, None, :], tip[:, None, :], head_center[:, None, :]], axis=1)
    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=30, bitrate=2800)
    trail = []
    with writer.saving(fig, str(output), dpi=140):
        for i in range(len(ref_body)):
            ax.clear()
            set_equal_axes(ax, pts)
            ax.view_init(elev=16, azim=-68)
            draw_skeleton(ax, ref_body[i])
            trail.append(tip[i])
            trail_arr = np.asarray(trail[-45:])
            ax.plot(trail_arr[:, 0], trail_arr[:, 1], trail_arr[:, 2], color="#00a8ff", linewidth=2.0, label="tip trail")
            ax.plot(
                [anchor[i, 0], head_center[i, 0], tip[i, 0]],
                [anchor[i, 1], head_center[i, 1], tip[i, 1]],
                [anchor[i, 2], head_center[i, 2], tip[i, 2]],
                color="#20bf6b",
                linewidth=3.0,
                label="racket shaft/head marker",
            )
            ax.scatter([anchor[i, 0]], [anchor[i, 1]], [anchor[i, 2]], color="#fed330", s=38, label="anchor")
            ax.scatter([head_center[i, 0]], [head_center[i, 1]], [head_center[i, 2]], color="#20bf6b", s=38, label="head center")
            ax.scatter([tip[i, 0]], [tip[i, 1]], [tip[i, 2]], color="#00a8ff", s=42, label="tip marker")
            ax.set_title(
                f"{sequence}\nmode = reference_only_corrected_geometry | source frame {int(source_idx[i])}",
                fontsize=10,
            )
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:5], labels[:5], loc="upper right")
            writer.grab_frame()
    plt.close(fig)


def summarize_aligned(
    args: argparse.Namespace,
    transform: dict[str, Any],
    ref_body: np.ndarray,
    anchor: np.ndarray,
    tip: np.ndarray,
    head_center: np.ndarray,
    source_indices: np.ndarray,
    source_field: str,
    source_geom_frames: int,
    diagnostic_count: int,
    dropped: int,
    entry: dict[str, Any],
) -> dict[str, Any]:
    right_hand_dist = np.linalg.norm(anchor - ref_body[:, 23], axis=1)
    right_wrist_dist = np.linalg.norm(anchor - ref_body[:, 21], axis=1)
    tip_to_anchor = np.linalg.norm(tip - anchor, axis=1)
    head_to_anchor = np.linalg.norm(head_center - anchor, axis=1)
    validation_pass = bool(
        np.nanmean(right_hand_dist) < 0.35
        and np.nanpercentile(right_hand_dist, 90) < 0.55
        and np.nanstd(tip_to_anchor) < 0.05
        and np.nanmean(tip_to_anchor) > 0.25
    )
    return {
        "sequence": args.sequence,
        "frame_count": int(len(ref_body)),
        "diagnostic_path": str(args.diagnostic),
        "converted_motion": str(args.converted_motion),
        "source_geometry_npz": str(args.source_geometry_npz) if args.source_geometry_npz else None,
        "PHC_coordinate_transform_used": transform,
        "groundfix_applied": entry.get("ground_fix", {}),
        "diagnostic_source_frame_field_used": source_field,
        "first_10_source_indices": source_indices[:10].astype(int).tolist(),
        "last_10_source_indices": source_indices[-10:].astype(int).tolist(),
        "source_geometry_frame_count": int(source_geom_frames),
        "diagnostic_record_count": int(diagnostic_count),
        "dropped_invalid_frames": int(dropped),
        "anchor_min": anchor.min(axis=0).tolist(),
        "anchor_max": anchor.max(axis=0).tolist(),
        "tip_min": tip.min(axis=0).tolist(),
        "tip_max": tip.max(axis=0).tolist(),
        "anchor_to_right_hand_distance_mean": float(np.mean(right_hand_dist)),
        "anchor_to_right_hand_distance_p90": float(np.percentile(right_hand_dist, 90)),
        "anchor_to_right_hand_distance_max": float(np.max(right_hand_dist)),
        "anchor_to_right_wrist_distance_mean": float(np.mean(right_wrist_dist)),
        "tip_to_anchor_length_mean": float(np.mean(tip_to_anchor)),
        "tip_to_anchor_length_std": float(np.std(tip_to_anchor)),
        "tip_to_anchor_length_p90": float(np.percentile(tip_to_anchor, 90)),
        "head_center_to_anchor_length_mean": float(np.mean(head_to_anchor)),
        "anchor_definition": "source_racket_handle_anchor_world transformed from OBJ local [0,0,0] when available; source_racket_anchor_world/joint24 is retained separately and is not assumed to be the handle.",
        "validation_status": "passed" if validation_pass else "failed",
        "validation_reason": (
            "Anchor stays near PHC reference right hand and tip-to-anchor racket length is stable."
            if validation_pass
            else "Anchor/right-hand plausibility or tip-to-anchor length stability failed; do not use for official racket accuracy."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--converted_motion", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--source_geometry_npz", type=Path, default=None)
    parser.add_argument("--output_npz", required=True, type=Path)
    parser.add_argument("--output_summary", required=True, type=Path)
    parser.add_argument("--output_video", type=Path, default=None)
    parser.add_argument(
        "--mode",
        default="traced_source_geometry_to_phc",
        choices=[
            "traced_source_geometry_to_phc",
            "debug_world_plus_groundfix",
            "debug_local_plus_root_trans_offset",
            "debug_local_plus_trans_orig_plus_groundfix",
        ],
    )
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--allow_step_as_source_idx_for_single_motion", action="store_true")
    args = parser.parse_args()

    motion = joblib.load(args.converted_motion)
    if args.sequence not in motion:
        raise KeyError(args.sequence)
    entry = motion[args.sequence]
    records = load_records(args.diagnostic, args.sequence)
    ref_body_all = load_ref_body(records)
    diagnostic_source_idx, source_field = extract_source_frame_indices(
        records,
        allow_step=args.allow_step_as_source_idx_for_single_motion,
    )

    if args.mode == "traced_source_geometry_to_phc":
        if args.source_geometry_npz is None:
            raise ValueError("--source_geometry_npz is required for traced_source_geometry_to_phc")
        geom, transform, source_geom_idx = convert_source_geometry(args.source_geometry_npz, entry)
    else:
        if args.dataset_root is None:
            raise ValueError("--dataset_root is required for debug NPZ modes")
        raw_tip = load_debug_raw_tip(args.dataset_root, args.sequence)
        geom, transform = convert_debug_tip(raw_tip, entry, args.mode)
        source_geom_idx = np.arange(len(geom["tip"]), dtype=np.int64)

    idx_to_pos = {int(frame): pos for pos, frame in enumerate(source_geom_idx.tolist())}
    source_pos = np.asarray([idx_to_pos.get(int(frame), -1) for frame in diagnostic_source_idx], dtype=np.int64)
    valid = source_pos >= 0
    if not np.any(valid):
        raise ValueError("no diagnostic source frame indices overlap with source geometry frames")

    ref_body = ref_body_all[valid]
    source_indices = diagnostic_source_idx[valid]
    source_pos = source_pos[valid]
    anchor = geom["anchor"][source_pos]
    joint24 = geom["joint24"][source_pos]
    tip = geom["tip"][source_pos]
    head_center = geom["head_center"][source_pos]

    summary = summarize_aligned(
        args=args,
        transform=transform,
        ref_body=ref_body,
        anchor=anchor,
        tip=tip,
        head_center=head_center,
        source_indices=source_indices,
        source_field=source_field,
        source_geom_frames=len(source_geom_idx),
        diagnostic_count=len(records),
        dropped=int((~valid).sum()),
        entry=entry,
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        sequence=np.asarray(args.sequence),
        source_frame_idx=source_indices.astype(np.int32),
        reference_racket_anchor_phc_world=anchor.astype(np.float32),
        reference_racket_joint24_phc_world=joint24.astype(np.float32),
        reference_racket_tip_phc_world=tip.astype(np.float32),
        reference_racket_head_center_phc_world=head_center.astype(np.float32),
        reference_body_pos=ref_body.astype(np.float32),
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_video:
        render_reference_video(args.output_video, args.sequence, ref_body, anchor, tip, head_center, source_indices, args.stride)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
