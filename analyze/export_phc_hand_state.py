#!/usr/bin/env python3
"""Export PHC rollout/reference right hand state from diagnostic JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar",
    "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist",
    "R_Wrist", "L_Hand", "R_Hand",
]


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-8)


def geometric_frames(body: np.ndarray) -> np.ndarray:
    hand = body[:, 23]
    wrist = body[:, 21]
    elbow = body[:, 19]
    x_axis = normalize(hand - wrist)
    y_seed = normalize(wrist - elbow)
    z_axis = normalize(np.cross(x_axis, y_seed))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=-1)


def load_records(path: Path, sequence: str) -> tuple[np.ndarray, np.ndarray, list[int], np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = []
    ref = []
    steps = []
    mpjpe = []
    root = []
    body_rot = []
    ref_body_rot = []
    for rec in payload["records"]:
        keys = rec.get("motion_keys", [])
        if keys and keys[0] != sequence:
            continue
        body.append(np.asarray(rec["body_pos"][0], dtype=np.float64))
        ref.append(np.asarray(rec["ref_body_pos"][0], dtype=np.float64))
        steps.append(int(rec["step"]))
        mpjpe.append(float(rec["mpjpe"][0]))
        root.append(float(rec["root_error"][0]))
        if "body_rot" in rec:
            body_rot.append(np.asarray(rec["body_rot"][0], dtype=np.float64))
        if "ref_body_rot" in rec:
            ref_body_rot.append(np.asarray(rec["ref_body_rot"][0], dtype=np.float64))
    if not body:
        raise ValueError(f"no records for {sequence} in {path}")
    body_rot_arr = np.stack(body_rot) if len(body_rot) == len(body) else None
    ref_body_rot_arr = np.stack(ref_body_rot) if len(ref_body_rot) == len(body) else None
    return np.stack(body), np.stack(ref), steps, np.asarray(mpjpe), np.asarray(root), body_rot_arr, ref_body_rot_arr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--completed", required=True)
    parser.add_argument("--termination_frame", default="")
    parser.add_argument("--output_npz", required=True, type=Path)
    parser.add_argument("--output_summary", required=True, type=Path)
    args = parser.parse_args()

    body, ref, steps, mpjpe, root, body_rot, ref_body_rot = load_records(args.diagnostic, args.sequence)
    rollout_hand_pos = body[:, 23]
    reference_hand_pos = ref[:, 23]
    rollout_wrist_pos = body[:, 21]
    reference_wrist_pos = ref[:, 21]
    rollout_geom = geometric_frames(body)
    reference_geom = geometric_frames(ref)

    true_rollout_quat = body_rot[:, 23] if body_rot is not None else None
    true_reference_quat = ref_body_rot[:, 23] if ref_body_rot is not None else None

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        sequence=np.asarray(args.sequence),
        frame_idx=np.asarray(steps, dtype=np.int32),
        right_hand_body_name=np.asarray("R_Hand"),
        right_wrist_body_name=np.asarray("R_Wrist"),
        rollout_hand_position_world=rollout_hand_pos.astype(np.float32),
        reference_hand_position_world=reference_hand_pos.astype(np.float32),
        rollout_wrist_position_world=rollout_wrist_pos.astype(np.float32),
        reference_wrist_position_world=reference_wrist_pos.astype(np.float32),
        rollout_hand_orientation_world_matrix_proxy=rollout_geom.astype(np.float32),
        reference_hand_orientation_world_matrix_proxy=reference_geom.astype(np.float32),
        rollout_hand_orientation_world_quat=true_rollout_quat.astype(np.float32) if true_rollout_quat is not None else np.empty((0, 4), dtype=np.float32),
        reference_hand_orientation_world_quat=true_reference_quat.astype(np.float32) if true_reference_quat is not None else np.empty((0, 4), dtype=np.float32),
        body_mpjpe=mpjpe.astype(np.float32),
        root_error=root.astype(np.float32),
    )
    summary = {
        "sequence": args.sequence,
        "frame_count": len(steps),
        "right_hand_body_name": "R_Hand",
        "right_wrist_body_name": "R_Wrist",
        "completed": args.completed.lower() == "true",
        "termination_frame": int(args.termination_frame) if args.termination_frame else None,
        "has_simulator_hand_quaternion": true_rollout_quat is not None,
        "has_reference_hand_quaternion": true_reference_quat is not None,
        "orientation_export_note": "Simulator/reference quaternions are exported when diagnostic JSON contains body_rot/ref_body_rot. Otherwise only a geometric frame proxy from R_Elbow/R_Wrist/R_Hand positions is exported.",
    }
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
