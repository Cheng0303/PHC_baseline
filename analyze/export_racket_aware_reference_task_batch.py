#!/usr/bin/env python3
"""Batch-export racket-aware reference task NPZ files.

This script builds sequence-level reference targets only. It does not run PHC
rollouts, train a policy, or assume a fixed passive racket attachment.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation


REQUIRED_FIELDS = [
    "sequence",
    "source_frame_idx",
    "valid_mask",
    "reference_body_pos",
    "racket_pose_parameter",
    "racket_handle_phc_world",
    "racket_tip_phc_world",
    "racket_head_center_phc_world",
    "racket_long_axis_phc_world",
    "root_position_phc_world",
    "root_rotation_phc_world_matrix",
    "root_rotation_phc_world_quat_xyzw",
    "racket_handle_root_local",
    "racket_tip_root_local",
    "racket_long_axis_root_local",
    "source_anatomical_rhand_world",
    "source_racket_transform_world",
    "dynamic_hand_to_racket_transform",
    "tip_in_hand_frame",
    "handle_in_hand_frame",
    "head_center_in_hand_frame",
    "source_racket_handle_world",
    "source_racket_tip_world",
    "source_racket_head_center_world",
    "racket_handle_in_racket_frame",
    "racket_tip_in_racket_frame",
    "racket_head_center_in_racket_frame",
]


def safe_name(sequence: str) -> str:
    return sequence.replace("/", "_")


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


def transform_markers(local_points: np.ndarray, canonical_wrist: np.ndarray, mats: np.ndarray) -> np.ndarray:
    points = local_points[None, :, :] + canonical_wrist[:, None, :]
    rot = mats[:, :3, :3]
    trans = mats[:, :3, 3][:, None, :]
    return np.einsum("tij,tnj->tni", rot, points) + trans


def apply_newracket_global_rotation(
    pose: np.ndarray,
    trans: np.ndarray,
    racket_pose: np.ndarray,
    axis: str = "x",
    degrees: float = -90.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror humenv/data_preparation/convert_new_racket_to_amass_npz.py."""
    rot = Rotation.from_euler(axis, degrees, degrees=True)
    pose_out = pose.copy()
    pose_out[:, :3] = (rot * Rotation.from_rotvec(pose[:, :3])).as_rotvec()
    return pose_out, rot.apply(trans), rot.apply(racket_pose)


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize(q)
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


def load_custom_smpl(root: Path):
    resolved = root.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from submodules import smplx  # type: ignore

    return smplx


def load_phc_skeleton(phc_root: Path):
    resolved = phc_root.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree  # type: ignore

    mjcf = resolved / "egoquest/data/assets/mjcf/smpl_humanoid_1.xml"
    return SkeletonState, SkeletonTree.from_mjcf(str(mjcf))


def completed_sequences(metrics_csv: Path) -> list[str]:
    rows = []
    with metrics_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq = row.get("sequence") or row.get("sequence_name") or row.get("motion_key") or row.get("key")
            completed = str(row.get("completed", row.get("complete", ""))).lower() == "true"
            valid = str(row.get("valid_export", "true")).lower() != "false"
            if seq and completed and valid:
                rows.append(seq)
    return sorted(set(rows))


def boolish(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def integrity_check(path: Path) -> tuple[bool, str, dict[str, float | int]]:
    data = np.load(path, allow_pickle=True)
    missing = [field for field in REQUIRED_FIELDS if field not in data.files]
    if missing:
        return False, f"missing fields: {missing}", {}
    frame_count = int(data["source_frame_idx"].shape[0])
    if frame_count < 10:
        return False, f"frame count too small: {frame_count}", {"frame_count": frame_count}
    for field in REQUIRED_FIELDS:
        arr = data[field]
        if arr.dtype.kind in "fiu" and not np.all(np.isfinite(arr)):
            return False, f"non-finite values in {field}", {"frame_count": frame_count}
        if hasattr(arr, "shape") and arr.shape and field != "sequence" and arr.shape[0] != frame_count:
            return False, f"frame length mismatch in {field}: {arr.shape[0]} vs {frame_count}", {"frame_count": frame_count}
    axis_world = np.linalg.norm(data["racket_long_axis_phc_world"], axis=1)
    axis_local = np.linalg.norm(data["racket_long_axis_root_local"], axis=1)
    if not np.allclose(axis_world, 1.0, atol=1e-3):
        return False, "racket_long_axis_phc_world norm is not approximately 1", {"frame_count": frame_count}
    if not np.allclose(axis_local, 1.0, atol=1e-3):
        return False, "racket_long_axis_root_local norm is not approximately 1", {"frame_count": frame_count}
    length = np.linalg.norm(data["racket_tip_phc_world"] - data["racket_handle_phc_world"], axis=1)
    if abs(float(length.mean()) - 0.672026) > 0.02:
        return False, f"racket length mean unexpected: {length.mean()}", {"frame_count": frame_count}
    if float(length.std()) > 1e-3:
        return False, f"racket length is not stable: std={length.std()}", {"frame_count": frame_count}
    return True, "", {
        "frame_count": frame_count,
        "racket_length_mean": float(length.mean()),
        "racket_length_std": float(length.std()),
        "long_axis_norm_mean": float(axis_local.mean()),
    }


def replay_check(path: Path) -> tuple[bool, dict[str, float]]:
    data = np.load(path, allow_pickle=True)
    hand = np.asarray(data["source_anatomical_rhand_world"], dtype=np.float64)
    dyn = np.asarray(data["dynamic_hand_to_racket_transform"], dtype=np.float64)
    replay = np.matmul(hand, dyn)
    marker_local = np.stack(
        [
            data["racket_handle_in_racket_frame"],
            data["racket_tip_in_racket_frame"],
            data["racket_head_center_in_racket_frame"],
        ],
        axis=1,
    ).astype(np.float64)
    rot = replay[:, :3, :3]
    trans = replay[:, :3, 3][:, None, :]
    pred = np.einsum("tij,tnj->tni", rot, marker_local) + trans
    ref = np.stack(
        [
            data["source_racket_handle_world"],
            data["source_racket_tip_world"],
            data["source_racket_head_center_world"],
        ],
        axis=1,
    ).astype(np.float64)
    err = np.linalg.norm(pred - ref, axis=-1)
    pred_axis = normalize(pred[:, 1] - pred[:, 0])
    ref_axis = normalize(ref[:, 1] - ref[:, 0])
    axis_err = np.degrees(np.arccos(np.clip(np.sum(pred_axis * ref_axis, axis=1), -1.0, 1.0)))
    stats = {
        "handle_error_max_m": float(err[:, 0].max()),
        "tip_error_mean_m": float(err[:, 1].mean()),
        "tip_error_max_m": float(err[:, 1].max()),
        "head_center_error_max_m": float(err[:, 2].max()),
        "long_axis_angle_error_p90_deg": float(np.percentile(axis_err, 90)),
    }
    passed = bool(
        stats["tip_error_mean_m"] < 1e-5
        and stats["tip_error_max_m"] < 1e-4
        and stats["handle_error_max_m"] < 1e-4
        and stats["long_axis_angle_error_p90_deg"] < 1e-3
    )
    return passed, stats


def export_task_for_sequence(
    sequence: str,
    source_pth: Path,
    entry: dict[str, Any],
    smpl_model: Any,
    smpl_device: torch.device,
    skeleton_state_cls: Any,
    skeleton_tree: Any,
    marker_local: np.ndarray,
    output_npz: Path,
    output_summary_json: Path,
) -> None:
    checkpoint = torch.load(source_pth, map_location=smpl_device)
    trans_np = checkpoint["trans"].detach().cpu().numpy()
    pose_np = checkpoint["body_pose"].detach().cpu().numpy()
    racket_pose_np = checkpoint["racket_pose"].detach().cpu().numpy()
    mask_np = checkpoint.get("mask", torch.ones(len(trans_np), device=smpl_device)).detach().cpu().numpy().astype(bool)
    pose_np, trans_np, racket_pose_np = apply_newracket_global_rotation(pose_np, trans_np, racket_pose_np)
    valid_idx = np.where(mask_np)[0]
    pose_np = pose_np[valid_idx]
    trans_np = trans_np[valid_idx]
    racket_pose_np = racket_pose_np[valid_idx]
    total_frames = len(trans_np)
    if total_frames < 10:
        raise ValueError(f"too few valid source frames: {total_frames}")

    beta = checkpoint["beta"].detach().cpu().numpy()
    beta_t = torch.from_numpy(beta).expand(total_frames, -1).to(smpl_device)
    pose_t = torch.from_numpy(pose_np).to(smpl_device)
    trans_t = torch.from_numpy(trans_np).to(smpl_device)
    racket_pose_t = torch.from_numpy(racket_pose_np).to(smpl_device)

    with torch.no_grad():
        racket_smpl = smpl_model.forward(
            betas=beta_t,
            global_orient=pose_t[:, :3],
            transl=trans_t,
            body_pose=pose_t[:, 3:],
            racket_pose=racket_pose_t,
        )
        body_smpl = smpl_model.forward(
            betas=beta_t,
            global_orient=pose_t[:, :3],
            transl=trans_t,
            body_pose=pose_t[:, 3:],
        )
        canonical_smpl = smpl_model.forward(beta=beta_t)

    body_A = body_smpl.A.detach().cpu().numpy()
    racket_A = racket_smpl.A.detach().cpu().numpy()
    source_hand = body_A[:, 23]
    racket_transform = racket_A[:, -1]
    canonical_wrist = canonical_smpl.joints[:, 21, :].detach().cpu().numpy()
    marker_world = transform_markers(marker_local, canonical_wrist, racket_transform)
    source_handle, source_tip, source_head = marker_world[:, 0], marker_world[:, 1], marker_world[:, 2]

    pose_quat = torch.from_numpy(np.asarray(entry["pose_quat"], dtype=np.float32))
    root_trans = entry["root_trans_offset"].detach().cpu().float() if torch.is_tensor(entry["root_trans_offset"]) else torch.from_numpy(np.asarray(entry["root_trans_offset"], dtype=np.float32))
    phc_state = skeleton_state_cls.from_rotation_and_root_translation(skeleton_tree, pose_quat, root_trans, is_local=True)
    body_pos = phc_state.global_translation.detach().cpu().numpy().astype(np.float64)
    root_quat = phc_state.global_rotation[:, 0].detach().cpu().numpy().astype(np.float64)
    root_rot = quat_xyzw_to_matrix(root_quat)

    root_trans_np = root_trans.detach().cpu().numpy().astype(np.float64)
    trans_orig = np.asarray(entry["trans_orig"], dtype=np.float64)
    n = min(total_frames, len(root_trans_np), len(trans_orig), len(body_pos))
    if n < 10:
        raise ValueError(f"aligned frame count too small: {n}")
    source_frame_idx = valid_idx[:n].astype(np.int32)
    root_delta = root_trans_np[:n] - trans_orig[:n]
    handle_phc = source_handle[:n] + root_delta
    tip_phc = source_tip[:n] + root_delta
    head_phc = source_head[:n] + root_delta
    long_axis_phc = normalize(tip_phc - handle_phc)
    root_position = body_pos[:n, 0]
    root_rot = root_rot[:n]
    root_rot_t = np.swapaxes(root_rot, 1, 2)
    handle_root_local = np.einsum("tij,tj->ti", root_rot_t, handle_phc - root_position)
    tip_root_local = np.einsum("tij,tj->ti", root_rot_t, tip_phc - root_position)
    long_axis_root_local = normalize(np.einsum("tij,tj->ti", root_rot_t, long_axis_phc))

    source_hand = source_hand[:n]
    racket_transform = racket_transform[:n]
    dynamic = np.matmul(invert_transform(source_hand), racket_transform)
    tip_hand = apply_inverse_to_points(source_hand, source_tip[:n])
    handle_hand = apply_inverse_to_points(source_hand, source_handle[:n])
    head_hand = apply_inverse_to_points(source_hand, source_head[:n])
    handle_racket_frame = apply_inverse_to_points(racket_transform, source_handle[:n])
    tip_racket_frame = apply_inverse_to_points(racket_transform, source_tip[:n])
    head_racket_frame = apply_inverse_to_points(racket_transform, source_head[:n])

    fields = {
        "sequence": np.asarray(sequence),
        "source_frame_idx": source_frame_idx,
        "valid_mask": np.ones(n, dtype=bool),
        "reference_body_pos": body_pos[:n].astype(np.float32),
        "racket_pose_parameter": racket_pose_np[:n].astype(np.float32),
        "racket_handle_phc_world": handle_phc.astype(np.float32),
        "racket_tip_phc_world": tip_phc.astype(np.float32),
        "racket_head_center_phc_world": head_phc.astype(np.float32),
        "racket_long_axis_phc_world": long_axis_phc.astype(np.float32),
        "root_position_phc_world": root_position.astype(np.float32),
        "root_rotation_phc_world_matrix": root_rot.astype(np.float32),
        "root_rotation_phc_world_quat_xyzw": root_quat[:n].astype(np.float32),
        "racket_handle_root_local": handle_root_local.astype(np.float32),
        "racket_tip_root_local": tip_root_local.astype(np.float32),
        "racket_long_axis_root_local": long_axis_root_local.astype(np.float32),
        "source_anatomical_rhand_world": source_hand.astype(np.float32),
        "source_racket_transform_world": racket_transform.astype(np.float32),
        "dynamic_hand_to_racket_transform": dynamic.astype(np.float32),
        "tip_in_hand_frame": tip_hand.astype(np.float32),
        "handle_in_hand_frame": handle_hand.astype(np.float32),
        "head_center_in_hand_frame": head_hand.astype(np.float32),
        "source_racket_handle_world": source_handle[:n].astype(np.float32),
        "source_racket_tip_world": source_tip[:n].astype(np.float32),
        "source_racket_head_center_world": source_head[:n].astype(np.float32),
        "racket_handle_in_racket_frame": handle_racket_frame.astype(np.float32),
        "racket_tip_in_racket_frame": tip_racket_frame.astype(np.float32),
        "racket_head_center_in_racket_frame": head_racket_frame.astype(np.float32),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **fields)

    length = np.linalg.norm(tip_phc - handle_phc, axis=1)
    tip_dev = np.linalg.norm(tip_hand - tip_hand.mean(axis=0, keepdims=True), axis=1)
    summary = {
        "sequence": sequence,
        "frame_count": int(n),
        "source_files_used": {
            "source_pth": str(source_pth),
            "converted_motion_entry": "phc_baseline/converted/badminton_phc_motion_groundfix.pkl",
        },
        "coordinate_transform_mode": "batch_traced_source_geometry_to_phc_reference; source point + (root_trans_offset - trans_orig)",
        "racket_length_mean": float(length.mean()),
        "racket_length_std": float(length.std()),
        "dynamic_attachment": {
            "tip_local_deviation_mean": float(tip_dev.mean()),
            "tip_local_deviation_p90": float(np.percentile(tip_dev, 90)),
            "fixed_passive_attachment_plausible": False,
        },
        "fields_written": sorted(fields.keys()),
        "warnings": [
            "fixed passive attachment remains rejected; dynamic_hand_to_racket_transform is time-varying reference data.",
            "PHC rollout racket accuracy is not computed by this batch exporter.",
        ],
    }
    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "selection_rank",
        "frame_count",
        "completed_body_baseline",
        "source_pth_exists",
        "task_export_passed",
        "integrity_check_passed",
        "dynamic_replay_checked",
        "dynamic_replay_passed",
        "npz_path",
        "summary_json_path",
        "failure_reason",
    ]
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {"rows": rows}
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_audit(args: argparse.Namespace, rows: list[dict[str, Any]], completed_count: int, existing_count: int) -> None:
    attempted = len(rows)
    success = sum(boolish(row["task_export_passed"]) for row in rows)
    integrity = sum(boolish(row["integrity_check_passed"]) for row in rows)
    checked = sum(boolish(row["dynamic_replay_checked"]) for row in rows)
    replay_pass = sum(boolish(row["dynamic_replay_passed"]) for row in rows)
    reasons = Counter(row["failure_reason"] or "passed" for row in rows if row["failure_reason"])
    status = "blocked_insufficient_data" if integrity < 10 else ("preliminary_only" if integrity < 30 else "first_formal_diagnostic_permitted")
    selected = [row["sequence"] for row in rows if boolish(row["integrity_check_passed"])]
    audit = {
        "task_npz_count_existing": existing_count,
        "task_sequences_existing": sorted(
            str(np.load(path, allow_pickle=True)["sequence"].item())
            for path in args.existing_task_dir.glob("*_racket_aware_reference_task.npz")
        ) if args.existing_task_dir and args.existing_task_dir.exists() else [],
        "completed_body_sequences_available": completed_count,
        "deterministic_candidate_count_attempted": attempted,
        "successful_task_export_count": success,
        "integrity_passed_count": integrity,
        "dynamic_replay_checked_count": checked,
        "dynamic_replay_passed_count": replay_pass,
        "dynamic_replay_failed_count": checked - replay_pass,
        "final_eligible_diagnostic_sequence_count": integrity,
        "reliable_diagnostic_evaluation_status": status,
        "selected_sequences_for_diagnostic": selected,
        "selection_rule": (
            f"completed=True sequences sorted by sequence name; first {args.candidate_limit}, "
            f"expanded to first {args.expand_candidate_limit} if fewer than {args.min_successful} integrity-passed exports"
        ),
        "missing_or_failed_sequences": [
            {"sequence": row["sequence"], "failure_reason": row["failure_reason"]}
            for row in rows
            if row["failure_reason"]
        ],
        "excluded_failure_reasons": dict(reasons),
        "warnings": [
            "This dataset is for offline reference-level predictability diagnostics only.",
            "Dynamic replay is source/reference transform validation, not PHC simulated rollout accuracy.",
        ],
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Body-Only Predictability Dataset Audit",
        "",
        f"- Existing complete racket-aware task NPZ before this run: `{existing_count}`",
        f"- Completed body baseline sequences available: `{completed_count}`",
        f"- Deterministic candidates attempted: `{attempted}`",
        f"- Successful task exports: `{success}`",
        f"- Integrity-passed tasks: `{integrity}`",
        f"- Dynamic replay checked / passed / failed: `{checked}` / `{replay_pass}` / `{checked - replay_pass}`",
        f"- Final eligible diagnostic clips: `{integrity}`",
        f"- Evaluation status: `{status}`",
        "",
        "Grouped failure reasons:",
    ]
    if reasons:
        for reason, count in reasons.most_common():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines += [
        "",
        "Scope: offline reference-level diagnostic dataset preparation only; not PHC simulated rollout accuracy.",
    ]
    args.audit_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", required=True, type=Path)
    parser.add_argument("--converted_motion", required=True, type=Path)
    parser.add_argument("--phc_root", required=True, type=Path)
    parser.add_argument("--source_pth_root", required=True, type=Path)
    parser.add_argument("--custom_smpl_model_path", required=True, type=Path)
    parser.add_argument("--smpl_model_path", required=True, type=Path)
    parser.add_argument("--racket_markers_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--audit_json", required=True, type=Path)
    parser.add_argument("--audit_md", required=True, type=Path)
    parser.add_argument("--existing_task_dir", type=Path, default=Path("phc_baseline/racket_calibration/racket_aware_reference_task"))
    parser.add_argument("--candidate_limit", type=int, default=100)
    parser.add_argument("--expand_candidate_limit", type=int, default=200)
    parser.add_argument("--min_successful", type=int, default=30)
    parser.add_argument("--gender", default="male")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    tasks_dir = args.output_dir / "tasks"
    summaries_dir = args.output_dir / "summaries"
    rows: list[dict[str, Any]] = []
    existing_count = len(list(args.existing_task_dir.glob("*_racket_aware_reference_task.npz"))) if args.existing_task_dir.exists() else 0
    completed = completed_sequences(args.metrics_csv)
    motion = joblib.load(args.converted_motion)
    marker_cfg = json.loads(args.racket_markers_json.read_text(encoding="utf-8"))
    marker_local = np.asarray(
        [
            marker_cfg["handle_anchor_local_xyz"],
            marker_cfg["tip_marker_local_xyz"],
            marker_cfg["head_center_local_xyz"],
        ],
        dtype=np.float64,
    )
    skeleton_state_cls, skeleton_tree = load_phc_skeleton(args.phc_root)
    smplx = load_custom_smpl(args.custom_smpl_model_path)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    limit = args.candidate_limit
    for pass_idx in range(2):
        candidates = completed[:limit]
        rows = []
        smpl_model_cache: dict[int, Any] = {}
        for rank, sequence in enumerate(candidates, start=1):
            source_pth = args.source_pth_root / f"{sequence}.pth"
            out_npz = tasks_dir / f"{safe_name(sequence)}_racket_aware_reference_task.npz"
            out_summary = summaries_dir / f"{safe_name(sequence)}_racket_aware_reference_task_summary.json"
            row: dict[str, Any] = {
                "sequence": sequence,
                "selection_rank": rank,
                "frame_count": 0,
                "completed_body_baseline": True,
                "source_pth_exists": source_pth.exists(),
                "task_export_passed": False,
                "integrity_check_passed": False,
                "dynamic_replay_checked": False,
                "dynamic_replay_passed": False,
                "npz_path": str(out_npz),
                "summary_json_path": str(out_summary),
                "failure_reason": "",
            }
            try:
                if not source_pth.exists():
                    raise FileNotFoundError(f"source .pth missing: {source_pth}")
                if sequence not in motion:
                    raise KeyError(f"sequence missing from converted motion: {sequence}")
                source_meta = torch.load(source_pth, map_location="cpu")
                raw_frames = int(source_meta["trans"].shape[0])
                mask_meta = source_meta.get("mask", torch.ones(raw_frames))
                valid_frames = int(mask_meta.detach().cpu().numpy().astype(bool).sum())
                if valid_frames not in smpl_model_cache:
                    smpl_model_cache[valid_frames] = smplx.SMPL(
                        model_path=str(args.smpl_model_path.resolve()),
                        gender=args.gender,
                        batch_size=valid_frames,
                    ).to(device).eval()
                export_task_for_sequence(
                    sequence=sequence,
                    source_pth=source_pth,
                    entry=motion[sequence],
                    smpl_model=smpl_model_cache[valid_frames],
                    smpl_device=device,
                    skeleton_state_cls=skeleton_state_cls,
                    skeleton_tree=skeleton_tree,
                    marker_local=marker_local,
                    output_npz=out_npz,
                    output_summary_json=out_summary,
                )
                row["task_export_passed"] = True
                ok, reason, stats = integrity_check(out_npz)
                row["integrity_check_passed"] = ok
                row["frame_count"] = int(stats.get("frame_count", 0))
                if not ok:
                    raise ValueError(f"integrity check failed: {reason}")
                replay_ok, _ = replay_check(out_npz)
                row["dynamic_replay_checked"] = True
                row["dynamic_replay_passed"] = replay_ok
                if not replay_ok:
                    raise ValueError("dynamic replay failed")
            except Exception as exc:  # noqa: BLE001 - manifest must preserve per-sequence failures
                row["failure_reason"] = str(exc)
            rows.append(row)

        write_manifest(args.output_dir, rows)
        passed = sum(boolish(row["integrity_check_passed"]) for row in rows)
        if passed >= args.min_successful or limit >= args.expand_candidate_limit:
            break
        limit = args.expand_candidate_limit

    write_manifest(args.output_dir, rows)
    write_audit(args, rows, completed_count=len(completed), existing_count=existing_count)
    payload = {
        "attempted": len(rows),
        "integrity_passed": sum(boolish(row["integrity_check_passed"]) for row in rows),
        "dynamic_replay_passed": sum(boolish(row["dynamic_replay_passed"]) for row in rows),
        "manifest": str(args.output_dir / "manifest.csv"),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
