#!/usr/bin/env python3
"""Export source racket geometry from the original custom SMPL+racket pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation


def safe_name(sequence: str) -> str:
    return sequence.replace("/", "_")


def load_custom_smpl(custom_smpl_model_path: Path):
    root = custom_smpl_model_path.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from submodules import smplx  # type: ignore

    return smplx


def transform_points_like_original(local_points: np.ndarray, canonical_wrist: np.ndarray, mats: np.ndarray) -> np.ndarray:
    points = local_points[None, :, :] + canonical_wrist[:, None, :]
    rot = mats[:, :3, :3]
    trans = mats[:, :3, 3][:, None, :]
    return np.einsum("tij,tnj->tni", rot, points) + trans


def apply_newracket_global_rotation(
    pose: np.ndarray,
    trans: np.ndarray,
    racket_pose: np.ndarray,
    axis: str,
    degrees: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mirror humenv/data_preparation/convert_new_racket_to_amass_npz.py."""
    rot = Rotation.from_euler(axis, degrees, degrees=True)
    pose_out = pose.copy()
    pose_out[:, :3] = (rot * Rotation.from_rotvec(pose[:, :3])).as_rotvec()
    return pose_out, rot.apply(trans), rot.apply(racket_pose)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_pth", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--gender", default="male")
    parser.add_argument("--custom_smpl_model_path", required=True, type=Path)
    parser.add_argument("--smpl_model_path", default=None, type=Path)
    parser.add_argument("--racket_obj", required=True, type=Path)
    parser.add_argument("--marker_config", required=True, type=Path)
    parser.add_argument("--output_npz", required=True, type=Path)
    parser.add_argument("--output_summary_json", required=True, type=Path)
    parser.add_argument("--mask_invalid_frames", action="store_true")
    parser.add_argument("--apply_global_rotation", action="store_true")
    parser.add_argument("--rotation_axis", default="x", choices=["x", "y", "z"])
    parser.add_argument("--rotation_deg", default=-90.0, type=float)
    args = parser.parse_args()

    smplx = load_custom_smpl(args.custom_smpl_model_path)
    comp_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.source_pth, map_location=comp_device)
    trans_np = checkpoint["trans"].detach().cpu().numpy()
    total_frames = int(trans_np.shape[0])
    pose_np = checkpoint["body_pose"].detach().cpu().numpy()
    racket_pose_np = checkpoint["racket_pose"].detach().cpu().numpy()
    if args.apply_global_rotation:
        pose_np, trans_np, racket_pose_np = apply_newracket_global_rotation(
            pose=pose_np,
            trans=trans_np,
            racket_pose=racket_pose_np,
            axis=args.rotation_axis,
            degrees=args.rotation_deg,
        )
    trans = torch.from_numpy(trans_np).to(comp_device)
    beta = checkpoint["beta"].expand(total_frames, -1).to(comp_device)
    pose = torch.from_numpy(pose_np).to(comp_device)
    racket_pose = torch.from_numpy(racket_pose_np).to(comp_device)
    mask = checkpoint.get("mask", torch.ones(total_frames, device=comp_device))
    mask_np = mask.detach().cpu().numpy().astype(bool)

    smpl_model_path = args.smpl_model_path or (args.custom_smpl_model_path / "body_models/human_model_files/smpl")
    smpl_model = smplx.SMPL(
        model_path=str(smpl_model_path),
        gender=args.gender,
        batch_size=total_frames,
    ).to(comp_device).eval()

    with torch.no_grad():
        live_smpl = smpl_model.forward(
            betas=beta,
            global_orient=pose[:, :3],
            transl=trans,
            body_pose=pose[:, 3:],
            racket_pose=racket_pose,
        )
        canonical_smpl = smpl_model.forward(beta=beta)

    joints = live_smpl.joints[:, :25, :].detach().cpu().numpy()
    mats = live_smpl.A[:, -1, :, :].detach().cpu().numpy()
    canonical_wrist = canonical_smpl.joints[:, 21, :].detach().cpu().numpy()
    source_body = joints[:, :24, :]
    anchor = joints[:, 24, :]

    mesh = trimesh.load(args.racket_obj, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    marker_cfg = json.loads(args.marker_config.read_text(encoding="utf-8"))
    marker_local = np.asarray(
        [
            marker_cfg["handle_anchor_local_xyz"],
            marker_cfg["tip_marker_local_xyz"],
            marker_cfg["head_center_local_xyz"],
        ],
        dtype=np.float64,
    )
    marker_world = transform_points_like_original(marker_local, canonical_wrist, mats)
    transformed_vertices = transform_points_like_original(vertices.astype(np.float64), canonical_wrist, mats)

    frame_idx = np.arange(total_frames, dtype=np.int32)
    if args.mask_invalid_frames:
        valid = mask_np
        frame_idx = frame_idx[valid]
        source_body = source_body[valid]
        anchor = anchor[valid]
        mats = mats[valid]
        canonical_wrist_out = canonical_wrist[valid]
        transformed_vertices = transformed_vertices[valid]
        marker_world = marker_world[valid]
        racket_pose_out = racket_pose.detach().cpu().numpy()[valid]
        mask_out = mask_np[valid]
    else:
        canonical_wrist_out = canonical_wrist
        racket_pose_out = racket_pose.detach().cpu().numpy()
        mask_out = mask_np

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        sequence=np.asarray(args.sequence),
        source_frame_idx=frame_idx,
        source_mask=mask_out.astype(bool),
        source_body_joints_world=source_body.astype(np.float32),
        source_racket_anchor_world=anchor.astype(np.float32),
        source_racket_transform_world=mats.astype(np.float32),
        source_racket_vertices_world=transformed_vertices.astype(np.float32),
        source_racket_faces=faces,
        source_racket_pose_parameter=racket_pose_out.astype(np.float32),
        source_racket_tip_world=marker_world[:, 1, :].astype(np.float32),
        source_racket_head_center_world=marker_world[:, 2, :].astype(np.float32),
        source_racket_handle_anchor_world=marker_world[:, 0, :].astype(np.float32),
        canonical_wrist_position=canonical_wrist_out.astype(np.float32),
    )

    length = np.linalg.norm(marker_world[:, 1, :] - marker_world[:, 0, :], axis=1)
    summary = {
        "sequence": args.sequence,
        "source_pth": str(args.source_pth),
        "source_coordinate_preprocessing": {
            "apply_global_rotation": args.apply_global_rotation,
            "rotation_axis": args.rotation_axis if args.apply_global_rotation else None,
            "rotation_deg": args.rotation_deg if args.apply_global_rotation else None,
            "matches": "humenv/data_preparation/convert_new_racket_to_amass_npz.py _apply_global_rotation",
        },
        "frame_count": int(len(frame_idx)),
        "raw_frame_count": total_frames,
        "mask_valid_count": int(mask_np.sum()),
        "body_pose_shape": list(pose.shape),
        "racket_pose_parameter_shape": list(racket_pose.shape),
        "racket_anchor_shape": list(anchor.shape),
        "racket_transform_shape": list(mats.shape),
        "racket_vertices_shape": list(transformed_vertices.shape),
        "live_smpl_joints_shape": list(live_smpl.joints.shape),
        "live_smpl_A_shape": list(live_smpl.A.shape),
        "obj_local_bounding_box": {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
        "transformed_world_bounding_box": {"min": transformed_vertices.min(axis=(0, 1)).tolist(), "max": transformed_vertices.max(axis=(0, 1)).tolist()},
        "tip_to_anchor_length_mean": float(np.mean(length)),
        "tip_to_anchor_length_std": float(np.std(length)),
        "tip_to_anchor_length_p90": float(np.percentile(length, 90)),
        "notes": "source_racket_anchor_world is custom SMPL joint 24. It is an attachment/anchor point, not assumed to be the racket tip. source_racket_tip_world is transformed from the OBJ marker definition using the original process_racket_obj formula.",
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
