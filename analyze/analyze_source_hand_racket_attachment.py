#!/usr/bin/env python3
"""Analyze whether source racket geometry is fixed relative to custom SMPL hand."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation


JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar",
    "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist",
    "R_Wrist", "L_Hand", "R_Hand",
]


def safe_name(sequence: str) -> str:
    return sequence.replace("/", "_")


def load_custom_smpl(root: Path):
    resolved = root.resolve()
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    from submodules import smplx  # type: ignore

    return smplx


def apply_newracket_global_rotation(
    pose: np.ndarray,
    trans: np.ndarray,
    racket_pose: np.ndarray,
    axis: str = "x",
    degrees: float = -90.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rot = Rotation.from_euler(axis, degrees, degrees=True)
    pose_out = pose.copy()
    pose_out[:, :3] = (rot * Rotation.from_rotvec(pose[:, :3])).as_rotvec()
    return pose_out, rot.apply(trans), rot.apply(racket_pose)


def transform_markers(local_points: np.ndarray, canonical_wrist: np.ndarray, mats: np.ndarray) -> np.ndarray:
    points = local_points[None, :, :] + canonical_wrist[:, None, :]
    rot = mats[:, :3, :3]
    trans = mats[:, :3, 3][:, None, :]
    return np.einsum("tij,tnj->tni", rot, points) + trans


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


def quat_angle_deg(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-8)
    w = np.clip(np.abs(q[:, 3]), -1.0, 1.0)
    return np.degrees(2.0 * np.arccos(w))


def rotation_deviation_deg(rot: np.ndarray, reference: np.ndarray) -> np.ndarray:
    rel = Rotation.from_matrix(reference[None, :, :].transpose(0, 2, 1) @ rot)
    return quat_angle_deg(rel.as_quat())


def local_stats(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    mean = points.mean(axis=0)
    dev = np.linalg.norm(points - mean[None, :], axis=1)
    return mean, dev, {
        "mean_deviation": float(dev.mean()),
        "std_norm": float(np.linalg.norm(points.std(axis=0))),
        "p90_deviation": float(np.percentile(dev, 90)),
        "max_deviation": float(dev.max()),
    }


def finite_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    if len(a) < 3 or np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def plot_series(output_dir: Path, sequence: str, tip_local: np.ndarray, rot_dev: np.ndarray, racket_pose_norm: np.ndarray) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(tip_local))
    tip_mean = tip_local.mean(axis=0)
    tip_dev = np.linalg.norm(tip_local - tip_mean[None, :], axis=1)

    specs = [
        ("tip_local_xyz", "tip local xyz (m)", [tip_local[:, 0], tip_local[:, 1], tip_local[:, 2]], ["x", "y", "z"]),
        ("tip_local_deviation_norm", "tip local deviation norm (m)", [tip_dev], ["deviation"]),
        ("relative_rotation_deviation_deg", "relative rotation deviation (deg)", [rot_dev], ["angle"]),
        ("racket_pose_axis_angle_norm", "racket_pose axis-angle norm", [racket_pose_norm], ["norm"]),
    ]
    for suffix, title, ys, labels in specs:
        fig, ax = plt.subplots(figsize=(9, 4), dpi=140)
        for y, label in zip(ys, labels):
            ax.plot(x, y, label=label)
        ax.set_title(f"{sequence} - {title}")
        ax.set_xlabel("frame")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(output_dir / f"{suffix}.png")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_pth", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--gender", default="male")
    parser.add_argument("--custom_smpl_model_path", required=True, type=Path)
    parser.add_argument("--smpl_model_path", default=None, type=Path)
    parser.add_argument("--racket_obj", required=True, type=Path)
    parser.add_argument("--racket_markers_json", required=True, type=Path)
    parser.add_argument("--output_npz", required=True, type=Path)
    parser.add_argument("--output_summary_json", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--output_plot_dir", required=True, type=Path)
    parser.add_argument("--attachment_index", default=23, type=int)
    args = parser.parse_args()

    smplx = load_custom_smpl(args.custom_smpl_model_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.source_pth, map_location=device)
    trans_np = checkpoint["trans"].detach().cpu().numpy()
    pose_np = checkpoint["body_pose"].detach().cpu().numpy()
    racket_pose_np = checkpoint["racket_pose"].detach().cpu().numpy()
    mask_np = checkpoint.get("mask", torch.ones(len(trans_np), device=device)).detach().cpu().numpy().astype(bool)
    pose_np, trans_np, racket_pose_np = apply_newracket_global_rotation(pose_np, trans_np, racket_pose_np)
    valid_idx = np.where(mask_np)[0]
    pose_np = pose_np[valid_idx]
    trans_np = trans_np[valid_idx]
    racket_pose_np = racket_pose_np[valid_idx]
    total_frames = len(trans_np)

    beta = checkpoint["beta"].detach().cpu().numpy()
    beta_t = torch.from_numpy(beta).expand(total_frames, -1).to(device)
    pose_t = torch.from_numpy(pose_np).to(device)
    trans_t = torch.from_numpy(trans_np).to(device)
    racket_pose_t = torch.from_numpy(racket_pose_np).to(device)
    smpl_model_path = args.smpl_model_path or (args.custom_smpl_model_path / "body_models/human_model_files/smpl")
    smpl_model = smplx.SMPL(model_path=str(smpl_model_path), gender=args.gender, batch_size=total_frames).to(device).eval()

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
    source_hand = body_A[:, args.attachment_index]
    source_wrist = body_A[:, 21]
    racket_transform = racket_A[:, -1]
    racket_parent_frame = racket_A[:, 23]
    custom_joint24 = racket_smpl.joints[:, 24, :].detach().cpu().numpy()
    canonical_wrist = canonical_smpl.joints[:, 21, :].detach().cpu().numpy()
    markers = json.loads(args.racket_markers_json.read_text(encoding="utf-8"))
    local_markers = np.asarray(
        [
            markers["handle_anchor_local_xyz"],
            markers["tip_marker_local_xyz"],
            markers["head_center_local_xyz"],
        ],
        dtype=np.float64,
    )
    marker_world = transform_markers(local_markers, canonical_wrist, racket_transform)
    handle_world, tip_world, head_world = marker_world[:, 0], marker_world[:, 1], marker_world[:, 2]

    hand_to_racket = np.matmul(invert_transform(source_hand), racket_transform)
    wrist_to_racket = np.matmul(invert_transform(source_wrist), racket_transform)
    parent_to_racket = np.matmul(invert_transform(racket_parent_frame), racket_transform)
    tip_in_hand = apply_inverse_to_points(source_hand, tip_world)
    anchor_in_hand = apply_inverse_to_points(source_hand, handle_world)
    head_in_hand = apply_inverse_to_points(source_hand, head_world)
    joint24_in_hand = apply_inverse_to_points(source_hand, custom_joint24)

    tip_mean, tip_dev, tip_stats = local_stats(tip_in_hand)
    anchor_mean, anchor_dev, anchor_stats = local_stats(anchor_in_hand)
    head_mean, head_dev, head_stats = local_stats(head_in_hand)
    rel_rot = hand_to_racket[:, :3, :3]
    rot_dev = rotation_deviation_deg(rel_rot, Rotation.from_matrix(rel_rot).mean().as_matrix())
    first_rot_dev = rotation_deviation_deg(rel_rot, rel_rot[0])
    parent_rot_dev = rotation_deviation_deg(parent_to_racket[:, :3, :3], parent_to_racket[0, :3, :3])
    racket_pose_norm = np.linalg.norm(racket_pose_np, axis=1)
    pose_change = np.linalg.norm(np.diff(racket_pose_np, axis=0), axis=1)
    rel_change = np.linalg.norm(np.diff(tip_in_hand, axis=0), axis=1)

    summary = {
        "sequence": args.sequence,
        "source_pth": str(args.source_pth),
        "frame_count": int(total_frames),
        "source_preprocessing": {"global_rotation_axis": "x", "global_rotation_deg": -90.0},
        "body_joint_mapping": [
            {"name": name, "joint_index": i, "A_index": i} for i, name in enumerate(JOINT_NAMES)
        ] + [{"name": "RacketJoint24", "joint_index": 24, "A_index": 24}],
        "selected_attachment_source_body_name": JOINT_NAMES[args.attachment_index],
        "selected_attachment_A_index": int(args.attachment_index),
        "right_wrist_A_index": 21,
        "racket_transform_A_index": int(racket_A.shape[1] - 1),
        "implementation_note": (
            "body-only SMPL forward is used for anatomical hand/wrist transforms because racket_pose forward "
            "overwrites rot_mats[:, -1] before adding the racket child joint."
        ),
        "tip_local_mean": tip_mean.tolist(),
        "tip_local_position_stats": tip_stats,
        "anchor_local_mean": anchor_mean.tolist(),
        "anchor_local_position_stats": anchor_stats,
        "head_center_local_mean": head_mean.tolist(),
        "head_center_local_position_stats": head_stats,
        "relative_rotation_angle_deviation_deg": {
            "mean": float(rot_dev.mean()),
            "p90": float(np.percentile(rot_dev, 90)),
            "max": float(rot_dev.max()),
            "from_first_p90": float(np.percentile(first_rot_dev, 90)),
            "from_first_max": float(first_rot_dev.max()),
        },
        "racket_parent_to_racket_rotation_deviation_deg": {
            "p90": float(np.percentile(parent_rot_dev, 90)),
            "max": float(parent_rot_dev.max()),
        },
        "racket_pose_axis_angle_norm": {
            "mean": float(racket_pose_norm.mean()),
            "std": float(racket_pose_norm.std()),
            "p90": float(np.percentile(racket_pose_norm, 90)),
            "max": float(racket_pose_norm.max()),
        },
        "racket_pose_change_to_tip_local_change_corr": finite_corr(pose_change, rel_change),
        "fixed_attachment_plausible": bool(tip_stats["p90_deviation"] < 0.05 and np.percentile(rot_dev, 90) < 10.0),
        "time_varying_racket_control_required_candidate": bool(tip_stats["p90_deviation"] > 0.10 or np.percentile(rot_dev, 90) > 20.0),
    }

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        sequence=np.asarray(args.sequence),
        source_frame_idx=valid_idx.astype(np.int32),
        T_source_hand=source_hand.astype(np.float32),
        T_source_wrist=source_wrist.astype(np.float32),
        T_source_racket=racket_transform.astype(np.float32),
        T_hand_to_racket=hand_to_racket.astype(np.float32),
        T_wrist_to_racket=wrist_to_racket.astype(np.float32),
        T_racket_parent_to_racket=parent_to_racket.astype(np.float32),
        source_racket_tip_world=tip_world.astype(np.float32),
        source_racket_anchor_world=handle_world.astype(np.float32),
        source_racket_joint24_world=custom_joint24.astype(np.float32),
        source_racket_head_center_world=head_world.astype(np.float32),
        tip_in_hand_frame=tip_in_hand.astype(np.float32),
        anchor_in_hand_frame=anchor_in_hand.astype(np.float32),
        joint24_in_hand_frame=joint24_in_hand.astype(np.float32),
        head_center_in_hand_frame=head_in_hand.astype(np.float32),
        relative_rotation_angle_deviation_deg=rot_dev.astype(np.float32),
        racket_pose_parameter=racket_pose_np.astype(np.float32),
    )
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plot_series(args.output_plot_dir, args.sequence, tip_in_hand, rot_dev, racket_pose_norm)

    lines = [
        f"# Source Hand-Racket Attachment: {args.sequence}",
        "",
        f"- Selected anatomical source body: `{summary['selected_attachment_source_body_name']}` index `{args.attachment_index}`.",
        "- Anatomical body-only SMPL forward is used for the hand transform.",
        f"- Tip local p90 deviation: {tip_stats['p90_deviation']:.6f} m.",
        f"- Relative rotation p90 deviation: {summary['relative_rotation_angle_deviation_deg']['p90']:.6f} deg.",
        f"- Fixed attachment plausible: {summary['fixed_attachment_plausible']}.",
        f"- Time-varying racket-control candidate: {summary['time_varying_racket_control_required_candidate']}.",
    ]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
