#!/usr/bin/env python3
"""CLI wrapper for PHC's official SMPL-to-humanoid motion conversion."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phc-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--smpl-data-dir", default="data/smpl")
    args = parser.parse_args()

    phc_root = args.phc_root.resolve()
    os.chdir(phc_root)
    sys.path.insert(0, str(phc_root))

    import joblib
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as sRot
    from tqdm import tqdm

    from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree
    from smpl_sim.smpllib.smpl_joint_names import SMPL_BONE_ORDER_NAMES, SMPL_MUJOCO_NAMES
    from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot as LocalRobot

    robot_cfg = {
        "mesh": False,
        "model": "smpl",
        "upright_start": True,
        "body_params": {},
        "joint_params": {},
        "geom_params": {},
        "actuator_params": {},
    }
    smpl_local_robot = LocalRobot(robot_cfg, data_dir=args.smpl_data_dir)
    amass_data = joblib.load(args.input)

    smpl_2_mujoco = [SMPL_BONE_ORDER_NAMES.index(q) for q in SMPL_MUJOCO_NAMES if q in SMPL_BONE_ORDER_NAMES]

    xml_path = phc_root / "egoquest/data/assets/mjcf/smpl_humanoid_1.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    full_motion_dict = {}
    for key_name in tqdm(amass_data.keys(), desc="Converting SMPL to PHC motion"):
        entry = amass_data[key_name]
        pose_aa = entry["pose_aa"].copy()
        root_trans = entry["trans"].copy()
        batch_size = pose_aa.shape[0]

        beta = entry["beta"].copy() if "beta" in entry else entry["betas"].copy()
        if len(beta.shape) == 2:
            beta = beta[0]

        gender = entry.get("gender", "neutral")
        fps = entry["fps"]
        if isinstance(gender, np.ndarray):
            gender = gender.item()
        if isinstance(gender, bytes):
            gender = gender.decode("utf-8")
        gender_number = {"neutral": [0], "male": [1], "female": [2]}.get(gender)
        if gender_number is None:
            raise ValueError(f"Gender not supported for {key_name}: {gender}")

        pose_aa = np.concatenate([pose_aa[:, :66], np.zeros((batch_size, 6))], axis=1)
        pose_aa_mj = pose_aa.reshape(-1, 24, 3)[..., smpl_2_mujoco, :].copy()
        pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(batch_size, 24, 4)

        # Matches the released PHC converter behavior: published SMPL humanoid
        # checkpoints use the neutral model with upright_start coordinate handling.
        gender_number, beta[:], gender = [0], 0, "neutral"

        smpl_local_robot.load_from_skeleton(
            betas=torch.from_numpy(beta[None,]), gender=gender_number, objs_info=None
        )
        smpl_local_robot.write_xml(str(xml_path))
        skeleton_tree = SkeletonTree.from_mjcf(str(xml_path))
        root_trans_offset = torch.from_numpy(root_trans) + skeleton_tree.local_translation[0]

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat),
            root_trans_offset,
            is_local=True,
        )

        pose_quat_global = (
            sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy())
            * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()
        ).as_quat().reshape(batch_size, -1, 4)

        new_sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            torch.from_numpy(pose_quat_global),
            root_trans_offset,
            is_local=False,
        )
        pose_quat = new_sk_state.local_rotation.numpy()

        full_motion_dict[key_name] = {
            "pose_quat_global": pose_quat_global,
            "pose_quat": pose_quat,
            "trans_orig": root_trans,
            "root_trans_offset": root_trans_offset,
            "beta": beta,
            "gender": gender,
            "pose_aa": pose_aa,
            "fps": fps,
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(full_motion_dict, args.output)


if __name__ == "__main__":
    main()
