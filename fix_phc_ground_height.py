#!/usr/bin/env python3
"""Apply a simple vertical ground-height offset to converted PHC motions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phc-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--target-min-z", default=0.02, type=float)
    parser.add_argument("--axis", default=2, type=int)
    args = parser.parse_args()

    phc_root = args.phc_root.resolve()
    os.chdir(phc_root)
    sys.path.insert(0, str(phc_root))

    from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

    data = joblib.load(args.input)
    skeleton_tree = SkeletonTree.from_mjcf("egoquest/data/assets/mjcf/smpl_humanoid_1.xml")

    report = {}
    for key, entry in data.items():
        pose_quat = entry["pose_quat"]
        if not isinstance(pose_quat, torch.Tensor):
            pose_quat_t = torch.from_numpy(pose_quat)
        else:
            pose_quat_t = pose_quat

        root = entry["root_trans_offset"]
        root_t = root.clone() if isinstance(root, torch.Tensor) else torch.from_numpy(root).clone()

        sk_state = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree,
            pose_quat_t,
            root_t,
            is_local=True,
        )
        body_pos = sk_state.global_translation
        current_min = float(body_pos[..., args.axis].min().item())
        offset = args.target_min_z - current_min

        root_t[:, args.axis] += offset
        entry["root_trans_offset"] = root_t
        entry["ground_fix"] = {
            "axis": args.axis,
            "target_min_z": args.target_min_z,
            "original_global_body_min_z": current_min,
            "applied_vertical_offset": offset,
        }
        report[key] = entry["ground_fix"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, args.output)

    print(f"wrote fixed motion: {args.output}")
    for key, info in report.items():
        print(
            f"{key}: min_z {info['original_global_body_min_z']:.6f} "
            f"offset {info['applied_vertical_offset']:.6f} "
            f"target {info['target_min_z']:.6f}"
        )


if __name__ == "__main__":
    main()
