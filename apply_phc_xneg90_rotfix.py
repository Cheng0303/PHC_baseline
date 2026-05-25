#!/usr/bin/env python3
"""Apply x_-90deg coordinate rotfix to an already-converted PHC motion."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation as sRot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phc-root", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    phc_root = args.phc_root.resolve()
    os.chdir(phc_root)
    sys.path.insert(0, str(phc_root))

    from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

    data = joblib.load(args.input)
    tree = SkeletonTree.from_mjcf("egoquest/data/assets/mjcf/smpl_humanoid_1.xml")
    rot = sRot.from_euler("x", -90, degrees=True)

    out_data = {}
    for key, entry in data.items():
        out = dict(entry)
        root = entry["root_trans_offset"]
        root_np = root.cpu().numpy() if isinstance(root, torch.Tensor) else np.asarray(root)
        pose_global = np.asarray(entry["pose_quat_global"])

        root_fixed = rot.apply(root_np)
        pose_global_fixed = (
            rot * sRot.from_quat(pose_global.reshape(-1, 4))
        ).as_quat().reshape(pose_global.shape)

        sk_state = SkeletonState.from_rotation_and_root_translation(
            tree,
            torch.from_numpy(pose_global_fixed),
            torch.from_numpy(root_fixed).float(),
            is_local=False,
        )

        out["root_trans_offset"] = torch.from_numpy(root_fixed).float()
        out["pose_quat_global"] = pose_global_fixed
        out["pose_quat"] = sk_state.local_rotation.numpy()
        out["coord_rotfix"] = {
            "source": "humenv/kintwin/humenv_amass_rotfix_single",
            "rotation": "x_-90deg applied after PHC conversion",
            "mapping": "x->x, y->z, z->-y before floor/height correction",
        }
        out_data[key] = out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out_data, args.output)
    print(f"wrote PHC x_-90 rotfixed motion: {args.output}")
    for key, entry in out_data.items():
        root = entry["root_trans_offset"].numpy()
        print(f"{key}: root min {root.min(axis=0).tolist()} max {root.max(axis=0).tolist()}")


if __name__ == "__main__":
    main()
