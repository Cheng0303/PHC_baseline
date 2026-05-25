#!/usr/bin/env python3
"""Apply the KinTwin x_-90deg SMPL coordinate rotfix before PHC conversion."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as sRot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--key", default=None)
    args = parser.parse_args()

    data = joblib.load(args.input)
    if args.key:
        data = {args.key: data[args.key]}

    rot = sRot.from_euler("x", -90, degrees=True)
    fixed = {}
    for key, entry in data.items():
        out = dict(entry)
        pose = np.asarray(entry["pose_aa"]).copy()
        trans = np.asarray(entry["trans"]).copy()

        root_rot = sRot.from_rotvec(pose[:, 0:3])
        pose[:, 0:3] = (rot * root_rot).as_rotvec()
        trans = rot.apply(trans)

        out["pose_aa"] = pose
        out["trans"] = trans
        out["coord_rotfix"] = {
            "source": "humenv/kintwin/humenv_amass_rotfix_single",
            "rotation": "x_-90deg",
            "mapping": "x->x, y->z, z->-y before downstream floor/height correction",
        }
        fixed[key] = out

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fixed, args.output)
    print(f"wrote {len(fixed)} rotfixed SMPL motions to {args.output}")
    for key in fixed:
        trans = fixed[key]["trans"]
        print(f"{key}: trans min {trans.min(axis=0).tolist()} max {trans.max(axis=0).tolist()}")


if __name__ == "__main__":
    main()
