#!/usr/bin/env python3
"""Inspect raw SMPL-like badminton motion files before PHC conversion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np


POSE_KEYS = ("pose_aa", "poses", "pose", "root_orient")
TRANS_KEYS = ("trans", "transl", "translation", "root_trans")
BETA_KEYS = ("beta", "betas")
FPS_KEYS = ("fps", "mocap_framerate", "frame_rate")
GENDER_KEYS = ("gender",)


def _to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _to_python(value.item())
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Path):
        return str(value)
    return value


def _safe_shape(value: Any) -> list[int] | None:
    try:
        return list(np.asarray(value).shape)
    except Exception:
        return None


def _first_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> tuple[str | None, Any]:
    for key in keys:
        if key in mapping:
            return key, mapping[key]
    return None, None


def _load_motion_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if path.suffix in {".pkl", ".joblib"}:
        obj = joblib.load(path)
        if isinstance(obj, dict):
            return obj
        return {"_object_type": type(obj).__name__, "_object": obj}
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _is_sequence_dict(obj: dict[str, Any]) -> bool:
    return any(key in obj for key in POSE_KEYS) and any(key in obj for key in TRANS_KEYS)


def _summarize_sequence(name: str, rel_path: str, raw: dict[str, Any]) -> dict[str, Any]:
    pose_key, pose = _first_present(raw, POSE_KEYS)
    trans_key, trans = _first_present(raw, TRANS_KEYS)
    beta_key, beta = _first_present(raw, BETA_KEYS)
    fps_key, fps = _first_present(raw, FPS_KEYS)
    gender_key, gender = _first_present(raw, GENDER_KEYS)

    pose_arr = np.asarray(pose) if pose_key is not None else None
    trans_arr = np.asarray(trans) if trans_key is not None else None
    beta_arr = np.asarray(beta) if beta_key is not None else None

    frame_count = None
    if pose_arr is not None and pose_arr.ndim >= 1:
        frame_count = int(pose_arr.shape[0])
    elif trans_arr is not None and trans_arr.ndim >= 1:
        frame_count = int(trans_arr.shape[0])

    finite_arrays = []
    for arr in (pose_arr, trans_arr, beta_arr):
        if arr is not None and np.issubdtype(arr.dtype, np.number):
            finite_arrays.append(arr)

    has_nan = any(np.isnan(arr).any() for arr in finite_arrays)
    has_inf = any(np.isinf(arr).any() for arr in finite_arrays)

    trans_min = trans_max = None
    vertical_axis_guess = None
    pelvis_height_or_vertical_range = None
    if trans_arr is not None and trans_arr.ndim == 2 and trans_arr.shape[1] >= 3:
        trans_min = np.min(trans_arr[:, :3], axis=0).tolist()
        trans_max = np.max(trans_arr[:, :3], axis=0).tolist()
        ranges = np.ptp(trans_arr[:, :3], axis=0)
        # Heuristic only: locomotion often has larger horizontal range, vertical is
        # usually the axis with plausible human pelvis height and smaller variation.
        axis_names = ["x", "y", "z"]
        candidate = int(np.argmin(ranges))
        vertical_axis_guess = axis_names[candidate]
        pelvis_height_or_vertical_range = {
            "axis_guess": vertical_axis_guess,
            "min": float(trans_min[candidate]),
            "max": float(trans_max[candidate]),
            "range": float(ranges[candidate]),
            "all_axis_ranges": ranges.tolist(),
        }

    pose_dim = None
    if pose_arr is not None:
        if pose_arr.ndim == 2:
            pose_dim = int(pose_arr.shape[1])
        elif pose_arr.ndim >= 3:
            pose_dim = int(np.prod(pose_arr.shape[1:]))

    axis_angle_candidate = pose_dim is not None and pose_dim % 3 == 0
    smpl_24_joint_72d = pose_dim == 72
    contains_hand_or_smplx_dims = pose_dim is not None and pose_dim > 72

    return {
        "sequence_name": name,
        "file": rel_path,
        "keys": sorted(raw.keys()),
        "frame_count": frame_count,
        "pose_key": pose_key,
        "pose_shape": _safe_shape(pose),
        "trans_key": trans_key,
        "trans_shape": _safe_shape(trans),
        "beta_key": beta_key,
        "beta_shape": _safe_shape(beta),
        "gender": _to_python(gender) if gender_key else None,
        "fps_key": fps_key,
        "fps": _to_python(fps) if fps_key else None,
        "is_axis_angle_candidate": bool(axis_angle_candidate),
        "is_smpl_24_joint_72d_pose": bool(smpl_24_joint_72d),
        "contains_hand_or_smplx_dims": bool(contains_hand_or_smplx_dims),
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
        "root_translation_min": trans_min,
        "root_translation_max": trans_max,
        "pelvis_height_or_vertical_axis_check": pelvis_height_or_vertical_range,
        "can_map_to_phc_adapter": bool(pose_key and trans_key and beta_key and fps_key),
        "notes": [],
    }


def inspect_source(source: Path, max_sequences: int | None) -> dict[str, Any]:
    files: list[Path]
    if source.is_dir():
        files = sorted(
            path
            for path in source.rglob("*")
            if path.suffix.lower() in {".npz", ".pkl", ".joblib"}
        )
    else:
        files = [source]

    sequences: list[dict[str, Any]] = []
    total_files = len(files)
    for path in files[: max_sequences or total_files]:
        raw = _load_motion_file(path)
        rel = str(path.relative_to(source)) if source.is_dir() else path.name
        if _is_sequence_dict(raw):
            sequences.append(_summarize_sequence(path.stem, rel, raw))
            continue

        for key, value in raw.items():
            if isinstance(value, dict) and _is_sequence_dict(value):
                sequences.append(_summarize_sequence(str(key), f"{rel}:{key}", value))

    missing_fps = sum(1 for item in sequences if item["fps"] is None)
    convertible = sum(1 for item in sequences if item["can_map_to_phc_adapter"])
    pose_dims = sorted({item["pose_shape"][-1] for item in sequences if item["pose_shape"]})

    return {
        "source_path": str(source),
        "source_exists": source.exists(),
        "total_motion_files": total_files,
        "inspected_sequence_count": len(sequences),
        "max_sequences": max_sequences,
        "summary": {
            "convertible_sequence_count": convertible,
            "missing_fps_count": missing_fps,
            "unique_pose_last_dims": pose_dims,
            "all_have_fps": missing_fps == 0,
            "all_convertible_by_adapter": len(sequences) > 0 and convertible == len(sequences),
        },
        "sequences": sequences,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()

    report = inspect_source(args.source, args.max_sequences)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
