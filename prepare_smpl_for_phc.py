#!/usr/bin/env python3
"""Prepare raw badminton SMPL NPZ/PKL files for PHC's SMPL converter."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from tqdm import tqdm


def _read_motion(path: Path) -> dict[str, Any]:
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as data:
            return {key: data[key] for key in data.files}
    if path.suffix in {".pkl", ".joblib"}:
        obj = joblib.load(path)
        if not isinstance(obj, dict):
            raise TypeError(f"{path} did not contain a dict")
        return obj
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _get(raw: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    raise KeyError(f"missing any of keys: {keys}")


def _scalar_str(value: Any, default: str) -> str:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.shape == ():
        value = arr.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def convert_entry(raw: dict[str, Any], default_gender: str) -> dict[str, Any]:
    pose = np.asarray(_get(raw, "pose_aa", "poses"))
    trans = np.asarray(_get(raw, "trans", "transl"))
    beta = np.asarray(_get(raw, "beta", "betas"))

    if pose.ndim != 2:
        raise ValueError(f"pose must be [T, D], got {pose.shape}")
    if pose.shape[1] < 72:
        raise ValueError(f"pose must contain at least SMPL body 72D, got {pose.shape}")
    if pose.shape[1] > 72:
        pose = pose[:, :72]
    if trans.ndim != 2 or trans.shape[1] < 3:
        raise ValueError(f"trans must be [T, 3+], got {trans.shape}")
    trans = trans[:, :3]

    fps = raw.get("fps", raw.get("mocap_framerate", raw.get("frame_rate")))
    if fps is None:
        raise ValueError("fps is missing; refusing to guess a framerate")
    fps_arr = np.asarray(fps)
    fps_value = float(fps_arr.item() if fps_arr.shape == () else fps_arr.reshape(-1)[0])

    gender = _scalar_str(raw.get("gender"), default_gender).lower()
    if gender not in {"neutral", "male", "female"}:
        gender = default_gender

    return {
        "pose_aa": pose.astype(np.float32),
        "trans": trans.astype(np.float32),
        "beta": beta.astype(np.float32),
        "gender": gender,
        "fps": fps_value,
    }


def iter_motion_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        path
        for path in source.rglob("*")
        if path.suffix.lower() in {".npz", ".pkl", ".joblib"}
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--default-gender", default="neutral", choices=["neutral", "male", "female"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    files = iter_motion_files(args.source)
    if args.limit is not None:
        files = files[: args.limit]

    out: dict[str, dict[str, Any]] = {}
    skipped: dict[str, str] = {}
    for path in tqdm(files, desc="Preparing PHC input"):
        try:
            raw = _read_motion(path)
            if "pose_aa" in raw or "poses" in raw:
                key = str(path.relative_to(args.source).with_suffix("")) if args.source.is_dir() else path.stem
                out[key] = convert_entry(raw, args.default_gender)
                continue
            for name, value in raw.items():
                if isinstance(value, dict) and ("pose_aa" in value or "poses" in value):
                    out[f"{path.stem}/{name}"] = convert_entry(value, args.default_gender)
        except Exception as exc:
            skipped[str(path)] = str(exc)

    if not out:
        raise RuntimeError(f"No convertible sequences found. Skipped: {skipped}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, args.output)
    if skipped:
        skipped_path = args.output.with_suffix(".skipped.json")
        import json

        skipped_path.write_text(json.dumps(skipped, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
