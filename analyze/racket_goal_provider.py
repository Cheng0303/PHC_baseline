#!/usr/bin/env python3
"""Sidecar provider for explicit root-local racket goals.

This module is intentionally read-only with respect to PHC.  It does not
augment the pretrained observation, action space, reward, or checkpoint.  The
primary controller-facing goal is explicit geometry:

    [handle_root_local, tip_root_local, long_axis_root_local]  # 9D

The source ``racket_pose_parameter`` remains available only through an
explicit diagnostic accessor.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


GOAL_FIELDS = (
    "racket_handle_root_local",
    "racket_tip_root_local",
    "racket_long_axis_root_local",
)
REQUIRED_FIELDS = (
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
    "racket_handle_root_local",
    "racket_tip_root_local",
    "racket_long_axis_root_local",
    "source_anatomical_rhand_world",
    "source_racket_transform_world",
    "dynamic_hand_to_racket_transform",
)


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def scalar_string(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return str(value.item() if value.shape == () else value.tolist())
    return str(value)


@dataclass(frozen=True)
class TaskRecord:
    sequence: str
    npz_path: Path
    summary_json_path: Path | None
    row: dict[str, str]


class RacketGoalProvider:
    """Lazy sidecar lookup for validated racket-aware task targets."""

    goal_dim = 9

    def __init__(
        self,
        manifest_csv: Path,
        *,
        require_integrity_pass: bool = True,
        require_dynamic_replay_pass: bool = True,
    ) -> None:
        self.manifest_csv = Path(manifest_csv)
        self.records = self._load_manifest(require_integrity_pass, require_dynamic_replay_pass)
        self._cache: dict[str, dict[str, Any]] = {}

    @property
    def sequence_names(self) -> list[str]:
        return sorted(self.records)

    def _load_manifest(self, require_integrity_pass: bool, require_dynamic_replay_pass: bool) -> dict[str, TaskRecord]:
        if not self.manifest_csv.exists():
            raise FileNotFoundError(self.manifest_csv)
        records: dict[str, TaskRecord] = {}
        with self.manifest_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                sequence = row["sequence"]
                if not boolish(row.get("task_export_passed", "")):
                    continue
                if require_integrity_pass and not boolish(row.get("integrity_check_passed", "")):
                    continue
                if require_dynamic_replay_pass and not boolish(row.get("dynamic_replay_passed", "")):
                    continue
                npz_path = Path(row["npz_path"])
                if not npz_path.is_absolute():
                    npz_path = self.manifest_csv.parents[3] / npz_path if str(npz_path).startswith("phc_baseline/") else self.manifest_csv.parent / npz_path
                summary = row.get("summary_json_path", "")
                summary_path = Path(summary) if summary else None
                if summary_path is not None and not summary_path.is_absolute():
                    summary_path = self.manifest_csv.parents[3] / summary_path if str(summary_path).startswith("phc_baseline/") else self.manifest_csv.parent / summary_path
                records[sequence] = TaskRecord(sequence, npz_path, summary_path, row)
        if not records:
            raise ValueError(f"no eligible task records in {self.manifest_csv}")
        return records

    def _load_sequence(self, sequence: str) -> dict[str, Any]:
        if sequence in self._cache:
            return self._cache[sequence]
        if sequence not in self.records:
            raise KeyError(f"missing racket-aware task sequence: {sequence}")
        path = self.records[sequence].npz_path
        if not path.exists():
            raise FileNotFoundError(path)
        npz = np.load(path, allow_pickle=True)
        data = {key: npz[key] for key in npz.files}
        missing = [key for key in REQUIRED_FIELDS if key not in data]
        if missing:
            raise KeyError(f"{path} missing required fields: {missing}")
        if scalar_string(data["sequence"]) != sequence:
            raise ValueError(f"sequence mismatch in {path}: {scalar_string(data['sequence'])} != {sequence}")

        n = len(np.asarray(data["source_frame_idx"]))
        for key in REQUIRED_FIELDS:
            if key == "sequence":
                continue
            value = np.asarray(data[key])
            if len(value) != n:
                raise ValueError(f"{sequence} field {key} has len {len(value)}, expected {n}")
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                raise ValueError(f"{sequence} field {key} contains NaN/Inf")

        source_frames = np.asarray(data["source_frame_idx"], dtype=np.int64)
        if len(np.unique(source_frames)) != len(source_frames):
            raise ValueError(f"{sequence} has duplicate source_frame_idx values")
        data["_source_frame_to_row"] = {int(frame): i for i, frame in enumerate(source_frames.tolist())}
        data["_source_frame_min"] = int(source_frames.min())
        data["_source_frame_max"] = int(source_frames.max())
        self._cache[sequence] = data
        return data

    def _resolve_rows(
        self,
        sequence: str,
        frames: Iterable[int] | np.ndarray,
        *,
        by_source_frame: bool = True,
        clip: bool = False,
        nearest_if_missing: bool = False,
    ) -> np.ndarray:
        data = self._load_sequence(sequence)
        frames_arr = np.asarray(list(frames) if not isinstance(frames, np.ndarray) else frames, dtype=np.int64).reshape(-1)
        if not by_source_frame:
            if clip:
                frames_arr = np.clip(frames_arr, 0, len(data["source_frame_idx"]) - 1)
            if ((frames_arr < 0) | (frames_arr >= len(data["source_frame_idx"]))).any():
                raise IndexError(f"row frame out of range for {sequence}")
            return frames_arr.astype(np.int64)

        source_frames = np.asarray(data["source_frame_idx"], dtype=np.int64)
        mapping: dict[int, int] = data["_source_frame_to_row"]
        rows: list[int] = []
        for frame in frames_arr.tolist():
            frame_i = int(frame)
            if clip:
                frame_i = int(np.clip(frame_i, data["_source_frame_min"], data["_source_frame_max"]))
            if frame_i in mapping:
                rows.append(mapping[frame_i])
                continue
            if nearest_if_missing:
                pos = int(np.searchsorted(source_frames, frame_i))
                pos = int(np.clip(pos, 0, len(source_frames) - 1))
                if pos > 0 and abs(int(source_frames[pos - 1]) - frame_i) <= abs(int(source_frames[pos]) - frame_i):
                    pos -= 1
                rows.append(pos)
                continue
            raise KeyError(f"{sequence} has no source_frame_idx={frame_i}")
        return np.asarray(rows, dtype=np.int64)

    def get_goals_v1(
        self,
        sequence: str,
        frames: Iterable[int] | np.ndarray,
        *,
        by_source_frame: bool = True,
    ) -> np.ndarray:
        data = self._load_sequence(sequence)
        rows = self._resolve_rows(sequence, frames, by_source_frame=by_source_frame)
        parts = [np.asarray(data[field], dtype=np.float32)[rows] for field in GOAL_FIELDS]
        return np.concatenate(parts, axis=-1)

    def get_goal_v1(self, sequence: str, frame: int, *, by_source_frame: bool = True) -> np.ndarray:
        return self.get_goals_v1(sequence, np.asarray([frame]), by_source_frame=by_source_frame)[0]

    def get_future_goals_v1(
        self,
        sequence: str,
        frames: Iterable[int] | np.ndarray,
        offsets: Iterable[int] | np.ndarray,
        *,
        by_source_frame: bool = True,
        clip: bool = True,
        nearest_if_missing: bool = True,
    ) -> np.ndarray:
        data = self._load_sequence(sequence)
        frames_arr = np.asarray(list(frames) if not isinstance(frames, np.ndarray) else frames, dtype=np.int64).reshape(-1)
        offsets_arr = np.asarray(list(offsets) if not isinstance(offsets, np.ndarray) else offsets, dtype=np.int64).reshape(-1)
        stacked = []
        for offset in offsets_arr.tolist():
            target = frames_arr + int(offset)
            rows = self._resolve_rows(
                sequence,
                target,
                by_source_frame=by_source_frame,
                clip=clip,
                nearest_if_missing=nearest_if_missing,
            )
            parts = [np.asarray(data[field], dtype=np.float32)[rows] for field in GOAL_FIELDS]
            stacked.append(np.concatenate(parts, axis=-1))
        return np.concatenate(stacked, axis=-1)

    def get_source_racket_pose_parameter(
        self,
        sequence: str,
        frames: Iterable[int] | np.ndarray,
        *,
        by_source_frame: bool = True,
    ) -> np.ndarray:
        data = self._load_sequence(sequence)
        rows = self._resolve_rows(sequence, frames, by_source_frame=by_source_frame)
        return np.asarray(data["racket_pose_parameter"], dtype=np.float32)[rows]


def smoke_test(provider: RacketGoalProvider, future_offsets: list[int], sample_frames: int) -> dict[str, Any]:
    errors = []
    future_shapes = []
    for sequence in provider.sequence_names:
        data = provider._load_sequence(sequence)
        frames = np.asarray(data["source_frame_idx"], dtype=np.int64)
        if sample_frames > 0 and len(frames) > sample_frames:
            pick = np.linspace(0, len(frames) - 1, sample_frames).round().astype(np.int64)
            frames = frames[pick]
        goal = provider.get_goals_v1(sequence, frames)
        gt = np.concatenate([np.asarray(data[field], dtype=np.float32)[provider._resolve_rows(sequence, frames)] for field in GOAL_FIELDS], axis=-1)
        errors.append(float(np.max(np.abs(goal - gt))))
        if future_offsets:
            future = provider.get_future_goals_v1(sequence, frames, future_offsets)
            future_shapes.append(list(future.shape))
        _ = provider.get_source_racket_pose_parameter(sequence, frames)
    return {
        "clips_checked": len(provider.sequence_names),
        "max_abs_goal_error": float(max(errors) if errors else 0.0),
        "future_offsets": future_offsets,
        "future_shapes_sample": future_shapes[:5],
        "goal_schema": {
            "name": "g_racket_v1",
            "dim": 9,
            "fields": [
                "racket_handle_root_local[0:3]",
                "racket_tip_root_local[3:6]",
                "racket_long_axis_root_local[6:9]",
            ],
            "dtype": "float32",
        },
    }


def parse_offsets(text: str) -> list[int]:
    if not text:
        return []
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_csv", required=True, type=Path)
    parser.add_argument("--output_summary_json", type=Path, default=None)
    parser.add_argument("--future_offsets", default="0,1,2")
    parser.add_argument("--sample_frames", type=int, default=5)
    args = parser.parse_args()

    provider = RacketGoalProvider(args.manifest_csv)
    summary = smoke_test(provider, parse_offsets(args.future_offsets), args.sample_frames)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_summary_json:
        args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
