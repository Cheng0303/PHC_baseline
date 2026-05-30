#!/usr/bin/env python3
"""PHC motion-time aligned racket goal adapter.

This mirrors MotionLibBase._calc_frame_blend for time-to-frame lookup, then
interpolates the explicit geometric racket goal.  It does not modify PHC
runtime code or feed augmented observations to a pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from racket_goal_provider import RacketGoalProvider  # noqa: E402


@dataclass(frozen=True)
class MotionMeta:
    sequence: str
    num_frames: int
    fps: float
    dt: float
    length: float


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("cannot normalize near-zero racket long-axis vector")
    return v / norm


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


class RacketGoalMotionTimeAdapter:
    """Racket goal lookup using PHC motion-time frame blending."""

    goal_dim = 9

    def __init__(
        self,
        *,
        manifest_csv: Path,
        phc_motion_file: Path,
        mapping_audit_csv: Path | None = None,
    ) -> None:
        self.provider = RacketGoalProvider(manifest_csv)
        self.motion_data = joblib.load(phc_motion_file)
        self.meta = self._load_motion_meta()
        self.mapping_convention = self._load_mapping_convention(mapping_audit_csv)

    def _load_motion_meta(self) -> dict[str, MotionMeta]:
        meta = {}
        for sequence in self.provider.sequence_names:
            if sequence not in self.motion_data:
                raise KeyError(f"{sequence} missing from PHC motion file")
            entry = self.motion_data[sequence]
            n = int(len(entry["pose_aa"]))
            fps = float(entry.get("fps", 30))
            meta[sequence] = MotionMeta(sequence, n, fps, 1.0 / fps, (n - 1) / fps)
        return meta

    def _load_mapping_convention(self, mapping_audit_csv: Path | None) -> dict[str, str]:
        if mapping_audit_csv is None:
            return {sequence: "row_index_aligned" for sequence in self.provider.sequence_names}
        conventions: dict[str, str] = {}
        with mapping_audit_csv.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if not boolish(row.get("eligible_for_runtime_time_adapter")):
                    continue
                conventions[row["sequence"]] = row["mapping_convention_determined"]
        missing = sorted(set(self.provider.sequence_names) - set(conventions))
        if missing:
            raise ValueError(f"mapping audit missing eligible conventions for {len(missing)} sequences, e.g. {missing[:5]}")
        bad = {seq: conv for seq, conv in conventions.items() if conv != "row_index_aligned"}
        if bad:
            raise ValueError(f"this adapter currently supports row_index_aligned mappings only: {list(bad.items())[:5]}")
        return conventions

    @staticmethod
    def calc_frame_blend(motion_times: np.ndarray, meta: MotionMeta) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        time = np.asarray(motion_times, dtype=np.float64).copy()
        if not np.isfinite(time).all():
            raise ValueError("motion_times contains NaN/Inf")
        phase = np.clip(time / meta.length, 0.0, 1.0) if meta.length > 0 else np.zeros_like(time)
        time[time < 0] = 0.0
        frame_idx0 = np.floor(phase * (meta.num_frames - 1)).astype(np.int64)
        frame_idx1 = np.minimum(frame_idx0 + 1, meta.num_frames - 1).astype(np.int64)
        blend = np.clip((time - frame_idx0 * meta.dt) / meta.dt, 0.0, 1.0)
        return frame_idx0, frame_idx1, blend.astype(np.float64)

    def _goal_rows(self, sequence: str, rows: np.ndarray) -> np.ndarray:
        return self.provider.get_goals_v1(sequence, rows, by_source_frame=False).astype(np.float64)

    def get_goal_v1_by_motion_time(
        self,
        sequence_keys: Iterable[str] | np.ndarray,
        motion_times: Iterable[float] | np.ndarray,
        *,
        future_time_offsets: Iterable[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        sequence_arr = np.asarray(list(sequence_keys) if not isinstance(sequence_keys, np.ndarray) else sequence_keys, dtype=object).reshape(-1)
        time_arr = np.asarray(list(motion_times) if not isinstance(motion_times, np.ndarray) else motion_times, dtype=np.float64).reshape(-1)
        if len(sequence_arr) != len(time_arr):
            raise ValueError(f"sequence_keys length {len(sequence_arr)} != motion_times length {len(time_arr)}")
        offsets = np.asarray([0.0], dtype=np.float64) if future_time_offsets is None else np.asarray(list(future_time_offsets) if not isinstance(future_time_offsets, np.ndarray) else future_time_offsets, dtype=np.float64).reshape(-1)
        blocks = []
        for offset in offsets.tolist():
            goals = []
            for sequence, motion_time in zip(sequence_arr.tolist(), time_arr.tolist()):
                if sequence not in self.meta:
                    raise KeyError(f"unknown sequence {sequence}")
                meta = self.meta[sequence]
                idx0, idx1, blend = self.calc_frame_blend(np.asarray([motion_time + offset], dtype=np.float64), meta)
                g0 = self._goal_rows(sequence, idx0)[0]
                g1 = self._goal_rows(sequence, idx1)[0]
                b = float(blend[0])
                handle = (1.0 - b) * g0[0:3] + b * g1[0:3]
                tip = (1.0 - b) * g0[3:6] + b * g1[3:6]
                axis = normalize(((1.0 - b) * g0[6:9] + b * g1[6:9])[None, :])[0]
                goals.append(np.concatenate([handle, tip, axis]).astype(np.float32))
            blocks.append(np.stack(goals, axis=0))
        return np.concatenate(blocks, axis=-1)

    def get_goal_v1_by_motion_ids(
        self,
        motion_ids: Iterable[int] | np.ndarray,
        motion_times: Iterable[float] | np.ndarray,
        motion_data_keys: Iterable[str] | np.ndarray,
        *,
        future_time_offsets: Iterable[float] | np.ndarray | None = None,
    ) -> np.ndarray:
        keys = np.asarray(list(motion_data_keys) if not isinstance(motion_data_keys, np.ndarray) else motion_data_keys, dtype=object)
        ids = np.asarray(list(motion_ids) if not isinstance(motion_ids, np.ndarray) else motion_ids, dtype=np.int64)
        if ((ids < 0) | (ids >= len(keys))).any():
            raise IndexError("motion id out of range for motion_data_keys")
        sequence_keys = keys[ids]
        return self.get_goal_v1_by_motion_time(sequence_keys, motion_times, future_time_offsets=future_time_offsets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_csv", required=True, type=Path)
    parser.add_argument("--phc_motion_file", required=True, type=Path)
    parser.add_argument("--mapping_audit_csv", type=Path, default=None)
    parser.add_argument("--sample_count", type=int, default=5)
    args = parser.parse_args()

    adapter = RacketGoalMotionTimeAdapter(
        manifest_csv=args.manifest_csv,
        phc_motion_file=args.phc_motion_file,
        mapping_audit_csv=args.mapping_audit_csv,
    )
    seqs = adapter.provider.sequence_names[: args.sample_count]
    times = np.zeros(len(seqs), dtype=np.float64)
    goals = adapter.get_goal_v1_by_motion_time(seqs, times)
    summary = {
        "sample_sequences": seqs,
        "goal_shape": list(goals.shape),
        "goal_dtype": str(goals.dtype),
        "scope": "motion-time sidecar adapter smoke test; not PHC rollout accuracy",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
