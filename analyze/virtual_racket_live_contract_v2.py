#!/usr/bin/env python3
"""Live virtual racket V2 world-persistent contract utilities.

V2 keeps realized virtual racket state in PHC/world coordinates. Policy-facing
goals can remain root-local, but live state persistence and metrics use world
space to avoid root-frame drift.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from virtual_racket_control_contract import DEFAULT_MANIFEST, DEFAULT_OUT_DIR, EPS, axis_angle_error_deg, load_manifest, summarize


@dataclass
class LiveVirtualRacketStateV2:
    handle_phc_world: np.ndarray
    long_axis_phc_world: np.ndarray
    racket_length: np.ndarray | float

    @property
    def tip_phc_world(self) -> np.ndarray:
        return self.handle_phc_world + np.asarray(self.racket_length)[..., None] * self.long_axis_phc_world


class LiveVirtualRacketDynamicsV2:
    """World-persistent state with simulated/reference-root-local action."""

    action_dim = 6

    @staticmethod
    def normalize_axis(axis: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        if np.any(norm < EPS):
            raise ValueError("near-zero live racket axis")
        return axis / norm

    @staticmethod
    def rotate_axis(axis: np.ndarray, omega_world: np.ndarray, dt: float) -> np.ndarray:
        angle_vec = omega_world * dt
        angle = np.linalg.norm(angle_vec, axis=-1, keepdims=True)
        rot_axis = np.zeros_like(angle_vec)
        nonzero = angle[..., 0] > EPS
        rot_axis[nonzero] = angle_vec[nonzero] / angle[nonzero]
        cos = np.cos(angle)
        sin = np.sin(angle)
        cross = np.cross(rot_axis, axis)
        dot = np.sum(rot_axis * axis, axis=-1, keepdims=True)
        rotated = axis * cos + cross * sin + rot_axis * dot * (1.0 - cos)
        rotated[~nonzero] = axis[~nonzero]
        return LiveVirtualRacketDynamicsV2.normalize_axis(rotated)

    @staticmethod
    def local_action_to_world(action_local: np.ndarray, root_rot_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v_world = np.einsum("tij,tj->ti", root_rot_world, action_local[..., :3])
        omega_world = np.einsum("tij,tj->ti", root_rot_world, action_local[..., 3:6])
        return v_world, omega_world

    @staticmethod
    def step(state: LiveVirtualRacketStateV2, action_local: np.ndarray, root_rot_world: np.ndarray, dt: float) -> LiveVirtualRacketStateV2:
        v_world, omega_world = LiveVirtualRacketDynamicsV2.local_action_to_world(action_local, root_rot_world)
        handle_next = state.handle_phc_world + dt * v_world
        axis_next = LiveVirtualRacketDynamicsV2.rotate_axis(state.long_axis_phc_world, omega_world, dt)
        return LiveVirtualRacketStateV2(handle_next, axis_next, state.racket_length)

    @staticmethod
    def derive_oracle_action_local(state_t: LiveVirtualRacketStateV2, state_t1: LiveVirtualRacketStateV2, root_rot_world_t: np.ndarray, dt: float) -> np.ndarray:
        v_world = (state_t1.handle_phc_world - state_t.handle_phc_world) / dt
        a = LiveVirtualRacketDynamicsV2.normalize_axis(state_t.long_axis_phc_world)
        b = LiveVirtualRacketDynamicsV2.normalize_axis(state_t1.long_axis_phc_world)
        cross = np.cross(a, b)
        sin = np.linalg.norm(cross, axis=-1, keepdims=True)
        cos = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)
        angle = np.arctan2(sin, cos)
        rot_axis_world = np.zeros_like(a)
        nonzero = sin[..., 0] > EPS
        rot_axis_world[nonzero] = cross[nonzero] / sin[nonzero]
        omega_world = rot_axis_world * angle / dt
        v_local = np.einsum("tji,tj->ti", root_rot_world_t, v_world)
        omega_local = np.einsum("tji,tj->ti", root_rot_world_t, omega_world)
        return np.concatenate([v_local, omega_local], axis=-1)


def load_live_target(npz_path: Path) -> tuple[LiveVirtualRacketStateV2, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    handle = np.asarray(data["racket_handle_phc_world"], dtype=np.float64)
    tip = np.asarray(data["racket_tip_phc_world"], dtype=np.float64)
    axis = LiveVirtualRacketDynamicsV2.normalize_axis(np.asarray(data["racket_long_axis_phc_world"], dtype=np.float64))
    root_rot = np.asarray(data["root_rotation_phc_world_matrix"], dtype=np.float64)
    length = np.linalg.norm(tip - handle, axis=-1)
    return LiveVirtualRacketStateV2(handle, axis, length), tip, root_rot, length


def validate_world_targets(rows: list[dict[str, str]], out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_rows = []
    all_lengths, all_tip_errors, all_axis_norms = [], [], []
    for row in rows:
        state, tip, _root_rot, length = load_live_target(Path(row["npz_path"]))
        derived = state.tip_phc_world
        tip_err = np.linalg.norm(derived - tip, axis=-1)
        axis_norm = np.linalg.norm(state.long_axis_phc_world, axis=-1)
        all_lengths.append(length)
        all_tip_errors.append(tip_err)
        all_axis_norms.append(axis_norm)
        per_rows.append(
            {
                "sequence": row["sequence"],
                "frame_count": len(length),
                "length_mean_m": float(np.mean(length)),
                "length_std_m": float(np.std(length)),
                "direct_tip_vs_derived_tip_max_error_m": float(np.max(tip_err)),
                "axis_norm_max_abs_error_from_1": float(np.max(np.abs(axis_norm - 1.0))),
                "passed": bool(np.max(tip_err) < 1e-5 and np.max(np.abs(axis_norm - 1.0)) < 1e-5),
            }
        )
    lengths = np.concatenate(all_lengths)
    tip_errors = np.concatenate(all_tip_errors)
    axis_norms = np.concatenate(all_axis_norms)
    summary = {
        "clips": len(rows),
        "frames": int(sum(r["frame_count"] for r in per_rows)),
        "state_frame": "PHC/world persistent realized state",
        "action_frame": "sim/reference-root-local 6D action converted to world by root rotation",
        "racket_length_m": summarize(lengths),
        "racket_length_global_std": float(np.std(lengths)),
        "direct_tip_vs_derived_tip_error_m": summarize(tip_errors),
        "axis_norm_abs_error_from_1": summarize(np.abs(axis_norms - 1.0)),
        "world_target_consistency_passed": bool(np.max(tip_errors) < 1e-5),
    }
    with (out_dir / "live_v2_world_target_consistency_per_sequence.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_rows)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    summary = validate_world_targets(rows, args.output_dir)
    (args.output_dir / "live_v2_world_target_consistency_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
