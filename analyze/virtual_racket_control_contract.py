#!/usr/bin/env python3
"""Virtual kinematic racket state/action contract utilities.

This module is offline-only. It defines geometry utilities and validates the
Goal V1 task dataset as a future virtual racket state target. It does not touch
PHC rewards, action spaces, policies, or physics.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "phc_baseline"
    / "racket_calibration"
    / "racket_aware_reference_task_cross_session_dataset"
    / "manifest.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "phc_baseline" / "reports" / "racket_calibration" / "virtual_racket_control"
EPS = 1e-8


@dataclass
class VirtualRacketStateV1:
    """Root-local virtual racket state.

    V1 does not claim full racket face orientation. It stores only the handle
    position and normalized long axis; tip is derived with a scalar length.
    """

    handle_root_local: np.ndarray
    long_axis_root_local: np.ndarray
    racket_length: np.ndarray | float

    @property
    def tip_root_local(self) -> np.ndarray:
        return self.handle_root_local + np.asarray(self.racket_length)[..., None] * self.long_axis_root_local


class VirtualRacketDynamicsV1:
    """Velocity-controlled kinematic transition for V1 state.

    action = [v_handle(3), omega_axis(3)].
    """

    action_dim = 6

    @staticmethod
    def normalize_axis(axis: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        if np.any(norm < EPS):
            raise ValueError("near-zero long-axis vector")
        return axis / norm

    @staticmethod
    def rotate_axis(axis: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
        angle_vec = omega * dt
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
        return VirtualRacketDynamicsV1.normalize_axis(rotated)

    @staticmethod
    def step(state: VirtualRacketStateV1, action: np.ndarray, dt: float) -> VirtualRacketStateV1:
        action = np.asarray(action, dtype=np.float64)
        handle_next = state.handle_root_local + dt * action[..., :3]
        axis_next = VirtualRacketDynamicsV1.rotate_axis(state.long_axis_root_local, action[..., 3:6], dt)
        return VirtualRacketStateV1(handle_next, axis_next, state.racket_length)

    @staticmethod
    def derive_oracle_action(state_t: VirtualRacketStateV1, state_t1: VirtualRacketStateV1, dt: float) -> np.ndarray:
        v_handle = (state_t1.handle_root_local - state_t.handle_root_local) / dt
        a = VirtualRacketDynamicsV1.normalize_axis(state_t.long_axis_root_local)
        b = VirtualRacketDynamicsV1.normalize_axis(state_t1.long_axis_root_local)
        cross = np.cross(a, b)
        sin = np.linalg.norm(cross, axis=-1, keepdims=True)
        cos = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)
        angle = np.arctan2(sin, cos)
        rot_axis = np.zeros_like(a)
        nonzero = sin[..., 0] > EPS
        rot_axis[nonzero] = cross[nonzero] / sin[nonzero]
        omega = rot_axis * angle / dt
        return np.concatenate([v_handle, omega], axis=-1)


def axis_angle_error_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> np.ndarray:
    a = VirtualRacketDynamicsV1.normalize_axis(axis_a)
    b = VirtualRacketDynamicsV1.normalize_axis(axis_b)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p95": float("nan"), "p99": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def load_manifest(manifest_csv: Path) -> list[dict[str, str]]:
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row.get("task_export_passed") == "True"
        and row.get("integrity_check_passed") == "True"
        and row.get("dynamic_replay_passed") == "True"
    ]


def load_target_state(npz_path: Path) -> tuple[VirtualRacketStateV1, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    handle = np.asarray(data["racket_handle_root_local"], dtype=np.float64)
    tip = np.asarray(data["racket_tip_root_local"], dtype=np.float64)
    axis = VirtualRacketDynamicsV1.normalize_axis(np.asarray(data["racket_long_axis_root_local"], dtype=np.float64))
    length = np.linalg.norm(tip - handle, axis=-1)
    state = VirtualRacketStateV1(handle, axis, length)
    return state, handle, tip, axis


def validate_goal_to_state(rows: Iterable[dict[str, str]], out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    per_rows: list[dict[str, object]] = []
    all_lengths, all_axis_norms, all_tip_errors = [], [], []
    invalid_count = 0
    total_frames = 0

    for row in rows:
        seq = row["sequence"]
        npz_path = REPO_ROOT / row["npz_path"]
        try:
            state, handle, tip, axis = load_target_state(npz_path)
            axis_norm = np.linalg.norm(axis, axis=-1)
            direct_vs_derived = np.linalg.norm(state.tip_root_local - tip, axis=-1)
            lengths = np.linalg.norm(tip - handle, axis=-1)
            finite = np.isfinite(handle).all() and np.isfinite(tip).all() and np.isfinite(axis).all()
            invalid = 0 if finite else len(lengths)
            invalid_count += invalid
            total_frames += len(lengths)
            all_lengths.append(lengths)
            all_axis_norms.append(axis_norm)
            all_tip_errors.append(direct_vs_derived)
            per_rows.append(
                {
                    "sequence": seq,
                    "frame_count": len(lengths),
                    "racket_length_mean": float(np.mean(lengths)),
                    "racket_length_std": float(np.std(lengths)),
                    "racket_length_max_deviation_from_sequence_mean": float(np.max(np.abs(lengths - np.mean(lengths)))),
                    "axis_norm_mean": float(np.mean(axis_norm)),
                    "axis_norm_max_abs_error_from_1": float(np.max(np.abs(axis_norm - 1.0))),
                    "direct_tip_vs_derived_tip_mean_error_m": float(np.mean(direct_vs_derived)),
                    "direct_tip_vs_derived_tip_p90_error_m": float(np.percentile(direct_vs_derived, 90)),
                    "direct_tip_vs_derived_tip_max_error_m": float(np.max(direct_vs_derived)),
                    "invalid_count": invalid,
                    "passed": bool(finite and np.max(np.abs(axis_norm - 1.0)) < 1e-5 and np.max(direct_vs_derived) < 1e-5),
                }
            )
        except Exception as exc:
            invalid_count += 1
            per_rows.append({"sequence": seq, "frame_count": 0, "invalid_count": 1, "passed": False, "failure_reason": repr(exc)})

    lengths_all = np.concatenate(all_lengths) if all_lengths else np.array([])
    axis_norms_all = np.concatenate(all_axis_norms) if all_axis_norms else np.array([])
    tip_errors_all = np.concatenate(all_tip_errors) if all_tip_errors else np.array([])
    summary = {
        "clips": len(per_rows),
        "frames": int(total_frames),
        "virtual_state_v1": {
            "fields": ["handle_root_local[3]", "long_axis_root_local[3]"],
            "dimension": 6,
            "derived_tip": "handle_root_local + racket_length * long_axis_root_local",
        },
        "racket_length_m": summarize(lengths_all),
        "racket_length_global_std": float(np.std(lengths_all)) if lengths_all.size else float("nan"),
        "axis_norm_abs_error_from_1": summarize(np.abs(axis_norms_all - 1.0)),
        "direct_tip_vs_derived_tip_error_m": summarize(tip_errors_all),
        "invalid_count": int(invalid_count),
        "consistency_passed": bool(invalid_count == 0 and tip_errors_all.size > 0 and np.max(tip_errors_all) < 1e-5),
    }

    per_csv = out_dir / "goal_to_virtual_state_per_sequence.csv"
    fieldnames = sorted({key for item in per_rows for key in item.keys()})
    with per_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_rows)
    (out_dir / "goal_to_virtual_state_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Goal V1 To Virtual State V1 Validation

This is an offline reference-level contract validation. It is not PHC rollout racket accuracy and not learned policy performance.

## Contract

- state: `handle_root_local[3] + long_axis_root_local[3]`
- state dimension: `6`
- derived tip: `handle + racket_length * long_axis`
- full racket face orientation: not claimed in V1

## Dataset

- clips: `{summary['clips']}`
- frames: `{summary['frames']}`
- invalid count: `{summary['invalid_count']}`
- consistency passed: `{summary['consistency_passed']}`

## Results

- racket length mean: `{summary['racket_length_m']['mean']:.9f} m`
- racket length std: `{summary['racket_length_global_std']:.9e} m`
- direct tip vs derived tip mean: `{summary['direct_tip_vs_derived_tip_error_m']['mean']:.9e} m`
- direct tip vs derived tip p90: `{summary['direct_tip_vs_derived_tip_error_m']['p90']:.9e} m`
- direct tip vs derived tip max: `{summary['direct_tip_vs_derived_tip_error_m']['max']:.9e} m`

The direct target tip and the derived V1 tip agree to numerical precision, so the V1 6D state is sufficient for the currently validated handle/tip/long-axis geometry target.
"""
    (out_dir / "goal_to_virtual_state_validation_report.md").write_text(report, encoding="utf-8")
    return summary


def audit_handle_body_consistency(rows: Iterable[dict[str, str]], out_dir: Path) -> dict[str, object]:
    """Diagnostic-only handle/body relation using PHC SMPL 24-body order.

    Assumption: reference_body_pos uses the same 24-body SMPL order used by the
    corrected task exporter; source-level analysis selected R_Wrist=21 and
    R_Hand=23. This is not used as a fixed attachment.
    """

    per_rows: list[dict[str, object]] = []
    hand_dists, wrist_dists = [], []
    for row in rows:
        seq = row["sequence"]
        data = np.load(REPO_ROOT / row["npz_path"], allow_pickle=True)
        body = np.asarray(data["reference_body_pos"], dtype=np.float64)
        handle_world = np.asarray(data["racket_handle_phc_world"], dtype=np.float64)
        root = np.asarray(data["root_position_phc_world"], dtype=np.float64)
        rot = np.asarray(data["root_rotation_phc_world_matrix"], dtype=np.float64)
        handle_root_local = np.asarray(data["racket_handle_root_local"], dtype=np.float64)
        hand_world = body[:, 23, :]
        wrist_world = body[:, 21, :]
        hand_root_local = np.einsum("tji,tj->ti", rot, hand_world - root)
        wrist_root_local = np.einsum("tji,tj->ti", rot, wrist_world - root)
        hand_dist = np.linalg.norm(handle_root_local - hand_root_local, axis=-1)
        wrist_dist = np.linalg.norm(handle_root_local - wrist_root_local, axis=-1)
        hand_dists.append(hand_dist)
        wrist_dists.append(wrist_dist)
        per_rows.append(
            {
                "sequence": seq,
                "frame_count": len(hand_dist),
                "handle_to_rhand_mean_m": float(np.mean(hand_dist)),
                "handle_to_rhand_p90_m": float(np.percentile(hand_dist, 90)),
                "handle_to_rhand_max_m": float(np.max(hand_dist)),
                "handle_to_rwrist_mean_m": float(np.mean(wrist_dist)),
                "handle_to_rwrist_p90_m": float(np.percentile(wrist_dist, 90)),
                "handle_to_rwrist_max_m": float(np.max(wrist_dist)),
            }
        )
    hand_all = np.concatenate(hand_dists)
    wrist_all = np.concatenate(wrist_dists)
    summary = {
        "clips": len(per_rows),
        "frame_convention": "PHC root-local distances; reference_body_pos assumed SMPL 24-body order with R_Wrist index 21 and R_Hand index 23.",
        "fixed_attachment_used": False,
        "handle_to_rhand_m": summarize(hand_all),
        "handle_to_rwrist_m": summarize(wrist_all),
        "interpretation": "Diagnostic soft-constraint candidate only; this does not restore fixed passive attachment.",
    }
    per_csv = out_dir / "handle_body_consistency_per_sequence.csv"
    with per_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_rows)
    (out_dir / "handle_body_consistency_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = f"""# Handle / Body Consistency Audit

This is a diagnostic-only audit for future soft constraints. It does not use or restore fixed passive racket attachment.

## Convention

- frame: PHC root-local distances
- body source: `reference_body_pos`
- assumed SMPL body indices: `R_Wrist=21`, `R_Hand=23`
- fixed passive transform: not used

## Results

- clips: `{summary['clips']}`
- handle-to-R_Hand mean: `{summary['handle_to_rhand_m']['mean']:.6f} m`
- handle-to-R_Hand p90: `{summary['handle_to_rhand_m']['p90']:.6f} m`
- handle-to-R_Hand max: `{summary['handle_to_rhand_m']['max']:.6f} m`
- handle-to-R_Wrist mean: `{summary['handle_to_rwrist_m']['mean']:.6f} m`
- handle-to-R_Wrist p90: `{summary['handle_to_rwrist_m']['p90']:.6f} m`
- handle-to-R_Wrist max: `{summary['handle_to_rwrist_m']['max']:.6f} m`

Future reward/metric use should compare realized virtual handle-to-body distance against the reference dynamic distance distribution, not force a single fixed attachment transform.
"""
    (out_dir / "handle_body_consistency_audit.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    rows = load_manifest(args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    validate_goal_to_state(rows, args.output_dir)
    audit_handle_body_consistency(rows, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
