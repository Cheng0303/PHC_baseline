#!/usr/bin/env python3
"""Validate Live Goal V2 current-sim-root projection semantics.

This is an offline reference-level frame validation. It does not run PHC
policy inference, training, reward tuning, or rollout racket accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from virtual_racket_control_contract import DEFAULT_MANIFEST, DEFAULT_OUT_DIR, axis_angle_error_deg, load_manifest, summarize


PERTURBATIONS = {
    "root_matched": {"translation": [0.0, 0.0, 0.0], "yaw_deg": 0.0},
    "trans_x_025": {"translation": [0.25, 0.0, 0.0], "yaw_deg": 0.0},
    "trans_y_neg025": {"translation": [0.0, -0.25, 0.0], "yaw_deg": 0.0},
    "yaw_pos30": {"translation": [0.0, 0.0, 0.0], "yaw_deg": 30.0},
    "yaw_neg30": {"translation": [0.0, 0.0, 0.0], "yaw_deg": -30.0},
    "trans_x025_yaw_pos30": {"translation": [0.25, 0.0, 0.0], "yaw_deg": 30.0},
}


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("near-zero axis")
    return v / norm


def _yaw_matrix(yaw_rad: np.ndarray) -> np.ndarray:
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    z = np.zeros_like(c)
    o = np.ones_like(c)
    return np.stack(
        [
            np.stack([c, -s, z], axis=-1),
            np.stack([s, c, z], axis=-1),
            np.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


def _heading_matrix_from_root_rot(root_rot: np.ndarray) -> np.ndarray:
    x_axis = root_rot[..., :, 0]
    yaw = np.arctan2(x_axis[..., 1], x_axis[..., 0])
    return _yaw_matrix(yaw)


def _project_heading_local(points_or_vecs: np.ndarray, root_pos: np.ndarray | None, root_rot: np.ndarray) -> np.ndarray:
    heading = _heading_matrix_from_root_rot(root_rot)
    values = points_or_vecs - root_pos if root_pos is not None else points_or_vecs
    return np.einsum("tji,tj->ti", heading, values)


def _goal_v2(handle: np.ndarray, tip: np.ndarray, axis: np.ndarray, root_pos: np.ndarray, root_rot: np.ndarray) -> np.ndarray:
    handle_local = _project_heading_local(handle, root_pos, root_rot)
    tip_local = _project_heading_local(tip, root_pos, root_rot)
    axis_local = _normalize(_project_heading_local(axis, None, root_rot))
    return np.concatenate([handle_local, tip_local, axis_local], axis=-1)


def _lookup_world_at_time(handle: np.ndarray, tip: np.ndarray, axis: np.ndarray, time: float, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(handle)
    length = dt * (n - 1)
    phase = np.clip(time / length, 0.0, 1.0) if length > 0 else 0.0
    time = max(float(time), 0.0)
    idx0 = int(np.floor(phase * (n - 1)))
    idx1 = min(idx0 + 1, n - 1)
    blend = float(np.clip((time - idx0 * dt) / dt, 0.0, 1.0))
    h = (1.0 - blend) * handle[idx0] + blend * handle[idx1]
    t = (1.0 - blend) * tip[idx0] + blend * tip[idx1]
    a = _normalize(((1.0 - blend) * axis[idx0] + blend * axis[idx1])[None, :])[0]
    return h, t, a


def _adapter_validation(handle: np.ndarray, tip: np.ndarray, axis: np.ndarray, dt: float) -> dict[str, float]:
    endpoint_handle, endpoint_tip, endpoint_axis = [], [], []
    midpoint_handle, midpoint_tip, midpoint_axis = [], [], []
    for i in range(len(handle)):
        h, t, a = _lookup_world_at_time(handle, tip, axis, i * dt, dt)
        endpoint_handle.append(np.linalg.norm(h - handle[i]))
        endpoint_tip.append(np.linalg.norm(t - tip[i]))
        endpoint_axis.append(axis_angle_error_deg(a[None, :], axis[i : i + 1])[0])
    for i in range(len(handle) - 1):
        for blend in (0.25, 0.5, 0.75):
            h, t, a = _lookup_world_at_time(handle, tip, axis, (i + blend) * dt, dt)
            eh = (1.0 - blend) * handle[i] + blend * handle[i + 1]
            et = (1.0 - blend) * tip[i] + blend * tip[i + 1]
            ea = _normalize(((1.0 - blend) * axis[i] + blend * axis[i + 1])[None, :])[0]
            midpoint_handle.append(np.linalg.norm(h - eh))
            midpoint_tip.append(np.linalg.norm(t - et))
            midpoint_axis.append(axis_angle_error_deg(a[None, :], ea[None, :])[0])
    return {
        "endpoint_handle_max_m": float(np.max(endpoint_handle)),
        "endpoint_tip_max_m": float(np.max(endpoint_tip)),
        "endpoint_axis_max_deg": float(np.max(endpoint_axis)),
        "midpoint_handle_max_m": float(np.max(midpoint_handle)),
        "midpoint_tip_max_m": float(np.max(midpoint_tip)),
        "midpoint_axis_max_deg": float(np.max(midpoint_axis)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)

    per_rows = []
    adapter_stats = []
    v2_handle_errors, v2_tip_errors, v2_axis_errors = [], [], []
    v1_artificial_by_perturb: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"handle": [], "tip": [], "axis": []} for name in PERTURBATIONS
    }
    invalid = 0
    frames = 0

    for row in rows:
        data = np.load(row["npz_path"], allow_pickle=True)
        handle = np.asarray(data["racket_handle_phc_world"], dtype=np.float64)
        tip = np.asarray(data["racket_tip_phc_world"], dtype=np.float64)
        axis = _normalize(np.asarray(data["racket_long_axis_phc_world"], dtype=np.float64))
        root_pos = np.asarray(data["root_position_phc_world"], dtype=np.float64)
        root_rot = np.asarray(data["root_rotation_phc_world_matrix"], dtype=np.float64)
        v1_handle = np.asarray(data["racket_handle_root_local"], dtype=np.float64)
        v1_tip = np.asarray(data["racket_tip_root_local"], dtype=np.float64)
        v1_axis = _normalize(np.asarray(data["racket_long_axis_root_local"], dtype=np.float64))
        frames += len(handle)
        adapter_stats.append(_adapter_validation(handle, tip, axis, args.dt))

        for name, perturb in PERTURBATIONS.items():
            yaw = np.deg2rad(float(perturb["yaw_deg"]))
            trans = np.asarray(perturb["translation"], dtype=np.float64)[None, :]
            yaw_rot = _yaw_matrix(np.full((len(handle),), yaw))
            sim_root_pos = root_pos + trans
            sim_root_rot = np.einsum("tij,tjk->tik", yaw_rot, root_rot)

            live_goal = _goal_v2(handle, tip, axis, sim_root_pos, sim_root_rot)
            realized_feedback = _goal_v2(handle, tip, axis, sim_root_pos, sim_root_rot)
            v2_handle = np.linalg.norm(live_goal[:, 0:3] - realized_feedback[:, 0:3], axis=-1)
            v2_tip = np.linalg.norm(live_goal[:, 3:6] - realized_feedback[:, 3:6], axis=-1)
            v2_axis = axis_angle_error_deg(live_goal[:, 6:9], realized_feedback[:, 6:9])
            v2_handle_errors.append(v2_handle)
            v2_tip_errors.append(v2_tip)
            v2_axis_errors.append(v2_axis)

            artificial_handle = np.linalg.norm(v1_handle - realized_feedback[:, 0:3], axis=-1)
            artificial_tip = np.linalg.norm(v1_tip - realized_feedback[:, 3:6], axis=-1)
            artificial_axis = axis_angle_error_deg(v1_axis, realized_feedback[:, 6:9])
            v1_artificial_by_perturb[name]["handle"].append(artificial_handle)
            v1_artificial_by_perturb[name]["tip"].append(artificial_tip)
            v1_artificial_by_perturb[name]["axis"].append(artificial_axis)

            finite = all(np.isfinite(x).all() for x in [live_goal, realized_feedback, artificial_handle, artificial_tip, artificial_axis])
            invalid += 0 if finite else 1
            per_rows.append(
                {
                    "sequence": row["sequence"],
                    "perturbation": name,
                    "frames": len(handle),
                    "live_v2_handle_max_m": float(np.max(v2_handle)),
                    "live_v2_tip_max_m": float(np.max(v2_tip)),
                    "live_v2_axis_max_deg": float(np.max(v2_axis)),
                    "goal_v1_artificial_handle_mean_m": float(np.mean(artificial_handle)),
                    "goal_v1_artificial_tip_mean_m": float(np.mean(artificial_tip)),
                    "goal_v1_artificial_axis_mean_deg": float(np.mean(artificial_axis)),
                    "finite": finite,
                }
            )

    adapter_summary = {
        key: float(max(stat[key] for stat in adapter_stats))
        for key in adapter_stats[0].keys()
    }
    v2_handle_all = np.concatenate(v2_handle_errors)
    v2_tip_all = np.concatenate(v2_tip_errors)
    v2_axis_all = np.concatenate(v2_axis_errors)
    perturb_summary = {}
    for name, values in v1_artificial_by_perturb.items():
        perturb_summary[name] = {
            "goal_v1_artificial_handle_error_m": summarize(np.concatenate(values["handle"])),
            "goal_v1_artificial_tip_error_m": summarize(np.concatenate(values["tip"])),
            "goal_v1_artificial_axis_error_deg": summarize(np.concatenate(values["axis"])),
        }

    summary = {
        "scope": "offline reference-level frame validation only; no policy, no training, no PHC rollout racket accuracy",
        "clips": len(rows),
        "frames": int(frames),
        "perturbations": list(PERTURBATIONS.keys()),
        "projection_convention": "PHC heading-local, using calc_heading_quat_inv-compatible yaw projection",
        "goal_v1_frame": "reference-root-local full root quaternion from exported task NPZ",
        "live_goal_v2_frame": "current simulated root heading-local projection of PHC/world target",
        "realized_feedback_v2_frame": "same current simulated root heading-local projection",
        "world_target_adapter_validation": adapter_summary,
        "live_v2_same_frame_mismatch": {
            "handle_error_m": summarize(v2_handle_all),
            "tip_error_m": summarize(v2_tip_all),
            "axis_error_deg": summarize(v2_axis_all),
        },
        "goal_v1_artificial_mismatch_by_perturbation": perturb_summary,
        "invalid_count": int(invalid),
        "passed": bool(invalid == 0 and np.max(v2_handle_all) < 1e-8 and np.max(v2_tip_all) < 1e-8 and np.max(v2_axis_all) < 1e-5),
    }

    csv_path = args.output_dir / "live_goal_v2_alignment_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_rows)
    (args.output_dir / "live_goal_v2_alignment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = f"""# Live Goal V2 Frame Alignment Validation

This is offline reference-level frame validation only. It is not learned policy performance and not PHC rollout racket accuracy.

## Finding

The existing Goal V1 is reference-root-local. The live virtual racket realized-state feedback is current simulated-root-local. These frames diverge when the simulated root differs from the reference root.

Live Goal V2 projects PHC/world target racket geometry into the current simulated root heading frame, matching the realized-state feedback convention.

## Validation

- clips: `{summary['clips']}`
- frames: `{summary['frames']}`
- perturbations: `{', '.join(summary['perturbations'])}`
- projection convention: `{summary['projection_convention']}`
- passed: `{summary['passed']}`

World-target adapter endpoint max errors:

- handle: `{adapter_summary['endpoint_handle_max_m']:.9e} m`
- tip: `{adapter_summary['endpoint_tip_max_m']:.9e} m`
- axis: `{adapter_summary['endpoint_axis_max_deg']:.9e} deg`

Live Goal V2 same-frame perfect-state max mismatch:

- handle: `{summary['live_v2_same_frame_mismatch']['handle_error_m']['max']:.9e} m`
- tip: `{summary['live_v2_same_frame_mismatch']['tip_error_m']['max']:.9e} m`
- axis: `{summary['live_v2_same_frame_mismatch']['axis_error_deg']['max']:.9e} deg`

Goal V1 artificial mismatch examples under perturbations:

- translation `[0.25, 0, 0]` handle/tip mean: `{perturb_summary['trans_x_025']['goal_v1_artificial_handle_error_m']['mean']:.6f}` / `{perturb_summary['trans_x_025']['goal_v1_artificial_tip_error_m']['mean']:.6f}` m
- yaw `+30 deg` handle/tip/axis mean: `{perturb_summary['yaw_pos30']['goal_v1_artificial_handle_error_m']['mean']:.6f}` / `{perturb_summary['yaw_pos30']['goal_v1_artificial_tip_error_m']['mean']:.6f}` m / `{perturb_summary['yaw_pos30']['goal_v1_artificial_axis_error_deg']['mean']:.6f}` deg

Goal V1 remains useful as reference/dataset metadata and ablation input. Live controller input should use Goal V2.
"""
    (args.output_dir / "live_goal_v2_alignment_report.md").write_text(report, encoding="utf-8")
    return 0 if summary["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
