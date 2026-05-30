#!/usr/bin/env python3
"""No-training oracle replay validation for Virtual Racket Dynamics V1."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from virtual_racket_control_contract import (
    DEFAULT_MANIFEST,
    DEFAULT_OUT_DIR,
    VirtualRacketDynamicsV1,
    VirtualRacketStateV1,
    axis_angle_error_deg,
    load_manifest,
    load_target_state,
    summarize,
)


def _state_errors(realized: VirtualRacketStateV1, target: VirtualRacketStateV1, target_tip_direct: np.ndarray) -> dict[str, np.ndarray]:
    handle_error = np.linalg.norm(realized.handle_root_local - target.handle_root_local, axis=-1)
    tip_error = np.linalg.norm(realized.tip_root_local - target_tip_direct, axis=-1)
    axis_error = axis_angle_error_deg(realized.long_axis_root_local, target.long_axis_root_local)
    return {"handle": handle_error, "tip": tip_error, "axis_deg": axis_error}


def replay_one(npz_path: Path, dt: float) -> dict[str, np.ndarray]:
    target_state, _handle, target_tip_direct, _axis = load_target_state(npz_path)
    n = len(target_state.handle_root_local)
    length = np.asarray(target_state.racket_length)

    oracle_handle = np.zeros_like(target_state.handle_root_local)
    oracle_axis = np.zeros_like(target_state.long_axis_root_local)
    null_handle = np.zeros_like(target_state.handle_root_local)
    null_axis = np.zeros_like(target_state.long_axis_root_local)
    oracle_handle[0] = target_state.handle_root_local[0]
    oracle_axis[0] = target_state.long_axis_root_local[0]
    null_handle[0] = target_state.handle_root_local[0]
    null_axis[0] = target_state.long_axis_root_local[0]

    actions = []
    dyn = VirtualRacketDynamicsV1()
    for t in range(n - 1):
        curr = VirtualRacketStateV1(target_state.handle_root_local[t], target_state.long_axis_root_local[t], length[t])
        nxt = VirtualRacketStateV1(target_state.handle_root_local[t + 1], target_state.long_axis_root_local[t + 1], length[t + 1])
        action = dyn.derive_oracle_action(curr, nxt, dt)
        actions.append(action)
        realized_next = dyn.step(VirtualRacketStateV1(oracle_handle[t], oracle_axis[t], length[t]), action, dt)
        oracle_handle[t + 1] = realized_next.handle_root_local
        oracle_axis[t + 1] = realized_next.long_axis_root_local
        null_next = dyn.step(VirtualRacketStateV1(null_handle[t], null_axis[t], length[t]), np.zeros(6), dt)
        null_handle[t + 1] = null_next.handle_root_local
        null_axis[t + 1] = null_next.long_axis_root_local

    actions_arr = np.asarray(actions, dtype=np.float64) if actions else np.zeros((0, 6), dtype=np.float64)
    oracle = VirtualRacketStateV1(oracle_handle, oracle_axis, length)
    null = VirtualRacketStateV1(null_handle, null_axis, length)
    oracle_errors = _state_errors(oracle, target_state, target_tip_direct)
    null_errors = _state_errors(null, target_state, target_tip_direct)
    return {
        "oracle_handle_error": oracle_errors["handle"],
        "oracle_tip_error": oracle_errors["tip"],
        "oracle_axis_error_deg": oracle_errors["axis_deg"],
        "null_handle_error": null_errors["handle"],
        "null_tip_error": null_errors["tip"],
        "null_axis_error_deg": null_errors["axis_deg"],
        "action": actions_arr,
        "target_tip": target_tip_direct,
        "oracle_tip": oracle.tip_root_local,
        "null_tip": null.tip_root_local,
    }


def _model_summary(prefix: str, handle: np.ndarray, tip: np.ndarray, axis: np.ndarray) -> dict[str, float]:
    out = {}
    for name, values in [("handle_error", handle), ("tip_error", tip), ("long_axis_angle_error", axis)]:
        stats = summarize(values)
        suffix = "_deg" if name == "long_axis_angle_error" else "_m"
        for k, v in stats.items():
            out[f"{prefix}_{name}_{k}{suffix}"] = v
    return out


def _write_plots(out_dir: Path, rows: list[dict[str, str]], per_results: dict[str, dict[str, np.ndarray]]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    oracle_tip = np.concatenate([r["oracle_tip_error"] for r in per_results.values()])
    null_tip = np.concatenate([r["null_tip_error"] for r in per_results.values()])
    plt.figure(figsize=(7, 4))
    plt.boxplot([oracle_tip, null_tip], labels=["oracle", "null"], showfliers=False)
    plt.ylabel("tip error (m)")
    plt.title("virtual kinematic contract validation only\nnot learned policy performance; not PHC rollout racket accuracy")
    plt.tight_layout()
    plt.savefig(plot_dir / "oracle_vs_null_tip_error_comparison.png", dpi=160)
    plt.close()

    actions = np.concatenate([r["action"] for r in per_results.values() if len(r["action"])], axis=0)
    action_mag = np.linalg.norm(actions, axis=-1)
    handle_speed = np.linalg.norm(actions[:, :3], axis=-1)
    omega_mag = np.linalg.norm(actions[:, 3:6], axis=-1)
    plt.figure(figsize=(7, 4))
    plt.hist(action_mag, bins=80, alpha=0.6, label="full action")
    plt.hist(handle_speed, bins=80, alpha=0.6, label="handle speed")
    plt.hist(omega_mag, bins=80, alpha=0.6, label="axis omega")
    plt.legend()
    plt.xlabel("magnitude")
    plt.title("oracle action magnitude distribution\nvirtual kinematic contract validation only")
    plt.tight_layout()
    plt.savefig(plot_dir / "oracle_action_magnitude_distribution.png", dpi=160)
    plt.close()

    rep_seq = rows[-1]["sequence"]
    rep = per_results[rep_seq]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rep["target_tip"][:, 0], rep["target_tip"][:, 1], rep["target_tip"][:, 2], label="target")
    ax.plot(rep["oracle_tip"][:, 0], rep["oracle_tip"][:, 1], rep["oracle_tip"][:, 2], "--", label="oracle")
    ax.plot(rep["null_tip"][:, 0], rep["null_tip"][:, 1], rep["null_tip"][:, 2], ":", label="null")
    ax.set_title(f"representative virtual replay: {rep_seq}\nnot learned policy performance; not PHC rollout racket accuracy")
    ax.legend()
    plt.tight_layout()
    safe = rep_seq.replace("/", "_")
    plt.savefig(plot_dir / f"representative_trajectory_state_replay_{safe}.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)

    per_csv_rows = []
    per_results = {}
    all_oracle_handle, all_oracle_tip, all_oracle_axis = [], [], []
    all_null_handle, all_null_tip, all_null_axis = [], [], []
    all_actions = []
    invalid = 0

    for row in rows:
        seq = row["sequence"]
        result = replay_one(Path(row["npz_path"]), args.dt)
        per_results[seq] = result
        all_oracle_handle.append(result["oracle_handle_error"])
        all_oracle_tip.append(result["oracle_tip_error"])
        all_oracle_axis.append(result["oracle_axis_error_deg"])
        all_null_handle.append(result["null_handle_error"])
        all_null_tip.append(result["null_tip_error"])
        all_null_axis.append(result["null_axis_error_deg"])
        if len(result["action"]):
            all_actions.append(result["action"])
        finite = all(np.isfinite(v).all() for k, v in result.items() if isinstance(v, np.ndarray))
        invalid += 0 if finite else 1
        row_out = {"sequence": seq, "frame_count": len(result["oracle_tip_error"]), "finite": finite}
        row_out.update(_model_summary("oracle", result["oracle_handle_error"], result["oracle_tip_error"], result["oracle_axis_error_deg"]))
        row_out.update(_model_summary("null", result["null_handle_error"], result["null_tip_error"], result["null_axis_error_deg"]))
        per_csv_rows.append(row_out)

    oracle_handle = np.concatenate(all_oracle_handle)
    oracle_tip = np.concatenate(all_oracle_tip)
    oracle_axis = np.concatenate(all_oracle_axis)
    null_handle = np.concatenate(all_null_handle)
    null_tip = np.concatenate(all_null_tip)
    null_axis = np.concatenate(all_null_axis)
    actions = np.concatenate(all_actions, axis=0)
    action_mag = np.linalg.norm(actions, axis=-1)
    handle_speed = np.linalg.norm(actions[:, :3], axis=-1)
    omega_mag = np.linalg.norm(actions[:, 3:6], axis=-1)

    summary = {
        "clips": len(rows),
        "frames": int(sum(int(r["frame_count"]) for r in per_csv_rows)),
        "dt_seconds": args.dt,
        "dynamics": {
            "state_dim": 6,
            "action_dim": 6,
            "action_fields": ["handle_linear_velocity_root_local[3]", "long_axis_angular_velocity_root_local[3]"],
            "no_physics_collision_or_mass": True,
        },
        "oracle": {
            "handle_error_m": summarize(oracle_handle),
            "tip_error_m": summarize(oracle_tip),
            "long_axis_angle_error_deg": summarize(oracle_axis),
        },
        "null_action": {
            "handle_error_m": summarize(null_handle),
            "tip_error_m": summarize(null_tip),
            "long_axis_angle_error_deg": summarize(null_axis),
        },
        "oracle_action": {
            "full_action_magnitude": summarize(action_mag),
            "handle_speed_m_per_s": summarize(handle_speed),
            "axis_omega_rad_per_s": summarize(omega_mag),
        },
        "invalid_sequence_count": int(invalid),
        "oracle_passed": bool(invalid == 0 and np.max(oracle_tip) < 1e-4 and np.max(oracle_axis) < 1e-3),
        "null_is_nontrivial": bool(np.mean(null_tip) > 0.05 and np.mean(null_axis) > 5.0),
    }

    results_csv = args.output_dir / "oracle_replay_results.csv"
    fieldnames = sorted({k for r in per_csv_rows for k in r.keys()})
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_csv_rows)

    (args.output_dir / "oracle_replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_plots(args.output_dir, rows, per_results)

    report = f"""# Virtual Racket Dynamics V1 Oracle Replay

This is virtual kinematic contract validation only. It is not learned policy performance and not PHC rollout racket accuracy.

## Contract

- state: `handle_root_local[3] + long_axis_root_local[3]`
- action: `handle linear velocity[3] + long-axis angular velocity[3]`
- action dimension: `6`
- dt: `{args.dt}` seconds
- no mass, inertia, collision, shuttle, or physical racket

## Oracle Replay

- clips: `{summary['clips']}`
- frames: `{summary['frames']}`
- oracle passed: `{summary['oracle_passed']}`
- oracle handle error mean/max: `{summary['oracle']['handle_error_m']['mean']:.9e}` / `{summary['oracle']['handle_error_m']['max']:.9e}` m
- oracle tip error mean/max: `{summary['oracle']['tip_error_m']['mean']:.9e}` / `{summary['oracle']['tip_error_m']['max']:.9e}` m
- oracle long-axis error mean/max: `{summary['oracle']['long_axis_angle_error_deg']['mean']:.9e}` / `{summary['oracle']['long_axis_angle_error_deg']['max']:.9e}` deg

## Null-Action Baseline

- null is nontrivial: `{summary['null_is_nontrivial']}`
- null handle error mean/p90/max: `{summary['null_action']['handle_error_m']['mean']:.6f}` / `{summary['null_action']['handle_error_m']['p90']:.6f}` / `{summary['null_action']['handle_error_m']['max']:.6f}` m
- null tip error mean/p90/max: `{summary['null_action']['tip_error_m']['mean']:.6f}` / `{summary['null_action']['tip_error_m']['p90']:.6f}` / `{summary['null_action']['tip_error_m']['max']:.6f}` m
- null long-axis error mean/p90/max: `{summary['null_action']['long_axis_angle_error_deg']['mean']:.6f}` / `{summary['null_action']['long_axis_angle_error_deg']['p90']:.6f}` / `{summary['null_action']['long_axis_angle_error_deg']['max']:.6f}` deg

## Oracle Action Distribution

- full action magnitude mean/p95/p99/max: `{summary['oracle_action']['full_action_magnitude']['mean']:.6f}` / `{summary['oracle_action']['full_action_magnitude']['p95']:.6f}` / `{summary['oracle_action']['full_action_magnitude']['p99']:.6f}` / `{summary['oracle_action']['full_action_magnitude']['max']:.6f}`
- handle speed mean/p95/p99/max: `{summary['oracle_action']['handle_speed_m_per_s']['mean']:.6f}` / `{summary['oracle_action']['handle_speed_m_per_s']['p95']:.6f}` / `{summary['oracle_action']['handle_speed_m_per_s']['p99']:.6f}` / `{summary['oracle_action']['handle_speed_m_per_s']['max']:.6f}` m/s
- axis omega mean/p95/p99/max: `{summary['oracle_action']['axis_omega_rad_per_s']['mean']:.6f}` / `{summary['oracle_action']['axis_omega_rad_per_s']['p95']:.6f}` / `{summary['oracle_action']['axis_omega_rad_per_s']['p99']:.6f}` / `{summary['oracle_action']['axis_omega_rad_per_s']['max']:.6f}` rad/s

The oracle replay shows the kinematic state/action contract can exactly express the reference target trajectory when given ground-truth consecutive target transitions. The null-action baseline shows the target trajectory is dynamic and cannot be tracked by simply holding the initial virtual racket state.
"""
    (args.output_dir / "oracle_replay_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
