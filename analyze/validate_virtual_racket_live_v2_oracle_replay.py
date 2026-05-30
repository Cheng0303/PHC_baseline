#!/usr/bin/env python3
"""No-training oracle replay validation for Live Virtual Racket V2.

This validates the world-persistent virtual racket contract. It uses reference
root rotations to express oracle actions in root-local coordinates, then
integrates the realized virtual racket state in PHC/world coordinates.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from virtual_racket_control_contract import DEFAULT_MANIFEST, DEFAULT_OUT_DIR, axis_angle_error_deg, load_manifest, summarize
from virtual_racket_live_contract_v2 import LiveVirtualRacketDynamicsV2, LiveVirtualRacketStateV2, load_live_target, validate_world_targets


def _errors(realized: LiveVirtualRacketStateV2, target: LiveVirtualRacketStateV2, target_tip: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "handle": np.linalg.norm(realized.handle_phc_world - target.handle_phc_world, axis=-1),
        "tip": np.linalg.norm(realized.tip_phc_world - target_tip, axis=-1),
        "axis_deg": axis_angle_error_deg(realized.long_axis_phc_world, target.long_axis_phc_world),
    }


def replay_one(npz_path: Path, dt: float) -> dict[str, np.ndarray]:
    target, target_tip, root_rot, length = load_live_target(npz_path)
    n = len(target.handle_phc_world)
    dyn = LiveVirtualRacketDynamicsV2()

    oracle_handle = np.zeros_like(target.handle_phc_world)
    oracle_axis = np.zeros_like(target.long_axis_phc_world)
    null_handle = np.zeros_like(target.handle_phc_world)
    null_axis = np.zeros_like(target.long_axis_phc_world)
    oracle_handle[0] = target.handle_phc_world[0]
    oracle_axis[0] = target.long_axis_phc_world[0]
    null_handle[0] = target.handle_phc_world[0]
    null_axis[0] = target.long_axis_phc_world[0]

    actions = []
    for t in range(n - 1):
        curr_target = LiveVirtualRacketStateV2(target.handle_phc_world[t : t + 1], target.long_axis_phc_world[t : t + 1], length[t : t + 1])
        next_target = LiveVirtualRacketStateV2(
            target.handle_phc_world[t + 1 : t + 2],
            target.long_axis_phc_world[t + 1 : t + 2],
            length[t + 1 : t + 2],
        )
        action = dyn.derive_oracle_action_local(curr_target, next_target, root_rot[t : t + 1], dt)[0]
        actions.append(action)

        oracle_curr = LiveVirtualRacketStateV2(oracle_handle[t : t + 1], oracle_axis[t : t + 1], length[t : t + 1])
        oracle_next = dyn.step(oracle_curr, action[None, :], root_rot[t : t + 1], dt)
        oracle_handle[t + 1] = oracle_next.handle_phc_world[0]
        oracle_axis[t + 1] = oracle_next.long_axis_phc_world[0]

        null_curr = LiveVirtualRacketStateV2(null_handle[t : t + 1], null_axis[t : t + 1], length[t : t + 1])
        null_next = dyn.step(null_curr, np.zeros((1, 6), dtype=np.float64), root_rot[t : t + 1], dt)
        null_handle[t + 1] = null_next.handle_phc_world[0]
        null_axis[t + 1] = null_next.long_axis_phc_world[0]

    actions_arr = np.asarray(actions, dtype=np.float64) if actions else np.zeros((0, 6), dtype=np.float64)
    oracle = LiveVirtualRacketStateV2(oracle_handle, oracle_axis, length)
    null = LiveVirtualRacketStateV2(null_handle, null_axis, length)
    oracle_errors = _errors(oracle, target, target_tip)
    null_errors = _errors(null, target, target_tip)
    return {
        "oracle_handle_error": oracle_errors["handle"],
        "oracle_tip_error": oracle_errors["tip"],
        "oracle_axis_error_deg": oracle_errors["axis_deg"],
        "null_handle_error": null_errors["handle"],
        "null_tip_error": null_errors["tip"],
        "null_axis_error_deg": null_errors["axis_deg"],
        "action": actions_arr,
        "target_tip": target_tip,
        "oracle_tip": oracle.tip_phc_world,
        "null_tip": null.tip_phc_world,
    }


def _metric_summary(prefix: str, handle: np.ndarray, tip: np.ndarray, axis: np.ndarray) -> dict[str, float]:
    out = {}
    for name, values, unit in [
        ("handle_error", handle, "m"),
        ("tip_error", tip, "m"),
        ("long_axis_angle_error", axis, "deg"),
    ]:
        for stat, value in summarize(values).items():
            out[f"{prefix}_{name}_{stat}_{unit}"] = value
    return out


def _write_plots(out_dir: Path, rows: list[dict[str, str]], results: dict[str, dict[str, np.ndarray]]) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    oracle_tip = np.concatenate([r["oracle_tip_error"] for r in results.values()])
    null_tip = np.concatenate([r["null_tip_error"] for r in results.values()])
    plt.figure(figsize=(7, 4))
    plt.boxplot([oracle_tip, null_tip], labels=["oracle", "null"], showfliers=False)
    plt.ylabel("world tip error (m)")
    plt.title("virtual kinematic contract validation only\nnot learned policy performance; not PHC rollout racket accuracy")
    plt.tight_layout()
    plt.savefig(plot_dir / "live_v2_oracle_vs_null_tip_error_comparison.png", dpi=160)
    plt.close()

    actions = np.concatenate([r["action"] for r in results.values() if len(r["action"])], axis=0)
    plt.figure(figsize=(7, 4))
    plt.hist(np.linalg.norm(actions[:, :3], axis=-1), bins=80, alpha=0.65, label="handle velocity")
    plt.hist(np.linalg.norm(actions[:, 3:6], axis=-1), bins=80, alpha=0.65, label="axis omega")
    plt.legend()
    plt.xlabel("root-local action magnitude")
    plt.title("oracle action distribution\nvirtual kinematic contract validation only")
    plt.tight_layout()
    plt.savefig(plot_dir / "live_v2_oracle_action_magnitude_distribution.png", dpi=160)
    plt.close()

    rep_seq = rows[-1]["sequence"]
    rep = results[rep_seq]
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(rep["target_tip"][:, 0], rep["target_tip"][:, 1], rep["target_tip"][:, 2], label="target")
    ax.plot(rep["oracle_tip"][:, 0], rep["oracle_tip"][:, 1], rep["oracle_tip"][:, 2], "--", label="oracle")
    ax.plot(rep["null_tip"][:, 0], rep["null_tip"][:, 1], rep["null_tip"][:, 2], ":", label="null")
    ax.set_title(f"Live V2 representative replay: {rep_seq}\nnot learned policy performance; not PHC rollout racket accuracy")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"live_v2_representative_trajectory_{rep_seq.replace('/', '_')}.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dt", type=float, default=1.0 / 30.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(args.manifest)

    target_summary = validate_world_targets(rows, args.output_dir)
    per_rows = []
    results = {}
    oracle_handle_all, oracle_tip_all, oracle_axis_all = [], [], []
    null_handle_all, null_tip_all, null_axis_all = [], [], []
    actions_all = []
    invalid = 0

    for row in rows:
        seq = row["sequence"]
        result = replay_one(Path(row["npz_path"]), args.dt)
        results[seq] = result
        finite = all(np.isfinite(value).all() for value in result.values() if isinstance(value, np.ndarray))
        invalid += 0 if finite else 1
        oracle_handle_all.append(result["oracle_handle_error"])
        oracle_tip_all.append(result["oracle_tip_error"])
        oracle_axis_all.append(result["oracle_axis_error_deg"])
        null_handle_all.append(result["null_handle_error"])
        null_tip_all.append(result["null_tip_error"])
        null_axis_all.append(result["null_axis_error_deg"])
        if len(result["action"]):
            actions_all.append(result["action"])
        csv_row = {"sequence": seq, "frame_count": len(result["oracle_tip_error"]), "finite": finite}
        csv_row.update(_metric_summary("oracle", result["oracle_handle_error"], result["oracle_tip_error"], result["oracle_axis_error_deg"]))
        csv_row.update(_metric_summary("null", result["null_handle_error"], result["null_tip_error"], result["null_axis_error_deg"]))
        per_rows.append(csv_row)

    oracle_handle = np.concatenate(oracle_handle_all)
    oracle_tip = np.concatenate(oracle_tip_all)
    oracle_axis = np.concatenate(oracle_axis_all)
    null_handle = np.concatenate(null_handle_all)
    null_tip = np.concatenate(null_tip_all)
    null_axis = np.concatenate(null_axis_all)
    actions = np.concatenate(actions_all, axis=0)

    summary = {
        "clips": len(rows),
        "frames": int(sum(row["frame_count"] for row in per_rows)),
        "dt_seconds": args.dt,
        "contract": {
            "version": "v2_world_internal_root_local_action",
            "internal_state_frame": "PHC/world",
            "internal_state_fields": ["handle_phc_world[3]", "long_axis_phc_world[3]"],
            "internal_state_dim": 6,
            "action_frame": "reference/sim root local, converted to world by root rotation",
            "action_fields": ["v_handle_root_local[3]", "omega_axis_root_local[3]"],
            "action_dim": 6,
            "policy_target_goal_remains": "Goal V1 root-local geometry [9]",
            "no_physics_collision_or_mass": True,
        },
        "world_target_consistency": target_summary,
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
            "handle_speed_root_local_m_per_s": summarize(np.linalg.norm(actions[:, :3], axis=-1)),
            "axis_omega_root_local_rad_per_s": summarize(np.linalg.norm(actions[:, 3:6], axis=-1)),
            "full_action_magnitude": summarize(np.linalg.norm(actions, axis=-1)),
        },
        "invalid_sequence_count": int(invalid),
        "oracle_passed": bool(invalid == 0 and np.max(oracle_tip) < 1e-4 and np.max(oracle_axis) < 1e-3),
        "null_is_nontrivial": bool(np.mean(null_tip) > 0.05 and np.mean(null_axis) > 5.0),
        "scope": "offline reference-level contract validation only; no policy, no training, no PHC rollout racket accuracy",
    }

    csv_path = args.output_dir / "live_v2_oracle_replay_results.csv"
    fieldnames = sorted({key for row in per_rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_rows)

    (args.output_dir / "live_v2_oracle_replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_plots(args.output_dir, rows, results)

    report = f"""# Live Virtual Racket V2 Oracle Replay

This is virtual kinematic contract validation only. It is not learned policy performance and not PHC rollout racket accuracy.

## Frame Semantics

- Internal realized state: PHC/world coordinates.
- Policy-facing target remains Goal V1 root-local geometry.
- Action: root-local 6D velocity command converted to world by the root rotation for integration.
- No mass, inertia, collision, shuttle, or physical racket is introduced.

## Dataset

- clips: `{summary['clips']}`
- frames: `{summary['frames']}`
- dt: `{summary['dt_seconds']}` seconds
- world target consistency passed: `{target_summary['world_target_consistency_passed']}`
- oracle passed: `{summary['oracle_passed']}`
- null action nontrivial: `{summary['null_is_nontrivial']}`

## Oracle Replay

- handle error mean/max: `{summary['oracle']['handle_error_m']['mean']:.9e}` / `{summary['oracle']['handle_error_m']['max']:.9e}` m
- tip error mean/max: `{summary['oracle']['tip_error_m']['mean']:.9e}` / `{summary['oracle']['tip_error_m']['max']:.9e}` m
- long-axis error mean/max: `{summary['oracle']['long_axis_angle_error_deg']['mean']:.9e}` / `{summary['oracle']['long_axis_angle_error_deg']['max']:.9e}` deg

## Null Action Baseline

- handle error mean/p90/max: `{summary['null_action']['handle_error_m']['mean']:.6f}` / `{summary['null_action']['handle_error_m']['p90']:.6f}` / `{summary['null_action']['handle_error_m']['max']:.6f}` m
- tip error mean/p90/max: `{summary['null_action']['tip_error_m']['mean']:.6f}` / `{summary['null_action']['tip_error_m']['p90']:.6f}` / `{summary['null_action']['tip_error_m']['max']:.6f}` m
- long-axis error mean/p90/max: `{summary['null_action']['long_axis_angle_error_deg']['mean']:.6f}` / `{summary['null_action']['long_axis_angle_error_deg']['p90']:.6f}` / `{summary['null_action']['long_axis_angle_error_deg']['max']:.6f}` deg

## Relation To V1

Root-local V1 remains the controller-facing target representation. Live V2 changes the realized virtual object persistence frame to PHC/world coordinates so that future simulated hand consistency and world-space tracking metrics have stable semantics.
"""
    (args.output_dir / "live_v2_oracle_replay_report.md").write_text(report, encoding="utf-8")
    return 0 if summary["oracle_passed"] and target_summary["world_target_consistency_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
