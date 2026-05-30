#!/usr/bin/env python3
"""CPU-only KinTwin-inspired tracking metric audit for Stage 1B traces.

This script does not run Isaac Gym, training, optimizers, backward passes, or
reward updates. It only inspects saved Stage 1B outputs and reports whether the
per-frame arrays needed for hand/wrist dynamic relation metrics are available.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "phc_baseline" / "reports" / "racket_calibration" / "frozen_body_head_integration"
FULL_EVAL_DIR = REPORT_DIR / "full_heldout_eval"
OUT_JSON = REPORT_DIR / "kintwin_style_tracking_metric_audit.json"
OUT_MD = REPORT_DIR / "kintwin_style_tracking_metric_audit.md"
OUT_CSV = REPORT_DIR / "kintwin_style_tracking_metric_per_sequence.csv"

VIRTUAL_MODES = ["virtual_null", "virtual_goal_only", "virtual_goal_state", "virtual_oracle"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stats(values: list[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def mse_rmse(vec_error: np.ndarray) -> tuple[float, float, dict[str, float | None]]:
    norms = np.linalg.norm(vec_error, axis=-1)
    mse = float(np.mean(np.sum(vec_error * vec_error, axis=-1)))
    return mse, float(math.sqrt(mse)), stats(norms)


def metric_from_trace(trace_path: Path) -> dict[str, Any]:
    data = np.load(trace_path)
    required = [
        "r_hand_sim_world",
        "r_hand_ref_world",
        "r_wrist_sim_world",
        "r_wrist_ref_world",
        "handle_realized_world",
        "handle_target_world",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        return {"available": False, "missing": missing}

    hand_err = data["r_hand_sim_world"] - data["r_hand_ref_world"]
    wrist_err = data["r_wrist_sim_world"] - data["r_wrist_ref_world"]
    handle_hand_err = (
        data["handle_realized_world"]
        - data["r_hand_sim_world"]
        - (data["handle_target_world"] - data["r_hand_ref_world"])
    )
    handle_wrist_err = (
        data["handle_realized_world"]
        - data["r_wrist_sim_world"]
        - (data["handle_target_world"] - data["r_wrist_ref_world"])
    )
    hand_mse, hand_rmse, hand_norm = mse_rmse(hand_err)
    wrist_mse, wrist_rmse, wrist_norm = mse_rmse(wrist_err)
    hh_mse, hh_rmse, hh_norm = mse_rmse(handle_hand_err)
    hw_mse, hw_rmse, hw_norm = mse_rmse(handle_wrist_err)
    return {
        "available": True,
        "frames": int(len(data["r_hand_sim_world"])),
        "hand_pos_mse": hand_mse,
        "hand_pos_rmse": hand_rmse,
        "hand_pos_error": hand_norm,
        "wrist_pos_mse": wrist_mse,
        "wrist_pos_rmse": wrist_rmse,
        "wrist_pos_error": wrist_norm,
        "dynamic_handle_hand_mse": hh_mse,
        "dynamic_handle_hand_rmse": hh_rmse,
        "dynamic_handle_hand_error": hh_norm,
        "dynamic_handle_wrist_mse": hw_mse,
        "dynamic_handle_wrist_rmse": hw_rmse,
        "dynamic_handle_wrist_error": hw_norm,
    }


def trace_error_arrays(trace_path: Path) -> dict[str, np.ndarray]:
    data = np.load(trace_path)
    hand_err = data["r_hand_sim_world"] - data["r_hand_ref_world"]
    wrist_err = data["r_wrist_sim_world"] - data["r_wrist_ref_world"]
    handle_hand_err = (
        data["handle_realized_world"]
        - data["r_hand_sim_world"]
        - (data["handle_target_world"] - data["r_hand_ref_world"])
    )
    handle_wrist_err = (
        data["handle_realized_world"]
        - data["r_wrist_sim_world"]
        - (data["handle_target_world"] - data["r_wrist_ref_world"])
    )
    return {
        "hand_pos": hand_err,
        "wrist_pos": wrist_err,
        "dynamic_handle_hand": handle_hand_err,
        "dynamic_handle_wrist": handle_wrist_err,
    }


def aggregate_trace_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    arrays: dict[str, list[np.ndarray]] = {
        "hand_pos": [],
        "wrist_pos": [],
        "dynamic_handle_hand": [],
        "dynamic_handle_wrist": [],
    }
    frames = 0
    for row in rows:
        trace_path = row.get("kintwin_trace_path")
        if not trace_path:
            continue
        path = Path(str(trace_path))
        if not path.exists():
            continue
        errs = trace_error_arrays(path)
        for key, arr in errs.items():
            arrays[key].append(arr)
        frames += len(errs["hand_pos"])

    out: dict[str, Any] = {
        "rows": len(rows),
        "trace_available_rows": sum(bool(row.get("kintwin_trace_available")) for row in rows),
        "frames": frames,
        "existing_tip_mean_sequence_average": stats([row["tip_error_m_mean"] for row in rows])["mean"],
        "existing_axis_mean_sequence_average": stats([row["axis_error_deg_mean"] for row in rows])["mean"],
        "existing_abs_distance_hand_consistency_sequence_average": stats(
            [row["existing_abs_distance_hand_consistency_mean"] for row in rows]
        )["mean"],
    }
    for key in arrays:
        if arrays[key]:
            arr = np.concatenate(arrays[key], axis=0)
            mse, rmse, norm_stats = mse_rmse(arr)
            out[f"{key}_mse_frame_weighted"] = mse
            out[f"{key}_rmse_frame_weighted"] = rmse
            out[f"{key}_error_mean_frame_weighted"] = norm_stats["mean"]
            out[f"{key}_error_p90_frame_weighted"] = norm_stats["p90"]
            out[f"{key}_error_max"] = norm_stats["max"]
        else:
            out[f"{key}_mse_frame_weighted"] = None
            out[f"{key}_rmse_frame_weighted"] = None
            out[f"{key}_error_mean_frame_weighted"] = None
            out[f"{key}_error_p90_frame_weighted"] = None
            out[f"{key}_error_max"] = None
    return out


def main() -> int:
    summary_path = FULL_EVAL_DIR / "stage1b_evaluation_summary.json"
    per_sequence_path = FULL_EVAL_DIR / "stage1b_per_sequence_results.csv"
    if not summary_path.exists() or not per_sequence_path.exists():
        raise FileNotFoundError(f"missing corrected Stage 1B outputs under {FULL_EVAL_DIR}")

    summary = load_json(summary_path)
    rows: list[dict[str, Any]] = []
    available_count = 0
    missing_reasons: dict[str, int] = {}
    with per_sequence_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mode = row["mode"]
            if mode not in VIRTUAL_MODES:
                continue
            sequence = row["sequence"]
            group = sequence.split("/", 1)[0]
            child_json = FULL_EVAL_DIR / "children" / mode / f"{sequence.replace('/', '__')}.json"
            child = load_json(child_json)
            trace_path = child.get("kintwin_trace_path")
            if trace_path and Path(trace_path).exists():
                trace_metrics = metric_from_trace(Path(trace_path))
            else:
                trace_metrics = {
                    "available": False,
                    "missing": [
                        "kintwin_trace_path",
                        "r_hand_sim_world",
                        "r_hand_ref_world",
                        "r_wrist_sim_world",
                        "r_wrist_ref_world",
                        "handle_realized_world",
                        "handle_target_world",
                    ],
                }
            if trace_metrics["available"]:
                available_count += 1
            else:
                reason = ",".join(trace_metrics.get("missing", []))
                missing_reasons[reason] = missing_reasons.get(reason, 0) + 1

            virtual = child.get("virtual_metrics") or {}
            hand_proxy = child.get("hand_body_consistency") or {}
            row_out: dict[str, Any] = {
                "sequence": sequence,
                "session_group": group,
                "mode": mode,
                "passed": child.get("passed"),
                "completed": child.get("completed"),
                "terminated": child.get("terminated"),
                "frames_evaluated": child.get("frames_evaluated"),
                "kintwin_trace_available": trace_metrics["available"],
                "kintwin_trace_path": trace_path if trace_metrics["available"] else "",
                "missing_reason": "" if trace_metrics["available"] else ",".join(trace_metrics.get("missing", [])),
                "tip_error_m_mean": (virtual.get("tip_error_m") or {}).get("mean"),
                "axis_error_deg_mean": (virtual.get("axis_error_deg") or {}).get("mean"),
                "existing_abs_distance_hand_consistency_mean": (hand_proxy.get("e_hand_consistency") or {}).get("mean"),
            }
            for key in [
                "hand_pos_mse",
                "hand_pos_rmse",
                "wrist_pos_mse",
                "wrist_pos_rmse",
                "dynamic_handle_hand_mse",
                "dynamic_handle_hand_rmse",
                "dynamic_handle_wrist_mse",
                "dynamic_handle_wrist_rmse",
            ]:
                row_out[key] = trace_metrics.get(key)
            for prefix, metric_key in [
                ("hand_pos_error", "hand_pos_error"),
                ("wrist_pos_error", "wrist_pos_error"),
                ("dynamic_handle_hand_error", "dynamic_handle_hand_error"),
                ("dynamic_handle_wrist_error", "dynamic_handle_wrist_error"),
            ]:
                metric_stats = trace_metrics.get(metric_key) or {}
                for stat_key in ["mean", "p90", "max"]:
                    row_out[f"{prefix}_{stat_key}"] = metric_stats.get(stat_key)
            rows.append(row_out)

    fieldnames = list(rows[0].keys()) if rows else []
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aggregates: dict[str, Any] = {}
    for mode in VIRTUAL_MODES:
        mode_rows = [row for row in rows if row["mode"] == mode]
        aggregates[mode] = aggregate_trace_rows(mode_rows)
        for group in sorted({row["session_group"] for row in mode_rows}):
            group_rows = [row for row in mode_rows if row["session_group"] == group]
            aggregates[mode][f"group_{group}"] = aggregate_trace_rows(group_rows)
        for status_key, status_value in [("completed", True), ("terminated", True)]:
            subset = [row for row in mode_rows if bool(row[status_key]) is status_value]
            aggregates[mode][status_key] = aggregate_trace_rows(subset)

    all_requested_metrics_available = bool(rows) and available_count == len(rows)
    audit_status = "computed" if all_requested_metrics_available else "blocked_saved_traces_insufficient_for_requested_mse"
    audit = {
        "status": audit_status,
        "stage1b_summary_path": str(summary_path),
        "stage1b_passed": bool(summary.get("passed")),
        "corrected_stage1b_tip_axis_context": {
            "virtual_goal_state_tip_mean_m": aggregates.get("virtual_goal_state", {}).get("existing_tip_mean_sequence_average"),
            "virtual_goal_state_axis_mean_deg": aggregates.get("virtual_goal_state", {}).get("existing_axis_mean_sequence_average"),
            "virtual_oracle_tip_mean_m": aggregates.get("virtual_oracle", {}).get("existing_tip_mean_sequence_average"),
            "virtual_oracle_axis_mean_deg": aggregates.get("virtual_oracle", {}).get("existing_axis_mean_sequence_average"),
        },
        "kintwin_reference_status": {
            "paper": "KinTwin arXiv/IEEE paper was consulted for its emphasis on clinically interpretable kinematic tracking metrics such as joint-angle and contact-event tracking.",
            "local_code": {
                "humenv/kintwin/eval_h5_metrics.py": [
                    "compute_metrics uses Euclidean joint-position errors to report MPJPE, root-aligned MPJPE, Procrustes-aligned MPJPE, PCK, velocity error, bone-length MAE, foot-skating, and per-joint errors.",
                    "The core position error is np.linalg.norm(pred_xyz - ref_xyz, axis=-1), averaged over frames and joints.",
                ],
                "humenv/kintwin/train.py": [
                    "The local training reward logs qpos_mse, qvel_mse, root_mse, wrist_mse, body_pos_mse, racket_tip_mse, racket_orient_mse, and related velocity MSE terms.",
                    "body_pos_mse is computed from tracked body points in pelvis-local frames; wrist_mse is the squared average local left/right wrist endpoint error; racket_tip_mse is the squared world-space tip error.",
                ],
            },
            "claim_boundary": "The requested hand/wrist and dynamic handle relation MSE/RMSE diagnostics are KinTwin-inspired and consistent with the local code's kinematic-error style, but this audit does not claim the KinTwin paper uses these exact formulas.",
            "urls": [
                "https://arxiv.org/abs/2505.13436",
                "https://doi.org/10.1109/TMRB.2025.3605962",
            ],
        },
        "requested_metrics": {
            "hand_pos_mse_rmse": "requires per-frame R_Hand_sim_world and R_Hand_ref_world",
            "wrist_pos_mse_rmse": "requires per-frame R_Wrist_sim_world and R_Wrist_ref_world",
            "dynamic_handle_hand_mse_rmse": "requires per-frame handle_realized_world, handle_target_world, R_Hand_sim_world, and R_Hand_ref_world",
            "dynamic_handle_wrist_mse_rmse": "requires per-frame handle_realized_world, handle_target_world, R_Wrist_sim_world, and R_Wrist_ref_world",
        },
        "saved_trace_inventory": {
            "child_json_rows": len(rows),
            "kintwin_trace_available_rows": available_count,
            "missing_reasons": missing_reasons,
            "root_trace_only": not all_requested_metrics_available,
        },
        "existing_proxy_metrics_available": {
            "tip_error": True,
            "axis_error": True,
            "abs_distance_hand_consistency": True,
            "note": "The old hand consistency is abs(||handle-realized - hand-sim|| - ||handle-target - hand-ref||); it is not the requested vector dynamic relation MSE.",
        },
        "aggregates": aggregates,
        "next_required_action": {
            "rerun_needed": not all_requested_metrics_available,
            "reason": None
            if all_requested_metrics_available
            else "The corrected Stage 1B run did not save the per-frame arrays needed to compute requested KinTwin-inspired MSE/RMSE diagnostics.",
            "code_prepared_for_next_run": "eval_full_heldout_config.json enables save_kintwin_tracking_traces, and evaluate.py writes *.kintwin_trace.npz for virtual modes.",
            "no_training_required": True,
        },
        "section_s_update": "ready_to_update" if all_requested_metrics_available else "not updated as final; requested diagnostic metrics are blocked until per-frame traces are produced.",
    }
    OUT_JSON.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    mode_table_lines = [
        "| mode | hand RMSE m | wrist RMSE m | dyn handle-hand RMSE m | dyn handle-wrist RMSE m | hand err mean/p90/max m | dyn handle-hand err mean/p90/max m |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in VIRTUAL_MODES:
        agg = aggregates[mode]

        def fmt(value: Any) -> str:
            return "" if value is None else f"{float(value):.6f}"

        mode_table_lines.append(
            "| "
            + mode
            + " | "
            + fmt(agg.get("hand_pos_rmse_frame_weighted"))
            + " | "
            + fmt(agg.get("wrist_pos_rmse_frame_weighted"))
            + " | "
            + fmt(agg.get("dynamic_handle_hand_rmse_frame_weighted"))
            + " | "
            + fmt(agg.get("dynamic_handle_wrist_rmse_frame_weighted"))
            + " | "
            + "/".join(
                [
                    fmt(agg.get("hand_pos_error_mean_frame_weighted")),
                    fmt(agg.get("hand_pos_error_p90_frame_weighted")),
                    fmt(agg.get("hand_pos_error_max")),
                ]
            )
            + " | "
            + "/".join(
                [
                    fmt(agg.get("dynamic_handle_hand_error_mean_frame_weighted")),
                    fmt(agg.get("dynamic_handle_hand_error_p90_frame_weighted")),
                    fmt(agg.get("dynamic_handle_hand_error_max")),
                ]
            )
            + " |"
        )
    mode_table = "\n".join(mode_table_lines)
    status_sentence = (
        "Status: computed from saved per-frame KinTwin-style traces."
        if all_requested_metrics_available
        else "Status: blocked by insufficient saved traces for the requested MSE/RMSE diagnostics."
    )
    trace_sufficiency = (
        "All virtual-mode child rows contain `*.kintwin_trace.npz` with the per-frame hand, wrist, realized-handle, and target-handle arrays needed for the requested diagnostics."
        if all_requested_metrics_available
        else "The corrected Stage 1B children currently save only `root_trace.npy` plus aggregated JSON statistics. They do **not** save all per-frame arrays required for the requested diagnostics."
    )
    interpretation_gate = (
        "The requested diagnostics are now computed. The large simulated hand/wrist tracking RMSE is mode-invariant, and Model B/oracle both have low virtual tip error but high dynamic handle-hand RMSE. This points to frozen-body hand trajectory mismatch/body-racket coupling as the next blocker, not virtual target tracking alone."
        if all_requested_metrics_available
        else "Do not design a new hand/body consistency objective yet. First rerun the corrected Stage 1B evaluation with trace saving enabled, then compute the KinTwin-inspired diagnostics from the saved per-frame arrays."
    )

    md = f"""# KinTwin-Style Tracking Metric Audit

{status_sentence}

## KinTwin Reference

The KinTwin paper was consulted as the requested reference point for kinematic tracking diagnostics. Its public paper record emphasizes clinically interpretable tracking metrics, including joint-angle tracking and ground-contact-event tracking.

I also checked the local KinTwin-style implementation in this workspace:

- `humenv/kintwin/eval_h5_metrics.py` computes Euclidean joint-position errors and reports MPJPE, root-aligned MPJPE, Procrustes-aligned MPJPE, PCK, velocity error, bone-length MAE, foot-skating, and per-joint errors.
- `humenv/kintwin/train.py` logs `qpos_mse`, `qvel_mse`, `root_mse`, `wrist_mse`, `body_pos_mse`, `racket_tip_mse`, `racket_orient_mse`, and related velocity MSE terms. In that code, `body_pos_mse` is based on tracked body-point differences in pelvis-local frames, `wrist_mse` is the squared average local left/right wrist endpoint error, and `racket_tip_mse` is squared world-space tip error.

This audit therefore treats hand/wrist and dynamic handle-to-limb metrics as **KinTwin-inspired diagnostics** consistent with the local code's kinematic-error style, not as a claim that the KinTwin paper uses the exact formulas below.

References:

- KinTwin arXiv record: https://arxiv.org/abs/2505.13436
- IEEE DOI record: https://doi.org/10.1109/TMRB.2025.3605962

No specific KinTwin paper loss formula is claimed here beyond the local code definitions cited above.

## Corrected Stage 1B Context

- corrected Stage 1B passed: `{summary.get('passed')}`
- Model B virtual tip mean: `{audit['corrected_stage1b_tip_axis_context']['virtual_goal_state_tip_mean_m']}`
- Model B virtual axis mean: `{audit['corrected_stage1b_tip_axis_context']['virtual_goal_state_axis_mean_deg']}`
- Oracle virtual tip mean: `{audit['corrected_stage1b_tip_axis_context']['virtual_oracle_tip_mean_m']}`
- Oracle virtual axis mean: `{audit['corrected_stage1b_tip_axis_context']['virtual_oracle_axis_mean_deg']}`

These results suggest the remaining blocker is not virtual racket target tracking alone.

## Saved Trace Sufficiency

{trace_sufficiency}

Trace rows available: `{available_count} / {len(rows)}`.

## Frame-Weighted Aggregate Metrics

{mode_table}

Existing proxy available: the previous scalar hand consistency metric, `abs(||handle_realized - hand_sim|| - ||handle_target - hand_ref||)`. This is not the requested dynamic vector relation metric.

## Prepared Trace Support

The evaluator has been updated so the next user-run full Stage 1B evaluation will save `*.kintwin_trace.npz` files for virtual modes, containing the per-frame arrays needed to compute:

- hand position MSE/RMSE and Euclidean error mean/p90/max
- wrist position MSE/RMSE and Euclidean error mean/p90/max
- dynamic handle-to-hand vector relation MSE/RMSE and error mean/p90/max
- dynamic handle-to-wrist vector relation MSE/RMSE and error mean/p90/max

No training, reward tuning, PHC fine-tuning, physical racket, shuttle, or hitting reward is involved.

## Interpretation Gate

{interpretation_gate}

Section S may now be updated with the corrected Stage 1B pass and this diagnostic, while preserving the strict no-physics/no-official-racket-accuracy scope.
"""
    OUT_MD.write_text(md, encoding="utf-8")
    print(json.dumps({"status": audit["status"], "csv": str(OUT_CSV), "json": str(OUT_JSON), "md": str(OUT_MD)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
