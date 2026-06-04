#!/usr/bin/env python3
"""Stage 1D CPU audit for frozen-body rollout/body-metric mismatch.

This script only reads saved Stage 1B traces and existing reports/configs. It
does not import Isaac Gym, run GPU evaluation, train, optimize, backpropagate,
tune rewards, or modify any checkpoint.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PHC_BASELINE = WORKSPACE_ROOT / "phc_baseline"
PHC_ROOT = WORKSPACE_ROOT / "humenv" / "data_preparation" / "PHC"
REPORT_DIR = PHC_BASELINE / "reports" / "racket_calibration" / "frozen_body_head_integration"
FULL_EVAL_DIR = REPORT_DIR / "full_heldout_eval"
CHILD_DIR = FULL_EVAL_DIR / "children"

STAGE1B_CONFIG = PHC_BASELINE / "configs" / "frozen_body_virtual_racket_stage1b" / "eval_full_heldout_config.json"
EARLIER_BASELINE_JSON = PHC_BASELINE / "reports" / "phc_newracket_dataset_metrics_fixed.json"
EARLIER_BASELINE_LOG = PHC_BASELINE / "reports" / "phc_newracket_dataset_metrics_fixed_tmux.log"
EARLIER_BASELINE_SCRIPT = PHC_BASELINE / "run_phc_badminton_dataset_metrics.sh"
USER_REPRO_JSON = REPORT_DIR / "stage1d_user_run_original_baseline_repro_full.json"
USER_REPRO_LOG = REPORT_DIR / "stage1d_user_run_original_baseline_repro_full.log"
SAVED_HYDRA_CONFIG = PHC_ROOT / "output" / "HumanoidIm" / "phc_comp_3" / ".hydra" / "config.yaml"
SAVED_HYDRA_OVERRIDES = PHC_ROOT / "output" / "HumanoidIm" / "phc_comp_3" / ".hydra" / "overrides.yaml"

ALL_MODES = ["body_only", "virtual_null", "virtual_goal_only", "virtual_goal_state", "virtual_oracle"]
VIRTUAL_MODES = ["virtual_null", "virtual_goal_only", "virtual_goal_state", "virtual_oracle"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def stats(values: list[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p50": None, "p90": None, "max": None, "min": None}
    return {
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
    }


def quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    return (
        v * (2.0 * q_w * q_w - 1.0)
        + np.cross(q_vec, v) * q_w * 2.0
        + q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    )


def heading_local_vector_xyzw(vec: np.ndarray, root_rot_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(root_rot_xyzw, dtype=np.float64)
    vec = np.asarray(vec, dtype=np.float64)
    ref = np.zeros_like(vec, dtype=np.float64)
    ref[..., 0] = 1.0
    rot_dir = quat_rotate_xyzw(q, ref)
    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    c = np.cos(heading)
    s = np.sin(heading)
    out = vec.copy()
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    out[..., 0] = c * x + s * y
    out[..., 1] = -s * x + c * y
    return out


def child_json_path(mode: str, sequence: str) -> Path:
    return CHILD_DIR / mode / f"{sequence.replace('/', '__')}.json"


def load_child_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    csv_path = FULL_EVAL_DIR / "stage1b_per_sequence_results.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            child_path = child_json_path(row["mode"], row["sequence"])
            child = load_json(child_path) if child_path.exists() else {}
            rows.append({**row, "child_path": str(child_path), "child": child})
    return rows


def load_trace_from_child(child: dict[str, Any]) -> dict[str, np.ndarray] | None:
    path = child.get("validity_trace_path")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p, allow_pickle=True)
    return {key: data[key] for key in data.files}


def trace_metrics(trace: dict[str, np.ndarray]) -> dict[str, Any]:
    body_sim = np.asarray(trace["body_sim_world"], dtype=np.float64)
    body_ref = np.asarray(trace["body_ref_world"], dtype=np.float64)
    root_sim = np.asarray(trace["root_sim_world"], dtype=np.float64)
    root_ref = np.asarray(trace["root_ref_world"], dtype=np.float64)
    root_sim_rot = np.asarray(trace["root_sim_rot"], dtype=np.float64)
    root_ref_rot = np.asarray(trace["root_ref_rot"], dtype=np.float64)

    world_body_err = np.linalg.norm(body_sim - body_ref, axis=-1)
    root_err = np.linalg.norm(root_sim - root_ref, axis=-1)
    root_aligned_err = np.linalg.norm((body_sim - root_sim[:, None, :]) - (body_ref - root_ref[:, None, :]), axis=-1)

    sim_heading = heading_local_vector_xyzw(body_sim - root_sim[:, None, :], np.repeat(root_sim_rot[:, None, :], body_sim.shape[1], axis=1))
    ref_heading = heading_local_vector_xyzw(body_ref - root_ref[:, None, :], np.repeat(root_ref_rot[:, None, :], body_ref.shape[1], axis=1))
    heading_local_err = np.linalg.norm(sim_heading - ref_heading, axis=-1)

    root_offset = root_sim - root_ref
    constant_offset = np.mean(root_offset, axis=0)
    constant_root_offset_err = np.linalg.norm((body_sim - constant_offset[None, None, :]) - body_ref, axis=-1)
    median_offset = np.median(root_offset, axis=0)
    median_root_offset_err = np.linalg.norm((body_sim - median_offset[None, None, :]) - body_ref, axis=-1)

    per_body_mean = np.mean(world_body_err, axis=0)
    names = [str(x) for x in trace.get("body_names", np.asarray([f"body_{i}" for i in range(body_sim.shape[1])]))]

    motion_time = np.asarray(trace.get("motion_time", []), dtype=np.float64).reshape(-1)
    progress = np.asarray(trace.get("progress_buf", []), dtype=np.float64).reshape(-1)
    sampled_motion_id = np.asarray(trace.get("sampled_motion_id", []), dtype=np.int64).reshape(-1)
    reset = np.asarray(trace.get("reset_buf_after_step", []), dtype=np.bool_).reshape(-1)

    return {
        "frames": int(body_sim.shape[0]),
        "body_count": int(body_sim.shape[1]),
        "world_mpjpe": stats(np.mean(world_body_err, axis=-1)),
        "root_error": stats(root_err),
        "root_aligned_mpjpe": stats(np.mean(root_aligned_err, axis=-1)),
        "heading_local_mpjpe": stats(np.mean(heading_local_err, axis=-1)),
        "constant_root_offset_corrected_mpjpe": stats(np.mean(constant_root_offset_err, axis=-1)),
        "median_root_offset_corrected_mpjpe": stats(np.mean(median_root_offset_err, axis=-1)),
        "root_offset_vector_mean": [float(x) for x in constant_offset],
        "root_offset_vector_median": [float(x) for x in median_offset],
        "root_offset_norm": stats(np.linalg.norm(root_offset, axis=-1)),
        "root_sim_range": {"min": [float(x) for x in np.min(root_sim, axis=0)], "max": [float(x) for x in np.max(root_sim, axis=0)]},
        "root_ref_range": {"min": [float(x) for x in np.min(root_ref, axis=0)], "max": [float(x) for x in np.max(root_ref, axis=0)]},
        "motion_time": {"min": float(np.min(motion_time)) if motion_time.size else None, "max": float(np.max(motion_time)) if motion_time.size else None},
        "progress_buf": {"min": float(np.min(progress)) if progress.size else None, "max": float(np.max(progress)) if progress.size else None},
        "sampled_motion_ids": sorted({int(x) for x in sampled_motion_id.tolist()}) if sampled_motion_id.size else [],
        "reset_count": int(np.sum(reset)) if reset.size else None,
        "per_body_mean_error": {names[i]: float(per_body_mean[i]) for i in range(min(len(names), len(per_body_mean)))},
    }


def weighted_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    weights = []
    for row in rows:
        value = row.get(key)
        frames = row.get("frames")
        if value is not None and frames:
            values.append(float(value))
            weights.append(float(frames))
    if not values:
        return None
    return float(np.average(values, weights=weights))


def summarize_current_body_only(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_sequence: list[dict[str, Any]] = []
    body_rows = [r for r in rows if r["mode"] == "body_only"]
    for row in body_rows:
        trace = load_trace_from_child(row["child"])
        if trace is None:
            continue
        m = trace_metrics(trace)
        child = row["child"]
        per_sequence.append(
            {
                "sequence": row["sequence"],
                "session_group": row["sequence"].split("/")[0],
                "completed": bool(child.get("completed")),
                "terminated": bool(child.get("terminated")),
                "frames": m["frames"],
                "world_mpjpe_mean": m["world_mpjpe"]["mean"],
                "root_error_mean": m["root_error"]["mean"],
                "root_aligned_mpjpe_mean": m["root_aligned_mpjpe"]["mean"],
                "heading_local_mpjpe_mean": m["heading_local_mpjpe"]["mean"],
                "constant_root_offset_corrected_mpjpe_mean": m["constant_root_offset_corrected_mpjpe"]["mean"],
                "median_root_offset_corrected_mpjpe_mean": m["median_root_offset_corrected_mpjpe"]["mean"],
                "root_offset_norm_mean": m["root_offset_norm"]["mean"],
                "root_offset_mean_x": m["root_offset_vector_mean"][0],
                "root_offset_mean_y": m["root_offset_vector_mean"][1],
                "root_offset_mean_z": m["root_offset_vector_mean"][2],
                "motion_time_min": m["motion_time"]["min"],
                "motion_time_max": m["motion_time"]["max"],
                "progress_min": m["progress_buf"]["min"],
                "progress_max": m["progress_buf"]["max"],
                "reset_count": m["reset_count"],
            }
        )

    aggregate = {
        "sequence_count": len(per_sequence),
        "frame_count": int(sum(int(r["frames"]) for r in per_sequence)),
        "completed_count": int(sum(1 for r in per_sequence if r["completed"])),
        "terminated_count": int(sum(1 for r in per_sequence if r["terminated"])),
        "frame_weighted": {
            "world_mpjpe_mean": weighted_mean(per_sequence, "world_mpjpe_mean"),
            "root_error_mean": weighted_mean(per_sequence, "root_error_mean"),
            "root_aligned_mpjpe_mean": weighted_mean(per_sequence, "root_aligned_mpjpe_mean"),
            "heading_local_mpjpe_mean": weighted_mean(per_sequence, "heading_local_mpjpe_mean"),
            "constant_root_offset_corrected_mpjpe_mean": weighted_mean(per_sequence, "constant_root_offset_corrected_mpjpe_mean"),
            "median_root_offset_corrected_mpjpe_mean": weighted_mean(per_sequence, "median_root_offset_corrected_mpjpe_mean"),
            "root_offset_norm_mean": weighted_mean(per_sequence, "root_offset_norm_mean"),
        },
        "completed_frame_weighted": {},
        "terminated_frame_weighted": {},
    }
    for label, pred in [("completed_frame_weighted", lambda r: r["completed"]), ("terminated_frame_weighted", lambda r: r["terminated"])]:
        subset = [r for r in per_sequence if pred(r)]
        aggregate[label] = {
            "sequence_count": len(subset),
            "frames": int(sum(int(r["frames"]) for r in subset)),
            "world_mpjpe_mean": weighted_mean(subset, "world_mpjpe_mean"),
            "root_error_mean": weighted_mean(subset, "root_error_mean"),
            "root_aligned_mpjpe_mean": weighted_mean(subset, "root_aligned_mpjpe_mean"),
            "heading_local_mpjpe_mean": weighted_mean(subset, "heading_local_mpjpe_mean"),
            "constant_root_offset_corrected_mpjpe_mean": weighted_mean(subset, "constant_root_offset_corrected_mpjpe_mean"),
        }
    return aggregate, per_sequence


def mode_invariance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sequences = sorted({r["sequence"] for r in rows})
    out_rows = []
    max_diff = 0.0
    missing = []
    for seq in sequences:
        base_row = next((r for r in rows if r["mode"] == "body_only" and r["sequence"] == seq), None)
        if base_row is None:
            missing.append({"sequence": seq, "mode": "body_only"})
            continue
        base = load_trace_from_child(base_row["child"])
        if base is None:
            missing.append({"sequence": seq, "mode": "body_only", "reason": "missing_trace"})
            continue
        for mode in VIRTUAL_MODES:
            row = next((r for r in rows if r["mode"] == mode and r["sequence"] == seq), None)
            trace = load_trace_from_child(row["child"]) if row else None
            if trace is None:
                missing.append({"sequence": seq, "mode": mode, "reason": "missing_trace"})
                continue
            n = min(base["body_sim_world"].shape[0], trace["body_sim_world"].shape[0])
            diff = float(
                max(
                    np.max(np.abs(base["body_sim_world"][:n] - trace["body_sim_world"][:n])),
                    np.max(np.abs(base["body_ref_world"][:n] - trace["body_ref_world"][:n])),
                    np.max(np.abs(base["root_sim_world"][:n] - trace["root_sim_world"][:n])),
                    np.max(np.abs(base["root_ref_world"][:n] - trace["root_ref_world"][:n])),
                )
            )
            max_diff = max(max_diff, diff)
            out_rows.append({"sequence": seq, "mode": mode, "max_abs_diff": diff, "frames_compared": n})
    return {"max_abs_diff": max_diff, "missing": missing, "rows": out_rows[:20], "passed": max_diff == 0.0 and not missing}


def join_earlier_baseline(per_sequence: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not EARLIER_BASELINE_JSON.exists():
        return {"available": False, "reason": "missing earlier baseline JSON"}, []
    baseline = load_json(EARLIER_BASELINE_JSON)
    by_seq = {row["sequence_name"]: row for row in baseline.get("per_sequence", [])}
    joined = []
    unmatched = []
    for row in per_sequence:
        old = by_seq.get(row["sequence"])
        if old is None:
            unmatched.append(row["sequence"])
            continue
        joined.append(
            {
                **row,
                "earlier_completed": bool(old.get("completed")),
                "earlier_mean_mpjpe": float(old.get("mean_mpjpe")),
                "earlier_mean_root_error": float(old.get("mean_root_error")),
                "earlier_frame_count": int(old.get("frame_count")) if old.get("frame_count") is not None else None,
                "earlier_count": int(old.get("count")) if old.get("count") is not None else None,
                "earlier_termination_frame": old.get("termination_frame"),
                "current_minus_earlier_mpjpe": float(row["world_mpjpe_mean"]) - float(old.get("mean_mpjpe")),
                "current_minus_earlier_root_error": float(row["root_error_mean"]) - float(old.get("mean_root_error")),
            }
        )
    summary = {
        "available": True,
        "baseline_json": str(EARLIER_BASELINE_JSON),
        "baseline_summary": baseline.get("summary", {}),
        "matched_stage1b_sequences": len(joined),
        "unmatched_stage1b_sequences": unmatched,
        "earlier_completed_count_for_matched": int(sum(1 for r in joined if r["earlier_completed"])),
        "current_completed_count_for_matched": int(sum(1 for r in joined if r["completed"])),
        "earlier_mean_mpjpe_for_matched_sequence_average": float(np.mean([r["earlier_mean_mpjpe"] for r in joined])) if joined else None,
        "earlier_mean_root_for_matched_sequence_average": float(np.mean([r["earlier_mean_root_error"] for r in joined])) if joined else None,
        "current_mean_mpjpe_for_matched_sequence_average": float(np.mean([r["world_mpjpe_mean"] for r in joined])) if joined else None,
        "current_mean_root_for_matched_sequence_average": float(np.mean([r["root_error_mean"] for r in joined])) if joined else None,
    }
    return summary, joined


def summarize_baseline_json_for_sequences(path: Path, sequences: list[str]) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "path": str(path), "reason": "missing"}
    payload = load_json(path)
    by_seq = {row["sequence_name"]: row for row in payload.get("per_sequence", [])}
    matched = []
    missing = []
    for seq in sorted(set(sequences)):
        row = by_seq.get(seq)
        if row is None:
            missing.append(seq)
        else:
            matched.append(row)

    def mean(key: str) -> float | None:
        values = [float(row[key]) for row in matched if row.get(key) is not None]
        return float(np.mean(values)) if values else None

    return {
        "available": True,
        "path": str(path),
        "summary": payload.get("summary", {}),
        "matched_stage1b_sequences": len(matched),
        "missing_stage1b_sequences": missing,
        "completed_count_for_matched": int(sum(1 for row in matched if row.get("completed"))),
        "mean_mpjpe_for_matched": mean("mean_mpjpe"),
        "mean_root_error_for_matched": mean("mean_root_error"),
        "log_path": str(USER_REPRO_LOG) if path == USER_REPRO_JSON else None,
    }


def read_text(path: Path, max_chars: int = 5000) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def config_diff_summary() -> dict[str, Any]:
    cfg = load_json(STAGE1B_CONFIG) if STAGE1B_CONFIG.exists() else {}
    baseline_script = read_text(EARLIER_BASELINE_SCRIPT, 12000) or ""
    saved_overrides = read_text(SAVED_HYDRA_OVERRIDES, 12000) or ""
    saved_config = read_text(SAVED_HYDRA_CONFIG, 12000) or ""
    baseline_log = read_text(EARLIER_BASELINE_LOG, 4000000) or ""
    return {
        "stage1b_config_path": str(STAGE1B_CONFIG),
        "stage1b_config_core": {
            "original_body_env_config": cfg.get("original_body_env_config"),
            "virtual_racket_env_config": cfg.get("virtual_racket_env_config"),
            "motion_file": cfg.get("motion_file"),
            "confirmed_body_checkpoint": cfg.get("confirmed_body_checkpoint"),
            "body_primitive_model": cfg.get("body_primitive_model"),
            "confirmed_body_obs_dim": cfg.get("confirmed_body_obs_dim"),
            "confirmed_body_action_dim": cfg.get("confirmed_body_action_dim"),
            "im_eval": cfg.get("im_eval"),
            "zero_out_far": cfg.get("zero_out_far"),
            "enable_early_termination": cfg.get("enable_early_termination"),
            "state_init": cfg.get("state_init"),
            "recovery_episode_prob": cfg.get("recovery_episode_prob"),
            "fall_init_prob": cfg.get("fall_init_prob"),
            "getup_schedule": cfg.get("getup_schedule"),
            "force_deterministic_reset_after_init": cfg.get("force_deterministic_reset_after_init"),
            "stop_sequence_on_reset": cfg.get("stop_sequence_on_reset"),
        },
        "earlier_baseline_script_path": str(EARLIER_BASELINE_SCRIPT),
        "earlier_baseline_script_evidence": {
            "uses_run_hydra_player": "phc/run_hydra.py" in baseline_script,
            "uses_env_im_getup_mcp": "env=env_im_getup_mcp" in baseline_script,
            "uses_phc_comp_3_exp": "exp_name=phc_comp_3" in baseline_script,
            "uses_num_prim_3": "env.num_prim=3" in baseline_script,
            "uses_phc_3_primitive_model": "output/HumanoidIm/phc_3/Humanoid.pth" in baseline_script,
            "uses_compact_metrics_path": "PHC_COMPACT_METRICS_PATH" in baseline_script,
            "uses_im_eval": "im_eval=True" in baseline_script,
            "uses_test": "test=True" in baseline_script,
            "default_num_envs": "128",
        },
        "earlier_baseline_log_evidence": {
            "found_phc_comp_3_checkpoint": "output/HumanoidIm/phc_comp_3/Humanoid.pth" in baseline_log,
            "loaded_phc_3_primitive": "output/HumanoidIm/phc_3/Humanoid.pth" in baseline_log,
            "running_mean_std_934": "RunningMeanStd:  (934,)" in baseline_log,
            "compact_metrics_completed": "Compact metrics" in baseline_log or "Compact metrics written" in baseline_log,
        },
        "saved_hydra_config_path": str(SAVED_HYDRA_CONFIG),
        "saved_hydra_overrides_path": str(SAVED_HYDRA_OVERRIDES),
        "saved_hydra_override_evidence": {
            "exists": SAVED_HYDRA_OVERRIDES.exists(),
            "motion_file_mentions_selected_rollout": "selected_rollouts" in saved_overrides,
            "num_envs_1": "env.num_envs=1" in saved_overrides,
            "games_num_1": "games_num=1" in saved_overrides,
            "enable_early_termination_false": "env.enableEarlyTermination=False" in saved_overrides,
        },
        "differences_relevant_to_stage1d": [
            "Earlier global baseline used the original PHC RL Games player path through phc/run_hydra.py; Stage 1B uses a custom Python evaluator and manual frozen-actor forward.",
            "Earlier baseline evaluated the full motion file with env.num_envs default 128; Stage 1B creates a one-motion temporary pickle per sequence and uses env.num_envs=1 per child.",
            "Stage 1B forces deterministic reset after task init, stateInit=Start, recovery/fall probabilities 0, getup_schedule=False, and stop-on-reset behavior.",
            "The saved phc_comp_3 Hydra directory currently points to a selected_rollouts single-clip eval, so it is weak provenance for the original global baseline command.",
            "Both paths use env_im_getup_mcp, im_mcp_big, smpl_humanoid, num_prim=3, the phc_comp_3 checkpoint family, and phc_3 primitive model evidence.",
        ],
    }


def classify(current: dict[str, Any], baseline_join: dict[str, Any], config_diff: dict[str, Any]) -> dict[str, Any]:
    fw = current["frame_weighted"]
    world = fw.get("world_mpjpe_mean")
    root_aligned = fw.get("root_aligned_mpjpe_mean")
    const_offset = fw.get("constant_root_offset_corrected_mpjpe_mean")
    earlier_mean = baseline_join.get("earlier_mean_mpjpe_for_matched_sequence_average")
    current_mean = baseline_join.get("current_mean_mpjpe_for_matched_sequence_average")
    earlier_completed = baseline_join.get("earlier_completed_count_for_matched")
    matched = baseline_join.get("matched_stage1b_sequences", 0)

    if world is not None and root_aligned is not None and const_offset is not None:
        if world > 1.0 and (root_aligned < 0.25 or const_offset < 0.25):
            return {
                "classification": "D1",
                "label": "root/global-origin/reset mismatch likely",
                "reason": "World MPJPE/root error are large but root-aligned or constant-offset-corrected MPJPE drops to a low range.",
                "gpu_body_only_reproduction_needed": True,
            }

    if matched and matched == 40 and earlier_completed == 40 and earlier_mean is not None and current_mean is not None:
        if earlier_mean < 0.15 and current_mean > 1.0:
            return {
                "classification": "D2",
                "label": "wrong checkpoint/config/eval path likely",
                "reason": "All Stage 1B held-out clips match earlier low-error completed baseline rows, but current custom Stage 1B body-only path has large MPJPE/root error and many terminations.",
                "gpu_body_only_reproduction_needed": True,
            }

    if matched and earlier_mean is not None and current_mean is not None and abs(earlier_mean - current_mean) < 0.2:
        return {
            "classification": "D3",
            "label": "held-out clips likely hard/failed under frozen body policy",
            "reason": "Current Stage 1B body metrics align with earlier per-sequence baseline metrics.",
            "gpu_body_only_reproduction_needed": False,
        }

    return {
        "classification": "D4",
        "label": "insufficient evidence",
        "reason": "Missing or inconclusive earlier baseline join/config/trace evidence.",
        "gpu_body_only_reproduction_needed": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def reproduction_command() -> str:
    out_json = REPORT_DIR / "stage1d_user_run_original_baseline_repro_full.json"
    log_path = REPORT_DIR / "stage1d_user_run_original_baseline_repro_full.log"
    return f"""# User-run GPU command, only if you want to reproduce the original PHC player baseline path.
# This is body-only PHC evaluation via the original compact-metrics script; no racket head,
# no optimizer/backward/training/reward tuning, and no physical racket/shuttle.
cd {WORKSPACE_ROOT}
nvidia-smi
phc_baseline/envs/phc_isaac/bin/python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("torch_cuda", torch.version.cuda)
PY
bash phc_baseline/run_phc_badminton_dataset_metrics.sh \\
  phc_baseline/converted/badminton_phc_motion_groundfix.pkl \\
  {out_json} \\
  128 \\
  1000000 2>&1 | tee {log_path}
python3 -m json.tool {out_json} >/dev/null
python3 - <<'PY'
import json
from pathlib import Path
p = Path("{out_json}")
d = json.loads(p.read_text())
print(json.dumps(d.get("summary", {{}}), indent=2))
PY
"""


def write_markdown(summary: dict[str, Any], per_sequence: list[dict[str, Any]], config_diff: dict[str, Any]) -> None:
    cls = summary["classification"]
    fw = summary["same_run_body_only"]["frame_weighted"]
    join = summary["earlier_baseline_join"]
    repro = summary["user_run_original_baseline_reproduction"]
    md = f"""# Stage 1D Body Rollout Mismatch Audit

Scope: CPU-only saved-trace and artifact audit. No GPU full evaluation, training, optimizer, backward pass, PPO/RL, reward update, checkpoint modification, physical racket, shuttle, or hitting reward was run.

## Classification

- outcome: `{cls['classification']} - {cls['label']}`
- reason: {cls['reason']}
- GPU body-only reproduction needed: `{cls['gpu_body_only_reproduction_needed']}`

## Same-Run Decomposition

Frame-weighted body-only metrics from corrected Stage 1B validity traces:

- world MPJPE mean: `{fw['world_mpjpe_mean']:.6f} m`
- root error mean: `{fw['root_error_mean']:.6f} m`
- root-aligned MPJPE mean: `{fw['root_aligned_mpjpe_mean']:.6f} m`
- heading-local MPJPE mean: `{fw['heading_local_mpjpe_mean']:.6f} m`
- constant-root-offset corrected MPJPE mean: `{fw['constant_root_offset_corrected_mpjpe_mean']:.6f} m`
- median-root-offset corrected MPJPE mean: `{fw['median_root_offset_corrected_mpjpe_mean']:.6f} m`
- completion / termination: `{summary['same_run_body_only']['completed_count']} / {summary['same_run_body_only']['terminated_count']}`

Root alignment and constant-offset correction do not by themselves restore the earlier low-error range unless those numbers are near the earlier baseline; this helps distinguish a pure origin offset from a broader body rollout/config mismatch.

## Earlier Baseline Join

- earlier baseline JSON: `{join.get('baseline_json')}`
- matched Stage 1B sequences: `{join.get('matched_stage1b_sequences')}`
- unmatched Stage 1B sequences: `{join.get('unmatched_stage1b_sequences')}`
- earlier completed count for matched: `{join.get('earlier_completed_count_for_matched')}`
- current completed count for matched: `{join.get('current_completed_count_for_matched')}`
- earlier matched mean MPJPE: `{join.get('earlier_mean_mpjpe_for_matched_sequence_average')}`
- current matched mean MPJPE: `{join.get('current_mean_mpjpe_for_matched_sequence_average')}`
- earlier matched mean root error: `{join.get('earlier_mean_root_for_matched_sequence_average')}`
- current matched mean root error: `{join.get('current_mean_root_for_matched_sequence_average')}`

The 40 held-out Stage 1B sequence keys match earlier compact-metrics rows. Those earlier rows are low-error and completed, while the current custom Stage 1B body-only path is high-error and terminates 19 of 40 clips. That argues against treating these clips as intrinsically hard based on the earlier artifact.

## User-Run Original Baseline Reproduction

- reproduction JSON: `{repro.get('path')}`
- reproduction available: `{repro.get('available')}`
- full dataset summary MPJPE/root: `{(repro.get('summary') or {}).get('dataset_mean_mpjpe')}` / `{(repro.get('summary') or {}).get('dataset_mean_root_error')}`
- matched Stage 1B sequences: `{repro.get('matched_stage1b_sequences')}`
- matched completed count: `{repro.get('completed_count_for_matched')}`
- matched mean MPJPE/root: `{repro.get('mean_mpjpe_for_matched')}` / `{repro.get('mean_root_error_for_matched')}`

This user-run reproduction uses the original compact-metrics path and confirms that the checkpoint/config family can reproduce the low body metrics outside the custom Stage 1B evaluator path.

## Config / Eval Path Findings

Key differences:

"""
    for item in config_diff["differences_relevant_to_stage1d"]:
        md += f"- {item}\n"
    md += f"""

Shared evidence:

- earlier baseline script uses `env_im_getup_mcp`, `im_mcp_big`, `smpl_humanoid`, `env.num_prim=3`, `im_eval=True`, and primitive model `output/HumanoidIm/phc_3/Humanoid.pth`.
- earlier baseline log references `output/HumanoidIm/phc_comp_3/Humanoid.pth`, loads `output/HumanoidIm/phc_3/Humanoid.pth`, and reports `RunningMeanStd: (934,)`.
- current Stage 1B contract remains `[934] -> [3]` body action and `[3]+[6]=[9]` env boundary; no `[4]` padding/truncation is involved.

## Interpretation

The current best classification is `{cls['classification']}`. The next gate should reproduce the body-only PHC player/evaluation path, preferably using the exact compact-metrics script or an equivalent subset export, before changing racket-head objectives or adding any hand/body reward.

## Optional User-Run Body-Only Reproduction Command

```bash
{reproduction_command().strip()}
```
"""
    (REPORT_DIR / "body_rollout_mismatch_stage1d_report.md").write_text(md, encoding="utf-8")

    diff_md = "# Stage 1D Config Diff\n\n"
    diff_md += "## Current Stage 1B Core Config\n\n```json\n"
    diff_md += json.dumps(config_diff["stage1b_config_core"], indent=2)
    diff_md += "\n```\n\n## Earlier Baseline Script Evidence\n\n```json\n"
    diff_md += json.dumps(config_diff["earlier_baseline_script_evidence"], indent=2)
    diff_md += "\n```\n\n## Earlier Baseline Log Evidence\n\n```json\n"
    diff_md += json.dumps(config_diff["earlier_baseline_log_evidence"], indent=2)
    diff_md += "\n```\n\n## Saved Hydra Override Evidence\n\n```json\n"
    diff_md += json.dumps(config_diff["saved_hydra_override_evidence"], indent=2)
    diff_md += "\n```\n\n"
    for item in config_diff["differences_relevant_to_stage1d"]:
        diff_md += f"- {item}\n"
    (REPORT_DIR / "body_rollout_mismatch_stage1d_config_diff.md").write_text(diff_md, encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_child_rows()
    current_summary, per_sequence = summarize_current_body_only(rows)
    mode_inv = mode_invariance(rows)
    baseline_join, joined_rows = join_earlier_baseline(per_sequence)
    user_repro = summarize_baseline_json_for_sequences(USER_REPRO_JSON, [row["sequence"] for row in per_sequence])
    config_diff = config_diff_summary()
    classification = classify(current_summary, baseline_join, config_diff)

    joined_by_seq = {r["sequence"]: r for r in joined_rows}
    per_sequence_rows = []
    for row in per_sequence:
        joined = joined_by_seq.get(row["sequence"], {})
        per_sequence_rows.append(
            {
                **row,
                "earlier_completed": joined.get("earlier_completed"),
                "earlier_mean_mpjpe": joined.get("earlier_mean_mpjpe"),
                "earlier_mean_root_error": joined.get("earlier_mean_root_error"),
                "earlier_frame_count": joined.get("earlier_frame_count"),
                "earlier_count": joined.get("earlier_count"),
                "earlier_termination_frame": joined.get("earlier_termination_frame"),
                "current_minus_earlier_mpjpe": joined.get("current_minus_earlier_mpjpe"),
                "current_minus_earlier_root_error": joined.get("current_minus_earlier_root_error"),
            }
        )

    timing = {
        "body_only_motion_time_ranges_from_traces": {
            "min": min((r["motion_time_min"] for r in per_sequence if r["motion_time_min"] is not None), default=None),
            "max": max((r["motion_time_max"] for r in per_sequence if r["motion_time_max"] is not None), default=None),
        },
        "progress_buf_ranges_from_traces": {
            "min": min((r["progress_min"] for r in per_sequence if r["progress_min"] is not None), default=None),
            "max": max((r["progress_max"] for r in per_sequence if r["progress_max"] is not None), default=None),
        },
        "note": "Validity traces show the custom Stage 1B per-sequence children use sampled_motion_id 0 after making one-motion temporary pickle files; this differs from the original full-motion-file player evaluation organization but not necessarily from reference timing inside each child.",
    }

    normalization = {
        "checkpoint_running_mean_dim": 934,
        "stage1b_body_obs_dim": 934,
        "stage1b_manual_actor_path": "evaluate.py normalizes body obs with checkpoint running_mean/running_var once before FrozenPHCBodyActor forward.",
        "original_player_path": "PHC RL Games player applies its normal observation normalization path.",
        "audit_status": "No dimension mismatch found, but Stage 1B uses a manual actor-forward reimplementation rather than the original player route; exact body-only reproduction is recommended.",
    }

    summary = {
        "status": "completed_cpu_only",
        "no_training": True,
        "no_optimizer": True,
        "no_backward": True,
        "no_ppo_rl": True,
        "no_reward_update": True,
        "no_gpu_full_eval_run_by_codex": True,
        "same_run_body_only": current_summary,
        "mode_invariance": mode_inv,
        "earlier_baseline_join": baseline_join,
        "user_run_original_baseline_reproduction": user_repro,
        "timing_motion_index_check": timing,
        "obs_normalization_actor_path_check": normalization,
        "config_diff": {
            "path": str(REPORT_DIR / "body_rollout_mismatch_stage1d_config_diff.json"),
            "key_differences": config_diff["differences_relevant_to_stage1d"],
        },
        "classification": classification,
        "recommended_next_gate": "Run an original PHC player/compact-metrics body-only reproduction for the same checkpoint/config family before changing racket-head objectives or adding hand/body rewards.",
        "optional_user_run_body_only_reproduction_command": reproduction_command(),
    }

    write_json(REPORT_DIR / "body_rollout_mismatch_stage1d_summary.json", summary)
    write_json(REPORT_DIR / "body_rollout_mismatch_stage1d_config_diff.json", config_diff)
    write_csv(REPORT_DIR / "body_rollout_mismatch_stage1d_per_sequence.csv", per_sequence_rows)
    write_markdown(summary, per_sequence_rows, config_diff)

    print(
        json.dumps(
            {
                "classification": classification["classification"],
                "label": classification["label"],
                "body_only_world_mpjpe": current_summary["frame_weighted"]["world_mpjpe_mean"],
                "body_only_root_error": current_summary["frame_weighted"]["root_error_mean"],
                "root_aligned_mpjpe": current_summary["frame_weighted"]["root_aligned_mpjpe_mean"],
                "matched_earlier_sequences": baseline_join.get("matched_stage1b_sequences"),
                "gpu_body_only_reproduction_needed": classification["gpu_body_only_reproduction_needed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
