#!/usr/bin/env python3
"""Audit Stage 1B hand/wrist metric provenance from saved traces.

CPU-only post-processing. This script does not run Isaac Gym, train, optimize,
backpropagate, tune rewards, or modify checkpoints.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = WORKSPACE_ROOT / "phc_baseline" / "reports" / "racket_calibration" / "frozen_body_head_integration"
FULL_EVAL_DIR = REPORT_DIR / "full_heldout_eval"
CHILD_DIR = FULL_EVAL_DIR / "children"
VIRTUAL_MODES = ["virtual_null", "virtual_goal_only", "virtual_goal_state", "virtual_oracle"]
ALL_MODES = ["body_only", *VIRTUAL_MODES]

SMPL_MUJOCO_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]


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


def vector_rmse(vec: np.ndarray) -> dict[str, Any]:
    vec = np.asarray(vec, dtype=np.float64)
    sq = np.sum(vec * vec, axis=-1)
    norms = np.sqrt(sq)
    mse = float(np.mean(sq)) if sq.size else None
    return {
        "mse": mse,
        "rmse": None if mse is None else float(math.sqrt(mse)),
        "euclidean_error": stats(norms),
    }


def quat_rotate_xyzw(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    q_vec = q[..., :3]
    q_w = q[..., 3:4]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.sum(q_vec * v, axis=-1, keepdims=True) * 2.0
    return a + b + c


def heading_local_vector_xyzw(vec: np.ndarray, root_rot_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(root_rot_xyzw, dtype=np.float64)
    ref = np.zeros_like(vec, dtype=np.float64)
    ref[..., 0] = 1.0
    rot_dir = quat_rotate_xyzw(q, ref)
    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    c = np.cos(heading)
    s = np.sin(heading)
    out = np.asarray(vec, dtype=np.float64).copy()
    x = out[..., 0].copy()
    y = out[..., 1].copy()
    out[..., 0] = c * x + s * y
    out[..., 1] = -s * x + c * y
    return out


def child_json_path(mode: str, sequence: str) -> Path:
    return CHILD_DIR / mode / f"{sequence.replace('/', '__')}.json"


def all_child_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_sequence = FULL_EVAL_DIR / "stage1b_per_sequence_results.csv"
    with per_sequence.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            child_path = child_json_path(row["mode"], row["sequence"])
            child = load_json(child_path) if child_path.exists() else {}
            rows.append({**row, "child_path": str(child_path), "child": child})
    return rows


def load_trace(path: str | None) -> dict[str, np.ndarray] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    data = np.load(p)
    return {key: data[key] for key in data.files}


def trace_schema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    examples: dict[str, Any] = {}
    for row in rows:
        trace = load_trace(row["child"].get("kintwin_trace_path"))
        if trace:
            examples = {
                key: {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "finite": bool(np.isfinite(value).all()) if np.issubdtype(value.dtype, np.number) else None,
                }
                for key, value in trace.items()
            }
            break
    required_for_current = [
        "r_hand_sim_world",
        "r_hand_ref_world",
        "r_wrist_sim_world",
        "r_wrist_ref_world",
        "handle_realized_world",
        "handle_target_world",
    ]
    required_for_exact_body_repro = [
        "body_sim_world",
        "body_ref_world",
        "root_sim_world",
        "root_ref_world",
        "root_sim_rot",
        "root_ref_rot",
        "motion_time",
        "progress_buf",
        "sampled_motion_id",
        "body_names",
        "r_hand_index",
        "r_wrist_index",
        "sequence",
        "mode",
        "completed",
        "terminated",
        "frames_evaluated",
        "step_index",
        "reset_buf_after_step",
    ]
    validity_examples: dict[str, Any] = {}
    for row in rows:
        validity = load_trace(row["child"].get("validity_trace_path"))
        if validity:
            validity_examples = {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in validity.items()
            }
            break
    validity_has_exact = all(k in validity_examples for k in required_for_exact_body_repro)
    return {
        "status": "inspected",
        "current_kintwin_trace_example_fields": examples,
        "current_fields_sufficient_for_requested_hand_wrist_dynamic_metrics": all(k in examples for k in required_for_current),
        "fields_missing_for_exact_official_mpjpe_reproduction": [] if validity_has_exact else [k for k in required_for_exact_body_repro if k not in examples],
        "validity_trace_example_fields": validity_examples,
        "validity_trace_present_in_saved_run": bool(validity_examples),
        "validity_trace_sufficient_for_exact_official_mpjpe_reproduction": validity_has_exact,
        "trace_field_semantics": {
            "r_hand_sim_world": "task._rigid_body_pos[:, R_Hand] after task.step; Isaac/PHC world meters, post-step.",
            "r_hand_ref_world": "motionlib rg_pos[:, R_Hand] queried by compute_body_tracking_metrics at progress_buf*dt + motion_start + offset; world meters.",
            "r_wrist_sim_world": "task._rigid_body_pos[:, R_Wrist] after task.step; Isaac/PHC world meters, post-step.",
            "r_wrist_ref_world": "motionlib rg_pos[:, R_Wrist] from the same query used for MPJPE in evaluate.py.",
            "handle_realized_world": "virtual racket persistent realized handle in PHC/world coordinates after the virtual kinematic update.",
            "handle_target_world": "Live V2 target handle world value queried with _current_motion_times(next_frame=False) after step.",
        },
        "official_mpjpe_reproduction_possible_from_current_npz": validity_has_exact,
        "reason": "saved kintwin traces do not include full body_sim/body_ref/root arrays or timing/body-name metadata; child JSON has aggregate evaluator-computed body metrics only."
        if not validity_has_exact
        else "validity traces include full body/ref/root/timing/body-name metadata for exact MPJPE recomputation.",
    }


def summarize_metric_by_rows(rows: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    values = []
    weights = []
    for row in rows:
        child = row["child"]
        metric = child.get(metric_key) or {}
        mean = metric.get("mean")
        frames = child.get("frames_evaluated")
        if mean is not None and frames:
            values.append(float(mean))
            weights.append(float(frames))
    if not values:
        return {"sequence_average": None, "frame_weighted_mean": None, "frames": 0}
    return {
        "sequence_average": float(np.mean(values)),
        "frame_weighted_mean": float(np.average(values, weights=weights)),
        "frames": int(np.sum(weights)),
    }


def same_run_body_metrics(rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    exact_rows: list[dict[str, Any]] = []
    out: dict[str, Any] = {
        "exact_official_mpjpe_reproduction_from_saved_npz": False,
        "uses_stage1b_evaluator_child_json_aggregates": True,
        "blocker_for_exact_reproduction": "current saved NPZ traces lack full body_sim/body_ref arrays and official player/evaluator per-frame info arrays",
        "prior_global_body_baseline": {
            "dataset_mean_mpjpe_m": 0.073808,
            "dataset_mean_root_error_m": 0.065351,
            "note": "Earlier global baseline is not the same saved Stage 1B held-out run/coverage and cannot by itself invalidate Stage 1B hand/wrist metrics.",
        },
    }
    per_seq: list[dict[str, Any]] = []
    for mode in ALL_MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        exact_mpjpe_values: list[float] = []
        exact_root_values: list[float] = []
        exact_weights: list[float] = []
        out[mode] = {
            "body_mpjpe": summarize_metric_by_rows(mode_rows, "body_mpjpe"),
            "root_error": summarize_metric_by_rows(mode_rows, "root_error"),
            "completed": sum(1 for r in mode_rows if bool(r["child"].get("completed"))),
            "terminated": sum(1 for r in mode_rows if bool(r["child"].get("terminated"))),
            "clips": len(mode_rows),
        }
        for r in mode_rows:
            child = r["child"]
            validity = load_trace(child.get("validity_trace_path"))
            exact_mpjpe_mean = None
            exact_root_mean = None
            exact_r_hand_rmse = None
            exact_r_wrist_rmse = None
            if validity and "body_sim_world" in validity and "body_ref_world" in validity:
                body_err = np.linalg.norm(validity["body_sim_world"] - validity["body_ref_world"], axis=-1)
                root_err = np.linalg.norm(validity["body_sim_world"][:, 0, :] - validity["body_ref_world"][:, 0, :], axis=-1)
                mpjpe_frames = body_err.mean(axis=-1)
                exact_mpjpe_mean = float(np.mean(mpjpe_frames))
                exact_root_mean = float(np.mean(root_err))
                exact_mpjpe_values.append(exact_mpjpe_mean)
                exact_root_values.append(exact_root_mean)
                exact_weights.append(float(len(mpjpe_frames)))
                names = [
                    x.decode("utf-8") if isinstance(x, bytes) else str(x)
                    for x in validity.get("body_names", [])
                ]
                if "R_Hand" in names:
                    idx = names.index("R_Hand")
                    exact_r_hand_rmse = vector_rmse(validity["body_sim_world"][:, idx, :] - validity["body_ref_world"][:, idx, :])["rmse"]
                if "R_Wrist" in names:
                    idx = names.index("R_Wrist")
                    exact_r_wrist_rmse = vector_rmse(validity["body_sim_world"][:, idx, :] - validity["body_ref_world"][:, idx, :])["rmse"]
                exact_rows.append(
                    {
                        "sequence": r["sequence"],
                        "mode": mode,
                        "exact_mpjpe_mean": exact_mpjpe_mean,
                        "exact_root_error_mean": exact_root_mean,
                        "exact_r_hand_rmse": exact_r_hand_rmse,
                        "exact_r_wrist_rmse": exact_r_wrist_rmse,
                    }
                )
            per_seq.append(
                {
                    "sequence": r["sequence"],
                    "session_group": r["sequence"].split("/", 1)[0],
                    "mode": mode,
                    "completed": child.get("completed"),
                    "terminated": child.get("terminated"),
                    "frames_evaluated": child.get("frames_evaluated"),
                    "body_mpjpe_mean": (child.get("body_mpjpe") or {}).get("mean"),
                    "root_error_mean": (child.get("root_error") or {}).get("mean"),
                    "exact_trace_mpjpe_mean": exact_mpjpe_mean,
                    "exact_trace_root_error_mean": exact_root_mean,
                    "exact_trace_r_hand_rmse": exact_r_hand_rmse,
                    "exact_trace_r_wrist_rmse": exact_r_wrist_rmse,
                }
            )
        if exact_weights:
            out[mode]["exact_trace_recomputed_body_mpjpe_frame_weighted"] = float(np.average(exact_mpjpe_values, weights=exact_weights))
            out[mode]["exact_trace_recomputed_root_error_frame_weighted"] = float(np.average(exact_root_values, weights=exact_weights))
            out["exact_official_mpjpe_reproduction_from_saved_npz"] = True
            out["blocker_for_exact_reproduction"] = None
    body_only = out.get("body_only", {})
    out["same_run_body_only_frame_weighted_mpjpe_m"] = (body_only.get("body_mpjpe") or {}).get("frame_weighted_mean")
    out["same_run_body_only_frame_weighted_root_error_m"] = (body_only.get("root_error") or {}).get("frame_weighted_mean")
    if (body_only or {}).get("exact_trace_recomputed_body_mpjpe_frame_weighted") is not None:
        out["same_run_body_only_exact_trace_mpjpe_m"] = body_only["exact_trace_recomputed_body_mpjpe_frame_weighted"]
        out["same_run_body_only_exact_trace_root_error_m"] = body_only["exact_trace_recomputed_root_error_frame_weighted"]
    return out, per_seq


def trace_metric_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    aggregate_arrays: dict[str, dict[str, list[np.ndarray]]] = {mode: {} for mode in VIRTUAL_MODES}
    for mode in VIRTUAL_MODES:
        aggregate_arrays[mode] = {
            "hand": [],
            "wrist": [],
            "dynamic_hand": [],
            "dynamic_wrist": [],
            "sim_segment": [],
            "ref_segment": [],
        }
    for row in rows:
        mode = row["mode"]
        if mode not in VIRTUAL_MODES:
            continue
        trace = load_trace(row["child"].get("kintwin_trace_path"))
        if not trace:
            continue
        hand = trace["r_hand_sim_world"] - trace["r_hand_ref_world"]
        wrist = trace["r_wrist_sim_world"] - trace["r_wrist_ref_world"]
        dyn_hand = trace["handle_realized_world"] - trace["r_hand_sim_world"] - (
            trace["handle_target_world"] - trace["r_hand_ref_world"]
        )
        dyn_wrist = trace["handle_realized_world"] - trace["r_wrist_sim_world"] - (
            trace["handle_target_world"] - trace["r_wrist_ref_world"]
        )
        sim_segment = np.linalg.norm(trace["r_hand_sim_world"] - trace["r_wrist_sim_world"], axis=-1)
        ref_segment = np.linalg.norm(trace["r_hand_ref_world"] - trace["r_wrist_ref_world"], axis=-1)
        for key, arr in [
            ("hand", hand),
            ("wrist", wrist),
            ("dynamic_hand", dyn_hand),
            ("dynamic_wrist", dyn_wrist),
        ]:
            aggregate_arrays[mode][key].append(arr)
        aggregate_arrays[mode]["sim_segment"].append(sim_segment)
        aggregate_arrays[mode]["ref_segment"].append(ref_segment)
        out_rows.append(
            {
                "sequence": row["sequence"],
                "session_group": row["sequence"].split("/", 1)[0],
                "mode": mode,
                "completed": row["child"].get("completed"),
                "terminated": row["child"].get("terminated"),
                "frames_evaluated": row["child"].get("frames_evaluated"),
                "hand_rmse_m": vector_rmse(hand)["rmse"],
                "wrist_rmse_m": vector_rmse(wrist)["rmse"],
                "dynamic_handle_hand_rmse_m": vector_rmse(dyn_hand)["rmse"],
                "dynamic_handle_wrist_rmse_m": vector_rmse(dyn_wrist)["rmse"],
                "hand_error_mean_m": vector_rmse(hand)["euclidean_error"]["mean"],
                "wrist_error_mean_m": vector_rmse(wrist)["euclidean_error"]["mean"],
                "sim_wrist_hand_segment_mean_m": stats(sim_segment)["mean"],
                "ref_wrist_hand_segment_mean_m": stats(ref_segment)["mean"],
            }
        )
    aggregates: dict[str, Any] = {}
    for mode, arrays in aggregate_arrays.items():
        mode_out: dict[str, Any] = {}
        for key in ["hand", "wrist", "dynamic_hand", "dynamic_wrist"]:
            arr = np.concatenate(arrays[key], axis=0) if arrays[key] else np.empty((0, 3))
            mode_out[key] = vector_rmse(arr)
        for key in ["sim_segment", "ref_segment"]:
            arr = np.concatenate(arrays[key], axis=0) if arrays[key] else np.empty((0,))
            mode_out[key] = stats(arr)
        aggregates[mode] = mode_out
    return out_rows, aggregates


def mode_invariance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_sequence: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for row in rows:
        if row["mode"] not in VIRTUAL_MODES:
            continue
        trace = load_trace(row["child"].get("kintwin_trace_path"))
        if trace:
            by_sequence.setdefault(row["sequence"], {})[row["mode"]] = trace
    fields = ["r_hand_sim_world", "r_hand_ref_world", "r_wrist_sim_world", "r_wrist_ref_world"]
    field_max: dict[str, float] = {field: 0.0 for field in fields}
    compared = 0
    for _seq, traces in by_sequence.items():
        if not all(mode in traces for mode in VIRTUAL_MODES):
            continue
        base = traces["virtual_null"]
        for mode in VIRTUAL_MODES[1:]:
            other = traces[mode]
            for field in fields:
                if base[field].shape == other[field].shape:
                    field_max[field] = max(field_max[field], float(np.max(np.abs(base[field] - other[field]))))
        compared += 1
    return {"sequences_compared": compared, "max_abs_diff_by_field": field_max, "passed": all(v == 0.0 for v in field_max.values())}


def timing_shift_tests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for mode in VIRTUAL_MODES:
        per_shift: dict[int, dict[str, list[np.ndarray]]] = {shift: {"hand": [], "wrist": []} for shift in range(-2, 3)}
        for row in rows:
            if row["mode"] != mode:
                continue
            trace = load_trace(row["child"].get("kintwin_trace_path"))
            if not trace:
                continue
            for shift in range(-2, 3):
                sim_h = trace["r_hand_sim_world"]
                ref_h = trace["r_hand_ref_world"]
                sim_w = trace["r_wrist_sim_world"]
                ref_w = trace["r_wrist_ref_world"]
                if shift > 0:
                    a_h, b_h = sim_h[:-shift], ref_h[shift:]
                    a_w, b_w = sim_w[:-shift], ref_w[shift:]
                elif shift < 0:
                    a_h, b_h = sim_h[-shift:], ref_h[:shift]
                    a_w, b_w = sim_w[-shift:], ref_w[:shift]
                else:
                    a_h, b_h = sim_h, ref_h
                    a_w, b_w = sim_w, ref_w
                if len(a_h):
                    per_shift[shift]["hand"].append(a_h - b_h)
                    per_shift[shift]["wrist"].append(a_w - b_w)
        mode_out: dict[str, Any] = {}
        best_hand = None
        best_wrist = None
        for shift in range(-2, 3):
            hand = np.concatenate(per_shift[shift]["hand"], axis=0) if per_shift[shift]["hand"] else np.empty((0, 3))
            wrist = np.concatenate(per_shift[shift]["wrist"], axis=0) if per_shift[shift]["wrist"] else np.empty((0, 3))
            hand_rmse = vector_rmse(hand)["rmse"]
            wrist_rmse = vector_rmse(wrist)["rmse"]
            mode_out[str(shift)] = {"hand_rmse_m": hand_rmse, "wrist_rmse_m": wrist_rmse, "frames": int(len(hand))}
            if hand_rmse is not None and (best_hand is None or hand_rmse < best_hand["hand_rmse_m"]):
                best_hand = {"shift": shift, "hand_rmse_m": hand_rmse}
            if wrist_rmse is not None and (best_wrist is None or wrist_rmse < best_wrist["wrist_rmse_m"]):
                best_wrist = {"shift": shift, "wrist_rmse_m": wrist_rmse}
        result[mode] = {"offsets": mode_out, "best_hand": best_hand, "best_wrist": best_wrist}
    return result


def position_range_tests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ["r_hand_sim_world", "r_hand_ref_world", "r_wrist_sim_world", "r_wrist_ref_world"]
    arrays: dict[str, list[np.ndarray]] = {field: [] for field in fields}
    for row in rows:
        if row["mode"] != "virtual_goal_state":
            continue
        trace = load_trace(row["child"].get("kintwin_trace_path"))
        if not trace:
            continue
        for field in fields:
            arrays[field].append(trace[field])
    out: dict[str, Any] = {}
    for field, chunks in arrays.items():
        arr = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 3))
        out[field] = {
            "axis_min": arr.min(axis=0).tolist() if len(arr) else None,
            "axis_max": arr.max(axis=0).tolist() if len(arr) else None,
            "finite": bool(np.isfinite(arr).all()) if len(arr) else False,
        }
    out["root_aligned_and_heading_aligned_tests"] = {
        "status": "blocked_by_missing_root_ref_root_rot_fields_in_current_saved_npz",
        "future_trace_export_patch": "evaluate.py now supports save_hand_wrist_validity_traces to write body/ref/root arrays for a user-run rerun.",
    }
    out["axis_permutation_unit_search"] = "not performed; current ranges and wrist-hand segment lengths are meter-scale, and arbitrary post-hoc transform search would overfit the diagnostic."
    return out


def validity_alignment_tests(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "blocked_by_missing_validity_trace",
        "world": {},
        "root_translation_aligned": {},
        "heading_local": {},
        "exact_trace_field_coverage": {"rows": 0, "frames": 0},
    }
    world_chunks: dict[str, list[np.ndarray]] = {"hand": [], "wrist": []}
    root_chunks: dict[str, list[np.ndarray]] = {"hand": [], "wrist": []}
    heading_chunks: dict[str, list[np.ndarray]] = {"hand": [], "wrist": []}
    rows_with_validity = 0
    frames = 0
    for row in rows:
        validity = load_trace(row["child"].get("validity_trace_path"))
        if not validity:
            continue
        required = [
            "body_sim_world",
            "body_ref_world",
            "root_sim_world",
            "root_ref_world",
            "root_sim_rot",
            "root_ref_rot",
            "body_names",
        ]
        if any(key not in validity for key in required):
            continue
        names = [
            x.decode("utf-8") if isinstance(x, bytes) else str(x)
            for x in validity["body_names"]
        ]
        if "R_Hand" not in names or "R_Wrist" not in names:
            continue
        hand_idx = names.index("R_Hand")
        wrist_idx = names.index("R_Wrist")
        body_sim = validity["body_sim_world"]
        body_ref = validity["body_ref_world"]
        root_sim = validity["root_sim_world"]
        root_ref = validity["root_ref_world"]
        root_sim_rot = validity["root_sim_rot"]
        root_ref_rot = validity["root_ref_rot"]
        for label, idx in [("hand", hand_idx), ("wrist", wrist_idx)]:
            sim = body_sim[:, idx, :]
            ref = body_ref[:, idx, :]
            world_chunks[label].append(sim - ref)
            root_chunks[label].append((sim - root_sim) - (ref - root_ref))
            sim_local = heading_local_vector_xyzw(sim - root_sim, root_sim_rot)
            ref_local = heading_local_vector_xyzw(ref - root_ref, root_ref_rot)
            heading_chunks[label].append(sim_local - ref_local)
        rows_with_validity += 1
        frames += int(len(body_sim))
    if rows_with_validity == 0:
        return out
    out["status"] = "computed_from_validity_trace"
    out["exact_trace_field_coverage"] = {"rows": rows_with_validity, "frames": frames}
    for target, chunks in [("world", world_chunks), ("root_translation_aligned", root_chunks), ("heading_local", heading_chunks)]:
        out[target] = {}
        for label in ["hand", "wrist"]:
            arr = np.concatenate(chunks[label], axis=0) if chunks[label] else np.empty((0, 3))
            out[target][label] = vector_rmse(arr)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(payloads: dict[str, Any]) -> None:
    schema = payloads["schema"]
    mapping = payloads["mapping"]
    same = payloads["same_run"]
    hypo = payloads["hypothesis"]
    conclusion = payloads["conclusion"]

    write_json(REPORT_DIR / "hand_wrist_metric_validity_trace_schema.json", schema)
    (REPORT_DIR / "hand_wrist_metric_validity_trace_schema.md").write_text(
        "# Hand/Wrist Metric Validity Trace Schema\n\n"
        f"Current saved `*.kintwin_trace.npz` fields: `{list(schema['current_kintwin_trace_example_fields'].keys())}`.\n\n"
        f"Enough for requested dynamic relation metrics: `{schema['current_fields_sufficient_for_requested_hand_wrist_dynamic_metrics']}`.\n\n"
        f"Enough for exact official MPJPE reproduction: `{schema['official_mpjpe_reproduction_possible_from_current_npz']}`.\n\n"
        "Missing for exact official reproduction: "
        f"`{schema['fields_missing_for_exact_official_mpjpe_reproduction']}`.\n",
        encoding="utf-8",
    )

    write_json(REPORT_DIR / "hand_wrist_body_index_mapping_audit.json", mapping)
    (REPORT_DIR / "hand_wrist_body_index_mapping_audit.md").write_text(
        "# Hand/Wrist Body Index Mapping Audit\n\n"
        f"Runtime/evaluator path uses body names from `task._body_names` and selects `R_Wrist`/`R_Hand` by name.\n\n"
        f"SMPL humanoid body-name fallback indexes: R_Wrist=`{mapping['smpl_mujoco_name_indices']['R_Wrist']}`, "
        f"R_Hand=`{mapping['smpl_mujoco_name_indices']['R_Hand']}`.\n\n"
        f"Segment sanity: sim wrist-hand mean `{mapping['segment_length_sanity']['sim_wrist_hand_segment_m']['mean']:.6f}` m, "
        f"ref wrist-hand mean `{mapping['segment_length_sanity']['ref_wrist_hand_segment_m']['mean']:.6f}` m.\n\n"
        f"Mapping status: `{mapping['status']}`.\n",
        encoding="utf-8",
    )

    write_json(REPORT_DIR / "same_run_body_metric_reproduction_summary.json", same)
    (REPORT_DIR / "same_run_body_metric_reproduction_report.md").write_text(
        "# Same-Run Body Metric Reproduction Report\n\n"
        "Exact official MPJPE reproduction from saved NPZ traces is blocked because full body/ref arrays were not saved in the corrected run.\n\n"
        "The Stage 1B evaluator child JSONs do contain same-run body metric aggregates. "
        f"Frame-weighted body-only MPJPE is `{same['same_run_body_only_frame_weighted_mpjpe_m']:.6f}` m and "
        f"root error is `{same['same_run_body_only_frame_weighted_root_error_m']:.6f}` m.\n\n"
        "This is not close to the earlier global PHC baseline summary `0.073808 m`, so the earlier aggregate cannot be used as a direct contradiction "
        "without proving identical checkpoint, motion split, state init, termination/evaluation coverage, and metric formula.\n",
        encoding="utf-8",
    )

    write_json(REPORT_DIR / "hand_wrist_alignment_hypothesis_tests_summary.json", hypo)
    (REPORT_DIR / "hand_wrist_alignment_hypothesis_tests_report.md").write_text(
        "# Hand/Wrist Alignment Hypothesis Tests\n\n"
        f"Mode invariance passed: `{hypo['mode_invariance']['passed']}`; max diffs: `{hypo['mode_invariance']['max_abs_diff_by_field']}`.\n\n"
        f"Validity-trace alignment status: `{hypo['validity_trace_alignment_tests']['status']}`.\n\n"
        "Root-aligned and heading-local tests are blocked by missing root/reference/root-rotation fields in the current saved traces when the status above is blocked.\n\n"
        f"Timing shift best offsets: `{hypo['timing_shift_tests_best_offsets']}`.\n\n"
        "No fixed attachment assumption is introduced here. Dynamic relation MSE is evaluated only after checking provenance.\n",
        encoding="utf-8",
    )

    write_json(REPORT_DIR / "hand_wrist_metric_validity_conclusion.json", conclusion)
    (REPORT_DIR / "hand_wrist_metric_validity_conclusion.md").write_text(
        "# Hand/Wrist Metric Validity Conclusion\n\n"
        f"Classification: `{conclusion['classification']}`.\n\n"
        f"{conclusion['summary']}\n\n"
        "For Model B/oracle, because `handle_realized ~= handle_target`, the dynamic handle-hand error is dominated by "
        "`hand_ref - hand_sim`. Therefore the dynamic RMSE being near hand RMSE is not additional evidence of a racket-head failure. "
        "For null action, lower dynamic relation RMSE can be accidental cancellation and is not a better holding result.\n\n"
        "Next gate: "
        f"{conclusion['next_gate']}\n",
        encoding="utf-8",
    )


def main() -> int:
    rows = all_child_rows()
    schema = trace_schema(rows)
    per_trace_rows, trace_aggregates = trace_metric_rows(rows)
    same_run, per_seq_body = same_run_body_metrics(rows)
    invariance = mode_invariance(rows)
    timing = timing_shift_tests(rows)
    ranges = position_range_tests(rows)
    validity_align = validity_alignment_tests(rows)

    segment_sim_chunks = []
    segment_ref_chunks = []
    for row in rows:
        if row["mode"] != "virtual_goal_state":
            continue
        trace = load_trace(row["child"].get("kintwin_trace_path"))
        if trace:
            segment_sim_chunks.append(np.linalg.norm(trace["r_hand_sim_world"] - trace["r_wrist_sim_world"], axis=-1))
            segment_ref_chunks.append(np.linalg.norm(trace["r_hand_ref_world"] - trace["r_wrist_ref_world"], axis=-1))
    segment_sim = np.concatenate(segment_sim_chunks) if segment_sim_chunks else np.empty((0,))
    segment_ref = np.concatenate(segment_ref_chunks) if segment_ref_chunks else np.empty((0,))
    mapping = {
        "status": "partially_verified_by_evaluator_code_and_body_name_convention_but_not_fully_independent_from_saved_trace_metadata",
        "stage1b_metric_code": "evaluate.py selects R_Hand/R_Wrist by name from task._body_names/body_names, then indexes both task._rigid_body_pos and motionlib rg_pos with the same index.",
        "smpl_mujoco_name_indices": {"R_Wrist": SMPL_MUJOCO_NAMES.index("R_Wrist"), "R_Hand": SMPL_MUJOCO_NAMES.index("R_Hand")},
        "nearby_limb_positions_available_in_current_trace": ["R_Wrist", "R_Hand"],
        "nearby_limb_positions_missing": ["R_Elbow", "R_Shoulder", "full body positions"],
        "segment_length_sanity": {
            "sim_wrist_hand_segment_m": stats(segment_sim),
            "ref_wrist_hand_segment_m": stats(segment_ref),
        },
        "remaining_mapping_risk": "current traces do not store body_names, hand/wrist indices, or full adjacent limb arrays; future validity traces should store those fields.",
    }

    timing_best = {
        mode: {
            "best_hand": timing[mode]["best_hand"],
            "best_wrist": timing[mode]["best_wrist"],
        }
        for mode in VIRTUAL_MODES
    }
    hypo = {
        "mode_invariance": invariance,
        "world_space_metrics": trace_aggregates,
        "timing_shift_tests": timing,
        "timing_shift_tests_best_offsets": timing_best,
        "position_range_and_unit_sanity": ranges,
        "validity_trace_alignment_tests": validity_align,
        "completed_vs_terminated_available": True,
        "root_aligned_heading_local_status": validity_align["status"],
    }
    same_mpjpe = same_run["same_run_body_only_frame_weighted_mpjpe_m"]
    same_root = same_run["same_run_body_only_frame_weighted_root_error_m"]
    conclusion = {
        "classification": "Outcome C: insufficient/ambiguous for confirmed hand-objective design",
        "diagnostic_bug_confirmed": False,
        "true_frozen_body_hand_mismatch_confirmed": False,
        "summary": (
            "The existing kintwin traces are enough to reproduce the large hand/wrist and dynamic relation diagnostics, "
            "and the hand/wrist arrays are mode-invariant. However, exact official MPJPE reproduction, root/heading alignment tests, "
            "and independent body-index provenance are blocked by missing full body/ref/root/timing metadata. The Stage 1B same-run "
            f"body metrics already report large frame-weighted MPJPE/root error ({same_mpjpe:.6f} m / {same_root:.6f} m), so the older "
            "0.073808 m global baseline cannot be used as a same-run contradiction."
        ),
        "section_s_status": "soften interpretation: corrected Stage 1B passed, but hand/wrist diagnostic validity audit remains pending before coupling objective design.",
        "user_run_trace_regeneration_needed": True,
        "next_gate": "rerun the same Stage 1B evaluation with save_hand_wrist_validity_traces enabled, then rerun this CPU audit to validate exact MPJPE/root/heading/body-index provenance.",
        "gpu_trace_regeneration_command": (
            "cd /train-data-1-hdd/guancheng/badminton_dataset && "
            "./phc_baseline/analyze/frozen_body_virtual_racket_stage1b/run_user_full_heldout_eval.sh && "
            "phc_baseline/envs/phc_isaac/bin/python phc_baseline/analyze/frozen_body_virtual_racket_stage1b/audit_hand_wrist_metric_validity.py"
        ),
    }

    write_csv(REPORT_DIR / "same_run_body_metric_per_sequence.csv", per_seq_body)
    write_csv(REPORT_DIR / "hand_wrist_alignment_hypothesis_tests_per_sequence.csv", per_trace_rows)
    write_reports(
        {
            "schema": schema,
            "mapping": mapping,
            "same_run": same_run,
            "hypothesis": hypo,
            "conclusion": conclusion,
        }
    )
    print(json.dumps({"classification": conclusion["classification"], "same_run_mpjpe": same_mpjpe, "report_dir": str(REPORT_DIR)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
