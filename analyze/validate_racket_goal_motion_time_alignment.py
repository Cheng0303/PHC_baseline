#!/usr/bin/env python3
"""Validate PHC motion-time aligned racket goal lookup.

This validator compares the adapter against independently computed endpoint,
mid-frame, clipping, and future-offset expectations.  It does not execute PHC
policy inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from racket_goal_motion_time_adapter import RacketGoalMotionTimeAdapter  # noqa: E402


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("near-zero axis during independent normalization")
    return v / norm


def independent_frame_blend(times: np.ndarray, num_frames: int, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0 / fps
    length = (num_frames - 1) / fps
    t = np.asarray(times, dtype=np.float64).copy()
    phase = np.clip(t / length, 0.0, 1.0) if length > 0 else np.zeros_like(t)
    t[t < 0] = 0.0
    idx0 = np.floor(phase * (num_frames - 1)).astype(np.int64)
    idx1 = np.minimum(idx0 + 1, num_frames - 1).astype(np.int64)
    blend = np.clip((t - idx0 * dt) / dt, 0.0, 1.0)
    return idx0, idx1, blend


def expected_from_rows(goal_rows: np.ndarray, idx0: np.ndarray, idx1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    g0 = goal_rows[idx0]
    g1 = goal_rows[idx1]
    b = blend[:, None]
    handle = (1.0 - b) * g0[:, 0:3] + b * g1[:, 0:3]
    tip = (1.0 - b) * g0[:, 3:6] + b * g1[:, 3:6]
    axis = normalize((1.0 - b) * g0[:, 6:9] + b * g1[:, 6:9])
    return np.concatenate([handle, tip, axis], axis=-1).astype(np.float32)


def l2_max(a: np.ndarray, b: np.ndarray, start: int, end: int) -> float:
    return float(np.linalg.norm(a[:, start:end] - b[:, start:end], axis=-1).max(initial=0.0))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_sequence(adapter: RacketGoalMotionTimeAdapter, sequence: str) -> dict[str, Any]:
    data = adapter.provider._load_sequence(sequence)
    meta = adapter.meta[sequence]
    goal_rows = np.concatenate(
        [
            np.asarray(data["racket_handle_root_local"], dtype=np.float32),
            np.asarray(data["racket_tip_root_local"], dtype=np.float32),
            np.asarray(data["racket_long_axis_root_local"], dtype=np.float32),
        ],
        axis=-1,
    )
    n = meta.num_frames
    dt = meta.dt
    frame_ids = np.arange(n, dtype=np.int64)
    endpoint_times = frame_ids.astype(np.float64) * dt
    adapter_endpoint = adapter.get_goal_v1_by_motion_time([sequence] * n, endpoint_times)
    expected_endpoint = goal_rows

    endpoint_handle_max = l2_max(adapter_endpoint, expected_endpoint, 0, 3)
    endpoint_tip_max = l2_max(adapter_endpoint, expected_endpoint, 3, 6)
    endpoint_axis_max = l2_max(adapter_endpoint, expected_endpoint, 6, 9)

    if n > 1:
        base = np.repeat(np.arange(n - 1, dtype=np.float64) * dt, 3)
        frac = np.tile(np.asarray([0.25, 0.5, 0.75], dtype=np.float64), n - 1)
        mid_times = base + frac * dt
        idx0, idx1, blend = independent_frame_blend(mid_times, n, meta.fps)
        expected_mid = expected_from_rows(goal_rows, idx0, idx1, blend)
        adapter_mid = adapter.get_goal_v1_by_motion_time([sequence] * len(mid_times), mid_times)
    else:
        mid_times = np.asarray([], dtype=np.float64)
        expected_mid = np.zeros((0, 9), dtype=np.float32)
        adapter_mid = np.zeros((0, 9), dtype=np.float32)

    mid_handle_max = l2_max(adapter_mid, expected_mid, 0, 3)
    mid_tip_max = l2_max(adapter_mid, expected_mid, 3, 6)
    mid_axis_max = l2_max(adapter_mid, expected_mid, 6, 9)

    clip_times = np.asarray([-dt, 0.0, meta.length, meta.length + 10 * dt], dtype=np.float64)
    c0, c1, cb = independent_frame_blend(clip_times, n, meta.fps)
    expected_clip = expected_from_rows(goal_rows, c0, c1, cb)
    adapter_clip = adapter.get_goal_v1_by_motion_time([sequence] * len(clip_times), clip_times)
    clipping_max = float(np.max(np.abs(adapter_clip - expected_clip)))

    future_offsets = np.asarray([0.0, dt, 2.0 * dt], dtype=np.float64)
    sample_rows = np.linspace(0, n - 1, min(9, n)).round().astype(np.int64)
    sample_times = sample_rows.astype(np.float64) * dt
    future = adapter.get_goal_v1_by_motion_time([sequence] * len(sample_times), sample_times, future_time_offsets=future_offsets)
    expected_blocks = []
    for offset in future_offsets:
        i0, i1, b = independent_frame_blend(sample_times + offset, n, meta.fps)
        expected_blocks.append(expected_from_rows(goal_rows, i0, i1, b))
    expected_future = np.concatenate(expected_blocks, axis=-1)
    future_max = float(np.max(np.abs(future - expected_future)))

    source_idx = np.asarray(data["source_frame_idx"], dtype=np.int64)
    contiguous = bool(np.all(np.diff(source_idx) == 1)) if len(source_idx) > 1 else True
    passed = bool(
        endpoint_handle_max < 1e-6
        and endpoint_tip_max < 1e-6
        and endpoint_axis_max < 1e-6
        and mid_handle_max < 1e-6
        and mid_tip_max < 1e-6
        and mid_axis_max < 1e-6
        and clipping_max < 1e-6
        and future_max < 1e-6
    )
    return {
        "sequence": sequence,
        "num_frames": n,
        "fps": meta.fps,
        "dt": dt,
        "source_frame_idx_contiguous": contiguous,
        "endpoint_frames_checked": n,
        "interpolation_queries_checked": int(len(mid_times)),
        "endpoint_handle_max_error": endpoint_handle_max,
        "endpoint_tip_max_error": endpoint_tip_max,
        "endpoint_axis_l2_max_error": endpoint_axis_max,
        "midpoint_handle_max_error": mid_handle_max,
        "midpoint_tip_max_error": mid_tip_max,
        "midpoint_axis_l2_max_error": mid_axis_max,
        "clipping_max_abs_error": clipping_max,
        "future_offsets_seconds": ",".join(f"{x:.10g}" for x in future_offsets.tolist()),
        "future_offsets_max_abs_error": future_max,
        "passed": passed,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def max_key(key: str) -> float:
        return float(max(float(row[key]) for row in rows)) if rows else 0.0

    passed = sum(1 for row in rows if row["passed"])
    return {
        "scope": "runtime-time sidecar validation; not PHC simulated rollout accuracy",
        "mapping_audit_required": True,
        "clips_checked": len(rows),
        "clips_passed": passed,
        "clips_failed": len(rows) - passed,
        "endpoint_frames_checked": int(sum(int(row["endpoint_frames_checked"]) for row in rows)),
        "interpolation_queries_checked": int(sum(int(row["interpolation_queries_checked"]) for row in rows)),
        "non_contiguous_clips_covered": sum(1 for row in rows if row["source_frame_idx_contiguous"] is False),
        "endpoint_handle_max_error": max_key("endpoint_handle_max_error"),
        "endpoint_tip_max_error": max_key("endpoint_tip_max_error"),
        "endpoint_axis_l2_max_error": max_key("endpoint_axis_l2_max_error"),
        "midpoint_handle_max_error": max_key("midpoint_handle_max_error"),
        "midpoint_tip_max_error": max_key("midpoint_tip_max_error"),
        "midpoint_axis_l2_max_error": max_key("midpoint_axis_l2_max_error"),
        "clipping_max_abs_error": max_key("clipping_max_abs_error"),
        "future_offsets_max_abs_error": max_key("future_offsets_max_abs_error"),
        "passed": passed == len(rows),
        "interpolation_convention": "handle/tip linear interpolation; long-axis linear interpolation followed by normalization; no full racket face orientation claim",
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Runtime-Time Alignment Validation",
        "",
        "Scope: motion-time sidecar validation only. This is not PHC simulated rollout accuracy.",
        "",
        f"- Clips checked / passed / failed: `{summary['clips_checked']}` / `{summary['clips_passed']}` / `{summary['clips_failed']}`",
        f"- Endpoint frames checked: `{summary['endpoint_frames_checked']}`",
        f"- Interpolation queries checked: `{summary['interpolation_queries_checked']}`",
        f"- Non-contiguous clips covered: `{summary['non_contiguous_clips_covered']}`",
        f"- Endpoint handle/tip/axis max error: `{summary['endpoint_handle_max_error']}` / `{summary['endpoint_tip_max_error']}` / `{summary['endpoint_axis_l2_max_error']}`",
        f"- Midpoint handle/tip/axis max error: `{summary['midpoint_handle_max_error']}` / `{summary['midpoint_tip_max_error']}` / `{summary['midpoint_axis_l2_max_error']}`",
        f"- Clipping max abs error: `{summary['clipping_max_abs_error']}`",
        f"- Future offsets max abs error: `{summary['future_offsets_max_abs_error']}`",
        f"- Passed: `{summary['passed']}`",
        "",
        "The adapter mirrors `MotionLibBase._calc_frame_blend()` for time clipping, frame pair lookup, and blend calculation. Goal V1 interpolation is geometric: handle and tip are linearly interpolated; the long axis is linearly interpolated and normalized. Full racket face orientation is not claimed.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_csv", required=True, type=Path)
    parser.add_argument("--phc_motion_file", required=True, type=Path)
    parser.add_argument("--mapping_audit_csv", required=True, type=Path)
    parser.add_argument("--output_summary_json", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    args = parser.parse_args()

    adapter = RacketGoalMotionTimeAdapter(
        manifest_csv=args.manifest_csv,
        phc_motion_file=args.phc_motion_file,
        mapping_audit_csv=args.mapping_audit_csv,
    )
    rows = [validate_sequence(adapter, sequence) for sequence in adapter.provider.sequence_names]
    summary = aggregate(rows)
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_report(args.output_report, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
