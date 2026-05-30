#!/usr/bin/env python3
"""Validate racket goal provider retrieval and frame alignment.

This is a no-training smoke validation.  It checks the sidecar interface
against exported racket-aware reference task NPZ files only; it does not run or
modify PHC policy inference.
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

from racket_goal_provider import GOAL_FIELDS, RacketGoalProvider, parse_offsets  # noqa: E402


def norm_rows(x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(x, axis=-1)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def validate_sequence(
    provider: RacketGoalProvider,
    sequence: str,
    *,
    future_offsets: list[int],
    max_frames_per_clip: int,
) -> dict[str, Any]:
    data = provider._load_sequence(sequence)
    frames_all = np.asarray(data["source_frame_idx"], dtype=np.int64)
    if max_frames_per_clip > 0 and len(frames_all) > max_frames_per_clip:
        pick = np.linspace(0, len(frames_all) - 1, max_frames_per_clip).round().astype(np.int64)
        frames = frames_all[pick]
    else:
        frames = frames_all

    rows = provider._resolve_rows(sequence, frames)
    goal = provider.get_goals_v1(sequence, frames)
    gt_parts = [np.asarray(data[field], dtype=np.float32)[rows] for field in GOAL_FIELDS]
    gt = np.concatenate(gt_parts, axis=-1)
    diff = goal - gt

    handle_err = norm_rows(diff[:, 0:3])
    tip_err = norm_rows(diff[:, 3:6])
    axis_err = norm_rows(diff[:, 6:9])
    finite_ok = np.isfinite(goal).all() and np.isfinite(gt).all()
    contiguous = bool(np.all(np.diff(frames_all) == 1)) if len(frames_all) > 1 else True

    future_ok = True
    future_shape = ""
    future_max_abs_error = 0.0
    if future_offsets:
        future = provider.get_future_goals_v1(sequence, frames, future_offsets)
        future_shape = "x".join(str(x) for x in future.shape)
        expected_blocks = []
        for offset in future_offsets:
            target = frames + int(offset)
            future_rows = provider._resolve_rows(
                sequence,
                target,
                clip=True,
                nearest_if_missing=True,
            )
            expected_blocks.append(np.concatenate([np.asarray(data[field], dtype=np.float32)[future_rows] for field in GOAL_FIELDS], axis=-1))
        expected = np.concatenate(expected_blocks, axis=-1)
        future_max_abs_error = float(np.max(np.abs(future - expected))) if future.size else 0.0
        future_ok = bool(future_max_abs_error < 1e-7)

    passed = bool(
        finite_ok
        and handle_err.max(initial=0.0) < 1e-7
        and tip_err.max(initial=0.0) < 1e-7
        and axis_err.max(initial=0.0) < 1e-7
        and future_ok
    )
    return {
        "sequence": sequence,
        "frame_count_total": int(len(frames_all)),
        "frames_checked": int(len(frames)),
        "source_frame_start": int(frames_all[0]),
        "source_frame_end": int(frames_all[-1]),
        "source_frame_contiguous": contiguous,
        "handle_retrieval_error_mean": float(handle_err.mean()) if len(handle_err) else 0.0,
        "handle_retrieval_error_max": float(handle_err.max()) if len(handle_err) else 0.0,
        "tip_retrieval_error_mean": float(tip_err.mean()) if len(tip_err) else 0.0,
        "tip_retrieval_error_max": float(tip_err.max()) if len(tip_err) else 0.0,
        "long_axis_retrieval_l2_error_mean": float(axis_err.mean()) if len(axis_err) else 0.0,
        "long_axis_retrieval_l2_error_max": float(axis_err.max()) if len(axis_err) else 0.0,
        "finite_ok": finite_ok,
        "future_offsets_checked": ",".join(str(x) for x in future_offsets),
        "future_stacked_shape": future_shape,
        "future_stacked_max_abs_error": future_max_abs_error,
        "passed": passed,
    }


def aggregate(rows: list[dict[str, Any]], future_offsets: list[int]) -> dict[str, Any]:
    passed_rows = [row for row in rows if row["passed"]]
    frames_checked = sum(int(row["frames_checked"]) for row in rows)
    continuous = sum(1 for row in rows if row["source_frame_contiguous"])
    non_contiguous = len(rows) - continuous

    def mean_key(key: str) -> float:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        return float(vals.mean()) if len(vals) else 0.0

    def max_key(key: str) -> float:
        vals = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        return float(vals.max()) if len(vals) else 0.0

    return {
        "scope": "no-training sidecar alignment validation; not PHC simulated rollout accuracy",
        "clips_checked": len(rows),
        "clips_passed": len(passed_rows),
        "clips_failed": len(rows) - len(passed_rows),
        "frames_checked": int(frames_checked),
        "handle_retrieval_error_mean": mean_key("handle_retrieval_error_mean"),
        "handle_retrieval_error_max": max_key("handle_retrieval_error_max"),
        "tip_retrieval_error_mean": mean_key("tip_retrieval_error_mean"),
        "tip_retrieval_error_max": max_key("tip_retrieval_error_max"),
        "long_axis_retrieval_l2_error_mean": mean_key("long_axis_retrieval_l2_error_mean"),
        "long_axis_retrieval_l2_error_max": max_key("long_axis_retrieval_l2_error_max"),
        "missing_sequence_count": 0,
        "non_finite_count": sum(0 if row["finite_ok"] else 1 for row in rows),
        "continuous_source_frame_sequences": continuous,
        "non_contiguous_source_frame_sequences": non_contiguous,
        "future_offsets_checked": future_offsets,
        "future_stacked_max_abs_error": max_key("future_stacked_max_abs_error"),
        "runtime_mapping_dry_run": {
            "status": "skipped",
            "reason": "no PHC rollout diagnostic JSON was supplied to this sidecar validator",
        },
        "goal_schema": {
            "name": "g_racket_v1",
            "shape": "[9] per frame, [B, 9] batched, [B, K*9] with future offsets",
            "fields": [
                "0:3 racket_handle_root_local",
                "3:6 racket_tip_root_local",
                "6:9 racket_long_axis_root_local",
            ],
            "dtype": "float32",
            "frame_convention": "source_frame_idx row lookup in exported racket-aware reference task NPZ",
        },
    }


def write_report(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Racket Goal Provider Alignment Validation",
        "",
        "Scope: no-training sidecar alignment validation. This is not PHC simulated rollout accuracy.",
        "",
        "## Result",
        "",
        f"- Clips checked / passed / failed: {summary['clips_checked']} / {summary['clips_passed']} / {summary['clips_failed']}",
        f"- Frames checked: {summary['frames_checked']}",
        f"- Handle retrieval error mean / max: {summary['handle_retrieval_error_mean']:.12g} / {summary['handle_retrieval_error_max']:.12g} m",
        f"- Tip retrieval error mean / max: {summary['tip_retrieval_error_mean']:.12g} / {summary['tip_retrieval_error_max']:.12g} m",
        f"- Long-axis retrieval L2 error mean / max: {summary['long_axis_retrieval_l2_error_mean']:.12g} / {summary['long_axis_retrieval_l2_error_max']:.12g}",
        f"- Non-finite clips: {summary['non_finite_count']}",
        f"- Continuous / non-contiguous source frame sequences: {summary['continuous_source_frame_sequences']} / {summary['non_contiguous_source_frame_sequences']}",
        f"- Future offset smoke test offsets: {summary['future_offsets_checked']}",
        f"- Future stacked max abs error: {summary['future_stacked_max_abs_error']:.12g}",
        "",
        "## Goal V1 Schema",
        "",
        "`g_racket_v1[t] = concat(handle_root_local, tip_root_local, long_axis_root_local)`.",
        "",
        "Field order:",
        "",
        "- `0:3` racket handle in PHC root-local coordinates",
        "- `3:6` racket tip in PHC root-local coordinates",
        "- `6:9` normalized racket long axis in PHC root-local coordinates",
        "",
        "`racket_pose_parameter` is available only through a diagnostic-only accessor and is not part of primary Goal V1.",
        "",
        "## Runtime Mapping Dry Run",
        "",
        f"- Status: {summary['runtime_mapping_dry_run']['status']}",
        f"- Reason: {summary['runtime_mapping_dry_run']['reason']}",
        "",
        "## Failed Clips",
        "",
    ]
    failed = [row for row in rows if not row["passed"]]
    if not failed:
        lines.append("None.")
    else:
        for row in failed:
            lines.append(f"- `{row['sequence']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_csv", required=True, type=Path)
    parser.add_argument("--output_summary_json", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--future_offsets", default="0,1,2")
    parser.add_argument("--max_frames_per_clip", type=int, default=0)
    args = parser.parse_args()

    provider = RacketGoalProvider(args.manifest_csv)
    future_offsets = parse_offsets(args.future_offsets)
    rows = [
        validate_sequence(provider, sequence, future_offsets=future_offsets, max_frames_per_clip=args.max_frames_per_clip)
        for sequence in provider.sequence_names
    ]
    summary = aggregate(rows, future_offsets)

    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_report(args.output_report, summary, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
