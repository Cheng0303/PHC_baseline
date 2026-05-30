#!/usr/bin/env python3
"""Dry-run synchronized racket goal tensors for a future obs extension.

This is an offline motion-time shadow validation.  It does not instantiate
Isaac Gym, does not modify PHC observations, and does not feed augmented
observations to the pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from racket_goal_motion_time_adapter import RacketGoalMotionTimeAdapter  # noqa: E402


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def group_of(sequence: str) -> str:
    return sequence.split("/")[0]


def dryrun_sequence(adapter: RacketGoalMotionTimeAdapter, sequence: str) -> dict[str, Any]:
    meta = adapter.meta[sequence]
    progress = np.arange(meta.num_frames, dtype=np.float64)
    query_times = (progress + 1.0) * meta.dt
    goals = adapter.get_goal_v1_by_motion_time([sequence] * len(query_times), query_times)
    finite = bool(np.isfinite(goals).all())
    return {
        "sequence": sequence,
        "session_group": group_of(sequence),
        "num_frames": meta.num_frames,
        "fps": meta.fps,
        "dt": meta.dt,
        "query_count": int(len(query_times)),
        "query_convention": "(progress_buf + 1) * dt, clipped by PHC-style motion-time lookup",
        "shadow_goal_shape": "x".join(str(x) for x in goals.shape),
        "shadow_goal_dtype": str(goals.dtype),
        "finite": finite,
        "goal_abs_max": float(np.max(np.abs(goals))) if goals.size else 0.0,
        "passed": finite and goals.shape == (meta.num_frames, 9),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Shadow Observation Extension Dry Run",
        "",
        "Scope: offline motion-time shadow validation only. No PHC env rollout, no policy observation mutation, and no pretrained checkpoint evaluation.",
        "",
        f"- Execution level: `{summary['execution_level']}`",
        f"- Clips checked / passed / failed: `{summary['clips_checked']}` / `{summary['clips_passed']}` / `{summary['clips_failed']}`",
        f"- Session groups checked: `{summary['session_groups_checked']}`",
        f"- Query times checked: `{summary['query_times_checked']}`",
        f"- Active convention: `{summary['active_query_convention']}`",
        f"- Original task obs dim estimate: `{summary['original_task_obs_dim_estimate']}`",
        f"- Racket goal dim: `{summary['racket_goal_dim']}`",
        f"- Proposed augmented task obs dim: `{summary['proposed_augmented_task_obs_dim']}`",
        f"- All finite: `{summary['all_finite']}`",
        "",
        "The original pretrained policy observation is not modified. The proposed augmented dimension is incompatible with the existing checkpoint as-is and would require new/adapted policy weights or a separate side branch.",
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
    rows = [dryrun_sequence(adapter, sequence) for sequence in adapter.provider.sequence_names]
    passed = sum(1 for row in rows if row["passed"])
    groups = sorted({row["session_group"] for row in rows})

    # The current SMPL task uses obs_v=6 and fut_tracks=False.  With full SMPL
    # body tracking, task obs dim is num_track_bodies * 1 * 24.
    first = adapter.provider._load_sequence(adapter.provider.sequence_names[0])
    num_track_bodies = int(np.asarray(first["reference_body_pos"]).shape[1])
    original_dim = num_track_bodies * 1 * 24
    goal_dim = 9
    summary = {
        "scope": "shadow observation extension dry-run; not PHC simulated rollout accuracy",
        "execution_level": "motion-library standalone; environment rollout hook not executed",
        "clips_checked": len(rows),
        "clips_passed": passed,
        "clips_failed": len(rows) - passed,
        "session_groups_checked": len(groups),
        "session_groups": groups,
        "query_times_checked": int(sum(int(row["query_count"]) for row in rows)),
        "active_query_convention": "(progress_buf + 1) * dt with PHC-style clipping; fut_tracks=False active case",
        "num_track_bodies_estimate": num_track_bodies,
        "original_task_obs_dim_estimate": original_dim,
        "racket_goal_dim": goal_dim,
        "proposed_augmented_task_obs_dim": original_dim + goal_dim,
        "future_stacked_goal_dim_example_k3": goal_dim * 3,
        "checkpoint_compatibility": "incompatible with existing pretrained checkpoint as-is; requires new/adapted policy weights or separate branch",
        "all_finite": all(bool(row["finite"]) for row in rows),
        "group_clip_counts": dict(Counter(row["session_group"] for row in rows)),
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, rows)
    write_report(args.output_report, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
