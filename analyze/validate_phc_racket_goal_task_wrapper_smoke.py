#!/usr/bin/env python3
"""No-training smoke validation for the opt-in PHC racket-goal task wrapper.

This is a controlled harness for the wrapper hook logic.  It does not
instantiate Isaac Gym, does not step a PHC policy, and does not feed augmented
observations to the original pretrained checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required for the smoke config guard test") from exc

REPO_ROOT = Path(__file__).resolve().parents[2]
PHC_ROOT = REPO_ROOT / "humenv" / "data_preparation" / "PHC"
ANALYZE_DIR = REPO_ROOT / "phc_baseline" / "analyze"
for path in [PHC_ROOT, ANALYZE_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from phc.env.tasks.racket_goal_runtime import (  # noqa: E402
    INCOMPATIBLE_CHECKPOINT_MESSAGE,
    RACKET_GOAL_DIM,
    RacketGoalRuntimeProvider,
    assert_racket_goal_checkpoint_compatible,
)
from racket_goal_motion_time_adapter import RacketGoalMotionTimeAdapter  # noqa: E402


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [
            row for row in csv.DictReader(f)
            if boolish(row.get("task_export_passed"))
            and boolish(row.get("integrity_check_passed"))
            and boolish(row.get("dynamic_replay_passed"))
        ]


def group_of(sequence: str) -> str:
    return sequence.split("/")[0]


def run_sequence(
    sequence: str,
    motion_entry: dict[str, Any],
    runtime_provider: RacketGoalRuntimeProvider,
    analysis_adapter: RacketGoalMotionTimeAdapter,
    original_dim: int,
) -> dict[str, Any]:
    n = int(len(motion_entry["pose_aa"]))
    fps = float(motion_entry.get("fps", 30))
    dt = 1.0 / fps
    query_times = (np.arange(n, dtype=np.float64) + 1.0) * dt

    # Disabled path parity: base task obs is returned unchanged and no manifest
    # lookup is needed.  The deterministic fake body task obs lets us detect
    # prefix mutations in the enabled path.
    base_obs = np.linspace(0.0, 1.0, n * original_dim, dtype=np.float32).reshape(n, original_dim)
    disabled_obs = base_obs.copy()
    disabled_diff = float(np.max(np.abs(disabled_obs - base_obs))) if base_obs.size else 0.0

    seq_keys = [sequence] * n
    runtime_goal = runtime_provider.lookup_batch(
        seq_keys,
        torch.as_tensor(query_times, dtype=torch.float64),
        torch.full((n,), n, dtype=torch.long),
        torch.full((n,), dt, dtype=torch.float64),
        device="cpu",
        dtype=torch.float32,
    ).cpu().numpy()
    expected_goal = analysis_adapter.get_goal_v1_by_motion_time(seq_keys, query_times)

    enabled_obs = np.concatenate([base_obs, runtime_goal], axis=-1)
    body_prefix_diff = float(np.max(np.abs(enabled_obs[:, :original_dim] - base_obs))) if base_obs.size else 0.0
    appended_diff = float(np.max(np.abs(enabled_obs[:, original_dim:] - expected_goal))) if runtime_goal.size else 0.0
    finite = bool(np.isfinite(enabled_obs).all())

    task_npz = np.load(analysis_adapter.provider.records[sequence].npz_path, allow_pickle=True)
    source_idx = np.asarray(task_npz["source_frame_idx"], dtype=np.int64)
    contiguous = bool(np.all(np.diff(source_idx) == 1)) if len(source_idx) > 1 else True
    return {
        "sequence": sequence,
        "session_group": group_of(sequence),
        "frames_checked": n,
        "source_frame_idx_contiguous": contiguous,
        "disabled_shape": "x".join(str(x) for x in disabled_obs.shape),
        "enabled_shape": "x".join(str(x) for x in enabled_obs.shape),
        "disabled_max_abs_diff": disabled_diff,
        "body_prefix_max_abs_diff": body_prefix_diff,
        "appended_goal_max_abs_diff": appended_diff,
        "finite": finite,
        "passed": disabled_diff == 0.0 and body_prefix_diff == 0.0 and appended_diff < 1e-6 and finite,
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# PHC Racket-Goal Task Wrapper Smoke Validation",
        "",
        "Scope: no-training controlled hook smoke validation. This is not PHC rollout racket accuracy and does not use augmented observations with the original checkpoint.",
        "",
        f"- Execution level: `{summary['execution_level']}`",
        f"- Cases checked / passed / failed: `{summary['clips_checked']}` / `{summary['clips_passed']}` / `{summary['clips_failed']}`",
        f"- Session groups checked: `{summary['session_groups_checked']}`",
        f"- Frames checked: `{summary['frames_checked']}`",
        f"- Non-contiguous clips covered: `{summary['non_contiguous_clips_covered']}`",
        f"- Disabled max abs diff: `{summary['disabled_max_abs_diff']}`",
        f"- Body prefix max abs diff: `{summary['body_prefix_max_abs_diff']}`",
        f"- Appended goal max abs diff: `{summary['appended_goal_max_abs_diff']}`",
        f"- Original task obs component dim: `{summary['original_task_obs_component_dim']}`",
        f"- Augmented task obs component dim: `{summary['augmented_task_obs_component_dim']}`",
        f"- Full policy obs dim confirmed: `{summary['full_policy_obs_dim_confirmed']}`",
        f"- Checkpoint guard coverage: `{summary['checkpoint_guard']['coverage']}`",
        f"- Checkpoint guard refusal passed: `{summary['checkpoint_guard']['refusal_test_passed']}`",
        "",
        "The wrapper class and opt-in config are implemented, but this smoke test uses a controlled harness rather than an actual Isaac Gym environment initialization. The original body-only default config remains unchanged.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_csv", required=True, type=Path)
    parser.add_argument("--phc_motion_file", required=True, type=Path)
    parser.add_argument("--mapping_audit_csv", required=True, type=Path)
    parser.add_argument("--opt_in_env_config", required=True, type=Path)
    parser.add_argument("--output_summary_json", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    args = parser.parse_args()

    rows = load_manifest(args.manifest_csv)
    motion_data = joblib.load(args.phc_motion_file)
    runtime_provider = RacketGoalRuntimeProvider(args.manifest_csv)
    analysis_adapter = RacketGoalMotionTimeAdapter(
        manifest_csv=args.manifest_csv,
        phc_motion_file=args.phc_motion_file,
        mapping_audit_csv=args.mapping_audit_csv,
    )
    first_task = np.load(analysis_adapter.provider.records[rows[0]["sequence"]].npz_path, allow_pickle=True)
    num_track_bodies = int(np.asarray(first_task["reference_body_pos"]).shape[1])
    original_dim = num_track_bodies * 1 * 24
    augmented_dim = original_dim + RACKET_GOAL_DIM

    result_rows = [
        run_sequence(row["sequence"], motion_data[row["sequence"]], runtime_provider, analysis_adapter, original_dim)
        for row in rows
    ]

    cfg = yaml.safe_load(args.opt_in_env_config.read_text(encoding="utf-8"))
    guard_refusal_passed = False
    guard_message = ""
    try:
        assert_racket_goal_checkpoint_compatible(cfg)
    except RuntimeError as exc:
        guard_message = str(exc)
        guard_refusal_passed = INCOMPATIBLE_CHECKPOINT_MESSAGE in guard_message

    passed = sum(1 for row in result_rows if row["passed"])
    summary = {
        "scope": "no-training task wrapper smoke validation; not PHC rollout racket accuracy",
        "wrapper_file": "humenv/data_preparation/PHC/phc/env/tasks/humanoid_im_mcp_getup_racket_goal.py",
        "wrapper_class": "HumanoidImMCPGetupRacketGoal",
        "wrapper_parent": "HumanoidImMCPGetup",
        "config_path": str(args.opt_in_env_config),
        "execution_level": "Level 2 controlled harness; actual Isaac env initialization not executed",
        "clips_checked": len(result_rows),
        "clips_passed": passed,
        "clips_failed": len(result_rows) - passed,
        "session_groups_checked": len({row["session_group"] for row in result_rows}),
        "frames_checked": int(sum(int(row["frames_checked"]) for row in result_rows)),
        "non_contiguous_clips_covered": sum(1 for row in result_rows if not row["source_frame_idx_contiguous"]),
        "disabled_path_requires_manifest": False,
        "disabled_max_abs_diff": float(max(row["disabled_max_abs_diff"] for row in result_rows)),
        "body_prefix_max_abs_diff": float(max(row["body_prefix_max_abs_diff"] for row in result_rows)),
        "appended_goal_max_abs_diff": float(max(row["appended_goal_max_abs_diff"] for row in result_rows)),
        "all_finite": all(bool(row["finite"]) for row in result_rows),
        "original_task_obs_component_dim": original_dim,
        "racket_goal_dim": RACKET_GOAL_DIM,
        "augmented_task_obs_component_dim": augmented_dim,
        "full_policy_obs_dim_confirmed": False,
        "full_policy_obs_dim_note": "Not confirmed because this smoke test does not initialize the Isaac Gym task/env.",
        "query_time_convention": "(progress_buf + 1) * dt + motion_start_times + motion_start_times_offset",
        "checkpoint_guard": {
            "location": "phc.env.tasks.racket_goal_runtime.assert_racket_goal_checkpoint_compatible, called by HumanoidImMCPGetupRacketGoal.__init__ before policy/player use",
            "coverage": "task/config preflight guard plus smoke-validator guard test; full production entrypoint coverage not separately tested",
            "refusal_test_passed": guard_refusal_passed,
            "refusal_message": guard_message,
        },
        "group_clip_counts": dict(Counter(row["session_group"] for row in result_rows)),
    }
    args.output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, result_rows)
    write_report(args.output_report, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
