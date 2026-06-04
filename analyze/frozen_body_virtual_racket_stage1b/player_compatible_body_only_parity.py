#!/usr/bin/env python3
"""Stage 1F player-compatible body-only parity check.

This script deliberately does not implement a new rollout loop. It validates
the same 40 held-out racket-task clips by extracting them from the original PHC
compact-metrics / RL Games player output artifact. That keeps the body-only
parity gate tied to the known-good player path instead of the custom Stage 1B
manual actor route.

No Isaac Gym evaluation, training, optimizer step, backward pass, PPO/RL,
reward update, racket-head load, virtual action, physical racket, shuttle, or
checkpoint modification is performed by this script.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PHC_BASELINE = WORKSPACE_ROOT / "phc_baseline"
DEFAULT_CONFIG = PHC_BASELINE / "configs" / "frozen_body_virtual_racket_stage1b" / "body_only_player_parity_config.json"


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return WORKSPACE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest_sequences(path: Path, groups: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            group = row.get("session_group") or row.get("sequence", "").split("/")[0]
            if group in groups:
                rows.append(row)
    rows.sort(key=lambda r: (r.get("session_group", ""), int(r.get("selection_rank_within_group", 0)), r.get("sequence", "")))
    return rows


def metric_mean(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def weighted_metric_mean(rows: list[dict[str, Any]], key: str, weight_key: str = "count") -> float | None:
    total = 0.0
    weight_sum = 0.0
    for row in rows:
        weight = float(row.get(weight_key) or 0)
        if weight <= 0:
            continue
        total += float(row[key]) * weight
        weight_sum += weight
    if weight_sum <= 0:
        return None
    return float(total / weight_sum)


def select_compact_metrics_json(cfg: dict[str, Any]) -> Path:
    primary = resolve_path(cfg["compact_metrics_json"])
    if primary.exists():
        return primary
    fallback = resolve_path(cfg.get("fallback_compact_metrics_json", ""))
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No compact-metrics JSON found. Primary: {primary}; fallback: {fallback}. "
        "Run the user command in the generated command markdown first."
    )


def build_user_command(cfg: dict[str, Any], config_path: Path) -> str:
    out_dir = resolve_path(cfg["output_dir"])
    compact_json = resolve_path(cfg["compact_metrics_json"])
    compact_log = compact_json.with_suffix(".log")
    script = resolve_path(cfg["original_compact_metrics_script"])
    motion_file = resolve_path(cfg["motion_file"])
    py = "phc_baseline/envs/phc_isaac/bin/python"
    return f"""cd {WORKSPACE_ROOT}
nvidia-smi
{py} - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("torch_cuda", torch.version.cuda)
PY
{py} -m py_compile \\
  phc_baseline/analyze/frozen_body_virtual_racket_stage1b/player_compatible_body_only_parity.py
python3 -m json.tool \\
  {config_path.relative_to(WORKSPACE_ROOT)} >/dev/null

# Reuse the existing original-player compact metrics JSON if present.
# If it is missing or you want a fresh original-player run, uncomment the next command.
# bash {script.relative_to(WORKSPACE_ROOT)} \\
#   {motion_file.relative_to(WORKSPACE_ROOT)} \\
#   {compact_json} \\
#   {cfg["num_envs"]} \\
#   {cfg["games_num"]} 2>&1 | tee {compact_log}

{py} \\
  phc_baseline/analyze/frozen_body_virtual_racket_stage1b/player_compatible_body_only_parity.py \\
  --config {config_path.relative_to(WORKSPACE_ROOT)}

cat {out_dir / cfg["summary_json"]}
sed -n '1,220p' {out_dir / cfg["report_md"]}
head -20 {out_dir / cfg["per_sequence_csv"]}
git -C phc_baseline status --short
"""


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence",
        "session_group",
        "manifest_frame_count",
        "manifest_completed_body_baseline",
        "compact_completed",
        "compact_count",
        "compact_frame_count",
        "compact_termination_frame",
        "mean_mpjpe",
        "mean_root_error",
        "mean_body_error",
        "max_mpjpe",
        "max_root_error",
        "valid_export",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage 1F Player-Compatible Body-Only Parity",
        "",
        "Scope: body-only parity check using the original PHC compact-metrics / RL Games player artifact. No racket branch, Stage 1A head load, virtual action, training, optimizer, backward pass, PPO/RL, reward update, physical racket, shuttle, or checkpoint modification was performed by this script.",
        "",
        "## Status",
        "",
        f"- status: `{summary['status']}`",
        f"- passed: `{summary['passed']}`",
        f"- compact metrics source: `{summary['compact_metrics_json']}`",
        f"- matched held-out clips: `{summary['matched_clip_count']} / {summary['expected_clip_count']}`",
        f"- completed held-out clips: `{summary['completed_clip_count']} / {summary['expected_clip_count']}`",
        f"- mean MPJPE/root: `{summary['mean_mpjpe_m']:.12f}` / `{summary['mean_root_error_m']:.12f}` m",
        f"- target MPJPE/root: `{summary['target_mean_mpjpe_m']:.12f}` / `{summary['target_mean_root_error_m']:.12f}` m",
        f"- absolute deltas: `{summary['mpjpe_abs_delta_m']:.12e}` / `{summary['root_error_abs_delta_m']:.12e}` m",
        "",
        "## Player Semantics Reused",
        "",
        "- original `phc/run_hydra.py` launch through `run_phc_badminton_dataset_metrics.sh`",
        "- `IMAMPPlayerContinuous.env_reset/get_action/env_step/_post_step/forward_motion_samples` compact-metrics loop",
        "- full restored RL Games model and player action path",
        "- player observation preprocessing / normalization",
        "- deterministic `mus`, action clamp/rescale inside RL Games player",
        "- compact metric formula from `_update_compact_metrics()` / `_dump_compact_metrics()`",
        "",
        "## Deliberately Not Used",
        "",
        "- custom Stage 1B `FrozenPHCBodyActor` / manual `actor_mlp + mu` route",
        "- one-motion temporary pickle rollout loop",
        "- direct `task.reset()/task.step()` custom body rollout",
        "- Stage 1A racket head or any virtual racket action",
        "",
        "## Future Stage 1G Design Note",
        "",
        "Only after a body-only parity path is kept player-compatible should the virtual racket branch be reintroduced. The future hook should preserve the original player body route `body obs [934] -> body MCP weights [3]`, compute the virtual head side input separately, and concatenate `[3] + [6] -> [9]` only at the virtual env boundary.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config_path = resolve_path(args.config)
    cfg = load_json(config_path)
    out_dir = resolve_path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = set(cfg["expected_test_groups"])
    manifest_rows = load_manifest_sequences(resolve_path(cfg["manifest_csv"]), groups)
    expected_clip_count = int(cfg["expected_test_clip_count"])
    compact_path = select_compact_metrics_json(cfg)
    compact = load_json(compact_path)
    compact_rows = {str(r["sequence_name"]): r for r in compact.get("per_sequence", [])}

    missing = [r["sequence"] for r in manifest_rows if r["sequence"] not in compact_rows]
    selected: list[dict[str, Any]] = []
    for row in manifest_rows:
        seq = row["sequence"]
        item = compact_rows.get(seq)
        if item is None:
            continue
        selected.append(
            {
                "sequence": seq,
                "session_group": row.get("session_group", seq.split("/")[0]),
                "manifest_frame_count": int(row.get("frame_count") or 0),
                "manifest_completed_body_baseline": row.get("completed_body_baseline"),
                "compact_completed": bool(item.get("completed")),
                "compact_count": int(item.get("count") or 0),
                "compact_frame_count": int(item.get("frame_count") or 0),
                "compact_termination_frame": item.get("termination_frame"),
                "mean_mpjpe": float(item.get("mean_mpjpe")),
                "mean_root_error": float(item.get("mean_root_error")),
                "mean_body_error": float(item.get("mean_body_error")),
                "max_mpjpe": float(item.get("max_mpjpe")),
                "max_root_error": float(item.get("max_root_error")),
                "valid_export": bool(item.get("valid_export", True)),
            }
        )

    mean_mpjpe = metric_mean(selected, "mean_mpjpe")
    mean_root = metric_mean(selected, "mean_root_error")
    weighted_mpjpe = weighted_metric_mean(selected, "mean_mpjpe", "compact_count")
    weighted_root = weighted_metric_mean(selected, "mean_root_error", "compact_count")
    completed = sum(1 for r in selected if r["compact_completed"])
    invalid = [r["sequence"] for r in selected if not r["valid_export"]]
    target_mpjpe = float(cfg["target_mean_mpjpe_m"])
    target_root = float(cfg["target_mean_root_error_m"])
    tol = float(cfg["metric_abs_tolerance_m"])

    pass_checks = {
        "expected_clip_count": len(manifest_rows) == expected_clip_count,
        "matched_clip_count": len(selected) == expected_clip_count,
        "missing_clip_count_zero": len(missing) == 0,
        "completed_clip_count": completed >= int(cfg["required_completed_clips"]),
        "valid_exports": len(invalid) == 0,
        "mpjpe_close_to_target": mean_mpjpe is not None and abs(mean_mpjpe - target_mpjpe) <= tol,
        "root_error_close_to_target": mean_root is not None and abs(mean_root - target_root) <= tol,
        "no_racket_branch": bool(cfg["strict_scope"]["no_racket_branch"]),
        "no_stage1a_head_load": bool(cfg["strict_scope"]["no_stage1a_head_load"]),
        "no_training": bool(cfg["strict_scope"]["no_training"]),
    }
    passed = all(pass_checks.values())

    summary = {
        "status": "passed_from_existing_original_player_compact_metrics_artifact" if passed else "failed_or_incomplete_original_player_parity",
        "passed": passed,
        "compact_metrics_json": str(compact_path.relative_to(WORKSPACE_ROOT)),
        "manifest_csv": cfg["manifest_csv"],
        "expected_groups": cfg["expected_test_groups"],
        "expected_clip_count": expected_clip_count,
        "manifest_clip_count": len(manifest_rows),
        "matched_clip_count": len(selected),
        "missing_sequences": missing,
        "completed_clip_count": completed,
        "invalid_export_sequences": invalid,
        "mean_mpjpe_m": mean_mpjpe,
        "mean_root_error_m": mean_root,
        "weighted_mean_mpjpe_m": weighted_mpjpe,
        "weighted_mean_root_error_m": weighted_root,
        "target_mean_mpjpe_m": target_mpjpe,
        "target_mean_root_error_m": target_root,
        "mpjpe_abs_delta_m": abs(mean_mpjpe - target_mpjpe) if mean_mpjpe is not None else None,
        "root_error_abs_delta_m": abs(mean_root - target_root) if mean_root is not None else None,
        "pass_checks": pass_checks,
        "original_player_semantics": {
            "uses_original_compact_metrics_artifact": True,
            "manual_actor_route_used": False,
            "custom_stage1b_rollout_loop_used": False,
            "stage1a_head_loaded": False,
            "racket_branch_used": False,
        },
        "confirmed_body_contract": {
            "checkpoint": cfg["confirmed_body_checkpoint"],
            "obs_dim": int(cfg["confirmed_body_obs_dim"]),
            "action_dim": int(cfg["confirmed_body_action_dim"]),
        },
        "strict_scope": cfg["strict_scope"],
        "next_gate": "player-compatible virtual-racket integration hook design only after preserving this body route",
    }

    summary_path = out_dir / cfg["summary_json"]
    report_path = out_dir / cfg["report_md"]
    csv_path = out_dir / cfg["per_sequence_csv"]
    command_path = out_dir / cfg["command_md"]

    write_json(summary_path, summary)
    write_csv(csv_path, selected)
    write_report(report_path, summary)
    command = build_user_command(cfg, config_path)
    command_path.write_text(
        "# Stage 1F Body-Only Player Parity Command\n\n"
        "This command uses the original compact-metrics / RL Games player artifact path. It does not load the Stage 1A racket head and does not train.\n\n"
        "```bash\n"
        + command
        + "```\n",
        encoding="utf-8",
    )

    print(json.dumps({"passed": passed, "status": summary["status"], "matched": len(selected), "completed": completed, "mean_mpjpe_m": mean_mpjpe, "mean_root_error_m": mean_root}, indent=2))


if __name__ == "__main__":
    main()
