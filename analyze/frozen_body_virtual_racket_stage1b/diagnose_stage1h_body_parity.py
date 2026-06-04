#!/usr/bin/env python3
"""Stage 1H-2 body parity diagnostic.

This runner compares only two player-compatible routes:

* body_only_player: original env_im_getup_mcp plus an opt-in passive recorder.
* virtual_null_player: virtual-racket env plus the Stage 1G body-IO proxy with
  zero virtual action.

It does not train, tune rewards, load Model A/B, run oracle, or use a physical
racket. The goal is to find where the body-only and virtual-null body routes
first diverge.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    WORKSPACE_ROOT
    / "phc_baseline"
    / "configs"
    / "frozen_body_virtual_racket_stage1b"
    / "stage1h_body_parity_diagnostic_config.json"
)


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_static_config(cfg: dict[str, Any]) -> dict[str, Any]:
    mode_names = [m["name"] for m in cfg["modes"]]
    checks = {
        "run_mode": cfg.get("run_mode") == "stage1h2_body_only_vs_virtual_null_diagnostic",
        "diagnostic_sequence_count_1_to_3": 1 <= len(cfg.get("diagnostic_sequences", [])) <= 3,
        "expected_modes_only": mode_names == ["body_only_player", "virtual_null_player"],
        "body_obs_dim_934": int(cfg["body_obs_dim"]) == 934,
        "body_action_dim_3": int(cfg["body_action_dim"]) == 3,
        "racket_head_input_dim_15": int(cfg["racket_head_input_dim"]) == 15,
        "racket_head_output_dim_6": int(cfg["racket_head_output_dim"]) == 6,
        "combined_action_dim_9": int(cfg["combined_action_dim"]) == 9,
        "combined_matches_parts": int(cfg["body_action_dim"]) + int(cfg["racket_head_output_dim"]) == int(cfg["combined_action_dim"]),
        "uses_original_player_get_action": cfg.get("use_original_player_get_action") is True,
        "manual_actor_route_disallowed": cfg.get("manual_actor_route_allowed") is False,
        "reward_disabled": cfg.get("enable_virtual_racket_reward") is False,
        "physical_racket_disabled": cfg.get("enable_physical_racket") is False,
        "shuttle_disabled": cfg.get("enable_shuttle") is False,
        "full_eval_disabled": cfg.get("full_heldout_evaluation") is False,
    }
    scope = cfg.get("strict_scope", {})
    for key in [
        "no_training",
        "no_optimizer",
        "no_backward",
        "no_ppo_rl",
        "no_reward_update",
        "no_phc_finetune",
        "no_physical_racket_or_shuttle",
        "no_model_a_b_or_oracle",
    ]:
        checks[f"strict_{key}"] = scope.get(key) is True
    return {"passed": all(checks.values()), "checks": checks}


def run_checked(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            handle.write(line)
        ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


def build_motion_subset(cfg: dict[str, Any]) -> None:
    cmd = [
        str(resolve(cfg["python"])),
        str(resolve("phc_baseline/select_phc_motion_subset.py")),
        "--input",
        str(resolve(cfg["motion_source"])),
        "--output",
        str(resolve(cfg["motion_output"])),
        "--keys",
        *cfg["diagnostic_sequences"],
    ]
    run_checked(cmd)


def base_env() -> dict[str, str]:
    env = os.environ.copy()
    env_root_lib = resolve("phc_baseline/envs/phc_isaac/lib")
    env_root_bin = resolve("phc_baseline/envs/phc_isaac/bin")
    env["LD_LIBRARY_PATH"] = f"{env_root_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PATH"] = f"{env_root_bin}:{env.get('PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(resolve("phc_baseline/torch_extensions"))
    return env


def run_mode(cfg: dict[str, Any], mode: dict[str, Any]) -> dict[str, Any]:
    out_dir = resolve(cfg["output_dir"])
    mode_name = mode["name"]
    compact_path = out_dir / f"stage1h2_{mode_name}_compact_metrics.json"
    diag_path = out_dir / (cfg["body_only_diag_json"] if mode_name == "body_only_player" else cfg["virtual_null_diag_json"])
    log_path = out_dir / f"stage1h2_{mode_name}.log"
    for path in (compact_path, diag_path):
        if path.exists():
            path.unlink()

    env = base_env()
    env["PHC_COMPACT_METRICS_PATH"] = str(compact_path)
    args = list(cfg["phc_run_hydra_common_args"])
    num_envs_override = os.environ.get("PHC_STAGE1H2_NUM_ENVS")
    if num_envs_override:
        replaced = False
        for idx, arg in enumerate(args):
            if str(arg).startswith("env.num_envs="):
                args[idx] = f"env.num_envs={int(num_envs_override)}"
                replaced = True
                break
        if not replaced:
            args.append(f"env.num_envs={int(num_envs_override)}")
    args.append(f"env={mode['env_config']}")
    if mode.get("body_diagnostic_proxy"):
        env["PHC_STAGE1H2_BODY_DIAGNOSTIC_PROXY"] = "1"
        env["PHC_STAGE1H2_BODY_DIAG_PATH"] = str(diag_path)
    if mode.get("virtual_racket_proxy"):
        env.update(
            {
                "PHC_STAGE1G_PLAYER_VIRTUAL_RACKET_PROXY": "1",
                "PHC_STAGE1G_REPO_ROOT": str(WORKSPACE_ROOT),
                "PHC_STAGE1G_HOOK_CONFIG": str(resolve(cfg["hook_config"])),
                "PHC_STAGE1G_BODY_OBS_DIM": str(cfg["body_obs_dim"]),
                "PHC_STAGE1G_RACKET_MODE": str(mode["racket_mode"]),
                "PHC_STAGE1G_PROXY_SMOKE_PATH": str(diag_path),
                "PHC_STAGE1G_PROXY_RECORD_METRICS": "0",
                "PHC_STAGE1H2_RECORD_DIAGNOSTICS": "1",
                "PHC_STAGE1G_FULL_HELDOUT_EVALUATION": "0",
            }
        )
    cmd = [
        str(resolve(cfg["python"])),
        "phc/run_hydra.py",
        *args,
        f"env.motion_file={resolve(cfg['motion_output'])}",
        cfg["primitive_actor_models_arg"],
    ]
    run_checked(cmd, cwd=resolve(cfg["phc_root"]), env=env, log_path=log_path)
    return {
        "mode": mode_name,
        "compact_metrics_json": str(compact_path.relative_to(WORKSPACE_ROOT)),
        "diagnostic_json": str(diag_path.relative_to(WORKSPACE_ROOT)),
        "run_log": str(log_path.relative_to(WORKSPACE_ROOT)),
    }


def _as_float_list(values: Any) -> list[float]:
    if not isinstance(values, list):
        return []
    return [float(v) for v in values]


def _vector(values: Any, idx: int) -> list[float] | None:
    if not isinstance(values, list) or idx >= len(values) or not isinstance(values[idx], list):
        return None
    return [float(v) for v in values[idx]]


def _scalar(values: Any, idx: int) -> float | None:
    if not isinstance(values, list) or idx >= len(values):
        return None
    value = values[idx]
    if isinstance(value, list):
        return None
    return float(value)


def _bool(values: Any, idx: int) -> bool | None:
    if not isinstance(values, list) or idx >= len(values):
        return None
    return bool(values[idx])


def _max_abs_vec_diff(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None or len(a) != len(b):
        return None
    return max(abs(x - y) for x, y in zip(a, b)) if a else 0.0


def flatten_records(payload: dict[str, Any], *, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step_idx, record in enumerate(payload.get("records", [])):
        keys = record.get("pre_sequence_keys") or []
        for env_idx, seq in enumerate(keys):
            pre_progress = _scalar(record.get("pre_progress_buf"), env_idx)
            post_progress = _scalar(record.get("post_progress_buf"), env_idx)
            action_vec = _vector(record.get("pre_body_action"), env_idx)
            row = {
                "mode": mode,
                "step_record_index": step_idx,
                "env_index": env_idx,
                "sequence": seq,
                "pre_progress_buf": pre_progress,
                "post_progress_buf": post_progress,
                "match_key": f"{seq}|{pre_progress}",
                "pre_reset": _bool(record.get("pre_reset_buf"), env_idx),
                "post_reset": _bool(record.get("post_reset_buf"), env_idx),
                "done": _bool(record.get("done"), env_idx),
                "pre_terminate": _bool(record.get("pre_terminate_buf"), env_idx),
                "post_terminate": _bool(record.get("post_terminate_buf"), env_idx),
                "pre_body_obs_checksum": _scalar(record.get("pre_body_obs_checksum"), env_idx),
                "post_body_obs_checksum": _scalar(record.get("post_body_obs_checksum"), env_idx),
                "pre_body_obs_norm": _scalar(record.get("pre_body_obs_norm"), env_idx),
                "post_body_obs_norm": _scalar(record.get("post_body_obs_norm"), env_idx),
                "pre_body_pos_checksum": _scalar(record.get("pre_body_pos_checksum"), env_idx),
                "post_body_pos_checksum": _scalar(record.get("post_body_pos_checksum"), env_idx),
                "pre_body_pos_norm": _scalar(record.get("pre_body_pos_norm"), env_idx),
                "post_body_pos_norm": _scalar(record.get("post_body_pos_norm"), env_idx),
                "pre_root_pos": _vector(record.get("pre_root_pos"), env_idx),
                "post_root_pos": _vector(record.get("post_root_pos"), env_idx),
                "body_action": action_vec,
                "body_action_norm": _scalar(record.get("pre_body_action_norm"), env_idx),
                "pre_motion_start_time": _scalar(record.get("pre_motion_start_times"), env_idx),
                "pre_motion_start_time_offset": _scalar(record.get("pre_motion_start_times_offset"), env_idx),
                "pre_sampled_motion_id": _scalar(record.get("pre_sampled_motion_ids"), env_idx),
            }
            rows.append(row)
    return rows


def summarize_compact(compact: dict[str, Any]) -> dict[str, Any]:
    rows = compact.get("per_sequence", [])
    summary = compact.get("summary", {})
    return {
        "sequence_count": len(rows),
        "completed_count": sum(1 for row in rows if row.get("completed")),
        "summary": summary,
    }


def compact_per_sequence_rows(mode_outputs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    body_by_seq: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    loaded = {}
    for mode_output in mode_outputs:
        compact = load_json(resolve(mode_output["compact_metrics_json"]))
        loaded[mode_output["mode"]] = compact
        if mode_output["mode"] == "body_only_player":
            body_by_seq = {
                row.get("sequence_name", row.get("sequence")): row
                for row in compact.get("per_sequence", [])
            }
    for mode_output in mode_outputs:
        mode = mode_output["mode"]
        for row in loaded[mode].get("per_sequence", []):
            seq = row.get("sequence_name", row.get("sequence"))
            body_ref = body_by_seq.get(seq)
            rows.append(
                {
                    "mode": mode,
                    "sequence": seq,
                    "completed": row.get("completed"),
                    "body_only_completed": body_ref.get("completed") if body_ref else None,
                    "frames": row.get("frame_count", row.get("count")),
                    "mean_mpjpe": row.get("mean_mpjpe"),
                    "mean_root_error": row.get("mean_root_error"),
                    "body_only_mean_mpjpe": body_ref.get("mean_mpjpe") if body_ref else None,
                    "body_only_mean_root_error": body_ref.get("mean_root_error") if body_ref else None,
                    "mpjpe_delta_vs_body_only": (float(row["mean_mpjpe"]) - float(body_ref["mean_mpjpe"])) if body_ref else None,
                    "root_delta_vs_body_only": (float(row["mean_root_error"]) - float(body_ref["mean_root_error"])) if body_ref else None,
                }
            )
    return rows, loaded


def compare_diagnostic_records(body_rows: list[dict[str, Any]], virtual_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    body_by_index = {
        (row["step_record_index"], row["env_index"]): row
        for row in body_rows
    }
    use_index_match = all(
        (row["step_record_index"], row["env_index"]) in body_by_index
        for row in virtual_rows
    )
    body_by_key: dict[str, list[dict[str, Any]]] = {}
    if not use_index_match:
        for row in body_rows:
            body_by_key.setdefault(str(row["match_key"]), []).append(row)
    compare_rows: list[dict[str, Any]] = []
    first_divergence = None
    for vrow in virtual_rows:
        if use_index_match:
            brow = body_by_index.get((vrow["step_record_index"], vrow["env_index"]))
        else:
            queue = body_by_key.get(str(vrow["match_key"])) or []
            brow = queue.pop(0) if queue else None
        if brow is None:
            continue
        action_diff = _max_abs_vec_diff(brow.get("body_action"), vrow.get("body_action"))
        root_pre_diff = _max_abs_vec_diff(brow.get("pre_root_pos"), vrow.get("pre_root_pos"))
        root_post_diff = _max_abs_vec_diff(brow.get("post_root_pos"), vrow.get("post_root_pos"))
        obs_diff = abs(float(brow["pre_body_obs_checksum"]) - float(vrow["pre_body_obs_checksum"]))
        body_pre_diff = abs(float(brow["pre_body_pos_checksum"]) - float(vrow["pre_body_pos_checksum"]))
        body_post_diff = abs(float(brow["post_body_pos_checksum"]) - float(vrow["post_body_pos_checksum"]))
        done_diff = brow.get("done") != vrow.get("done")
        reset_diff = brow.get("post_reset") != vrow.get("post_reset")
        row = {
            "sequence": vrow["sequence"],
            "pre_progress_buf": vrow["pre_progress_buf"],
            "body_step_record_index": brow["step_record_index"],
            "virtual_step_record_index": vrow["step_record_index"],
            "body_action_max_abs_diff": action_diff,
            "pre_obs_checksum_abs_diff": obs_diff,
            "pre_root_pos_max_abs_diff": root_pre_diff,
            "post_root_pos_max_abs_diff": root_post_diff,
            "pre_body_pos_checksum_abs_diff": body_pre_diff,
            "post_body_pos_checksum_abs_diff": body_post_diff,
            "done_diff": done_diff,
            "post_reset_diff": reset_diff,
            "body_done": brow.get("done"),
            "virtual_done": vrow.get("done"),
            "body_post_reset": brow.get("post_reset"),
            "virtual_post_reset": vrow.get("post_reset"),
            "body_pre_motion_start_time": brow.get("pre_motion_start_time"),
            "virtual_pre_motion_start_time": vrow.get("pre_motion_start_time"),
            "body_pre_motion_start_time_offset": brow.get("pre_motion_start_time_offset"),
            "virtual_pre_motion_start_time_offset": vrow.get("pre_motion_start_time_offset"),
        }
        compare_rows.append(row)
        if first_divergence is None:
            tol = cfg["diagnostic_tolerance"]
            diverged = (
                (action_diff is not None and action_diff > float(tol["action_max_abs"]))
                or obs_diff > float(tol["obs_checksum_abs"])
                or (root_pre_diff is not None and root_pre_diff > float(tol["root_pos_max_abs"]))
                or (root_post_diff is not None and root_post_diff > float(tol["root_pos_max_abs"]))
                or body_pre_diff > float(tol["body_pos_checksum_abs"])
                or body_post_diff > float(tol["body_pos_checksum_abs"])
                or done_diff
                or reset_diff
            )
            if diverged:
                first_divergence = row
    maxes = {
        "matched_frame_rows": len(compare_rows),
        "match_strategy": "step_record_index_env_index" if use_index_match else "sequence_progress_occurrence_queue",
        "first_divergence": first_divergence,
        "body_action_max_abs_diff": max((r["body_action_max_abs_diff"] or 0.0 for r in compare_rows), default=None),
        "pre_obs_checksum_max_abs_diff": max((r["pre_obs_checksum_abs_diff"] for r in compare_rows), default=None),
        "pre_root_pos_max_abs_diff": max((r["pre_root_pos_max_abs_diff"] or 0.0 for r in compare_rows), default=None),
        "post_root_pos_max_abs_diff": max((r["post_root_pos_max_abs_diff"] or 0.0 for r in compare_rows), default=None),
        "pre_body_pos_checksum_max_abs_diff": max((r["pre_body_pos_checksum_abs_diff"] for r in compare_rows), default=None),
        "post_body_pos_checksum_max_abs_diff": max((r["post_body_pos_checksum_abs_diff"] for r in compare_rows), default=None),
        "done_diff_count": sum(1 for r in compare_rows if r["done_diff"]),
        "post_reset_diff_count": sum(1 for r in compare_rows if r["post_reset_diff"]),
    }
    return compare_rows, maxes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 1H-2 Body Parity Diagnostic",
        "",
        f"Scope: {summary['strict_scope_label']}.",
        "",
        "## Status",
        "",
        f"- passed: `{summary['passed']}`",
        f"- status: `{summary['status']}`",
        f"- exception: `{summary['exception']}`",
        f"- modes compared: `{summary['modes']}`",
        f"- diagnostic sequences: `{summary['diagnostic_sequences']}`",
        "",
        "## Safety",
        "",
        f"- training / optimizer / backward / PPO-RL: `{summary['training']} / {summary['optimizer']} / {summary['backward']} / {summary['ppo_rl']}`",
        f"- reward update: `{summary['reward_update']}`",
        f"- physical racket or shuttle: `{summary['physical_racket_or_shuttle']}`",
        f"- Model A/B/oracle loaded: `{summary['model_a_b_or_oracle_loaded']}`",
        "",
        "## Compact Metrics",
        "",
    ]
    for mode, mode_summary in summary["mode_summaries"].items():
        compact = mode_summary.get("summary") or {}
        lines.append(
            f"- `{mode}`: clips `{mode_summary['sequence_count']}`, completed `{mode_summary['completed_count']}`, "
            f"MPJPE/root `{compact.get('dataset_mean_mpjpe')}` / `{compact.get('dataset_mean_root_error')}`"
        )
    lines.extend(
        [
            "",
            "## First Divergence",
            "",
            f"- first divergence: `{summary['diagnostic_comparison']['first_divergence']}`",
            f"- matched frame rows: `{summary['diagnostic_comparison']['matched_frame_rows']}`",
            f"- body action max diff: `{summary['diagnostic_comparison']['body_action_max_abs_diff']}`",
            f"- pre obs checksum max diff: `{summary['diagnostic_comparison']['pre_obs_checksum_max_abs_diff']}`",
            f"- pre root max diff: `{summary['diagnostic_comparison']['pre_root_pos_max_abs_diff']}`",
            f"- post root max diff: `{summary['diagnostic_comparison']['post_root_pos_max_abs_diff']}`",
            f"- done/reset diff counts: `{summary['diagnostic_comparison']['done_diff_count']}` / `{summary['diagnostic_comparison']['post_reset_diff_count']}`",
            "",
            "Interpretation: this diagnostic only compares `body_only_player` and `virtual_null_player`. It does not evaluate Model B or claim Stage 1H success.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def summarize(cfg: dict[str, Any], mode_outputs: list[dict[str, Any]], *, exception: str | None = None) -> dict[str, Any]:
    out_dir = resolve(cfg["output_dir"])
    compact_rows, loaded_compacts = compact_per_sequence_rows(mode_outputs)
    mode_summaries = {
        mode_output["mode"]: summarize_compact(loaded_compacts[mode_output["mode"]])
        for mode_output in mode_outputs
    }
    body_diag = load_json(resolve(mode_outputs[0]["diagnostic_json"])) if mode_outputs and Path(resolve(mode_outputs[0]["diagnostic_json"])).exists() else {}
    virtual_diag = load_json(resolve(mode_outputs[1]["diagnostic_json"])) if len(mode_outputs) > 1 and Path(resolve(mode_outputs[1]["diagnostic_json"])).exists() else {}
    body_rows = flatten_records(body_diag, mode="body_only_player")
    virtual_rows = flatten_records(virtual_diag, mode="virtual_null_player")
    compare_rows, compare_summary = compare_diagnostic_records(body_rows, virtual_rows, cfg) if body_rows and virtual_rows else ([], {"matched_frame_rows": 0, "first_divergence": None})

    compact_virtual_rows = [row for row in compact_rows if row["mode"] == "virtual_null_player"]
    compact_deltas = [
        max(abs(float(row["mpjpe_delta_vs_body_only"])), abs(float(row["root_delta_vs_body_only"])))
        for row in compact_virtual_rows
        if row.get("mpjpe_delta_vs_body_only") is not None
    ]
    body_parity_passed = bool(compact_deltas) and max(compact_deltas) <= float(cfg["body_parity_tolerance_m"]) and not any(
        row.get("completed") != row.get("body_only_completed")
        for row in compact_virtual_rows
    )
    passed = exception is None and body_parity_passed and compare_summary.get("first_divergence") is None
    summary = {
        "passed": passed,
        "status": "passed_stage1h2_body_parity_diagnostic" if passed else "failed_or_incomplete_stage1h2_body_parity_diagnostic",
        "exception": exception,
        "strict_scope_label": cfg["strict_scope_label"],
        "modes": [m["name"] for m in cfg["modes"]],
        "diagnostic_sequences": cfg["diagnostic_sequences"],
        "body_obs_dim": int(cfg["body_obs_dim"]),
        "body_action_dim": int(cfg["body_action_dim"]),
        "racket_head_input_dim": int(cfg["racket_head_input_dim"]),
        "racket_head_output_dim": int(cfg["racket_head_output_dim"]),
        "combined_action_dim": int(cfg["combined_action_dim"]),
        "training": False,
        "optimizer": False,
        "backward": False,
        "ppo_rl": False,
        "reward_update": False,
        "physical_racket_or_shuttle": False,
        "model_a_b_or_oracle_loaded": False,
        "mode_summaries": mode_summaries,
        "compact_body_parity_passed": body_parity_passed,
        "compact_max_delta_vs_body_only_m": max(compact_deltas) if compact_deltas else None,
        "diagnostic_records": {
            "body_only_records": len(body_diag.get("records", [])),
            "virtual_null_records": len(virtual_diag.get("records", [])),
            "body_only_frame_rows": len(body_rows),
            "virtual_null_frame_rows": len(virtual_rows),
        },
        "diagnostic_comparison": compare_summary,
        "outputs": mode_outputs,
    }
    write_json(out_dir / cfg["summary_json"], summary)
    write_csv(out_dir / cfg["per_frame_csv"], compare_rows)
    write_config_diff(out_dir / cfg["config_diff_json"], cfg)
    write_report(out_dir / cfg["report_md"], summary)
    return summary


def write_config_diff(path: Path, cfg: dict[str, Any]) -> None:
    payload = {
        "body_only_env_config": "env_im_getup_mcp",
        "virtual_null_env_config": "env_im_getup_mcp_virtual_racket_stage1b",
        "known_current_path_differences": {
            "task": ["HumanoidImMCPGetup", "HumanoidImMCPGetupVirtualRacket"],
            "public_obs_dim": [934, 949],
            "player_exposed_obs_dim": [934, 934],
            "public_action_dim": [3, 9],
            "player_exposed_action_dim": [3, 3],
            "virtual_racket_reward": [False, False],
            "virtual_racket_physics": [False, False],
        },
        "diagnostic_note": "Full YAML diff should be inspected if this diagnostic shows reset/state/action divergence.",
        "strict_scope": cfg["strict_scope"],
    }
    write_json(path, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    cfg = load_json(resolve(args.config))
    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    static = validate_static_config(cfg)
    if args.validate_only:
        print(json.dumps(static, indent=2))
        if not static["passed"]:
            raise SystemExit(1)
        return
    if not static["passed"]:
        raise RuntimeError(f"static config validation failed: {static}")

    mode_outputs: list[dict[str, Any]] = []
    exception = None
    if args.summarize_only:
        for mode in cfg["modes"]:
            mode_outputs.append(
                {
                    "mode": mode["name"],
                    "compact_metrics_json": f"{cfg['output_dir']}/stage1h2_{mode['name']}_compact_metrics.json",
                    "diagnostic_json": f"{cfg['output_dir']}/"
                    + (cfg["body_only_diag_json"] if mode["name"] == "body_only_player" else cfg["virtual_null_diag_json"]),
                    "run_log": f"{cfg['output_dir']}/stage1h2_{mode['name']}.log",
                }
            )
    else:
        try:
            build_motion_subset(cfg)
            for mode in cfg["modes"]:
                mode_outputs.append(run_mode(cfg, mode))
        except Exception as exc:
            exception = repr(exc)
            print(f"stage1h2 body parity diagnostic failed: {exception}", file=sys.stderr)
            raise
        finally:
            if mode_outputs:
                summarize(cfg, mode_outputs, exception=exception)
    if mode_outputs:
        summary = summarize(cfg, mode_outputs, exception=exception)
        print(json.dumps({k: summary[k] for k in ["passed", "status", "compact_max_delta_vs_body_only_m"]}, indent=2))


if __name__ == "__main__":
    main()
