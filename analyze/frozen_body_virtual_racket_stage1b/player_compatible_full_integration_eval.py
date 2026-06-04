#!/usr/bin/env python3
"""Stage 1H full player-compatible virtual-racket held-out evaluation.

This runner preserves the original PHC RL Games player body route. For virtual
branches, the opt-in Stage 1G proxy exposes body-only [934]/[3] IO to the player
and appends a no-physics virtual-racket action [6] at the env boundary.

No training, optimizer, reward update, physical racket, shuttle, or manual PHC
actor reconstruction is performed here.
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
DEFAULT_CONFIG = WORKSPACE_ROOT / "phc_baseline" / "configs" / "frozen_body_virtual_racket_stage1b" / "player_compatible_full_integration_config.json"


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
        "run_mode": cfg.get("run_mode") == "full_player_compatible_heldout_integration",
        "expected_groups": cfg.get("heldout_groups") == ["241217_2", "241226_1"],
        "expected_clip_count": int(cfg["expected_clip_count"]) == 40,
        "expected_modes": mode_names
        == [
            "body_only_player",
            "virtual_null_player",
            "virtual_goal_only_player",
            "virtual_goal_state_player",
            "virtual_oracle_player",
        ],
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
        "full_eval_enabled": cfg.get("full_heldout_evaluation") is True,
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


def select_heldout_sequences(cfg: dict[str, Any]) -> list[str]:
    rows = []
    groups = set(cfg["heldout_groups"])
    with resolve(cfg["manifest"]).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("task_export_passed") == "True"
                and row.get("integrity_check_passed") == "True"
                and row.get("dynamic_replay_passed") == "True"
                and row.get("session_group") in groups
            ):
                rows.append(row)
    rows = sorted(rows, key=lambda row: (row["session_group"], row["sequence"]))
    sequences = [row["sequence"] for row in rows]
    expected = int(cfg["expected_clip_count"])
    if len(sequences) != expected:
        raise RuntimeError(f"expected {expected} held-out clips from {cfg['heldout_groups']}, found {len(sequences)}")
    return sequences


def build_motion_subset(cfg: dict[str, Any], sequences: list[str]) -> None:
    cmd = [
        str(resolve(cfg["python"])),
        str(resolve("phc_baseline/select_phc_motion_subset.py")),
        "--input",
        str(resolve(cfg["motion_source"])),
        "--output",
        str(resolve(cfg["motion_output"])),
        "--keys",
        *sequences,
    ]
    run_checked(cmd)


def base_env(cfg: dict[str, Any]) -> dict[str, str]:
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
    compact_path = out_dir / f"stage1h_{mode_name}_compact_metrics.json"
    proxy_path = out_dir / f"stage1h_{mode_name}_proxy_records.json"
    log_path = out_dir / f"stage1h_{mode_name}.log"
    for path in (compact_path, proxy_path):
        if path.exists():
            path.unlink()

    env = base_env(cfg)
    env["PHC_COMPACT_METRICS_PATH"] = str(compact_path)
    args = list(cfg["phc_run_hydra_common_args"])
    if mode.get("proxy"):
        env.update(
            {
                "PHC_STAGE1G_PLAYER_VIRTUAL_RACKET_PROXY": "1",
                "PHC_STAGE1G_REPO_ROOT": str(WORKSPACE_ROOT),
                "PHC_STAGE1G_HOOK_CONFIG": str(resolve(cfg["hook_config"])),
                "PHC_STAGE1G_BODY_OBS_DIM": str(cfg["body_obs_dim"]),
                "PHC_STAGE1G_RACKET_MODE": str(mode["racket_mode"]),
                "PHC_STAGE1G_PROXY_SMOKE_PATH": str(proxy_path),
                "PHC_STAGE1G_PROXY_RECORD_METRICS": "1",
                "PHC_STAGE1G_FULL_HELDOUT_EVALUATION": "1",
            }
        )
        args.append(f"env={cfg['virtual_env_config']}")
    else:
        env.pop("PHC_STAGE1G_PLAYER_VIRTUAL_RACKET_PROXY", None)
        args.append(f"env={cfg['body_only_env_config']}")

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
        "proxy_records_json": str(proxy_path.relative_to(WORKSPACE_ROOT)) if proxy_path.exists() else None,
        "run_log": str(log_path.relative_to(WORKSPACE_ROOT)),
    }


def flatten_virtual_records(proxy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for record in proxy.get("records", []):
        keys = record.get("sequence_keys", [])
        n = len(keys)
        for idx in range(n):
            row = {
                "sequence": keys[idx],
                "progress_buf": _list_value(record.get("progress_buf"), idx),
                "handle_error_m": _list_value(record.get("handle_error_m"), idx),
                "tip_error_m": _list_value(record.get("tip_error_m"), idx),
                "axis_error_deg": _list_value(record.get("axis_error_deg"), idx),
                "action_magnitude": _list_value(record.get("racket_action_magnitude"), idx),
                "action_smoothness": _list_value(record.get("racket_action_smoothness"), idx),
            }
            rows.append(row)
    return rows


def _list_value(values: Any, idx: int) -> float | None:
    if not isinstance(values, list) or idx >= len(values):
        return None
    value = values[idx]
    if isinstance(value, list):
        return None
    return float(value)


def stats(values: list[float | None]) -> dict[str, float | None]:
    vals = sorted(float(v) for v in values if v is not None and np_isfinite(v))
    if not vals:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {
        "mean": sum(vals) / len(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "max": vals[-1],
    }


def np_isfinite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def percentile(vals: list[float], pct: float) -> float:
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    blend = pos - lo
    return vals[lo] * (1.0 - blend) + vals[hi] * blend


def summarize_outputs(cfg: dict[str, Any], mode_outputs: list[dict[str, Any]], *, exception: str | None = None) -> dict[str, Any]:
    out_dir = resolve(cfg["output_dir"])
    body_by_seq: dict[str, dict[str, Any]] = {}
    mode_summaries = {}
    per_sequence_rows = []
    virtual_rows = []
    for mode_output in mode_outputs:
        mode = mode_output["mode"]
        compact = load_json(resolve(mode_output["compact_metrics_json"])) if mode_output.get("compact_metrics_json") else {}
        per_seq = compact.get("per_sequence", [])
        mode_summaries[mode] = {
            "sequence_count": len(per_seq),
            "completed_count": sum(1 for row in per_seq if row.get("completed")),
            "summary": compact.get("summary", {}),
        }
        if mode == "body_only_player":
            body_by_seq = {row.get("sequence_name", row.get("sequence")): row for row in per_seq}
        for row in per_seq:
            seq = row.get("sequence_name", row.get("sequence"))
            body_ref = body_by_seq.get(seq) if body_by_seq else None
            per_sequence_rows.append(
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
                    "mpjpe_delta_vs_body_only": (float(row["mean_mpjpe"]) - float(body_ref["mean_mpjpe"])) if body_ref and row.get("mean_mpjpe") is not None else None,
                    "root_delta_vs_body_only": (float(row["mean_root_error"]) - float(body_ref["mean_root_error"])) if body_ref and row.get("mean_root_error") is not None else None,
                }
            )
        proxy_path = mode_output.get("proxy_records_json")
        if proxy_path:
            proxy = load_json(resolve(proxy_path))
            rows = flatten_virtual_records(proxy)
            for row in rows:
                row["mode"] = mode
            virtual_rows.extend(rows)

    body_parity = compute_body_parity(cfg, per_sequence_rows)
    virtual_summary = summarize_virtual_rows(virtual_rows)
    passed = (
        exception is None
        and all(mode in mode_summaries for mode in [m["name"] for m in cfg["modes"]])
        and mode_summaries.get("body_only_player", {}).get("sequence_count") == int(cfg["expected_clip_count"])
        and body_parity["passed"]
    )
    summary = {
        "passed": passed,
        "status": "passed_stage1h_player_full_integration" if passed else "failed_or_incomplete_stage1h_player_full_integration",
        "exception": exception,
        "strict_scope_label": cfg["strict_scope_label"],
        "full_heldout_evaluation": True,
        "heldout_groups": cfg["heldout_groups"],
        "expected_clip_count": cfg["expected_clip_count"],
        "modes": [m["name"] for m in cfg["modes"]],
        "body_checkpoint": cfg["body_checkpoint"],
        "body_obs_dim": int(cfg["body_obs_dim"]),
        "body_action_dim": int(cfg["body_action_dim"]),
        "racket_head_input_dim": int(cfg["racket_head_input_dim"]),
        "racket_head_output_dim": int(cfg["racket_head_output_dim"]),
        "combined_action_dim": int(cfg["combined_action_dim"]),
        "original_player_route_executed": True,
        "manual_actor_route_used": False,
        "augmented_obs_to_body_actor": False,
        "reward_enabled": False,
        "training": False,
        "optimizer": False,
        "backward": False,
        "ppo_rl": False,
        "reward_update": False,
        "physical_racket_or_shuttle": False,
        "mode_summaries": mode_summaries,
        "body_parity": body_parity,
        "virtual_metrics": virtual_summary,
        "outputs": mode_outputs,
    }
    write_json(out_dir / cfg["summary_json"], summary)
    write_csv(out_dir / cfg["per_sequence_csv"], per_sequence_rows)
    write_csv(out_dir / cfg["virtual_metrics_csv"], virtual_rows)
    write_csv(out_dir / cfg["results_csv"], mode_result_rows(summary))
    write_report(out_dir / cfg["report_md"], summary)
    return summary


def compute_body_parity(cfg: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    tol = float(cfg.get("body_parity_tolerance_m", 1e-5))
    deltas = []
    completed_mismatches = []
    for row in rows:
        if not row["mode"].startswith("virtual_"):
            continue
        if row.get("mpjpe_delta_vs_body_only") is not None:
            deltas.append(abs(float(row["mpjpe_delta_vs_body_only"])))
        if row.get("root_delta_vs_body_only") is not None:
            deltas.append(abs(float(row["root_delta_vs_body_only"])))
        body_completed = row.get("body_only_completed")
        if body_completed is not None and row.get("completed") != body_completed:
            completed_mismatches.append(row["sequence"])
    max_delta = max(deltas) if deltas else None
    return {
        "passed": max_delta is not None and max_delta <= tol and not completed_mismatches,
        "tolerance_m": tol,
        "max_metric_delta_vs_body_only_m": max_delta,
        "completion_mismatches": completed_mismatches,
    }


def summarize_virtual_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(row["mode"], []).append(row)
    return {
        mode: {
            "frame_rows": len(items),
            "handle_error_m": stats([r.get("handle_error_m") for r in items]),
            "tip_error_m": stats([r.get("tip_error_m") for r in items]),
            "axis_error_deg": stats([r.get("axis_error_deg") for r in items]),
            "action_magnitude": stats([r.get("action_magnitude") for r in items]),
            "action_smoothness": stats([r.get("action_smoothness") for r in items]),
        }
        for mode, items in sorted(by_mode.items())
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mode_result_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for mode, mode_summary in summary["mode_summaries"].items():
        virtual = summary["virtual_metrics"].get(mode, {})
        rows.append(
            {
                "mode": mode,
                "sequence_count": mode_summary.get("sequence_count"),
                "completed_count": mode_summary.get("completed_count"),
                "mean_mpjpe": (mode_summary.get("summary") or {}).get("dataset_mean_mpjpe"),
                "mean_root_error": (mode_summary.get("summary") or {}).get("dataset_mean_root_error"),
                "tip_error_mean_m": (virtual.get("tip_error_m") or {}).get("mean"),
                "tip_error_p90_m": (virtual.get("tip_error_m") or {}).get("p90"),
                "axis_error_mean_deg": (virtual.get("axis_error_deg") or {}).get("mean"),
                "axis_error_p90_deg": (virtual.get("axis_error_deg") or {}).get("p90"),
            }
        )
    return rows


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 1H Player-Compatible Full Held-Out Integration",
        "",
        f"Scope: {summary['strict_scope_label']}.",
        "",
        "## Status",
        "",
        f"- passed: `{summary['passed']}`",
        f"- status: `{summary['status']}`",
        f"- exception: `{summary['exception']}`",
        f"- full held-out evaluation: `{summary['full_heldout_evaluation']}`",
        f"- held-out groups: `{summary['heldout_groups']}`",
        f"- expected clips: `{summary['expected_clip_count']}`",
        "",
        "## Route",
        "",
        f"- original player route executed: `{summary['original_player_route_executed']}`",
        f"- manual actor route used: `{summary['manual_actor_route_used']}`",
        f"- augmented obs to body actor: `{summary['augmented_obs_to_body_actor']}`",
        f"- body obs/action: `{summary['body_obs_dim']} / {summary['body_action_dim']}`",
        f"- racket head input/output: `{summary['racket_head_input_dim']} / {summary['racket_head_output_dim']}`",
        f"- combined action dim: `{summary['combined_action_dim']}`",
        "",
        "## Body Parity",
        "",
        f"- passed: `{summary['body_parity']['passed']}`",
        f"- max compact-metric delta vs body_only: `{summary['body_parity']['max_metric_delta_vs_body_only_m']}`",
        f"- tolerance m: `{summary['body_parity']['tolerance_m']}`",
        "",
        "## Modes",
        "",
    ]
    for row in mode_result_rows(summary):
        lines.append(
            f"- `{row['mode']}`: clips `{row['sequence_count']}`, completed `{row['completed_count']}`, "
            f"MPJPE/root `{row['mean_mpjpe']}` / `{row['mean_root_error']}`, "
            f"tip mean `{row['tip_error_mean_m']}`, axis mean `{row['axis_error_mean_deg']}`"
        )
    lines.extend(
        [
            "",
            "## Safety",
            "",
            f"- reward enabled: `{summary['reward_enabled']}`",
            f"- training / optimizer / backward / PPO-RL: `{summary['training']} / {summary['optimizer']} / {summary['backward']} / {summary['ppo_rl']}`",
            f"- reward update: `{summary['reward_update']}`",
            f"- physical racket or shuttle: `{summary['physical_racket_or_shuttle']}`",
            "",
            "These are no-physics virtual-racket tracking metrics under the original PHC player body route. They are not physical racket accuracy and not official PHC rollout racket accuracy.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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

    exception = None
    mode_outputs: list[dict[str, Any]] = []
    if not args.summarize_only:
        try:
            sequences = select_heldout_sequences(cfg)
            build_motion_subset(cfg, sequences)
            for mode in cfg["modes"]:
                mode_outputs.append(run_mode(cfg, mode))
        except Exception as exc:
            exception = repr(exc)
            print(f"stage1h full integration failed: {exception}", file=sys.stderr)
            raise
        finally:
            summarize_outputs(cfg, mode_outputs, exception=exception)
    else:
        for mode in cfg["modes"]:
            mode_outputs.append(
                {
                    "mode": mode["name"],
                    "compact_metrics_json": f"{cfg['output_dir']}/stage1h_{mode['name']}_compact_metrics.json",
                    "proxy_records_json": f"{cfg['output_dir']}/stage1h_{mode['name']}_proxy_records.json" if mode.get("proxy") else None,
                    "run_log": f"{cfg['output_dir']}/stage1h_{mode['name']}.log",
                }
            )
        summarize_outputs(cfg, mode_outputs, exception=None)

    summary = load_json(out_dir / cfg["summary_json"])
    print(json.dumps({k: summary[k] for k in ["passed", "status", "combined_action_dim"]}, indent=2))


if __name__ == "__main__":
    main()
