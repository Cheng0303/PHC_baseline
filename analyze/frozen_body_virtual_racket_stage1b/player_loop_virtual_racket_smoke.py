#!/usr/bin/env python3
"""Stage 1G-2 user-run original-player-loop virtual-racket smoke.

This runner is intentionally narrow. It invokes the original PHC run_hydra /
RL Games player path with an opt-in env proxy that exposes body-only [934]/[3]
IO to the restored player, then appends the Stage 1A separate racket-head action
[6] at the virtual-racket env boundary. It does not train, tune rewards, or run
the full held-out evaluation.
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
DEFAULT_CONFIG = WORKSPACE_ROOT / "phc_baseline" / "configs" / "frozen_body_virtual_racket_stage1b" / "player_loop_virtual_racket_smoke_config.json"


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_static_config(cfg: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "run_mode_player_loop_smoke": cfg.get("run_mode") == "player_loop_smoke",
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
        "smoke_clip_count_1_to_3": 1 <= len(cfg.get("smoke_sequences", [])) <= 3,
        "non_contiguous_present": all(seq in cfg.get("smoke_sequences", []) for seq in cfg.get("expected_non_contiguous_smoke_sequences", [])),
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


def build_motion_subset(cfg: dict[str, Any]) -> None:
    python = str(resolve(cfg["python"]))
    cmd = [
        python,
        str(resolve("phc_baseline/select_phc_motion_subset.py")),
        "--input",
        str(resolve(cfg["motion_source"])),
        "--output",
        str(resolve(cfg["smoke_motion_output"])),
        "--keys",
        *cfg["smoke_sequences"],
    ]
    run_checked(cmd)


def run_static_hook_smoke(cfg: dict[str, Any]) -> None:
    python = str(resolve(cfg["python"]))
    cmd = [
        python,
        str(resolve("phc_baseline/analyze/frozen_body_virtual_racket_stage1b/player_compatible_virtual_racket_integration.py")),
        "--config",
        str(resolve(cfg["hook_config"])),
        "--run-mode",
        "smoke",
    ]
    run_checked(cmd)


def run_player_loop_smoke(cfg: dict[str, Any]) -> None:
    root = WORKSPACE_ROOT
    phc_root = resolve(cfg["phc_root"])
    python = str(resolve(cfg["python"]))
    out_dir = resolve(cfg["output_dir"])
    compact_json = out_dir / cfg["compact_metrics_json"]
    proxy_json = out_dir / cfg["proxy_records_json"]
    log_path = out_dir / cfg["run_log"]
    for path in (compact_json, proxy_json):
        if path.exists():
            path.unlink()
    env = os.environ.copy()
    env.update(
        {
            "PHC_STAGE1G_PLAYER_VIRTUAL_RACKET_PROXY": "1",
            "PHC_STAGE1G_REPO_ROOT": str(root),
            "PHC_STAGE1G_HOOK_CONFIG": str(resolve(cfg["hook_config"])),
            "PHC_STAGE1G_BODY_OBS_DIM": str(cfg["body_obs_dim"]),
            "PHC_STAGE1G_PROXY_SMOKE_PATH": str(proxy_json),
            "PHC_COMPACT_METRICS_PATH": str(compact_json),
        }
    )
    env_root_lib = resolve("phc_baseline/envs/phc_isaac/lib")
    env_root_bin = resolve("phc_baseline/envs/phc_isaac/bin")
    env["LD_LIBRARY_PATH"] = f"{env_root_lib}:{env.get('LD_LIBRARY_PATH', '')}"
    env["PATH"] = f"{env_root_bin}:{env.get('PATH', '')}"
    env["TORCH_EXTENSIONS_DIR"] = str(resolve("phc_baseline/torch_extensions"))
    cmd = [
        python,
        "phc/run_hydra.py",
        *cfg["phc_run_hydra_args"],
        f"env.motion_file={resolve(cfg['smoke_motion_output'])}",
        cfg["primitive_actor_models_arg"],
    ]
    run_checked(cmd, cwd=phc_root, env=env, log_path=log_path)


def summarize(cfg: dict[str, Any], *, attempted: bool, exception: str | None = None) -> dict[str, Any]:
    out_dir = resolve(cfg["output_dir"])
    compact_path = out_dir / cfg["compact_metrics_json"]
    proxy_path = out_dir / cfg["proxy_records_json"]
    compact = load_json(compact_path) if compact_path.exists() else {}
    proxy = load_json(proxy_path) if proxy_path.exists() else {}
    records = proxy.get("records", [])
    per_sequence = compact.get("per_sequence", [])
    preserve = [float(r.get("body_action_preservation_max_abs_diff", 0.0)) for r in records]
    racket_preserve = [float(r.get("racket_action_preservation_max_abs_diff", 0.0)) for r in records]
    finite_checks = []
    for r in records:
        finite_checks.extend(
            [
                bool(r.get("body_action_finite", False)),
                bool(r.get("head_input_finite", False)),
                bool(r.get("racket_action_finite", False)),
                bool(r.get("combined_action_finite", False)),
            ]
        )
    completed = sum(1 for row in per_sequence if row.get("completed"))
    static = validate_static_config(cfg)
    passed = (
        attempted
        and exception is None
        and static["passed"]
        and bool(records)
        and bool(per_sequence)
        and max(preserve or [1.0]) == 0.0
        and max(racket_preserve or [1.0]) == 0.0
        and all(finite_checks)
    )
    return {
        "passed": passed,
        "status": "passed_stage1g2_player_loop_smoke" if passed else "failed_or_incomplete_stage1g2_player_loop_smoke",
        "exception": exception,
        "strict_scope_label": cfg["strict_scope_label"],
        "attempted": attempted,
        "full_heldout_evaluation": False,
        "smoke_sequences": cfg["smoke_sequences"],
        "expected_non_contiguous_smoke_sequences": cfg["expected_non_contiguous_smoke_sequences"],
        "non_contiguous_clip_included": all(seq in cfg["smoke_sequences"] for seq in cfg["expected_non_contiguous_smoke_sequences"]),
        "original_player_route_executed": bool(proxy.get("original_player_route_executed", False)),
        "original_player_get_action_executed_inferred": bool(proxy.get("original_player_get_action_executed_inferred", False)),
        "original_player_method": "RL Games player get_action() inferred from run_hydra player loop env.step(action)",
        "manual_actor_route_used": bool(proxy.get("manual_actor_route_used", False)),
        "augmented_obs_to_body_actor": bool(proxy.get("augmented_obs_to_body_actor", False)),
        "body_obs_dim": int(cfg["body_obs_dim"]),
        "body_action_dim": int(cfg["body_action_dim"]),
        "racket_head_input_dim": int(cfg["racket_head_input_dim"]),
        "racket_head_output_dim": int(cfg["racket_head_output_dim"]),
        "combined_action_dim": int(cfg["combined_action_dim"]),
        "body_action_preservation_max_abs_diff": max(preserve) if preserve else None,
        "racket_action_preservation_max_abs_diff": max(racket_preserve) if racket_preserve else None,
        "virtual_steps": int(proxy.get("virtual_steps", 0)),
        "env_step_with_combined_action_executed": bool(records),
        "body_physics_step_executed": bool(records),
        "virtual_state_update_executed": all(bool(r.get("virtual_state_update_executed", False)) for r in records) if records else False,
        "reward_enabled": any(bool(r.get("reward_enabled", False)) for r in records),
        "training": False,
        "optimizer": False,
        "backward": False,
        "ppo_rl": False,
        "reward_update": False,
        "physical_racket_or_shuttle": False,
        "compact_metrics": {
            "sequence_count": len(per_sequence),
            "completed_count": completed,
            "summary": compact.get("summary", {}),
        },
        "static_config_validation": static,
        "outputs": {
            "compact_metrics_json": str(compact_path.relative_to(WORKSPACE_ROOT)),
            "proxy_records_json": str(proxy_path.relative_to(WORKSPACE_ROOT)),
            "run_log": str((out_dir / cfg["run_log"]).relative_to(WORKSPACE_ROOT)),
        },
        "next_gate": "full player-compatible held-out evaluation only if this smoke passes; otherwise fix the failing smoke check",
    }


def write_results_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    compact_path = resolve(summary["outputs"]["compact_metrics_json"])
    if compact_path.exists():
        rows = load_json(compact_path).get("per_sequence", [])
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sequence", "completed", "mean_mpjpe", "mean_root_error", "frames"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sequence": row.get("sequence", row.get("sequence_name", "")),
                    "completed": row.get("completed", ""),
                    "mean_mpjpe": row.get("mean_mpjpe", ""),
                    "mean_root_error": row.get("mean_root_error", ""),
                    "frames": row.get("frames", row.get("frame_count", row.get("count", ""))),
                }
            )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 1G-2 Player-Loop Virtual-Racket Smoke",
        "",
        f"Scope: {summary['strict_scope_label']}.",
        "",
        "## Status",
        "",
        f"- passed: `{summary['passed']}`",
        f"- status: `{summary['status']}`",
        f"- attempted: `{summary['attempted']}`",
        f"- exception: `{summary['exception']}`",
        f"- full held-out evaluation: `{summary['full_heldout_evaluation']}`",
        "",
        "## Route Checks",
        "",
        f"- original player route executed: `{summary['original_player_route_executed']}`",
        f"- original player `get_action()` executed: `{summary['original_player_get_action_executed_inferred']}`",
        f"- method/source: {summary['original_player_method']}",
        f"- manual actor route used: `{summary['manual_actor_route_used']}`",
        f"- augmented obs to body actor: `{summary['augmented_obs_to_body_actor']}`",
        "",
        "## Dimensions",
        "",
        f"- body obs/action: `{summary['body_obs_dim']} / {summary['body_action_dim']}`",
        f"- racket head input/output: `{summary['racket_head_input_dim']} / {summary['racket_head_output_dim']}`",
        f"- combined action dim: `{summary['combined_action_dim']}`",
        f"- body action preservation max abs diff: `{summary['body_action_preservation_max_abs_diff']}`",
        f"- racket action preservation max abs diff: `{summary['racket_action_preservation_max_abs_diff']}`",
        "",
        "## Execution",
        "",
        f"- smoke sequences: `{summary['smoke_sequences']}`",
        f"- non-contiguous clip included: `{summary['non_contiguous_clip_included']}`",
        f"- env step with combined action executed: `{summary['env_step_with_combined_action_executed']}`",
        f"- body physics step executed: `{summary['body_physics_step_executed']}`",
        f"- virtual state update executed: `{summary['virtual_state_update_executed']}`",
        f"- virtual steps: `{summary['virtual_steps']}`",
        f"- compact sequence count: `{summary['compact_metrics']['sequence_count']}`",
        f"- compact completed count: `{summary['compact_metrics']['completed_count']}`",
        "",
        "## Safety",
        "",
        f"- reward enabled: `{summary['reward_enabled']}`",
        f"- training / optimizer / backward / PPO-RL: `{summary['training']} / {summary['optimizer']} / {summary['backward']} / {summary['ppo_rl']}`",
        f"- reward update: `{summary['reward_update']}`",
        f"- physical racket or shuttle: `{summary['physical_racket_or_shuttle']}`",
        "",
        "## Next Gate",
        "",
        summary["next_gate"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    cfg = load_json(resolve(args.config))
    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        static = validate_static_config(cfg)
        print(json.dumps(static, indent=2))
        if not static["passed"]:
            raise SystemExit(1)
        return

    exception = None
    attempted = False
    if not args.summarize_only:
        try:
            static = validate_static_config(cfg)
            if not static["passed"]:
                raise RuntimeError(f"static config validation failed: {static}")
            run_static_hook_smoke(cfg)
            build_motion_subset(cfg)
            attempted = True
            run_player_loop_smoke(cfg)
        except Exception as exc:  # keep report artifacts useful after a failed user-run smoke
            exception = repr(exc)
            print(f"stage1g2 smoke failed: {exception}", file=sys.stderr)
            raise
        finally:
            summary = summarize(cfg, attempted=attempted, exception=exception)
            write_json(out_dir / cfg["summary_json"], summary)
            write_report(out_dir / cfg["report_md"], summary)
            write_results_csv(out_dir / cfg["results_csv"], summary)
    else:
        summary = summarize(cfg, attempted=True)
        write_json(out_dir / cfg["summary_json"], summary)
        write_report(out_dir / cfg["report_md"], summary)
        write_results_csv(out_dir / cfg["results_csv"], summary)

    print(json.dumps({k: summary[k] for k in ["passed", "status", "original_player_route_executed", "combined_action_dim"]}, indent=2))


if __name__ == "__main__":
    main()
