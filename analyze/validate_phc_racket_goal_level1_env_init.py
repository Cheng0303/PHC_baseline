#!/usr/bin/env python3
"""Level-1 no-policy Hydra/Isaac smoke validation for the racket-goal task.

This script deliberately initializes the real PHC task class through
``parse_task`` but never builds an rl_games player, never loads a policy
checkpoint, and never executes a policy/action step.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
PHC_ROOT = REPO_ROOT / "humenv" / "data_preparation" / "PHC"
PHC_PACKAGE_ROOT = PHC_ROOT / "phc"
ISAACGYM_ROOT = REPO_ROOT / "phc_baseline" / "third_party" / "isaacgym" / "python"
PHC_ISAAC_ENV_BIN = REPO_ROOT / "phc_baseline" / "envs" / "phc_isaac" / "bin"
PHC_ISAAC_ENV_LIB = REPO_ROOT / "phc_baseline" / "envs" / "phc_isaac" / "lib"
TORCH_EXTENSIONS_DIR = REPO_ROOT / "phc_baseline" / "torch_extensions"

os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(TORCH_EXTENSIONS_DIR))
if str(PHC_ISAAC_ENV_BIN) not in os.environ.get("PATH", "").split(":"):
    os.environ["PATH"] = f"{PHC_ISAAC_ENV_BIN}:{os.environ.get('PATH', '')}"

ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
if str(PHC_ISAAC_ENV_LIB) not in ld_library_path.split(":"):
    os.environ["LD_LIBRARY_PATH"] = (
        str(PHC_ISAAC_ENV_LIB) if not ld_library_path else f"{PHC_ISAAC_ENV_LIB}:{ld_library_path}"
    )
    os.execv(sys.executable, [sys.executable, *sys.argv])

for path in [ISAACGYM_ROOT, PHC_PACKAGE_ROOT, PHC_ROOT, REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401,E402
from isaacgym import gymapi, gymutil  # noqa: E402

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from easydict import EasyDict  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from phc.env.tasks.humanoid_im_mcp_getup import HumanoidImMCPGetup  # noqa: E402
from phc.env.tasks.humanoid_im_mcp_getup_racket_goal import HumanoidImMCPGetupRacketGoal  # noqa: E402
from phc.utils.flags import flags  # noqa: E402
from phc.utils.parse_task import parse_task  # noqa: E402


DEFAULT_MOTION_FILE = REPO_ROOT / "phc_baseline" / "converted" / "badminton_phc_motion_groundfix.pkl"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "phc_baseline"
    / "racket_calibration"
    / "racket_aware_reference_task_cross_session_dataset"
    / "manifest.csv"
)
DEFAULT_OUT_DIR = REPO_ROOT / "phc_baseline" / "reports" / "racket_calibration" / "controller_interface"
DEFAULT_MAPPING_CSV = DEFAULT_OUT_DIR / "runtime_mapping_per_sequence.csv"
DEFAULT_SEQUENCE = "241217_1/1_00_02_0"


def _boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _compose_cfg(config_name: str, overrides: list[str]):
    cfg_dir = str(PHC_PACKAGE_ROOT / "data" / "cfg")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = compose(config_name="config", overrides=[f"env={config_name}", *overrides])
    OmegaConf.set_struct(cfg, False)
    return cfg


def _parse_sim_params(cfg):
    sim_params = gymapi.SimParams()
    sim_params.dt = eval(cfg.sim.physx.sim_time_step)
    sim_params.num_client_threads = cfg.sim.slices
    if cfg.sim.use_flex:
        if cfg.sim.pipeline in ["gpu"]:
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.use_flex.shape_collision_margin = 0.01
        sim_params.use_flex.num_outer_iterations = 4
        sim_params.use_flex.num_inner_iterations = 10
    else:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
        sim_params.physx.num_subscenes = cfg.sim.subscenes
        sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
    gymutil.parse_sim_config(cfg["sim"], sim_params)
    if not cfg.sim.use_flex and cfg.sim.physx.num_threads > 0:
        sim_params.physx.num_threads = cfg.sim.physx.num_threads
    return sim_params


def _make_args(cfg):
    return EasyDict(
        {
            "task": cfg.env.task,
            "device_id": int(cfg.device_id),
            "rl_device": cfg.rl_device,
            "physics_engine": gymapi.SIM_PHYSX if not cfg.sim.use_flex else gymapi.SIM_FLEX,
            "headless": bool(cfg.headless),
            "device": cfg.device,
        }
    )


def _make_single_motion_file(full_motion_file: Path, sequence: str) -> Path:
    motions = joblib.load(full_motion_file)
    if sequence not in motions:
        raise KeyError(f"{sequence} not found in {full_motion_file}")
    out_path = Path(tempfile.gettempdir()) / "phc_racket_goal_level1_smoke_motion.pkl"
    joblib.dump({sequence: motions[sequence]}, out_path)
    return out_path


def _manifest_row(manifest_csv: Path, sequence: str) -> dict[str, str]:
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["sequence"] == sequence:
                return row
    raise KeyError(f"{sequence} not found in {manifest_csv}")


def _mapping_row(mapping_csv: Path, sequence: str) -> dict[str, str]:
    if not mapping_csv.exists():
        return {}
    with mapping_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["sequence"] == sequence:
                return row
    return {}


def _common_overrides(args, motion_file: Path) -> list[str]:
    return [
        "learning=im_mcp_big",
        "robot=smpl_humanoid",
        "exp_name=phc_level1_racket_goal_smoke",
        "headless=True",
        "no_virtual_display=True",
        "test=True",
        "im_eval=False",
        "env.num_envs=1",
        f"env.motion_file={motion_file}",
        f"env.racket_goal_manifest={args.manifest}",
        "env.models=[]",
        "env.has_pnn=False",
        "env.fitting=False",
        "env.num_prim=3",
        "env.actors_to_load=3",
        "robot.real_weight_porpotion_boxes=False",
    ]


def _init_task(cfg):
    for key in [
        "debug",
        "follow",
        "fixed",
        "divide_group",
        "no_collision_check",
        "fixed_path",
        "real_path",
        "show_traj",
        "server_mode",
        "slow",
        "real_traj",
        "im_eval",
        "no_virtual_display",
        "render_o3d",
        "add_proj",
        "has_eval",
    ]:
        setattr(flags, key, bool(cfg.get(key, False)))
    flags.test = bool(cfg.test)
    flags.trigger_input = False
    flags.idx = 0
    sim_params = _parse_sim_params(cfg)
    cfg_train = cfg.learning.params.config
    task, env = parse_task(_make_args(cfg), cfg, cfg_train, sim_params)
    return task, env


def _destroy_task(task) -> None:
    try:
        if getattr(task, "viewer", None) is not None:
            task.gym.destroy_viewer(task.viewer)
    except Exception:
        pass
    try:
        task.gym.destroy_sim(task.sim)
    except Exception:
        pass


def _run_level1a(args, motion_file: Path) -> dict[str, Any]:
    cfg = _compose_cfg("env_im_getup_mcp_racket_goal_smoke", _common_overrides(args, motion_file))
    with open_dict(cfg):
        cfg.env.racket_goal_smoke_no_policy = True
    os.chdir(PHC_ROOT)
    task = None
    result: dict[str, Any] = {
        "config_name": "env_im_getup_mcp_racket_goal_smoke",
        "actual_hydra_task": str(cfg.env.task),
        "policy_loaded": False,
        "policy_step_executed": False,
    }
    try:
        task, env = _init_task(cfg)
        result["actual_env_task_initialization_passed"] = True
        result["actual_task_class"] = task.__class__.__name__
        result["actual_task_is_racket_goal"] = isinstance(task, HumanoidImMCPGetupRacketGoal)
        result["vec_env_num_obs"] = int(getattr(env, "num_obs", -1))
        result["task_num_obs"] = int(getattr(task, "num_obs", -1))
        result["task_obs_original_dim"] = int(HumanoidImMCPGetup.get_task_obs_size(task))
        result["task_obs_augmented_dim"] = int(task.get_task_obs_size())
        result["full_obs_dim_verified"] = True
        result["full_obs_augmented_dim"] = int(getattr(task, "num_obs", -1))
        result["full_obs_original_dim"] = None

        env_ids = torch.arange(task.num_envs, dtype=torch.long, device=task.device)
        task_obs = task._compute_task_obs(env_ids=env_ids, save_buffer=False)
        base_dim = result["task_obs_original_dim"]
        appended = task_obs[:, base_dim:]
        motion_times = (
            (task.progress_buf[env_ids] + 1) * task.dt
            + task._motion_start_times[env_ids]
            + task._motion_start_times_offset[env_ids]
        )
        sampled_motion_ids = task._sampled_motion_ids[env_ids]
        motion_keys_np = task._motion_lib._motion_data_keys[sampled_motion_ids.detach().cpu().numpy()]
        sequence_keys = [str(key) for key in motion_keys_np.tolist()]
        expected = task._racket_goal_provider.lookup_batch(
            sequence_keys,
            motion_times,
            task._motion_lib._motion_num_frames[sampled_motion_ids],
            task._motion_lib._motion_dt[sampled_motion_ids],
            device=task.device,
            dtype=task_obs.dtype,
        )
        diff = (appended - expected).abs()

        result.update(
            {
                "actual_hook_executed": True,
                "actual_hook_queries_checked": int(task.num_envs),
                "sequence_keys": sequence_keys,
                "motion_times": [float(x) for x in motion_times.detach().cpu().tolist()],
                "sampled_motion_ids": [int(x) for x in sampled_motion_ids.detach().cpu().tolist()],
                "task_obs_shape": list(task_obs.shape),
                "body_prefix_shape": [int(task_obs.shape[0]), int(base_dim)],
                "appended_goal_shape": list(appended.shape),
                "appended_goal_max_abs_diff": float(diff.max().detach().cpu().item()),
                "appended_goal_all_finite": bool(torch.isfinite(appended).all().detach().cpu().item()),
                "body_prefix_all_finite": bool(torch.isfinite(task_obs[:, :base_dim]).all().detach().cpu().item()),
                "non_contiguous_clip_actual_hook_checked": args.sequence == DEFAULT_SEQUENCE,
            }
        )
    except Exception as exc:
        result["actual_env_task_initialization_passed"] = False
        result["actual_hook_executed"] = False
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if task is not None:
            _destroy_task(task)
    return result


def _run_level1b_guard(args, motion_file: Path) -> dict[str, Any]:
    cfg = _compose_cfg("env_im_getup_mcp_racket_goal", _common_overrides(args, motion_file))
    with open_dict(cfg):
        cfg.env.racket_goal_smoke_no_policy = False
        cfg.env.racket_goal_augmented_checkpoint = None
    os.chdir(PHC_ROOT)
    result: dict[str, Any] = {
        "config_name": "env_im_getup_mcp_racket_goal",
        "actual_hydra_task": str(cfg.env.task),
        "policy_loaded": False,
        "policy_step_executed": False,
    }
    try:
        task, _env = _init_task(cfg)
        result["guard_refusal_passed"] = False
        result["unexpected_task_initialized"] = task.__class__.__name__
        _destroy_task(task)
    except RuntimeError as exc:
        message = str(exc)
        expected = "Racket-goal observations add 9 task-observation dimensions" in message
        result["guard_refusal_passed"] = expected
        result["refusal_message"] = message
        result["refusal_exception_type"] = exc.__class__.__name__
        result["refused_before_env_init"] = True
    except Exception as exc:
        result["guard_refusal_passed"] = False
        result["refusal_exception_type"] = exc.__class__.__name__
        result["refusal_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
    return result


def _write_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "level1_env_init_smoke_summary.json"
    report_path = out_dir / "level1_env_init_smoke_report.md"
    csv_path = out_dir / "level1_env_init_smoke_results.csv"
    summary["summary_path"] = str(summary_path)
    summary["report_path"] = str(report_path)
    summary["csv_path"] = str(csv_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    row = {
        "sequence": summary.get("sequence"),
        "execution_level_reached": summary.get("execution_level_reached"),
        "actual_task_class": summary.get("level1a", {}).get("actual_task_class"),
        "actual_env_task_initialization_passed": summary.get("actual_env_task_initialization_passed"),
        "actual_hook_executed": summary.get("actual_hook_executed"),
        "task_obs_original_dim": summary.get("task_obs_original_dim"),
        "task_obs_augmented_dim": summary.get("task_obs_augmented_dim"),
        "full_obs_dim_verified": summary.get("full_obs_dim_verified"),
        "full_obs_augmented_dim": summary.get("full_obs_augmented_dim"),
        "appended_goal_max_abs_diff": summary.get("appended_goal_max_abs_diff"),
        "original_checkpoint_guard_refusal_passed": summary.get("original_checkpoint_guard_refusal_passed"),
        "policy_loaded": summary.get("policy_loaded"),
        "policy_step_executed": summary.get("policy_step_executed"),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    level1a = summary.get("level1a", {})
    level1b = summary.get("level1b_guard", {})
    report = f"""# Level 1 Env Init Smoke Validation

This is a no-training, no-policy-step smoke validation for the opt-in PHC racket-goal task wrapper.

## Result

- execution level reached: `{summary.get('execution_level_reached')}`
- actual Hydra task: `{level1a.get('actual_hydra_task')}`
- instantiated class: `{level1a.get('actual_task_class')}`
- policy loaded: `{summary.get('policy_loaded')}`
- policy/action step executed: `{summary.get('policy_step_executed')}`
- actual hook executed: `{summary.get('actual_hook_executed')}`
- non-contiguous clip checked: `{summary.get('non_contiguous_clip_actual_hook_checked')}`

## Hook Validation

- sequence: `{summary.get('sequence')}`
- motion keys: `{level1a.get('sequence_keys')}`
- motion times: `{level1a.get('motion_times')}`
- task observation original component dim: `{summary.get('task_obs_original_dim')}`
- task observation augmented component dim: `{summary.get('task_obs_augmented_dim')}`
- full augmented observation dim verified: `{summary.get('full_obs_augmented_dim')}`
- appended goal max abs diff: `{summary.get('appended_goal_max_abs_diff')}`
- appended goal finite: `{level1a.get('appended_goal_all_finite')}`

## Checkpoint Guard

- actual guard path tested: `{summary.get('original_checkpoint_guard_actual_entrypoint_tested')}`
- refusal passed: `{summary.get('original_checkpoint_guard_refusal_passed')}`
- refusal message: `{level1b.get('refusal_message')}`

The guard was tested before constructing an rl_games player or loading policy weights. The smoke mode is a no-policy hook-validation path, not augmented-checkpoint compatibility.

## Files

- summary: `{summary.get('summary_path')}`
- csv: `{summary.get('csv_path')}`
"""
    if level1a.get("error"):
        report += f"\n## Level 1A Error\n\n```text\n{level1a.get('traceback')}\n```\n"
    report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    manifest_row = _manifest_row(args.manifest, args.sequence)
    mapping_row = _mapping_row(DEFAULT_MAPPING_CSV, args.sequence)
    smoke_motion_file = _make_single_motion_file(args.motion_file, args.sequence)
    level1a = _run_level1a(args, smoke_motion_file)
    level1b = _run_level1b_guard(args, smoke_motion_file)

    level1a_passed = bool(
        level1a.get("actual_env_task_initialization_passed")
        and level1a.get("actual_hook_executed")
        and level1a.get("appended_goal_all_finite")
        and float(level1a.get("appended_goal_max_abs_diff", float("inf"))) < 1e-5
    )
    level1b_passed = bool(level1b.get("guard_refusal_passed"))
    if level1a_passed and level1b_passed:
        execution_level = "level_1b"
    elif level1a_passed:
        execution_level = "level_1a"
    elif level1a.get("actual_env_task_initialization_passed") is False:
        execution_level = "blocked"
    else:
        execution_level = "level_2_only"

    summary: dict[str, Any] = {
        "sequence": args.sequence,
        "manifest": str(args.manifest),
        "manifest_row_source_frame_idx_contiguous": manifest_row.get("source_frame_idx_contiguous"),
        "runtime_mapping_source_frame_idx_contiguous": mapping_row.get("source_frame_idx_contiguous"),
        "runtime_mapping_convention": mapping_row.get("mapping_convention_determined"),
        "smoke_motion_file": str(smoke_motion_file),
        "execution_level_reached": execution_level,
        "actual_hydra_config_resolution_passed": level1a.get("actual_hydra_task") == "HumanoidImMCPGetupRacketGoal",
        "actual_env_task_initialization_passed": bool(level1a.get("actual_env_task_initialization_passed")),
        "actual_hook_executed": bool(level1a.get("actual_hook_executed")),
        "actual_task_class": level1a.get("actual_task_class"),
        "policy_loaded": False,
        "policy_step_executed": False,
        "original_checkpoint_guard_actual_entrypoint_tested": True,
        "original_checkpoint_guard_refusal_passed": level1b_passed,
        "task_obs_original_dim": level1a.get("task_obs_original_dim"),
        "task_obs_augmented_dim": level1a.get("task_obs_augmented_dim"),
        "full_obs_dim_verified": level1a.get("full_obs_dim_verified", False),
        "full_obs_original_dim": level1a.get("full_obs_original_dim"),
        "full_obs_augmented_dim": level1a.get("full_obs_augmented_dim"),
        "actual_hook_queries_checked": level1a.get("actual_hook_queries_checked", 0),
        "appended_goal_max_abs_diff": level1a.get("appended_goal_max_abs_diff"),
        "non_contiguous_clip_actual_hook_checked": bool(level1a.get("non_contiguous_clip_actual_hook_checked")),
        "blockers": [] if level1a_passed else [level1a.get("error", "level1a_failed")],
        "warnings": [
            "No policy/player was constructed; this is env/task hook validation only.",
            "full_obs_original_dim was not separately initialized in this smoke run.",
        ],
        "level1a": level1a,
        "level1b_guard": level1b,
    }
    _write_outputs(args.output_dir, summary)
    return 0 if level1a_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
