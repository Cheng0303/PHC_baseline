#!/usr/bin/env python3
"""No-training smoke validation for the live virtual racket env scaffold.

This script initializes the actual opt-in PHC task class through Hydra/Isaac,
executes only scripted virtual racket state/action hooks, and never constructs
an rl_games player, loads a policy checkpoint, or runs learned policy actions.
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
    os.environ["LD_LIBRARY_PATH"] = str(PHC_ISAAC_ENV_LIB) if not ld_library_path else f"{PHC_ISAAC_ENV_LIB}:{ld_library_path}"
    os.execv(sys.executable, [sys.executable, *sys.argv])

for path in [ISAACGYM_ROOT, PHC_PACKAGE_ROOT, PHC_ROOT, REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401,E402
from isaacgym import gymapi, gymutil  # noqa: E402

import joblib  # noqa: E402
import torch  # noqa: E402
from easydict import EasyDict  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from phc.env.tasks.humanoid_im_mcp_getup import HumanoidImMCPGetup  # noqa: E402
from phc.env.tasks.humanoid_im_mcp_getup_racket_goal import HumanoidImMCPGetupRacketGoal  # noqa: E402
from phc.env.tasks.humanoid_im_mcp_getup_virtual_racket import HumanoidImMCPGetupVirtualRacket  # noqa: E402
from phc.env.tasks.virtual_racket_runtime import world_to_heading_local_vector  # noqa: E402
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
DEFAULT_OUT_DIR = REPO_ROOT / "phc_baseline" / "reports" / "racket_calibration" / "virtual_racket_control"
DEFAULT_SEQUENCE = "241217_1/1_00_02_0"


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
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.num_threads = 4
    sim_params.physx.use_gpu = cfg.sim.pipeline in ["gpu"]
    sim_params.physx.num_subscenes = cfg.sim.subscenes
    sim_params.physx.max_gpu_contact_pairs = 4 * 1024 * 1024
    sim_params.use_gpu_pipeline = cfg.sim.pipeline in ["gpu"]
    gymutil.parse_sim_config(cfg["sim"], sim_params)
    return sim_params


def _make_args(cfg):
    return EasyDict(
        {
            "task": cfg.env.task,
            "device_id": int(cfg.device_id),
            "rl_device": cfg.rl_device,
            "physics_engine": gymapi.SIM_PHYSX,
            "headless": bool(cfg.headless),
            "device": cfg.device,
        }
    )


def _make_single_motion_file(full_motion_file: Path, sequence: str) -> Path:
    motions = joblib.load(full_motion_file)
    if sequence not in motions:
        raise KeyError(f"{sequence} not found in {full_motion_file}")
    out_path = Path(tempfile.gettempdir()) / "phc_virtual_racket_smoke_motion.pkl"
    joblib.dump({sequence: motions[sequence]}, out_path)
    return out_path


def _common_overrides(args, motion_file: Path) -> list[str]:
    return [
        "learning=im_mcp_big",
        "robot=smpl_humanoid",
        "exp_name=phc_virtual_racket_scaffold_smoke",
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
        "env.num_prim=4",
        "env.actors_to_load=4",
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


def _lookup_target(task, env_ids: torch.Tensor, next_frame: bool) -> dict[str, torch.Tensor]:
    times = task._current_motion_times(env_ids, next_frame=next_frame)
    return task._lookup_virtual_racket_target(env_ids, times)


def _derive_oracle_action(task, current: dict[str, torch.Tensor], target: dict[str, torch.Tensor], env_ids: torch.Tensor) -> torch.Tensor:
    root_rot = task._rigid_body_rot[env_ids, 0, :]
    v_world = (target["handle"] - current["handle"]) / task.dt
    axis_a = current["axis"]
    axis_b = target["axis"]
    cross = torch.cross(axis_a, axis_b, dim=-1)
    sin = torch.linalg.norm(cross, dim=-1, keepdim=True)
    cos = torch.clamp(torch.sum(axis_a * axis_b, dim=-1, keepdim=True), -1.0, 1.0)
    angle = torch.atan2(sin, cos)
    rot_axis_world = torch.zeros_like(axis_a)
    nonzero = sin[..., 0] > 1e-8
    if bool(nonzero.any().detach().cpu().item()):
        rot_axis_world[nonzero] = cross[nonzero] / sin[nonzero]
    omega_world = rot_axis_world * angle / task.dt
    v_local = world_to_heading_local_vector(v_world, root_rot)
    omega_local = world_to_heading_local_vector(omega_world, root_rot)
    return torch.cat([v_local, omega_local], dim=-1)


def _run_smoke(args, motion_file: Path) -> dict[str, Any]:
    cfg = _compose_cfg("env_im_getup_mcp_virtual_racket_smoke", _common_overrides(args, motion_file))
    with open_dict(cfg):
        cfg.env.virtual_racket_smoke_no_policy = True
        cfg.env.racket_goal_smoke_no_policy = True
    os.chdir(PHC_ROOT)
    task = None
    result: dict[str, Any] = {
        "config_name": "env_im_getup_mcp_virtual_racket_smoke",
        "actual_hydra_task": str(cfg.env.task),
        "policy_loaded": False,
        "policy_step_executed": False,
    }
    try:
        task, env = _init_task(cfg)
        result["actual_env_task_initialization_passed"] = True
        result["actual_task_class"] = task.__class__.__name__
        result["actual_task_is_virtual_racket"] = isinstance(task, HumanoidImMCPGetupVirtualRacket)
        result["task_num_obs"] = int(getattr(task, "num_obs", -1))
        result["vec_env_num_obs"] = int(getattr(env, "num_obs", -1))
        result["body_only_task_obs_dim"] = int(HumanoidImMCPGetup.get_task_obs_size(task))
        result["goal_only_task_obs_dim"] = int(HumanoidImMCPGetupRacketGoal.get_task_obs_size(task))
        result["virtual_racket_task_obs_dim"] = int(task.get_task_obs_size())
        result["full_env_obs_dim"] = int(getattr(task, "num_obs", -1))
        result["original_mcp_action_dim"] = int(task.num_prim)
        result["augmented_action_dim"] = int(task.num_actions)

        env_ids = torch.arange(task.num_envs, dtype=torch.long, device=task.device)
        obs = task._compute_task_obs(env_ids=env_ids, save_buffer=False)
        body_dim = result["body_only_task_obs_dim"]
        goal_dim = result["goal_only_task_obs_dim"] - result["body_only_task_obs_dim"]
        live_goal_obs = obs[:, body_dim : body_dim + goal_dim]
        expected_live_goal = task._compute_live_racket_goal_obs(env_ids)
        realized_state_obs = obs[:, body_dim + goal_dim :]
        expected_realized_state = task._compute_virtual_racket_state_obs(env_ids)
        live_goal_diff = (live_goal_obs - expected_live_goal).abs()
        realized_state_diff = (realized_state_obs - expected_realized_state).abs()
        result["task_obs_shape"] = list(obs.shape)
        result["task_obs_all_finite"] = bool(torch.isfinite(obs).all().detach().cpu().item())

        initial_target = _lookup_target(task, env_ids, next_frame=False)
        reset_handle_err = torch.linalg.norm(task._virtual_racket_handle_world[env_ids] - initial_target["handle"], dim=-1)
        reset_axis_err = torch.linalg.norm(task._virtual_racket_axis_world[env_ids] - initial_target["axis"], dim=-1)

        current = {
            "handle": task._virtual_racket_handle_world[env_ids].clone(),
            "axis": task._virtual_racket_axis_world[env_ids].clone(),
        }
        next_target = _lookup_target(task, env_ids, next_frame=True)
        oracle_action = _derive_oracle_action(task, current, next_target, env_ids)
        task._step_virtual_racket(oracle_action, env_ids=env_ids)
        oracle_handle_err = torch.linalg.norm(task._virtual_racket_handle_world[env_ids] - next_target["handle"], dim=-1)
        oracle_axis_dot = torch.clamp(torch.sum(task._virtual_racket_axis_world[env_ids] * next_target["axis"], dim=-1), -1.0, 1.0)
        oracle_axis_err_deg = torch.rad2deg(torch.arccos(oracle_axis_dot))
        oracle_metrics = task._virtual_racket_metrics

        task._reset_virtual_racket_state(env_ids)
        zero_action = torch.zeros((task.num_envs, 6), device=task.device, dtype=obs.dtype)
        reward_before = task.rew_buf.clone()
        task._step_virtual_racket(zero_action, env_ids=env_ids)
        reward_after = task.rew_buf.clone()
        null_metrics = task._virtual_racket_metrics

        shape_mismatch_guard = False
        try:
            task.split_virtual_racket_action(torch.zeros((task.num_envs, task.num_actions - 1), device=task.device, dtype=obs.dtype))
        except RuntimeError:
            shape_mismatch_guard = True

        result.update(
            {
                "actual_hook_executed": True,
                "sequence_keys": [str(x) for x in task._motion_lib._motion_data_keys[task._sampled_motion_ids.detach().cpu().numpy()].tolist()],
                "motion_times_current": [float(x) for x in task._current_motion_times(env_ids, next_frame=False).detach().cpu().tolist()],
                "motion_times_next": [float(x) for x in task._current_motion_times(env_ids, next_frame=True).detach().cpu().tolist()],
                "live_goal_version": str(task._virtual_racket_live_goal_version),
                "live_goal_obs_shape": list(live_goal_obs.shape),
                "live_goal_projection_max_abs_diff": float(live_goal_diff.max().detach().cpu().item()),
                "realized_state_projection_max_abs_diff": float(realized_state_diff.max().detach().cpu().item()),
                "reset_handle_max_error_m": float(reset_handle_err.max().detach().cpu().item()),
                "reset_axis_l2_max_error": float(reset_axis_err.max().detach().cpu().item()),
                "oracle_action_shape": list(oracle_action.shape),
                "oracle_handle_max_error_m": float(oracle_handle_err.max().detach().cpu().item()),
                "oracle_axis_max_error_deg": float(oracle_axis_err_deg.max().detach().cpu().item()),
                "oracle_metric_tip_max_error_m": float(oracle_metrics["tip_error_m"].max().detach().cpu().item()),
                "null_metric_tip_max_error_m": float(null_metrics["tip_error_m"].max().detach().cpu().item()),
                "null_metric_axis_max_error_deg": float(null_metrics["axis_error_deg"].max().detach().cpu().item()),
                "reward_off_rew_buf_max_diff": float((reward_after - reward_before).abs().max().detach().cpu().item()),
                "shape_mismatch_guard_passed": shape_mismatch_guard,
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


def _run_guard(args, motion_file: Path, override: str | None = None) -> dict[str, Any]:
    cfg = _compose_cfg("env_im_getup_mcp_virtual_racket_smoke", _common_overrides(args, motion_file))
    with open_dict(cfg):
        cfg.env.virtual_racket_smoke_no_policy = False
        cfg.env.racket_goal_smoke_no_policy = False
        cfg.env.virtual_racket_augmented_checkpoint = None
        cfg.env.racket_goal_augmented_checkpoint = None
        if override == "source_pose":
            cfg.env.racket_goal_include_source_racket_pose = True
    os.chdir(PHC_ROOT)
    result: dict[str, Any] = {"override": override, "policy_loaded": False, "policy_step_executed": False}
    try:
        task, _env = _init_task(cfg)
        result["guard_refusal_passed"] = False
        result["unexpected_task_initialized"] = task.__class__.__name__
        _destroy_task(task)
    except Exception as exc:
        message = str(exc)
        if override == "source_pose":
            expected = "racket_pose_parameter is diagnostic-only" in message
        else:
            expected = "Virtual-racket state/action observations and actions are incompatible" in message
        result["guard_refusal_passed"] = expected
        result["refusal_exception_type"] = exc.__class__.__name__
        result["refusal_message"] = message
    return result


def _write_outputs(out_dir: Path, summary: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "env_scaffold_smoke_summary.json"
    csv_path = out_dir / "env_scaffold_smoke_results.csv"
    report_path = out_dir / "env_scaffold_smoke_report.md"
    live_summary_path = out_dir / "live_goal_v2_env_smoke_summary.json"
    live_csv_path = out_dir / "live_goal_v2_env_smoke_results.csv"
    live_report_path = out_dir / "live_goal_v2_env_smoke_report.md"
    summary["summary_path"] = str(summary_path)
    summary["csv_path"] = str(csv_path)
    summary["report_path"] = str(report_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    live_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    row = {
        "sequence": summary.get("sequence"),
        "execution_level_reached": summary.get("execution_level_reached"),
        "actual_task_class": summary.get("actual_task_class"),
        "body_only_task_obs_dim": summary.get("body_only_task_obs_dim"),
        "goal_only_task_obs_dim": summary.get("goal_only_task_obs_dim"),
        "virtual_racket_task_obs_dim": summary.get("virtual_racket_task_obs_dim"),
        "full_env_obs_dim": summary.get("full_env_obs_dim"),
        "original_mcp_action_dim": summary.get("original_mcp_action_dim"),
        "augmented_action_dim": summary.get("augmented_action_dim"),
        "live_goal_projection_max_abs_diff": summary.get("live_goal_projection_max_abs_diff"),
        "realized_state_projection_max_abs_diff": summary.get("realized_state_projection_max_abs_diff"),
        "reset_handle_max_error_m": summary.get("reset_handle_max_error_m"),
        "oracle_handle_max_error_m": summary.get("oracle_handle_max_error_m"),
        "oracle_axis_max_error_deg": summary.get("oracle_axis_max_error_deg"),
        "oracle_metric_tip_max_error_m": summary.get("oracle_metric_tip_max_error_m"),
        "null_metric_tip_max_error_m": summary.get("null_metric_tip_max_error_m"),
        "reward_off_rew_buf_max_diff": summary.get("reward_off_rew_buf_max_diff"),
        "checkpoint_guard_passed": summary.get("checkpoint_guard_passed"),
        "source_pose_guard_passed": summary.get("source_pose_guard_passed"),
        "shape_mismatch_guard_passed": summary.get("shape_mismatch_guard_passed"),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    with live_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    report = f"""# Live Virtual Racket Env Scaffold Smoke

This is a no-training scripted contract smoke. No player, policy checkpoint, learned action, reward tuning, physical racket, collision, shuttle, or PHC rollout racket accuracy is involved.

## Result

- execution level reached: `{summary.get('execution_level_reached')}`
- actual class: `{summary.get('actual_task_class')}`
- policy loaded: `{summary.get('policy_loaded')}`
- policy/action step executed: `{summary.get('policy_step_executed')}`
- sequence: `{summary.get('sequence')}`

## Dimensions

- body-only task obs dim: `{summary.get('body_only_task_obs_dim')}`
- goal-only task obs dim: `{summary.get('goal_only_task_obs_dim')}`
- virtual-racket task obs dim: `{summary.get('virtual_racket_task_obs_dim')}`
- full env obs dim: `{summary.get('full_env_obs_dim')}`
- original MCP action dim: `{summary.get('original_mcp_action_dim')}`
- augmented action dim: `{summary.get('augmented_action_dim')}`
- live goal version: `{summary.get('live_goal_version')}`

## Scripted Hooks

- Live Goal V2 projection max abs diff: `{summary.get('live_goal_projection_max_abs_diff')}`
- realized-state projection max abs diff: `{summary.get('realized_state_projection_max_abs_diff')}`
- reset handle max error: `{summary.get('reset_handle_max_error_m')}` m
- oracle handle max error: `{summary.get('oracle_handle_max_error_m')}` m
- oracle axis max error: `{summary.get('oracle_axis_max_error_deg')}` deg
- oracle metric tip max error: `{summary.get('oracle_metric_tip_max_error_m')}` m
- null metric tip max error: `{summary.get('null_metric_tip_max_error_m')}` m
- reward-off rew buf max diff: `{summary.get('reward_off_rew_buf_max_diff')}`

## Guards

- original checkpoint/augmented policy guard passed: `{summary.get('checkpoint_guard_passed')}`
- source racket pose primary inclusion guard passed: `{summary.get('source_pose_guard_passed')}`
- action shape mismatch guard passed: `{summary.get('shape_mismatch_guard_passed')}`

This validates scaffold semantics only. It does not validate learned policy performance.
"""
    report_path.write_text(report, encoding="utf-8")
    live_report_path.write_text(report, encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    live_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", default=DEFAULT_SEQUENCE)
    parser.add_argument("--motion_file", type=Path, default=DEFAULT_MOTION_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    smoke_motion_file = _make_single_motion_file(args.motion_file, args.sequence)
    smoke = _run_smoke(args, smoke_motion_file)
    checkpoint_guard = _run_guard(args, smoke_motion_file)
    source_pose_guard = _run_guard(args, smoke_motion_file, override="source_pose")

    smoke_passed = bool(
        smoke.get("actual_env_task_initialization_passed")
        and smoke.get("actual_hook_executed")
        and smoke.get("task_obs_all_finite")
        and float(smoke.get("live_goal_projection_max_abs_diff", float("inf"))) < 1e-5
        and float(smoke.get("realized_state_projection_max_abs_diff", float("inf"))) < 1e-5
        and float(smoke.get("reset_handle_max_error_m", float("inf"))) < 1e-5
        and float(smoke.get("oracle_metric_tip_max_error_m", float("inf"))) < 1e-4
        and float(smoke.get("reward_off_rew_buf_max_diff", float("inf"))) == 0.0
    )
    execution_level = "vr_1" if smoke_passed else "blocked"
    summary = {
        "sequence": args.sequence,
        "manifest": str(args.manifest),
        "smoke_motion_file": str(smoke_motion_file),
        "execution_level_reached": execution_level,
        "policy_loaded": False,
        "policy_step_executed": False,
        "actual_task_class": smoke.get("actual_task_class"),
        "actual_hydra_task": smoke.get("actual_hydra_task"),
        "actual_env_task_initialization_passed": smoke.get("actual_env_task_initialization_passed"),
        "actual_hook_executed": smoke.get("actual_hook_executed"),
        "body_only_task_obs_dim": smoke.get("body_only_task_obs_dim"),
        "goal_only_task_obs_dim": smoke.get("goal_only_task_obs_dim"),
        "virtual_racket_task_obs_dim": smoke.get("virtual_racket_task_obs_dim"),
        "full_env_obs_dim": smoke.get("full_env_obs_dim"),
        "original_mcp_action_dim": smoke.get("original_mcp_action_dim"),
        "augmented_action_dim": smoke.get("augmented_action_dim"),
        "reset_handle_max_error_m": smoke.get("reset_handle_max_error_m"),
        "reset_axis_l2_max_error": smoke.get("reset_axis_l2_max_error"),
        "live_goal_version": smoke.get("live_goal_version"),
        "live_goal_projection_max_abs_diff": smoke.get("live_goal_projection_max_abs_diff"),
        "realized_state_projection_max_abs_diff": smoke.get("realized_state_projection_max_abs_diff"),
        "oracle_handle_max_error_m": smoke.get("oracle_handle_max_error_m"),
        "oracle_axis_max_error_deg": smoke.get("oracle_axis_max_error_deg"),
        "oracle_metric_tip_max_error_m": smoke.get("oracle_metric_tip_max_error_m"),
        "null_metric_tip_max_error_m": smoke.get("null_metric_tip_max_error_m"),
        "null_metric_axis_max_error_deg": smoke.get("null_metric_axis_max_error_deg"),
        "reward_off_rew_buf_max_diff": smoke.get("reward_off_rew_buf_max_diff"),
        "shape_mismatch_guard_passed": smoke.get("shape_mismatch_guard_passed"),
        "checkpoint_guard_passed": checkpoint_guard.get("guard_refusal_passed"),
        "source_pose_guard_passed": source_pose_guard.get("guard_refusal_passed"),
        "checkpoint_guard": checkpoint_guard,
        "source_pose_guard": source_pose_guard,
        "smoke": smoke,
        "blockers": [] if smoke_passed else [smoke.get("error", "virtual_racket_smoke_failed")],
    }
    _write_outputs(args.output_dir, summary)
    return 0 if smoke_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
