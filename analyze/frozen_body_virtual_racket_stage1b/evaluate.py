#!/usr/bin/env python3
"""Stage 1B frozen-body + separate virtual-racket-head integration preflight.

This entrypoint is evaluation-only. It does not create an optimizer, call
backward, train, save trained weights, enable rewards, or modify PHC weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - fallback for minimal environments
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[3]
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

for path in [ISAACGYM_ROOT, PHC_PACKAGE_ROOT, PHC_ROOT, REPO_ROOT, REPO_ROOT / "phc_baseline" / "analyze"]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401,E402
from isaacgym import gymapi, gymutil  # noqa: E402

import joblib  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from easydict import EasyDict  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402

from phc.env.tasks.humanoid_im_mcp_getup import HumanoidImMCPGetup  # noqa: E402
from phc.env.tasks.humanoid_im_mcp_getup_virtual_racket import HumanoidImMCPGetupVirtualRacket  # noqa: E402
from phc.env.tasks.virtual_racket_runtime import (  # noqa: E402
    VIRTUAL_RACKET_ACTION_DIM,
    compute_virtual_racket_metrics,
    normalize_axis,
    world_to_heading_local_vector,
)
from phc.utils.flags import flags  # noqa: E402
from phc.utils.parse_task import parse_task  # noqa: E402

from virtual_racket_head_stage1a_utils import axis_angle_error_deg, step_dynamics, summarize  # noqa: E402


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int = 6):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrozenPHCBodyActor(nn.Module):
    """Minimal deterministic actor route for frozen PHC MCP checkpoint smoke."""

    def __init__(self, checkpoint_model: dict[str, torch.Tensor]):
        super().__init__()
        layers: list[nn.Module] = []
        linear_indices = sorted(
            {
                int(key.split(".")[2])
                for key in checkpoint_model
                if key.startswith("a2c_network.actor_mlp.") and key.endswith(".weight") and key.split(".")[2].isdigit()
            }
        )
        for idx in linear_indices:
            weight = checkpoint_model[f"a2c_network.actor_mlp.{idx}.weight"]
            bias = checkpoint_model[f"a2c_network.actor_mlp.{idx}.bias"]
            layer = nn.Linear(int(weight.shape[1]), int(weight.shape[0]))
            layer.weight.data.copy_(weight)
            layer.bias.data.copy_(bias)
            layers.extend([layer, nn.SiLU()])
        self.actor_mlp = nn.Sequential(*layers)
        mu_weight = checkpoint_model["a2c_network.mu.weight"]
        mu_bias = checkpoint_model["a2c_network.mu.bias"]
        self.mu = nn.Linear(int(mu_weight.shape[1]), int(mu_weight.shape[0]))
        self.mu.weight.data.copy_(mu_weight)
        self.mu.bias.data.copy_(mu_bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mu(self.actor_mlp(obs))


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def confirmed_body_action_dim_from_audit(cfg: dict[str, Any]) -> dict[str, Any] | None:
    audit_path = cfg.get("body_checkpoint_provenance_audit")
    if not audit_path:
        return None
    path = resolve(audit_path)
    if not path.exists():
        return None
    audit = load_json(path)
    resolution = audit.get("resolution", {})
    if resolution.get("case") not in {"Case A", "Case B"}:
        return None
    dim = resolution.get("recommended_body_action_dim")
    if dim is None:
        return None
    return {
        "audit_path": str(path),
        "case": resolution.get("case"),
        "recommended_body_action_dim": int(dim),
        "future_combined_action_dim": int(resolution.get("future_combined_action_dim")),
        "summary": resolution.get("summary"),
    }


def compose_cfg(
    config_name: str,
    cfg: dict[str, Any],
    motion_file: Path,
    sequence_count: int = 1,
    *,
    virtual: bool = False,
    full_body_physics: bool = False,
):
    cfg_dir = str(PHC_PACKAGE_ROOT / "data" / "cfg")
    body_dim = int(cfg["_resolved_original_mcp_action_dim"])
    body_primitive_model = cfg.get("body_primitive_model", "output/HumanoidIm/phc_3/Humanoid.pth")
    overrides = [
        f"learning={cfg['learning_config']}",
        f"robot={cfg['robot_config']}",
        "exp_name=stage1b_preflight_no_training",
        "headless=True",
        "no_virtual_display=True",
        "test=True",
        f"im_eval={bool(cfg.get('im_eval', False))}",
        f"env.num_envs={sequence_count}",
        f"env.motion_file={motion_file}",
        f"env.models={[body_primitive_model] if full_body_physics else []}",
        f"env.has_pnn={bool(full_body_physics)}",
        "env.fitting=False",
        f"env.num_prim={body_dim}",
        f"env.actors_to_load={body_dim}",
        f"env.zero_out_far={bool(cfg.get('zero_out_far', False))}",
        f"env.enableEarlyTermination={bool(cfg.get('enable_early_termination', False))}",
        f"env.stateInit={cfg.get('state_init', 'Start')}",
        f"env.recoveryEpisodeProb={float(cfg.get('recovery_episode_prob', 0.0))}",
        f"env.fallInitProb={float(cfg.get('fall_init_prob', 0.0))}",
        f"env.getup_schedule={bool(cfg.get('getup_schedule', False))}",
        "robot.real_weight_porpotion_boxes=False",
    ]
    if virtual:
        overrides.append(f"env.racket_goal_manifest={resolve(cfg['manifest'])}")
    with initialize_config_dir(config_dir=cfg_dir, version_base=None):
        hydra_cfg = compose(config_name="config", overrides=[f"env={config_name}", *overrides])
    OmegaConf.set_struct(hydra_cfg, False)
    return hydra_cfg


def parse_sim_params(cfg):
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


def make_args(cfg):
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


def init_task(hydra_cfg):
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
        setattr(flags, key, bool(hydra_cfg.get(key, False)))
    flags.test = bool(hydra_cfg.test)
    flags.trigger_input = False
    flags.idx = 0
    sim_params = parse_sim_params(hydra_cfg)
    task, env = parse_task(make_args(hydra_cfg), hydra_cfg, hydra_cfg.learning.params.config, sim_params)
    return task, env


def destroy_task(task) -> None:
    try:
        if getattr(task, "viewer", None) is not None:
            task.gym.destroy_viewer(task.viewer)
    except Exception:
        pass
    try:
        task.gym.destroy_sim(task.sim)
    except Exception:
        pass


def make_smoke_motion_file(full_motion_file: Path, sequences: list[str]) -> Path:
    motions = joblib.load(full_motion_file)
    missing = [sequence for sequence in sequences if sequence not in motions]
    if missing:
        raise KeyError(f"{missing} not found in {full_motion_file}")
    out_path = Path(tempfile.gettempdir()) / "phc_stage1b_preflight_motion.pkl"
    joblib.dump({sequence: motions[sequence] for sequence in sequences}, out_path)
    return out_path


def checkpoint_summary(path: Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    model = ckpt["model"]
    composer = [key for key in model if key.startswith("a2c_network.composer") and key.endswith(".weight")]
    out_dim = None
    if "a2c_network.mu.bias" in model:
        out_dim = int(model["a2c_network.mu.bias"].shape[0])
    return {
        "path": str(path),
        "exists": path.exists(),
        "running_mean_dim": int(ckpt["running_mean_std"]["running_mean"].shape[0]),
        "running_var_dim": int(ckpt["running_mean_std"]["running_var"].shape[0]),
        "composer_weight_count": len(composer),
        "actor_output_dim": out_dim,
        "has_optimizer_state": "optimizer" in ckpt,
        "loaded_for_inference_route_check_only": True,
        "weights_modified": False,
    }


def load_frozen_body_actor(path: Path, device: torch.device) -> tuple[FrozenPHCBodyActor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    ckpt = torch.load(path, map_location=device)
    actor = FrozenPHCBodyActor(ckpt["model"]).to(device)
    actor.eval()
    running = ckpt["running_mean_std"]
    stats = {
        "mean": running["running_mean"].to(device=device, dtype=torch.float32),
        "var": running["running_var"].to(device=device, dtype=torch.float32),
    }
    return actor, ckpt["model"], stats


def load_stage1a_head(cfg: dict[str, Any], device: torch.device) -> tuple[MLPHead, dict[str, Any], dict[str, torch.Tensor]]:
    ckpt = torch.load(resolve(cfg["model_b_checkpoint"]), map_location=device)
    model_meta = ckpt["metadata"]
    if int(model_meta["input_dim"]) != 15 or int(model_meta["output_dim"]) != 6:
        raise RuntimeError(f"unexpected Stage 1A head dims: {model_meta}")
    head = MLPHead(int(model_meta["input_dim"]), [int(v) for v in model_meta["hidden_dims"]], int(model_meta["output_dim"])).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    full_meta = load_json(resolve(cfg["stage1a_model_dir"]) / "checkpoint_metadata.json")
    norm = full_meta["normalization"]
    tensors = {
        "x_mean": torch.tensor(norm["x_mean"], dtype=torch.float32, device=device),
        "x_std": torch.tensor(norm["x_std"], dtype=torch.float32, device=device),
        "y_mean": torch.tensor(norm["y_mean"], dtype=torch.float32, device=device),
        "y_std": torch.tensor(norm["y_std"], dtype=torch.float32, device=device),
    }
    return head, model_meta, tensors


def load_stage1a_head_checkpoint(
    checkpoint_path: Path,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    expected_input_dim: int,
) -> tuple[MLPHead, dict[str, Any], dict[str, torch.Tensor]]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_meta = ckpt["metadata"]
    if int(model_meta["input_dim"]) != int(expected_input_dim) or int(model_meta["output_dim"]) != 6:
        raise RuntimeError(f"unexpected Stage 1A head dims in {checkpoint_path}: {model_meta}")
    head = MLPHead(int(model_meta["input_dim"]), [int(v) for v in model_meta["hidden_dims"]], int(model_meta["output_dim"])).to(device)
    head.load_state_dict(ckpt["state_dict"])
    head.eval()

    full_meta = load_json(resolve(cfg["stage1a_model_dir"]) / "checkpoint_metadata.json")
    norm = full_meta["normalization"]
    tensors = {
        "x_mean": torch.tensor(norm["x_mean"][:expected_input_dim], dtype=torch.float32, device=device),
        "x_std": torch.tensor(norm["x_std"][:expected_input_dim], dtype=torch.float32, device=device),
        "y_mean": torch.tensor(norm["y_mean"], dtype=torch.float32, device=device),
        "y_std": torch.tensor(norm["y_std"], dtype=torch.float32, device=device),
    }
    return head, model_meta, tensors


def run_body_actor(actor: FrozenPHCBodyActor, stats: dict[str, torch.Tensor], body_obs: torch.Tensor) -> torch.Tensor:
    normalized = (body_obs.to(dtype=torch.float32) - stats["mean"]) / torch.sqrt(stats["var"] + 1e-5)
    return actor(normalized)


def run_racket_head(head: MLPHead, norm: dict[str, torch.Tensor], head_input: torch.Tensor) -> torch.Tensor:
    x = (head_input.to(dtype=torch.float32) - norm["x_mean"]) / norm["x_std"].clamp_min(1e-8)
    y = head(x)
    return y * norm["y_std"] + norm["y_mean"]


def resolve_body_action_contract(cfg: dict[str, Any], body_ckpt: dict[str, Any]) -> dict[str, Any]:
    audit_contract = confirmed_body_action_dim_from_audit(cfg)
    expected = int(cfg["expected_original_mcp_action_dim"])
    if audit_contract is not None:
        expected = int(audit_contract["recommended_body_action_dim"])
    actor_output = int(body_ckpt["actor_output_dim"]) if body_ckpt.get("actor_output_dim") is not None else None
    source = cfg.get("body_action_dim_source", "config")
    return {
        "source": source,
        "audit_contract": audit_contract,
        "expected_original_mcp_action_dim": expected,
        "checkpoint_actor_output_dim": actor_output,
        "matches_checkpoint": actor_output == expected,
        "virtual_racket_action_dim": int(cfg["virtual_racket_action_dim"]),
        "expected_combined_action_dim": expected + int(cfg["virtual_racket_action_dim"]),
    }


def validate_head_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    metadata_path = resolve(cfg["stage1a_model_dir"]) / "checkpoint_metadata.json"
    meta = load_json(metadata_path)
    primary = meta["models"]["goal_state"]
    goal_only = meta["models"]["goal_only"]
    checks = {
        "metadata_path": str(metadata_path),
        "goal_version": meta.get("goal_version"),
        "state_version": meta.get("state_version"),
        "action_version": meta.get("action_version"),
        "body_path_frozen": bool(meta.get("body_path_frozen")),
        "no_physics_virtual_head_only": bool(meta.get("no_physics_virtual_head_only")),
        "primary_input_dim": int(primary["input_dim"]),
        "primary_output_dim": int(primary["output_dim"]),
        "goal_only_input_dim": int(goal_only["input_dim"]),
        "goal_only_output_dim": int(goal_only["output_dim"]),
    }
    checks["passed"] = (
        checks["goal_version"] == "v2_sim_root_projected_world_target"
        and checks["state_version"] == "v2_sim_root_projected_world_state"
        and checks["action_version"] == "v2_heading_local_velocity"
        and checks["body_path_frozen"]
        and checks["no_physics_virtual_head_only"]
        and checks["primary_input_dim"] == 15
        and checks["primary_output_dim"] == 6
        and checks["goal_only_input_dim"] == 9
        and checks["goal_only_output_dim"] == 6
    )
    return checks


def actual_obs_preflight(cfg: dict[str, Any], motion_file: Path, sequence_count: int) -> dict[str, Any]:
    result: dict[str, Any] = {"passed": False}
    os.chdir(PHC_ROOT)
    body_task = None
    virtual_task = None
    try:
        if bool(cfg.get("initialize_body_only_env_for_smoke", False)):
            body_cfg = compose_cfg(cfg["original_body_env_config"], cfg, motion_file, sequence_count=sequence_count, virtual=False)
            body_task, _body_env = init_task(body_cfg)
            result["body_only_env_initialization_attempted"] = True
            result["body_only_env_initialization_mode"] = "separate_isaac_sim"
            result["original_body_task_class"] = body_task.__class__.__name__
            result["actual_original_full_obs_dim"] = int(body_task.num_obs)
            result["actual_original_task_obs_dim"] = int(body_task.get_task_obs_size())
            result["actual_original_mcp_action_dim"] = int(body_task.num_actions)
            destroy_task(body_task)
            body_task = None
        else:
            result["body_only_env_initialization_attempted"] = False
            result["body_only_env_initialization_mode"] = "skipped_to_avoid_two_isaac_sim_initializations; same-state hook parity is checked inside the virtual env"

        virtual_cfg = compose_cfg(cfg["virtual_racket_env_config"], cfg, motion_file, sequence_count=sequence_count, virtual=True)
        with open_dict(virtual_cfg):
            virtual_cfg.env.virtual_racket_smoke_no_policy = True
            virtual_cfg.env.racket_goal_smoke_no_policy = True
            virtual_cfg.env.enable_virtual_racket_reward = False
        virtual_task, _virtual_env = init_task(virtual_cfg)
        result["virtual_task_class"] = virtual_task.__class__.__name__
        result["virtual_full_obs_dim"] = int(virtual_task.num_obs)
        result["virtual_task_obs_dim"] = int(virtual_task.get_task_obs_size())
        result["virtual_augmented_action_dim"] = int(virtual_task.num_actions)
        result["virtual_reward_enabled"] = bool(getattr(virtual_task, "_enable_virtual_racket_reward", True))
        if not bool(cfg.get("initialize_body_only_env_for_smoke", False)):
            result["original_body_task_class"] = "HumanoidImMCPGetup (same-state base hook inside HumanoidImMCPGetupVirtualRacket)"
            result["actual_original_full_obs_dim"] = None
            result["actual_original_task_obs_dim"] = None
            result["actual_original_mcp_action_dim"] = int(super(HumanoidImMCPGetupVirtualRacket, virtual_task).get_action_size())

        env_ids = torch.arange(virtual_task.num_envs, dtype=torch.long, device=virtual_task.device)
        exported_body_obs = virtual_task.get_original_body_policy_obs(env_ids)
        manual_body_obs = torch.cat(
            [
                virtual_task._compute_humanoid_obs(env_ids),
                HumanoidImMCPGetup._compute_task_obs(virtual_task, env_ids=env_ids, save_buffer=False),
            ],
            dim=-1,
        )
        full_aug_obs = virtual_task._compute_observations(env_ids=env_ids)
        head_input = virtual_task.get_virtual_racket_head_input(env_ids)
        body_actor, _body_model, body_stats = load_frozen_body_actor(resolve(cfg["original_phc_body_checkpoint"]), virtual_task.device)
        racket_head, racket_head_checkpoint_meta, racket_norm = load_stage1a_head(cfg, virtual_task.device)
        with torch.no_grad():
            body_action = run_body_actor(body_actor, body_stats, exported_body_obs)
            racket_action = run_racket_head(racket_head, racket_norm, head_input)
        combined = virtual_task.pack_combined_action(body_action, racket_action)

        wrong_body_shape_rejected = False
        wrong_racket_shape_rejected = False
        augmented_obs_to_body_actor_rejected = False
        try:
            virtual_task.pack_combined_action(torch.zeros((virtual_task.num_envs, body_action.shape[-1] + 1), device=virtual_task.device), racket_action)
        except RuntimeError:
            wrong_body_shape_rejected = True
        try:
            virtual_task.pack_combined_action(body_action, torch.zeros((virtual_task.num_envs, racket_action.shape[-1] + 1), device=virtual_task.device))
        except RuntimeError:
            wrong_racket_shape_rejected = True
        try:
            run_body_actor(body_actor, body_stats, full_aug_obs)
        except RuntimeError:
            augmented_obs_to_body_actor_rejected = True

        step_records = []
        for step_idx in range(int(cfg.get("smoke_step_count", 0))):
            body_obs_step = virtual_task.get_original_body_policy_obs(env_ids)
            head_input_step = virtual_task.get_virtual_racket_head_input(env_ids)
            with torch.no_grad():
                body_action_step = run_body_actor(body_actor, body_stats, body_obs_step)
                racket_action_step = run_racket_head(racket_head, racket_norm, head_input_step)
            combined_step = virtual_task.pack_combined_action(body_action_step, racket_action_step)
            # No-physics wiring smoke: validate both inference routes and the
            # env-boundary pack, then advance only the virtual racket state.
            # Do not call HumanoidImMCP.step(), which would run the PHC MCP
            # internal actor/physics path rather than this frozen-body route.
            virtual_task._step_virtual_racket(racket_action_step, env_ids=env_ids)
            step_records.append(
                {
                    "step": step_idx,
                    "body_physics_step_executed": False,
                    "virtual_racket_no_physics_step_executed": True,
                    "body_obs_shape": list(body_obs_step.shape),
                    "head_input_shape": list(head_input_step.shape),
                    "body_action_shape": list(body_action_step.shape),
                    "racket_action_shape": list(racket_action_step.shape),
                    "combined_action_shape": list(combined_step.shape),
                    "reward_enabled": bool(getattr(virtual_task, "_enable_virtual_racket_reward", True)),
                }
            )

        result.update(
            {
                "body_policy_obs_shape": list(exported_body_obs.shape),
                "body_policy_obs_all_finite": bool(torch.isfinite(exported_body_obs).all().detach().cpu().item()),
                "same_task_hook_body_obs_parity_max_abs_diff": float((exported_body_obs - manual_body_obs).abs().max().detach().cpu().item()),
                "body_policy_obs_dim_from_virtual_hook": int(exported_body_obs.shape[-1]),
                "augmented_full_obs_shape": list(full_aug_obs.shape),
                "augmented_obs_excluded_from_body_route": int(full_aug_obs.shape[-1]) != int(exported_body_obs.shape[-1]),
                "head_input_shape": list(head_input.shape),
                "stage1a_head_checkpoint_metadata": racket_head_checkpoint_meta,
                "body_action_shape": list(body_action.shape),
                "racket_action_shape": list(racket_action.shape),
                "body_actor_forward_executed": True,
                "racket_head_forward_executed": True,
                "combined_action_shape": list(combined.shape),
                "combined_action_dim": int(combined.shape[-1]),
                "wrong_body_action_shape_rejected": wrong_body_shape_rejected,
                "wrong_racket_action_shape_rejected": wrong_racket_shape_rejected,
                "augmented_obs_to_body_actor_rejected": augmented_obs_to_body_actor_rejected,
                "smoke_step_count": len(step_records),
                "smoke_steps": step_records,
            }
        )
    except Exception as exc:
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
    finally:
        if virtual_task is not None:
            destroy_task(virtual_task)
        if body_task is not None:
            destroy_task(body_task)
    return result


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def quick_virtual_reference_metrics(cfg: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    """Write blocked placeholders plus Stage 1A reference bars for context."""
    dataset = load_npz(resolve(cfg["dataset_dir"]) / "test_transitions_clean.npz")
    metrics = {}
    for name, actions in {
        "null_virtual_action": np.zeros_like(dataset["y"], dtype=np.float32),
        "oracle_virtual_action": dataset["y"].astype(np.float32),
    }.items():
        h, axis = step_dynamics(
            dataset["current_handle_world"].astype(np.float64),
            dataset["current_axis_world"].astype(np.float64),
            actions.astype(np.float64),
            dataset["root_rot_t_world"].astype(np.float64),
        )
        tip = h + dataset["racket_length"].astype(np.float64)[:, None] * axis
        metrics[name] = {
            "scope": "offline one-step contract context only; not frozen-body integration",
            "handle_error_m": summarize(np.linalg.norm(h - dataset["target_handle_next_world"], axis=-1)),
            "tip_error_m": summarize(np.linalg.norm(tip - dataset["target_tip_next_world"], axis=-1)),
            "axis_error_deg": summarize(axis_angle_error_deg(axis, dataset["target_axis_next_world"])),
        }
    plot_dir = report_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    labels = ["null", "oracle"]
    tips = [metrics["null_virtual_action"]["tip_error_m"]["mean"], metrics["oracle_virtual_action"]["tip_error_m"]["mean"]]
    axes = [metrics["null_virtual_action"]["axis_error_deg"]["mean"], metrics["oracle_virtual_action"]["axis_error_deg"]["mean"]]
    x = np.arange(len(labels))
    plt.bar(x - 0.15, tips, width=0.3, label="tip mean (m)")
    plt.bar(x + 0.15, axes, width=0.3, label="axis mean (deg)")
    plt.xticks(x, labels)
    plt.title("blocked before frozen-body integration\nno-physics virtual racket tracking under frozen-body PHC rollout; not physical racket accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "blocked_contract_context_tip_axis.png", dpi=160)
    plt.close()
    return metrics


def write_status_audit(report_dir: Path) -> dict[str, Any]:
    audit = {
        "status": "corrected",
        "current_status": {
            "stage1a_supervised_separate_head_training_executed": True,
            "stage1b_frozen_body_integration": "preflight_only_until_checkpoint_compatibility_passes",
            "goal_v1": "reference-root-local diagnostic/reference goal",
            "live_goal_v2": "default live controller target in current simulated-root heading-local frame",
            "racket_pose_parameter": "source diagnostic/auxiliary only",
            "no_ppo_rl": True,
            "no_phc_body_finetune": True,
            "no_physical_racket_or_shuttle": True,
            "no_official_phc_rollout_racket_accuracy": True,
        },
        "files_checked": [
            "phc_calibrated_racket_trajectory_report_v2.md",
            "racket_aware_task_interface_spec.md",
            "controller_interface/racket_aware_adaptation_experiment_plan.md",
            "controller_interface/separate_virtual_racket_head_training_readiness.md",
            "virtual_racket_control/racket_tracking_metric_and_future_reward_contract.md",
        ],
    }
    (report_dir / "report_status_consistency_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    md = """# Report Status Consistency Audit

Status: corrected.

Current status supersedes older future-only wording where present:

- Stage 1A separate-head supervised warm start has been executed.
- Goal V1 is a reference-root-local diagnostic/reference goal.
- Live Goal V2 is the default live controller target.
- `racket_pose_parameter` is source diagnostic/auxiliary only, not a primary controller input.
- No PPO/RL, no PHC body fine-tune, no physical racket/shuttle, and no official PHC rollout racket accuracy are claimed.

Historical sections may retain their original chronology, but current-status sections now distinguish superseded design notes from executed Stage 1A facts.
"""
    (report_dir / "report_status_consistency_audit.md").write_text(md, encoding="utf-8")
    return audit


def write_preflight_reports(report_dir: Path, summary: dict[str, Any]) -> None:
    (report_dir / "body_obs_compatibility_preflight.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md = f"""# Body Observation Compatibility Preflight

Scope: frozen-PHC-body + learned virtual-racket-head no-physics integration preflight only. No optimizer, backward pass, training, reward update, physical racket, shuttle, or official PHC rollout racket accuracy is involved.

## Result

- passed: `{summary['passed']}`
- blocker: `{summary.get('blocker')}`

## Actual Dims

- original body env full obs dim: `{summary.get('actual_original_full_obs_dim')}`
- virtual env full obs dim: `{summary.get('virtual_full_obs_dim')}`
- exported body-policy obs shape: `{summary.get('body_policy_obs_shape')}`
- same-task hook body obs parity max abs diff: `{summary.get('same_task_hook_body_obs_parity_max_abs_diff')}`
- original body env MCP action dim: `{summary.get('actual_original_mcp_action_dim')}`
- virtual combined action dim: `{summary.get('combined_action_dim')}`

## Checkpoint Routes

- original checkpoint path: `{summary['body_checkpoint']['path']}`
- original checkpoint input dim from running mean: `{summary['body_checkpoint']['running_mean_dim']}`
- original checkpoint actor output dim: `{summary['body_checkpoint']['actor_output_dim']}`
- Stage 1A head metadata passed: `{summary['head_metadata']['passed']}`
- augmented obs excluded from body route: `{summary.get('augmented_obs_excluded_from_body_route')}`

The original PHC checkpoint was loaded only for route/shape inspection. Its weights were not modified.
"""
    (report_dir / "body_obs_compatibility_preflight.md").write_text(md, encoding="utf-8")


def write_blocked_outputs(report_dir: Path, summary: dict[str, Any]) -> None:
    smoke = {
        "passed": False,
        "blocked_before_smoke_rollout": True,
        "reason": summary.get("blocker"),
        "no_training": True,
        "no_optimizer": True,
        "no_backward": True,
        "no_reward_update": True,
    }
    (report_dir / "integration_smoke_summary.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    (report_dir / "integration_smoke_report.md").write_text(
        "# Integration Smoke Report\n\nBlocked before rollout smoke because body observation/checkpoint compatibility preflight did not pass.\n",
        encoding="utf-8",
    )
    evaluation = {
        "passed": False,
        "blocked_before_full_heldout_evaluation": True,
        "reason": summary.get("blocker"),
        "coverage": {"groups": [], "clips": 0, "frames": 0, "modes": []},
        "body_parity_metrics": None,
        "virtual_metrics": None,
        "hand_body_consistency": {"measured": False, "reason": "integration rollout blocked before simulated hand trajectory was available"},
    }
    (report_dir / "stage1b_evaluation_summary.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    with (report_dir / "stage1b_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["status", "reason", "provenance_case", "body_action_dim", "virtual_racket_action_dim", "future_combined_action_dim", "groups", "clips", "frames", "modes"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "status": "not_run",
                "reason": "smoke_only_full_heldout_not_requested",
                "provenance_case": summary["body_action_contract"].get("audit_contract", {}).get("case"),
                "body_action_dim": summary["body_action_contract"]["expected_original_mcp_action_dim"],
                "virtual_racket_action_dim": summary["body_action_contract"]["virtual_racket_action_dim"],
                "future_combined_action_dim": summary["body_action_contract"]["expected_combined_action_dim"],
                "groups": "",
                "clips": 0,
                "frames": 0,
                "modes": "",
            }
        )
    with (report_dir / "stage1b_per_sequence_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "status", "reason", "provenance_case"])
        writer.writeheader()
        for sequence in summary.get("smoke_sequences", []):
            writer.writerow({"sequence": sequence, "status": "smoke_only", "reason": "full_heldout_not_requested", "provenance_case": summary["body_action_contract"].get("audit_contract", {}).get("case")})
    fieldnames = ["status", "reason", "groups", "clips", "frames", "modes"]
    with (report_dir / "stage1b_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({"status": "blocked", "reason": summary.get("blocker"), "groups": "", "clips": 0, "frames": 0, "modes": ""})
    with (report_dir / "stage1b_per_sequence_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "status", "reason"])
        writer.writeheader()
        writer.writerow({"sequence": "", "status": "blocked", "reason": summary.get("blocker")})
    (report_dir / "stage1b_evaluation_report.md").write_text(
        f"""# Stage 1B Evaluation Report

Stage 1B full held-out integration evaluation was blocked before rollout.

Reason: `{summary.get('blocker')}`

No optimizer, backward pass, training command, reward update, PHC body fine-tune, physical racket, shuttle, or official PHC rollout racket accuracy was run.
""",
        encoding="utf-8",
    )


def write_smoke_success_outputs(report_dir: Path, summary: dict[str, Any]) -> None:
    smoke = {
        "passed": True,
        "scope": "gated 1-3 clip frozen-body + separate virtual-racket-head GPU smoke only",
        "full_heldout_evaluation_run": False,
        "smoke_sequences": summary.get("smoke_sequences"),
        "smoke_sequence_count": summary.get("smoke_sequence_count"),
        "expected_non_contiguous_smoke_sequences": summary.get("expected_non_contiguous_smoke_sequences"),
        "non_contiguous_smoke_sequence_included": summary.get("non_contiguous_smoke_sequence_included"),
        "confirmed_body_checkpoint": summary["body_checkpoint"]["path"],
        "body_only_env_initialization_attempted": summary.get("body_only_env_initialization_attempted"),
        "body_only_env_initialization_mode": summary.get("body_only_env_initialization_mode"),
        "body_obs_dim": summary["body_checkpoint"]["running_mean_dim"],
        "body_action_dim": summary["body_action_contract"]["expected_original_mcp_action_dim"],
        "racket_head_input_dim": summary["head_metadata"]["primary_input_dim"],
        "racket_action_dim": summary["head_metadata"]["primary_output_dim"],
        "combined_action_dim": summary.get("combined_action_dim"),
        "body_policy_obs_shape": summary.get("body_policy_obs_shape"),
        "augmented_full_obs_shape": summary.get("augmented_full_obs_shape"),
        "augmented_obs_excluded_from_body_route": summary.get("augmented_obs_excluded_from_body_route"),
        "body_actor_forward_executed": summary.get("body_actor_forward_executed"),
        "racket_head_forward_executed": summary.get("racket_head_forward_executed"),
        "wrong_body_action_shape_rejected": summary.get("wrong_body_action_shape_rejected"),
        "wrong_racket_action_shape_rejected": summary.get("wrong_racket_action_shape_rejected"),
        "augmented_obs_to_body_actor_rejected": summary.get("augmented_obs_to_body_actor_rejected"),
        "smoke_step_count": summary.get("smoke_step_count"),
        "body_physics_step_executed": False,
        "virtual_racket_no_physics_steps_executed": summary.get("smoke_step_count"),
        "virtual_reward_enabled": summary.get("virtual_reward_enabled"),
        "no_training": True,
        "no_optimizer": True,
        "no_backward": True,
        "no_reward_update": True,
        "no_physical_racket_or_shuttle": True,
        "no_official_phc_rollout_racket_accuracy": True,
    }
    (report_dir / "integration_smoke_summary.json").write_text(json.dumps(smoke, indent=2), encoding="utf-8")
    md = f"""# Integration Smoke Report

Status: passed.

Scope: gated 1-3 clip frozen-body + separate virtual-racket-head GPU smoke only. This is not full held-out Stage 1B evaluation and not PHC rollout racket accuracy.

## Coverage

- smoke sequences: `{summary.get('smoke_sequences')}`
- smoke sequence count: `{summary.get('smoke_sequence_count')}`
- expected non-contiguous sequence included: `{summary.get('non_contiguous_smoke_sequence_included')}`
- smoke steps: `{summary.get('smoke_step_count')}`
- body-only env initialization attempted: `{summary.get('body_only_env_initialization_attempted')}`
- body-only env initialization mode: `{summary.get('body_only_env_initialization_mode')}`
- body physics step executed: `False`
- virtual racket no-physics steps executed: `{summary.get('smoke_step_count')}`

## Confirmed Routes

- frozen body checkpoint: `{summary['body_checkpoint']['path']}`
- body-policy obs shape: `{summary.get('body_policy_obs_shape')}`
- body action shape: `{summary.get('body_action_shape')}`
- Stage 1A head input shape: `{summary.get('head_input_shape')}`
- racket action shape: `{summary.get('racket_action_shape')}`
- combined action shape: `{summary.get('combined_action_shape')}`

The original PHC body actor received only the exported original body-policy observation. The augmented virtual-racket observation was rejected by the body actor route. The Stage 1A head received only Live Goal V2 plus realized-state feedback.

## Guards

- wrong body action shape rejected: `{summary.get('wrong_body_action_shape_rejected')}`
- wrong racket action shape rejected: `{summary.get('wrong_racket_action_shape_rejected')}`
- augmented observation excluded from body route: `{summary.get('augmented_obs_excluded_from_body_route')}`
- augmented observation to body actor rejected: `{summary.get('augmented_obs_to_body_actor_rejected')}`
- virtual reward enabled: `{summary.get('virtual_reward_enabled')}`

No optimizer, backward pass, training command, reward update, PHC body fine-tune, physical racket, shuttle, or official PHC rollout racket accuracy was run.
"""
    (report_dir / "integration_smoke_report.md").write_text(md, encoding="utf-8")

    evaluation = {
        "passed": False,
        "full_heldout_evaluation_attempted": False,
        "blocked_before_full_heldout_evaluation": True,
        "reason": "not_requested_this_round_smoke_only",
        "smoke_passed": True,
        "coverage": {"groups": [], "clips": 0, "frames": 0, "modes": []},
        "body_parity_metrics": None,
        "virtual_metrics": None,
        "hand_body_consistency": {"measured": False, "reason": "full held-out integration evaluation not run in this smoke-only round"},
    }
    (report_dir / "stage1b_evaluation_summary.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")


def select_heldout_sequences(cfg: dict[str, Any]) -> list[str]:
    rows = []
    with resolve(cfg["manifest"]).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("task_export_passed") == "True"
                and row.get("integrity_check_passed") == "True"
                and row.get("dynamic_replay_passed") == "True"
                and row.get("session_group") in set(cfg["expected_test_groups"])
            ):
                rows.append(row)
    rows = sorted(rows, key=lambda row: (row["session_group"], row["sequence"]))
    sequences = [row["sequence"] for row in rows]
    expected = int(cfg["expected_test_clip_count"])
    if len(sequences) != expected:
        raise RuntimeError(f"expected {expected} held-out clips from {cfg['expected_test_groups']}, found {len(sequences)}")
    return sequences


def tensor_stats(values: list[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": None, "p50": None, "p90": None, "max": None}
    return {"mean": float(np.mean(arr)), "p50": float(np.percentile(arr, 50)), "p90": float(np.percentile(arr, 90)), "max": float(np.max(arr))}


def derive_oracle_action_torch(task: HumanoidImMCPGetupVirtualRacket, env_ids: torch.Tensor) -> torch.Tensor:
    motion_times = task._current_motion_times(env_ids, next_frame=True)
    target = task._lookup_virtual_racket_target(env_ids, motion_times)
    current_handle = task._virtual_racket_handle_world[env_ids]
    current_axis = normalize_axis(task._virtual_racket_axis_world[env_ids])
    target_axis = normalize_axis(target["axis"])
    v_world = (target["handle"] - current_handle) / float(task.dt)
    cross = torch.cross(current_axis, target_axis, dim=-1)
    sin = torch.linalg.norm(cross, dim=-1, keepdim=True)
    cos = torch.clamp(torch.sum(current_axis * target_axis, dim=-1, keepdim=True), -1.0, 1.0)
    angle = torch.atan2(sin, cos)
    rot_axis = torch.zeros_like(current_axis)
    nonzero = sin[..., 0] > 1e-8
    if bool(nonzero.any().detach().cpu().item()):
        rot_axis[nonzero] = cross[nonzero] / sin[nonzero]
    omega_world = rot_axis * angle / float(task.dt)
    root_rot = task._rigid_body_rot[env_ids, 0, :]
    return torch.cat([world_to_heading_local_vector(v_world, root_rot), world_to_heading_local_vector(omega_world, root_rot)], dim=-1)


def compute_body_tracking_metrics(task, env_ids: torch.Tensor) -> dict[str, np.ndarray]:
    motion_times = task.progress_buf[env_ids] * task.dt + task._motion_start_times[env_ids] + task._motion_start_times_offset[env_ids]
    motion_res = task._get_state_from_motionlib_cache(task._sampled_motion_ids[env_ids], motion_times, task._global_offset[env_ids])
    ref_body_pos = motion_res["rg_pos"]
    ref_body_rot = motion_res["rb_rot"]
    body_pos = task._rigid_body_pos[env_ids]
    body_rot = task._rigid_body_rot[env_ids]
    mpjpe = torch.linalg.norm(body_pos - ref_body_pos, dim=-1).mean(dim=-1)
    root_error = torch.linalg.norm(body_pos[:, 0, :] - ref_body_pos[:, 0, :], dim=-1)
    return {
        "mpjpe": mpjpe.detach().cpu().numpy(),
        "root_error": root_error.detach().cpu().numpy(),
        "body_pos": body_pos.detach().cpu().numpy(),
        "body_rot": body_rot.detach().cpu().numpy(),
        "ref_body_pos": ref_body_pos.detach().cpu().numpy(),
        "ref_body_rot": ref_body_rot.detach().cpu().numpy(),
        "motion_time": motion_times.detach().cpu().numpy(),
    }


def body_policy_obs_for_task(task, env_ids: torch.Tensor) -> torch.Tensor:
    if isinstance(task, HumanoidImMCPGetupVirtualRacket):
        return task.get_original_body_policy_obs(env_ids)
    if int(task.obs_buf.shape[-1]) == int(task.num_obs):
        return task.obs_buf[env_ids]
    humanoid_obs = task._compute_humanoid_obs(env_ids)
    body_task_obs = HumanoidImMCPGetup._compute_task_obs(task, env_ids=env_ids, save_buffer=False)
    return torch.cat([humanoid_obs, body_task_obs], dim=-1)


def attempt_hand_consistency(task: HumanoidImMCPGetupVirtualRacket, env_ids: torch.Tensor) -> dict[str, Any]:
    names = getattr(task, "_body_names", None) or getattr(task, "body_names", None)
    if names is None:
        return {"available": False, "reason": "task exposes no body-name list for R_Hand lookup"}
    names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]
    if "R_Hand" not in names:
        return {"available": False, "reason": f"R_Hand not found in task body names; available count={len(names)}"}
    hand_idx = names.index("R_Hand")
    motion_times = task._current_motion_times(env_ids, next_frame=False)
    target = task._lookup_virtual_racket_target(env_ids, motion_times)
    body = compute_body_tracking_metrics(task, env_ids)
    sim_hand = task._rigid_body_pos[env_ids, hand_idx, :]
    ref_hand = torch.as_tensor(body["ref_body_pos"], device=task.device, dtype=task.obs_buf.dtype)[:, hand_idx, :]
    d_realized = torch.linalg.norm(task._virtual_racket_handle_world[env_ids] - sim_hand, dim=-1)
    d_target = torch.linalg.norm(target["handle"] - ref_hand, dim=-1)
    err = torch.abs(d_realized - d_target)
    return {
        "available": True,
        "body_name": "R_Hand",
        "body_index": int(hand_idx),
        "d_realized_hand": d_realized.detach().cpu().numpy(),
        "d_target_hand": d_target.detach().cpu().numpy(),
        "e_hand_consistency": err.detach().cpu().numpy(),
    }


def run_sequence_child(cfg: dict[str, Any], *, mode: str, sequence: str, child_output: Path) -> int:
    report_dir = child_output.parent
    report_dir.mkdir(parents=True, exist_ok=True)
    body_ckpt = checkpoint_summary(resolve(cfg["confirmed_body_checkpoint"]))
    body_contract = resolve_body_action_contract({**cfg, "expected_original_mcp_action_dim": cfg["confirmed_body_action_dim"], "virtual_racket_action_dim": cfg["racket_head_output_dim"]}, body_ckpt)
    cfg["_resolved_original_mcp_action_dim"] = body_contract["expected_original_mcp_action_dim"]
    motion_file = make_smoke_motion_file(resolve(cfg["motion_file"]), [sequence])
    env_config = cfg["original_body_env_config"] if mode == "body_only" else cfg["virtual_racket_env_config"]
    hydra_cfg = compose_cfg(env_config, cfg, motion_file, sequence_count=1, virtual=mode != "body_only", full_body_physics=True)
    with open_dict(hydra_cfg):
        hydra_cfg.env.enable_virtual_racket_reward = False
        if mode != "body_only":
            hydra_cfg.env.virtual_racket_smoke_no_policy = True
            hydra_cfg.env.racket_goal_smoke_no_policy = True
    os.chdir(PHC_ROOT)
    task = None
    try:
        task, _env = init_task(hydra_cfg)
        env_ids = torch.arange(task.num_envs, dtype=torch.long, device=task.device)
        if bool(cfg.get("force_deterministic_reset_after_init", True)):
            reset_seed = int(cfg["seed"]) + sum(ord(ch) for ch in sequence)
            torch.manual_seed(reset_seed)
            np.random.seed(reset_seed % (2**32 - 1))
            if hasattr(task, "_recovery_episode_prob"):
                task._recovery_episode_prob = 0
            if hasattr(task, "_fall_init_prob"):
                task._fall_init_prob = 0
            task.reset()
        body_actor, _body_model, body_stats = load_frozen_body_actor(resolve(cfg["confirmed_body_checkpoint"]), task.device)
        heads: dict[str, tuple[MLPHead, dict[str, Any], dict[str, torch.Tensor]]] = {}
        if mode == "virtual_goal_state":
            heads[mode] = load_stage1a_head_checkpoint(resolve(cfg["racket_head_checkpoint"]), cfg, task.device, expected_input_dim=15)
        elif mode == "virtual_goal_only":
            heads[mode] = load_stage1a_head_checkpoint(resolve(cfg["goal_only_head_checkpoint"]), cfg, task.device, expected_input_dim=9)

        sampled = task._sampled_motion_ids[env_ids]
        planned_steps = int(task._motion_lib._motion_num_frames[sampled][0].detach().cpu().item()) - 1
        if int(cfg.get("max_steps_per_clip", 0)) > 0:
            planned_steps = min(planned_steps, int(cfg["max_steps_per_clip"]))
        root_trace = []
        body_mpjpe: list[float] = []
        root_errors: list[float] = []
        handle_errors: list[float] = []
        tip_errors: list[float] = []
        axis_errors: list[float] = []
        action_norms: list[float] = []
        smoothness: list[float] = []
        hand_realized: list[float] = []
        hand_target: list[float] = []
        hand_error: list[float] = []
        kintwin_trace: dict[str, list[np.ndarray]] = {
            "r_hand_sim_world": [],
            "r_hand_ref_world": [],
            "r_wrist_sim_world": [],
            "r_wrist_ref_world": [],
            "handle_realized_world": [],
            "handle_target_world": [],
        }
        validity_trace: dict[str, list[np.ndarray]] = {
            "body_sim_world": [],
            "body_ref_world": [],
            "root_sim_world": [],
            "root_ref_world": [],
            "root_sim_rot": [],
            "root_ref_rot": [],
            "motion_time": [],
            "sampled_motion_id": [],
            "progress_buf": [],
            "step_index": [],
            "reset_buf_after_step": [],
            "handle_realized_world": [],
            "handle_target_world": [],
        }
        validity_body_names: list[str] | None = None
        validity_hand_idx: int | None = None
        validity_wrist_idx: int | None = None
        previous_action = None
        terminated = False

        for _step in range(planned_steps):
            body_obs = body_policy_obs_for_task(task, env_ids)
            with torch.no_grad():
                body_action = run_body_actor(body_actor, body_stats, body_obs)
            if mode == "body_only":
                task.step(body_action)
                virtual_action = None
            else:
                assert isinstance(task, HumanoidImMCPGetupVirtualRacket)
                if mode == "virtual_null":
                    virtual_action = torch.zeros((task.num_envs, VIRTUAL_RACKET_ACTION_DIM), device=task.device, dtype=body_action.dtype)
                elif mode == "virtual_oracle":
                    virtual_action = derive_oracle_action_torch(task, env_ids)
                elif mode == "virtual_goal_state":
                    head_input = task.get_virtual_racket_head_input(env_ids)
                    head, _meta, norm = heads[mode]
                    with torch.no_grad():
                        virtual_action = run_racket_head(head, norm, head_input)
                elif mode == "virtual_goal_only":
                    head_input = task._compute_live_racket_goal_obs(env_ids)
                    head, _meta, norm = heads[mode]
                    with torch.no_grad():
                        virtual_action = run_racket_head(head, norm, head_input)
                else:
                    raise RuntimeError(f"unknown mode {mode}")
                combined = task.pack_combined_action(body_action, virtual_action)
                task.step(combined)

            metrics = compute_body_tracking_metrics(task, env_ids)
            body_mpjpe.extend(metrics["mpjpe"].tolist())
            root_errors.extend(metrics["root_error"].tolist())
            root_trace.append(task._rigid_body_pos[env_ids, 0, :].detach().cpu().numpy().reshape(-1, 3)[0])
            if bool(cfg.get("save_hand_wrist_validity_traces", False)):
                names = getattr(task, "_body_names", None) or getattr(task, "body_names", None)
                if names is not None and validity_body_names is None:
                    validity_body_names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]
                    if "R_Hand" in validity_body_names:
                        validity_hand_idx = validity_body_names.index("R_Hand")
                    if "R_Wrist" in validity_body_names:
                        validity_wrist_idx = validity_body_names.index("R_Wrist")
                validity_trace["body_sim_world"].append(metrics["body_pos"][0])
                validity_trace["body_ref_world"].append(metrics["ref_body_pos"][0])
                validity_trace["root_sim_world"].append(metrics["body_pos"][0, 0, :])
                validity_trace["root_ref_world"].append(metrics["ref_body_pos"][0, 0, :])
                validity_trace["root_sim_rot"].append(metrics["body_rot"][0, 0, :])
                validity_trace["root_ref_rot"].append(metrics["ref_body_rot"][0, 0, :])
                validity_trace["motion_time"].append(np.asarray([metrics["motion_time"][0]], dtype=np.float32))
                validity_trace["sampled_motion_id"].append(task._sampled_motion_ids[env_ids].detach().cpu().numpy().astype(np.int64))
                validity_trace["progress_buf"].append(task.progress_buf[env_ids].detach().cpu().numpy().astype(np.float32))
                validity_trace["step_index"].append(np.asarray([_step], dtype=np.int64))
            if mode != "body_only":
                assert isinstance(task, HumanoidImMCPGetupVirtualRacket)
                vr = task._virtual_racket_metrics
                handle_errors.extend(vr["handle_error_m"].detach().cpu().numpy().tolist())
                tip_errors.extend(vr["tip_error_m"].detach().cpu().numpy().tolist())
                axis_errors.extend(vr["axis_error_deg"].detach().cpu().numpy().tolist())
                action_np = virtual_action.detach().cpu().numpy()
                action_norms.extend(np.linalg.norm(action_np, axis=-1).tolist())
                if previous_action is not None:
                    smoothness.extend(np.linalg.norm(action_np - previous_action, axis=-1).tolist())
                previous_action = action_np
                hand = attempt_hand_consistency(task, env_ids)
                if hand.get("available"):
                    hand_realized.extend(np.asarray(hand["d_realized_hand"]).reshape(-1).tolist())
                    hand_target.extend(np.asarray(hand["d_target_hand"]).reshape(-1).tolist())
                    hand_error.extend(np.asarray(hand["e_hand_consistency"]).reshape(-1).tolist())
                if bool(cfg.get("save_kintwin_tracking_traces", False)):
                    names = getattr(task, "_body_names", None) or getattr(task, "body_names", None)
                    if names is not None:
                        names = [name.decode("utf-8") if isinstance(name, bytes) else str(name) for name in names]
                        if "R_Hand" in names and "R_Wrist" in names:
                            body = compute_body_tracking_metrics(task, env_ids)
                            ref_body = body["ref_body_pos"][0]
                            hand_idx = names.index("R_Hand")
                            wrist_idx = names.index("R_Wrist")
                            target = task._lookup_virtual_racket_target(env_ids, task._current_motion_times(env_ids, next_frame=False))
                            kintwin_trace["r_hand_sim_world"].append(task._rigid_body_pos[env_ids, hand_idx, :].detach().cpu().numpy()[0])
                            kintwin_trace["r_hand_ref_world"].append(ref_body[hand_idx])
                            kintwin_trace["r_wrist_sim_world"].append(task._rigid_body_pos[env_ids, wrist_idx, :].detach().cpu().numpy()[0])
                            kintwin_trace["r_wrist_ref_world"].append(ref_body[wrist_idx])
                            kintwin_trace["handle_realized_world"].append(task._virtual_racket_handle_world[env_ids].detach().cpu().numpy()[0])
                            kintwin_trace["handle_target_world"].append(target["handle"].detach().cpu().numpy()[0])
                if bool(cfg.get("save_hand_wrist_validity_traces", False)):
                    target = task._lookup_virtual_racket_target(env_ids, task._current_motion_times(env_ids, next_frame=False))
                    validity_trace["handle_realized_world"].append(task._virtual_racket_handle_world[env_ids].detach().cpu().numpy()[0])
                    validity_trace["handle_target_world"].append(target["handle"].detach().cpu().numpy()[0])
            reset_now = bool(task.reset_buf.detach().cpu().numpy().any())
            if bool(cfg.get("save_hand_wrist_validity_traces", False)) and validity_trace["body_sim_world"]:
                validity_trace["reset_buf_after_step"].append(np.asarray([reset_now], dtype=np.bool_))
            if reset_now:
                terminated = True
                if bool(cfg.get("stop_sequence_on_reset", True)):
                    break

        root_trace_np = np.asarray(root_trace, dtype=np.float32)
        trace_path = child_output.with_suffix(".root_trace.npy")
        np.save(trace_path, root_trace_np)
        kintwin_trace_path = child_output.with_suffix(".kintwin_trace.npz")
        if bool(cfg.get("save_kintwin_tracking_traces", False)) and mode != "body_only" and kintwin_trace["r_hand_sim_world"]:
            np.savez_compressed(
                kintwin_trace_path,
                **{key: np.asarray(value, dtype=np.float32) for key, value in kintwin_trace.items()},
            )
            kintwin_trace_path_value = str(kintwin_trace_path)
        else:
            kintwin_trace_path_value = None
        validity_trace_path = child_output.with_suffix(".validity_trace.npz")
        if bool(cfg.get("save_hand_wrist_validity_traces", False)) and validity_trace["body_sim_world"]:
            arrays: dict[str, np.ndarray] = {}
            for key, value in validity_trace.items():
                if not value:
                    continue
                if key in {"sampled_motion_id", "step_index"}:
                    arrays[key] = np.asarray(value, dtype=np.int64)
                elif key == "reset_buf_after_step":
                    arrays[key] = np.asarray(value, dtype=np.bool_)
                else:
                    arrays[key] = np.asarray(value, dtype=np.float32)
            if validity_body_names is not None:
                arrays["body_names"] = np.asarray(validity_body_names)
            if validity_hand_idx is not None:
                arrays["r_hand_index"] = np.asarray([validity_hand_idx], dtype=np.int64)
            if validity_wrist_idx is not None:
                arrays["r_wrist_index"] = np.asarray([validity_wrist_idx], dtype=np.int64)
            arrays["sequence"] = np.asarray([sequence])
            arrays["mode"] = np.asarray([mode])
            arrays["completed"] = np.asarray([int(len(root_trace_np)) >= planned_steps and not terminated], dtype=np.bool_)
            arrays["terminated"] = np.asarray([terminated], dtype=np.bool_)
            arrays["frames_evaluated"] = np.asarray([int(len(root_trace_np))], dtype=np.int64)
            np.savez_compressed(validity_trace_path, **arrays)
            validity_trace_path_value = str(validity_trace_path)
        else:
            validity_trace_path_value = None
        result = {
            "sequence": sequence,
            "mode": mode,
            "passed": True,
            "frames_planned": planned_steps,
            "frames_evaluated": int(len(root_trace_np)),
            "completed": int(len(root_trace_np)) >= planned_steps and not terminated,
            "terminated": terminated,
            "body_physics_step_executed": int(len(root_trace_np)) > 0,
            "body_actor_forward_executed": True,
            "racket_head_forward_executed": mode in {"virtual_goal_only", "virtual_goal_state"},
            "reward_enabled": bool(getattr(task, "_enable_virtual_racket_reward", False)),
            "body_mpjpe": tensor_stats(body_mpjpe),
            "root_error": tensor_stats(root_errors),
            "virtual_metrics": None
            if mode == "body_only"
            else {
                "handle_error_m": tensor_stats(handle_errors),
                "tip_error_m": tensor_stats(tip_errors),
                "axis_error_deg": tensor_stats(axis_errors),
                "action_magnitude": tensor_stats(action_norms),
                "action_smoothness": tensor_stats(smoothness),
            },
            "hand_body_consistency": None
            if mode == "body_only"
            else {
                "available": bool(hand_error),
                "d_realized_hand": tensor_stats(hand_realized),
                "d_target_hand": tensor_stats(hand_target),
                "e_hand_consistency": tensor_stats(hand_error),
                "unresolved_reason": None if hand_error else "R_Hand mapping/body-name lookup unavailable or no virtual frames evaluated",
            },
            "root_trace_path": str(trace_path),
            "kintwin_trace_path": kintwin_trace_path_value,
            "validity_trace_path": validity_trace_path_value,
            "no_training": True,
            "no_optimizer": True,
            "no_backward": True,
            "no_reward_update": True,
            "no_physical_racket_or_shuttle": True,
            "no_official_phc_rollout_racket_accuracy": True,
        }
        child_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0
    except Exception as exc:
        result = {
            "sequence": sequence,
            "mode": mode,
            "passed": False,
            "error": repr(exc),
            "traceback": traceback.format_exc(),
            "no_training": True,
            "no_optimizer": True,
            "no_backward": True,
        }
        child_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 2
    finally:
        if task is not None:
            destroy_task(task)


def flatten_metric(prefix: str, stats: dict[str, Any] | None) -> dict[str, Any]:
    if not stats:
        return {f"{prefix}_{key}": None for key in ["mean", "p50", "p90", "max"]}
    return {f"{prefix}_{key}": stats.get(key) for key in ["mean", "p50", "p90", "max"]}


def aggregate_child_results(cfg: dict[str, Any], output_dir: Path, sequences: list[str], modes: list[str]) -> dict[str, Any]:
    rows = []
    for mode in modes:
        for sequence in sequences:
            child_json = output_dir / "children" / mode / f"{sequence.replace('/', '__')}.json"
            rows.append(load_json(child_json) if child_json.exists() else {"sequence": sequence, "mode": mode, "passed": False, "error": "missing child result"})

    parity_rows = []
    for sequence in sequences:
        base = next((row for row in rows if row["sequence"] == sequence and row["mode"] == "body_only"), None)
        base_trace = np.load(base["root_trace_path"]) if base and base.get("root_trace_path") and Path(base["root_trace_path"]).exists() else None
        for mode in modes:
            if mode == "body_only":
                continue
            row = next((item for item in rows if item["sequence"] == sequence and item["mode"] == mode), None)
            trace = np.load(row["root_trace_path"]) if row and row.get("root_trace_path") and Path(row["root_trace_path"]).exists() else None
            if base_trace is None or trace is None or len(base_trace) != len(trace):
                parity = {"sequence": sequence, "mode": mode, "available": False, "max_root_trace_abs_diff": None}
            else:
                parity = {"sequence": sequence, "mode": mode, "available": True, "max_root_trace_abs_diff": float(np.max(np.abs(base_trace - trace)))}
            parity_rows.append(parity)

    parity_available = [row["max_root_trace_abs_diff"] for row in parity_rows if row["available"]]
    parity_passed = bool(parity_available) and max(parity_available) <= float(cfg.get("body_parity_root_trace_tolerance", 1e-5))
    summary = {
        "passed": all(bool(row.get("passed")) for row in rows) and parity_passed,
        "evaluation_type": cfg["strict_scope_labels"]["scope"],
        "body_physics_step_executed": any(bool(row.get("body_physics_step_executed")) for row in rows),
        "body_actor_forward_executed": any(bool(row.get("body_actor_forward_executed")) for row in rows),
        "racket_head_forward_executed": any(bool(row.get("racket_head_forward_executed")) for row in rows),
        "confirmed_body_checkpoint": cfg["confirmed_body_checkpoint"],
        "original_body_obs_dim": cfg["confirmed_body_obs_dim"],
        "body_action_dim": cfg["confirmed_body_action_dim"],
        "racket_head_input_dim": cfg["racket_head_input_dim"],
        "racket_head_output_dim": cfg["racket_head_output_dim"],
        "combined_action_dim": cfg["combined_action_dim"],
        "test_groups": cfg["expected_test_groups"],
        "test_clips": len(sequences),
        "modes_completed": modes,
        "body_parity": {"passed": parity_passed, "root_trace_tolerance": cfg.get("body_parity_root_trace_tolerance", 1e-5), "rows": parity_rows},
        "reward_enabled": False,
        "training": False,
        "optimizer": False,
        "backward": False,
        "physical_racket_or_shuttle": False,
        "scope_statement": "no-physics virtual racket tracking under frozen-body PHC rollout; not physical racket accuracy",
    }
    return {"summary": summary, "rows": rows, "parity_rows": parity_rows}


def write_full_evaluation_outputs(cfg: dict[str, Any], output_dir: Path, aggregated: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregated["summary"]
    rows = aggregated["rows"]
    parity_rows = aggregated["parity_rows"]
    (output_dir / "stage1b_evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "stage1b_per_sequence_results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sequence",
            "mode",
            "passed",
            "frames_evaluated",
            "completed",
            "terminated",
            "body_mpjpe_mean",
            "root_error_mean",
            "tip_error_m_mean",
            "axis_error_deg_mean",
            "hand_consistency_mean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            virtual = row.get("virtual_metrics") or {}
            hand = row.get("hand_body_consistency") or {}
            writer.writerow(
                {
                    "sequence": row.get("sequence"),
                    "mode": row.get("mode"),
                    "passed": row.get("passed"),
                    "frames_evaluated": row.get("frames_evaluated"),
                    "completed": row.get("completed"),
                    "terminated": row.get("terminated"),
                    "body_mpjpe_mean": (row.get("body_mpjpe") or {}).get("mean"),
                    "root_error_mean": (row.get("root_error") or {}).get("mean"),
                    "tip_error_m_mean": (virtual.get("tip_error_m") or {}).get("mean"),
                    "axis_error_deg_mean": (virtual.get("axis_error_deg") or {}).get("mean"),
                    "hand_consistency_mean": (hand.get("e_hand_consistency") or {}).get("mean"),
                }
            )
    with (output_dir / "stage1b_body_parity_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sequence", "mode", "available", "max_root_trace_abs_diff"])
        writer.writeheader()
        writer.writerows(parity_rows)
    mode_rows = []
    for mode in cfg["modes"]:
        mode_items = [row for row in rows if row.get("mode") == mode and row.get("passed")]
        mode_rows.append(
            {
                "mode": mode,
                "clips": len(mode_items),
                "frames": int(sum(int(row.get("frames_evaluated", 0) or 0) for row in mode_items)),
                "body_mpjpe_mean": tensor_stats([(row.get("body_mpjpe") or {}).get("mean") for row in mode_items])["mean"],
                "root_error_mean": tensor_stats([(row.get("root_error") or {}).get("mean") for row in mode_items])["mean"],
                "tip_error_m_mean": tensor_stats([((row.get("virtual_metrics") or {}).get("tip_error_m") or {}).get("mean") for row in mode_items])["mean"],
                "axis_error_deg_mean": tensor_stats([((row.get("virtual_metrics") or {}).get("axis_error_deg") or {}).get("mean") for row in mode_items])["mean"],
                "hand_consistency_mean": tensor_stats([((row.get("hand_body_consistency") or {}).get("e_hand_consistency") or {}).get("mean") for row in mode_items])["mean"],
            }
        )
    with (output_dir / "stage1b_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(mode_rows[0].keys()))
        writer.writeheader()
        writer.writerows(mode_rows)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    virtual_modes = [row for row in mode_rows if row["mode"] != "body_only"]
    if virtual_modes:
        labels = [row["mode"] for row in virtual_modes]
        tips = [0.0 if row["tip_error_m_mean"] is None else row["tip_error_m_mean"] for row in virtual_modes]
        axes = [0.0 if row["axis_error_deg_mean"] is None else row["axis_error_deg_mean"] for row in virtual_modes]
        x = np.arange(len(labels))
        plt.figure(figsize=(9, 4))
        plt.bar(x - 0.18, tips, width=0.36, label="tip mean (m)")
        plt.bar(x + 0.18, axes, width=0.36, label="axis mean (deg)")
        plt.xticks(x, labels, rotation=20, ha="right")
        plt.title(cfg["strict_scope_labels"]["plot_label"])
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "tip_axis_mode_comparison.png", dpi=160)
        plt.close()
    report = f"""# Stage 1B Full Held-Out Evaluation Report

Status: {'passed' if summary['passed'] else 'completed_with_blocker_or_failure'}.

Scope: {summary['scope_statement']}

## Contract

- frozen body checkpoint: `{summary['confirmed_body_checkpoint']}`
- original body observation/action: `{summary['original_body_obs_dim']} -> {summary['body_action_dim']}`
- separate racket head input/output: `{summary['racket_head_input_dim']} -> {summary['racket_head_output_dim']}`
- combined env action dim: `{summary['combined_action_dim']}`
- reward enabled: `{summary['reward_enabled']}`
- training / optimizer / backward: `{summary['training']} / {summary['optimizer']} / {summary['backward']}`

## Coverage

- test groups: `{summary['test_groups']}`
- test clips: `{summary['test_clips']}`
- modes: `{summary['modes_completed']}`
- body physics step executed: `{summary['body_physics_step_executed']}`
- body parity passed: `{summary['body_parity']['passed']}`

Hand/body consistency is reported in `stage1b_per_sequence_results.csv`; unresolved mapping or empty values should block reward-tuning interpretation.

This is not physical racket accuracy and not official PHC rollout racket accuracy.
"""
    (output_dir / "stage1b_evaluation_report.md").write_text(report, encoding="utf-8")


def run_full_parent(config_path: Path, cfg: dict[str, Any]) -> int:
    body_ckpt = checkpoint_summary(resolve(cfg["confirmed_body_checkpoint"]))
    if int(body_ckpt["running_mean_dim"]) != int(cfg["confirmed_body_obs_dim"]):
        raise RuntimeError("confirmed body obs dim does not match checkpoint running mean")
    if int(body_ckpt["actor_output_dim"]) != int(cfg["confirmed_body_action_dim"]):
        raise RuntimeError("confirmed body action dim does not match checkpoint actor output")
    head_meta = validate_head_metadata({"stage1a_model_dir": cfg["stage1a_model_dir"]})
    if not head_meta["passed"]:
        raise RuntimeError("Stage 1A head metadata check failed")
    if int(cfg["confirmed_body_action_dim"]) + int(cfg["racket_head_output_dim"]) != int(cfg["combined_action_dim"]):
        raise RuntimeError("combined action dim must equal body action dim + racket action dim")
    sequences = select_heldout_sequences(cfg)
    output_dir = resolve(cfg["output_dir"])
    if bool(cfg.get("clean_output_dir_before_run", True)) and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    child_root = output_dir / "children"
    modes = list(cfg["modes"])
    commands = []
    planned = [(mode, sequence) for mode in modes for sequence in sequences]
    progress_iter = tqdm(planned, total=len(planned), desc="Stage 1B full held-out", unit="child") if tqdm is not None else planned
    for mode, sequence in progress_iter:
        if tqdm is not None:
            progress_iter.set_postfix(mode=mode, sequence=sequence)
        mode_dir = child_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        child_output = mode_dir / f"{sequence.replace('/', '__')}.json"
        log_output = mode_dir / f"{sequence.replace('/', '__')}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--config",
            str(config_path),
            "--child-mode",
            mode,
            "--child-sequence",
            sequence,
            "--child-output",
            str(child_output),
        ]
        commands.append(cmd)
        with log_output.open("w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0 and bool(cfg.get("abort_on_child_failure", True)):
            aggregated = aggregate_child_results(cfg, output_dir, sequences, modes)
            aggregated["summary"]["passed"] = False
            aggregated["summary"]["failed_child_command"] = " ".join(cmd)
            write_full_evaluation_outputs(cfg, output_dir, aggregated)
            return proc.returncode
    aggregated = aggregate_child_results(cfg, output_dir, sequences, modes)
    aggregated["summary"]["child_command_count"] = len(commands)
    write_full_evaluation_outputs(cfg, output_dir, aggregated)
    return 0 if aggregated["summary"]["passed"] else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--child-mode", type=str, default=None)
    parser.add_argument("--child-sequence", type=str, default=None)
    parser.add_argument("--child-output", type=Path, default=None)
    args = parser.parse_args()
    cfg = load_json(args.config)
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    if args.child_mode:
        if not args.child_sequence or args.child_output is None:
            raise ValueError("--child-mode requires --child-sequence and --child-output")
        return run_sequence_child(cfg, mode=args.child_mode, sequence=args.child_sequence, child_output=args.child_output)

    if cfg.get("run_mode") == "full_heldout_evaluation":
        return run_full_parent(args.config, cfg)

    report_dir = resolve(cfg["report_dir"])
    report_dir.mkdir(parents=True, exist_ok=True)
    write_status_audit(report_dir)

    smoke_sequences = list(cfg["smoke_sequences"])[: int(cfg.get("episode_limit", 3))]
    if not (1 <= len(smoke_sequences) <= 3):
        raise ValueError(f"expected 1-3 smoke sequences, got {len(smoke_sequences)}")
    motion_file = make_smoke_motion_file(resolve(cfg["motion_file"]), smoke_sequences)
    body_ckpt = checkpoint_summary(resolve(cfg["original_phc_body_checkpoint"]))
    body_contract = resolve_body_action_contract(cfg, body_ckpt)
    cfg["_resolved_original_mcp_action_dim"] = body_contract["expected_original_mcp_action_dim"]
    obs = actual_obs_preflight(cfg, motion_file, sequence_count=len(smoke_sequences))
    head_meta = validate_head_metadata(cfg)
    summary = {**obs, "body_checkpoint": body_ckpt, "body_action_contract": body_contract, "head_metadata": head_meta}
    expected_non_contiguous = list(cfg.get("expected_non_contiguous_smoke_sequences", []))
    summary["smoke_sequences"] = smoke_sequences
    summary["smoke_sequence_count"] = len(smoke_sequences)
    summary["expected_non_contiguous_smoke_sequences"] = expected_non_contiguous
    summary["non_contiguous_smoke_sequence_included"] = bool(set(smoke_sequences).intersection(expected_non_contiguous))

    expected_obs = int(body_ckpt["running_mean_dim"])
    expected_action = int(body_contract["expected_original_mcp_action_dim"])
    blockers = []
    if obs.get("error"):
        blockers.append(f"actual env initialization failed: {obs.get('error')}")
    if not obs.get("body_policy_obs_all_finite"):
        blockers.append("body policy observation is non-finite or unavailable")
    if obs.get("actual_original_full_obs_dim") is not None and int(obs.get("actual_original_full_obs_dim", -1)) != expected_obs:
        blockers.append(f"original env obs dim {obs.get('actual_original_full_obs_dim')} != checkpoint running_mean dim {expected_obs}")
    if obs.get("actual_original_full_obs_dim") is None and int(obs.get("body_policy_obs_dim_from_virtual_hook", -1)) != expected_obs:
        blockers.append(f"virtual hook body-policy obs dim {obs.get('body_policy_obs_dim_from_virtual_hook')} != checkpoint running_mean dim {expected_obs}")
    if int(obs.get("body_policy_obs_shape", [0, -1])[-1]) != expected_obs:
        blockers.append(f"exported body-policy obs dim {obs.get('body_policy_obs_shape')} != checkpoint input dim {expected_obs}")
    if int(obs.get("actual_original_mcp_action_dim", -1)) != expected_action:
        blockers.append(f"original env MCP action dim {obs.get('actual_original_mcp_action_dim')} != expected {expected_action}")
    if int(body_ckpt.get("actor_output_dim", -1)) != expected_action:
        blockers.append(f"body checkpoint actor output dim {body_ckpt.get('actor_output_dim')} != expected MCP action dim {expected_action}")
    if int(obs.get("combined_action_dim", -1)) != int(body_contract["expected_combined_action_dim"]):
        blockers.append(
            f"combined action dim {obs.get('combined_action_dim')} != expected {body_contract['expected_combined_action_dim']}"
        )
    if float(obs.get("same_task_hook_body_obs_parity_max_abs_diff", float("inf"))) > 1e-6:
        blockers.append("same-task hook body obs parity exceeded tolerance")
    if not bool(head_meta.get("passed")):
        blockers.append("Stage 1A head metadata compatibility failed")
    if "virtual_reward_enabled" in obs and bool(obs.get("virtual_reward_enabled")):
        blockers.append("virtual racket reward is enabled")
    if not summary["non_contiguous_smoke_sequence_included"]:
        blockers.append("no configured non-contiguous smoke sequence was included")
    if int(obs.get("body_action_shape", [0, -1])[-1]) != expected_action:
        blockers.append(f"body actor output shape {obs.get('body_action_shape')} != expected action dim {expected_action}")
    if int(obs.get("racket_action_shape", [0, -1])[-1]) != int(cfg["virtual_racket_action_dim"]):
        blockers.append(f"racket head output shape {obs.get('racket_action_shape')} != expected {cfg['virtual_racket_action_dim']}")
    if int(obs.get("head_input_shape", [0, -1])[-1]) != 15:
        blockers.append(f"racket head input shape {obs.get('head_input_shape')} != expected [*, 15]")
    if not bool(obs.get("body_actor_forward_executed")):
        blockers.append("frozen body actor forward was not executed")
    if not bool(obs.get("racket_head_forward_executed")):
        blockers.append("Stage 1A racket head forward was not executed")
    if not bool(obs.get("wrong_body_action_shape_rejected")):
        blockers.append("wrong body action shape guard did not reject")
    if not bool(obs.get("wrong_racket_action_shape_rejected")):
        blockers.append("wrong racket action shape guard did not reject")
    if not bool(obs.get("augmented_obs_to_body_actor_rejected")):
        blockers.append("augmented virtual observation was not rejected by body actor route")
    if int(obs.get("smoke_step_count", 0)) != int(cfg.get("smoke_step_count", 0)):
        blockers.append(f"smoke step count {obs.get('smoke_step_count')} != configured {cfg.get('smoke_step_count', 0)}")

    summary["passed"] = len(blockers) == 0
    summary["blockers"] = blockers
    summary["blocker"] = "; ".join(blockers) if blockers else None
    summary["checkpoint_routes"] = {
        "original_body_checkpoint_route": "original body-policy obs -> frozen MCP body action only",
        "augmented_obs_to_original_checkpoint": False,
        "original_checkpoint_outputs_combined_action": False,
        "stage1a_head_route": "Live Goal V2 + realized state V2 -> virtual action only",
    }
    summary["scope"] = cfg["strict_scope_labels"]
    summary["virtual_contract_context"] = quick_virtual_reference_metrics(cfg, report_dir)
    write_preflight_reports(report_dir, summary)

    if not summary["passed"]:
        write_blocked_outputs(report_dir, summary)
        return 2

    write_smoke_success_outputs(report_dir, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
