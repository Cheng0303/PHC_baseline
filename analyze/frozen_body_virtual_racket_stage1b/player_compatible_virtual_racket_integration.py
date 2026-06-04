#!/usr/bin/env python3
"""Stage 1G player-compatible virtual-racket integration hook.

This module is intentionally scoped as an action-boundary hook. The PHC body
action must be supplied by the original RL Games player route. This file never
reconstructs the PHC body actor and never feeds augmented virtual-racket
observations to the original body policy.

Smoke mode is CPU/static: it validates the Stage 1A head metadata/checkpoint,
computes a deterministic dummy racket action from a [15] hook input, and proves
that packing preserves an externally supplied body action [3] exactly while
forming [3]+[6]=[9]. It does not claim full Isaac/player rollout parity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PHC_BASELINE = WORKSPACE_ROOT / "phc_baseline"
DEFAULT_CONFIG = PHC_BASELINE / "configs" / "frozen_body_virtual_racket_stage1b" / "player_compatible_virtual_racket_integration_config.json"


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int):
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


def resolve(path_like: str | Path) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else WORKSPACE_ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_head_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    global_meta = load_json(resolve(cfg["checkpoint_metadata"]))
    checkpoint = torch.load(resolve(cfg["racket_head_checkpoint"]), map_location="cpu")
    ckpt_meta = checkpoint.get("metadata", {})
    goal_only_checkpoint = torch.load(resolve(cfg["goal_only_head_checkpoint"]), map_location="cpu") if cfg.get("goal_only_head_checkpoint") else None
    goal_only_ckpt_meta = goal_only_checkpoint.get("metadata", {}) if goal_only_checkpoint else {}
    primary = global_meta["models"]["goal_state"]
    goal_only = global_meta["models"].get("goal_only", {})

    checks = {
        "global_goal_version": global_meta.get("goal_version") == cfg["goal_version"],
        "global_state_version": global_meta.get("state_version") == cfg["state_version"],
        "global_action_version": global_meta.get("action_version") == cfg["action_version"],
        "body_path_frozen": global_meta.get("body_path_frozen") is True,
        "no_physics_virtual_head_only": global_meta.get("no_physics_virtual_head_only") is True,
        "primary_input_dim": int(primary.get("input_dim")) == int(cfg["racket_head_input_dim"]),
        "primary_output_dim": int(primary.get("output_dim")) == int(cfg["racket_head_output_dim"]),
        "checkpoint_input_dim": int(ckpt_meta.get("input_dim")) == int(cfg["racket_head_input_dim"]),
        "checkpoint_output_dim": int(ckpt_meta.get("output_dim")) == int(cfg["racket_head_output_dim"]),
        "checkpoint_model_name": ckpt_meta.get("model_name") == "goal_state",
    }
    if goal_only_checkpoint:
        checks.update(
            {
                "goal_only_global_input_dim": int(goal_only.get("input_dim")) == int(cfg.get("goal_only_head_input_dim", 9)),
                "goal_only_global_output_dim": int(goal_only.get("output_dim")) == int(cfg["racket_head_output_dim"]),
                "goal_only_checkpoint_input_dim": int(goal_only_ckpt_meta.get("input_dim")) == int(cfg.get("goal_only_head_input_dim", 9)),
                "goal_only_checkpoint_output_dim": int(goal_only_ckpt_meta.get("output_dim")) == int(cfg["racket_head_output_dim"]),
                "goal_only_checkpoint_model_name": goal_only_ckpt_meta.get("model_name") == "goal_only",
            }
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "global_metadata": {
            "goal_version": global_meta.get("goal_version"),
            "state_version": global_meta.get("state_version"),
            "action_version": global_meta.get("action_version"),
            "body_path_frozen": global_meta.get("body_path_frozen"),
            "no_physics_virtual_head_only": global_meta.get("no_physics_virtual_head_only"),
        },
        "checkpoint_metadata": ckpt_meta,
        "goal_only_checkpoint_metadata": goal_only_ckpt_meta,
        "normalization_available": all(k in global_meta.get("normalization", {}) for k in ["x_mean", "x_std", "y_mean", "y_std"]),
    }


class PlayerCompatibleVirtualRacketHook:
    """Side branch that appends racket action to externally supplied player action."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.body_dim = int(cfg["confirmed_body_action_dim"])
        self.head_input_dim = int(cfg["racket_head_input_dim"])
        self.racket_dim = int(cfg["racket_head_output_dim"])
        self.combined_dim = int(cfg["combined_action_dim"])
        if self.body_dim + self.racket_dim != self.combined_dim:
            raise ValueError("body_dim + racket_dim must equal combined_dim")
        global_meta = load_json(resolve(cfg["checkpoint_metadata"]))
        ckpt = torch.load(resolve(cfg["racket_head_checkpoint"]), map_location="cpu")
        ckpt_meta = ckpt["metadata"]
        self.model = self._load_model_from_checkpoint(ckpt)
        self.goal_only_model = None
        self.goal_only_input_dim = int(cfg.get("goal_only_head_input_dim", 9))
        if cfg.get("goal_only_head_checkpoint"):
            goal_only_ckpt = torch.load(resolve(cfg["goal_only_head_checkpoint"]), map_location="cpu")
            self.goal_only_model = self._load_model_from_checkpoint(goal_only_ckpt)
        norm = global_meta["normalization"]
        self.x_mean = torch.tensor(norm["x_mean"], dtype=torch.float32)
        self.x_std = torch.tensor(norm["x_std"], dtype=torch.float32)
        self.y_mean = torch.tensor(norm["y_mean"], dtype=torch.float32)
        self.y_std = torch.tensor(norm["y_std"], dtype=torch.float32)

    @staticmethod
    def _load_model_from_checkpoint(checkpoint: dict[str, Any]) -> MLPHead:
        meta = checkpoint["metadata"]
        model = MLPHead(int(meta["input_dim"]), list(meta["hidden_dims"]), int(meta["output_dim"]))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

    def predict_racket_action(self, head_input: torch.Tensor, *, model_name: str = "goal_state") -> torch.Tensor:
        if model_name == "goal_state":
            expected_dim = self.head_input_dim
            model = self.model
        elif model_name == "goal_only":
            if self.goal_only_model is None:
                raise RuntimeError("goal_only model requested but goal_only_head_checkpoint is not configured")
            expected_dim = self.goal_only_input_dim
            model = self.goal_only_model
        else:
            raise RuntimeError(f"unknown racket head model_name {model_name!r}")
        if head_input.shape[-1] != expected_dim:
            raise RuntimeError(f"expected {model_name} racket head input dim {expected_dim}, got {head_input.shape[-1]}")
        # Stage 1A stores one train-only normalization vector for the full
        # [goal,state] input. Model A uses the goal slice [0:9]; Model B uses all
        # 15 dims. Output normalization is shared.
        x_mean = self.x_mean[:expected_dim].to(head_input.device)
        x_std = self.x_std[:expected_dim].to(head_input.device)
        x = (head_input.float() - x_mean) / x_std
        with torch.no_grad():
            y_norm = model.to(head_input.device)(x)
        return y_norm * self.y_std.to(head_input.device) + self.y_mean.to(head_input.device)

    def pack(self, player_body_action: torch.Tensor, racket_action: torch.Tensor) -> torch.Tensor:
        if player_body_action.shape[-1] != self.body_dim:
            raise RuntimeError(f"expected externally supplied player body action dim {self.body_dim}, got {player_body_action.shape[-1]}")
        if racket_action.shape[-1] != self.racket_dim:
            raise RuntimeError(f"expected racket action dim {self.racket_dim}, got {racket_action.shape[-1]}")
        combined = torch.cat([player_body_action, racket_action], dim=-1)
        if combined.shape[-1] != self.combined_dim:
            raise RuntimeError(f"expected combined action dim {self.combined_dim}, got {combined.shape[-1]}")
        return combined


def smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    meta = validate_head_metadata(cfg)
    hook = PlayerCompatibleVirtualRacketHook(cfg)
    n = len(cfg["smoke_sequences"])
    body = torch.linspace(-0.2, 0.2, steps=n * int(cfg["confirmed_body_action_dim"]), dtype=torch.float32).reshape(n, -1)
    head_input = torch.linspace(-0.5, 0.5, steps=n * int(cfg["racket_head_input_dim"]), dtype=torch.float32).reshape(n, -1)
    racket = hook.predict_racket_action(head_input)
    combined = hook.pack(body, racket)
    preserve_diff = torch.max(torch.abs(combined[:, : hook.body_dim] - body)).item()

    guard_results: dict[str, bool] = {}
    try:
        hook.pack(torch.zeros(n, hook.body_dim + 1), racket)
        guard_results["wrong_body_action_shape_rejected"] = False
    except RuntimeError:
        guard_results["wrong_body_action_shape_rejected"] = True
    try:
        hook.pack(body, torch.zeros(n, hook.racket_dim + 1))
        guard_results["wrong_racket_action_shape_rejected"] = False
    except RuntimeError:
        guard_results["wrong_racket_action_shape_rejected"] = True
    try:
        hook.predict_racket_action(torch.zeros(n, hook.head_input_dim + 1))
        guard_results["wrong_head_input_shape_rejected"] = False
    except RuntimeError:
        guard_results["wrong_head_input_shape_rejected"] = True

    passed = (
        meta["passed"]
        and tuple(body.shape) == (n, int(cfg["confirmed_body_action_dim"]))
        and tuple(head_input.shape) == (n, int(cfg["racket_head_input_dim"]))
        and tuple(racket.shape) == (n, int(cfg["racket_head_output_dim"]))
        and tuple(combined.shape) == (n, int(cfg["combined_action_dim"]))
        and preserve_diff == 0.0
        and all(guard_results.values())
    )
    return {
        "passed": passed,
        "status": "passed_static_action_boundary_hook_smoke" if passed else "failed_static_action_boundary_hook_smoke",
        "coverage": "CPU/static hook smoke only; original RL Games player rollout is not executed by this script.",
        "original_player_route_required": True,
        "original_player_route_executed": False,
        "manual_actor_route_used": False,
        "augmented_obs_to_body_actor": False,
        "body_action_source": "external_original_rl_games_player_get_action_output_required",
        "body_action_shape": list(body.shape),
        "racket_head_input_shape": list(head_input.shape),
        "racket_action_shape": list(racket.shape),
        "combined_action_shape": list(combined.shape),
        "body_action_preservation_max_abs_diff": preserve_diff,
        "head_metadata": meta,
        "guard_results": guard_results,
        "smoke_sequences": cfg["smoke_sequences"],
        "expected_non_contiguous_smoke_sequences": cfg["expected_non_contiguous_smoke_sequences"],
        "reward_enabled": False,
        "physical_racket_or_shuttle": False,
        "training": False,
        "optimizer": False,
        "backward": False,
        "next_gate": "run the prepared user GPU smoke inside the original RL Games player loop, then prepare full held-out evaluation only if that smoke passes",
    }


def summarize_player_loop_smoke(cfg: dict[str, Any]) -> dict[str, Any]:
    out_dir = resolve(cfg["output_dir"])
    compact_path = out_dir / cfg["player_loop_smoke_compact_metrics_json"]
    proxy_path = out_dir / cfg["player_loop_smoke_proxy_records_json"]
    static_summary_path = out_dir / cfg["smoke_summary_json"]
    static_summary = load_json(static_summary_path) if static_summary_path.exists() else smoke(cfg)
    compact = load_json(compact_path) if compact_path.exists() else {}
    proxy = load_json(proxy_path) if proxy_path.exists() else {}
    per_sequence = compact.get("per_sequence", [])
    records = proxy.get("records", [])
    completed = sum(1 for row in per_sequence if row.get("completed"))
    mean_mpjpe = None
    mean_root = None
    if per_sequence:
        mean_mpjpe = sum(float(row["mean_mpjpe"]) for row in per_sequence) / len(per_sequence)
        mean_root = sum(float(row["mean_root_error"]) for row in per_sequence) / len(per_sequence)
    preserve = [float(r.get("body_action_preservation_max_abs_diff", 0.0)) for r in records]
    player_summary = {
        **static_summary,
        "status": "passed_player_loop_proxy_smoke" if records and per_sequence and max(preserve or [0.0]) == 0.0 else "player_loop_proxy_smoke_incomplete_or_failed",
        "passed": bool(records and per_sequence and max(preserve or [0.0]) == 0.0),
        "coverage": "GPU/user-run original RL Games player loop through Stage 1G proxy" if records else "player-loop records missing",
        "original_player_route_executed": bool(records),
        "proxy_records_path": str(proxy_path.relative_to(WORKSPACE_ROOT)),
        "compact_metrics_path": str(compact_path.relative_to(WORKSPACE_ROOT)),
        "virtual_steps": int(proxy.get("virtual_steps", 0)),
        "body_action_preservation_max_abs_diff": max(preserve) if preserve else None,
        "compact_metrics": {
            "sequence_count": len(per_sequence),
            "completed_count": completed,
            "mean_mpjpe_m": mean_mpjpe,
            "mean_root_error_m": mean_root,
            "summary": compact.get("summary", {}),
        },
    }
    return player_summary


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Stage 1G Player-Compatible Virtual-Racket Hook Smoke",
        "",
        "Scope: action-boundary hook smoke. This validates that an externally supplied original-player body action `[3]` can be preserved exactly while appending a Stage 1A Model B virtual-racket action `[6]` into `[9]`. It does not train, tune rewards, run PPO/RL, modify checkpoints, use a physical racket, or claim official PHC rollout racket accuracy.",
        "",
        "## Status",
        "",
        f"- passed: `{summary['passed']}`",
        f"- status: `{summary['status']}`",
        f"- coverage: {summary['coverage']}",
        f"- original player route required: `{summary['original_player_route_required']}`",
        f"- original player route executed here: `{summary['original_player_route_executed']}`",
        f"- manual actor route used: `{summary['manual_actor_route_used']}`",
        f"- augmented obs to body actor: `{summary['augmented_obs_to_body_actor']}`",
        "",
        "## Dimensions",
        "",
        f"- body action shape: `{summary['body_action_shape']}`",
        f"- racket head input shape: `{summary['racket_head_input_shape']}`",
        f"- racket action shape: `{summary['racket_action_shape']}`",
        f"- combined action shape: `{summary['combined_action_shape']}`",
        f"- body action preservation max abs diff: `{summary['body_action_preservation_max_abs_diff']}`",
        "",
        "## Guards",
        "",
    ]
    lines.extend([f"- {k}: `{v}`" for k, v in summary["guard_results"].items()])
    lines.extend(
        [
            "",
            "## Hook Design",
            "",
            "1. The original RL Games player remains responsible for `body obs [934] -> body action [3]`.",
            "2. The side branch calls the env helper for Live Goal V2 `[9]` plus realized state `[6]` in the future GPU integration.",
            "3. Stage 1A Model B maps `[15] -> [6]` after train-only normalization/denormalization from its checkpoint metadata.",
            "4. The hook packs `[3] + [6] -> [9]` and asserts the first three values are exactly the player body action.",
            "5. The virtual-racket env boundary must split `[9]`, route `[3]` to the MCP body path, and update the no-physics virtual racket state from `[6]` with reward disabled.",
            "",
            "## Next Gate",
            "",
        "If `original_player_route_executed` is false, run the user GPU smoke script. If it is true and body parity is acceptable, the next step is a full held-out player-compatible integration evaluation.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_full_eval_command(path: Path, cfg_path: Path) -> None:
    command = f"""# Run only after Stage 1G player-loop GPU smoke passes.
cd {WORKSPACE_ROOT}
nvidia-smi
phc_baseline/envs/phc_isaac/bin/python -m py_compile \\
  phc_baseline/analyze/frozen_body_virtual_racket_stage1b/player_compatible_virtual_racket_integration.py
python3 -m json.tool \\
  {cfg_path.relative_to(WORKSPACE_ROOT)} >/dev/null

# Placeholder: full held-out player-compatible integration must be implemented
# as a player-loop extension before running. Do not use the old custom
# evaluate.py manual actor route for the final Stage 1G evaluation.
"""
    path.write_text("# Stage 1G Full Evaluation Command\n\n```bash\n" + command + "```\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-mode", choices=["smoke", "summarize-player-loop-smoke"], default="smoke")
    args = parser.parse_args()
    cfg_path = resolve(args.config)
    cfg = load_json(cfg_path)
    out_dir = resolve(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = smoke(cfg) if args.run_mode == "smoke" else summarize_player_loop_smoke(cfg)
    summary_path = out_dir / cfg["smoke_summary_json"]
    report_path = out_dir / cfg["smoke_report_md"]
    full_cmd_path = out_dir / cfg["full_eval_command_md"]
    write_json(summary_path, summary)
    write_report(report_path, summary)
    write_full_eval_command(full_cmd_path, cfg_path)
    print(json.dumps({k: summary[k] for k in ["passed", "status", "coverage", "body_action_preservation_max_abs_diff"]}, indent=2))


if __name__ == "__main__":
    main()
