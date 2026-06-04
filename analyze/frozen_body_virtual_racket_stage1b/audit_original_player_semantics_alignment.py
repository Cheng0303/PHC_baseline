#!/usr/bin/env python3
"""Stage 1E audit: original PHC player semantics vs Stage 1B custom evaluator.

CPU-only static/provenance audit. This script reads source/config/report files
and writes a semantics diff plus the next body-only parity plan. It does not
run Isaac Gym, train, optimize, backpropagate, tune rewards, or modify
checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
PHC_BASELINE = WORKSPACE_ROOT / "phc_baseline"
PHC_ROOT = WORKSPACE_ROOT / "humenv" / "data_preparation" / "PHC"
REPORT_DIR = PHC_BASELINE / "reports" / "racket_calibration" / "frozen_body_head_integration"

STAGE1B_EVALUATOR = PHC_BASELINE / "analyze" / "frozen_body_virtual_racket_stage1b" / "evaluate.py"
STAGE1B_CONFIG = PHC_BASELINE / "configs" / "frozen_body_virtual_racket_stage1b" / "eval_full_heldout_config.json"
STAGE1D_SUMMARY = REPORT_DIR / "body_rollout_mismatch_stage1d_summary.json"
STAGE1D_REPORT = REPORT_DIR / "body_rollout_mismatch_stage1d_report.md"
USER_REPRO_JSON = REPORT_DIR / "stage1d_user_run_original_baseline_repro_full.json"
ORIGINAL_SCRIPT = PHC_BASELINE / "run_phc_badminton_dataset_metrics.sh"

IM_AMP_PLAYER = PHC_ROOT / "phc" / "learning" / "im_amp_players.py"
COMMON_PLAYER = PHC_ROOT / "phc" / "learning" / "common_player.py"
RL_GAMES_PLAYER = PHC_BASELINE / "envs" / "phc_isaac" / "lib" / "python3.8" / "site-packages" / "rl_games" / "algos_torch" / "players.py"
HUMANOID_IM_MCP = PHC_ROOT / "phc" / "env" / "tasks" / "humanoid_im_mcp.py"
HUMANOID_IM = PHC_ROOT / "phc" / "env" / "tasks" / "humanoid_im.py"
BASE_ENV_YAML = PHC_ROOT / "phc" / "data" / "cfg" / "env" / "env_im_getup_mcp.yaml"
VIRTUAL_ENV_YAML = PHC_ROOT / "phc" / "data" / "cfg" / "env" / "env_im_getup_mcp_virtual_racket_stage1b.yaml"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def contains(path: Path, needle: str) -> bool:
    return needle in read_text(path)


def build_audit() -> dict[str, Any]:
    cfg = load_json(STAGE1B_CONFIG)
    stage1d = load_json(STAGE1D_SUMMARY)
    repro = load_json(USER_REPRO_JSON)

    original_summary = repro.get("summary", {})
    stage1d_repro = stage1d.get("user_run_original_baseline_reproduction", {})
    same_run = stage1d.get("same_run_body_only", {})

    diff_rows = [
        {
            "topic": "policy inference route",
            "original_compact_metrics_path": "RL Games IMAMPPlayerContinuous.get_action() builds/restores the full model, preprocesses observations with running_mean_std, uses deterministic mus, then clips/rescales actions before env.step().",
            "stage1b_custom_evaluator": "evaluate.py manually reconstructs FrozenPHCBodyActor from actor_mlp + mu, normalizes obs by checkpoint stats, and returns raw actor output through run_body_actor().",
            "risk": "high",
            "evidence": [
                "rl_games/algos_torch/players.py PpoPlayerContinuous.get_action clamps [-1,1] and rescale_actions(actions_low, actions_high, action).",
                "evaluate.py FrozenPHCBodyActor only wraps actor_mlp + mu and run_body_actor returns actor(normalized) without clip/rescale.",
            ],
        },
        {
            "topic": "env/game loop",
            "original_compact_metrics_path": "Original player calls env_reset(), get_action(), env_step(), _post_step(), and forward_motion_samples() while collecting compact metrics until all motions are covered.",
            "stage1b_custom_evaluator": "Custom evaluator creates one-motion temporary pkl per sequence, directly calls task.reset(), then loops task.step() and stops the sequence on reset.",
            "risk": "high",
            "evidence": [
                "im_amp_players.py updates compact metrics, calls humanoid_env.forward_motion_samples(), and manages terminate_state over batched motions.",
                "evaluate.py make_smoke_motion_file() plus run_sequence_child() uses env.num_envs=1 per child and stop_sequence_on_reset.",
            ],
        },
        {
            "topic": "MCP action semantics",
            "original_compact_metrics_path": "Original body actor outputs MCP weights [3]; HumanoidImMCP.step(weights) then loads PNN/primitive actors and converts weights into humanoid PD actions.",
            "stage1b_custom_evaluator": "Custom evaluator also passes [3] into task.step(), but those [3] may not be numerically equivalent to the player output because inference preprocessing/action postprocessing differs.",
            "risk": "high",
            "evidence": [
                "humanoid_im_mcp.py step(weights) clamps obs, combines primitive actions by weights, then pre_physics_step(actions).",
                "Stage 1D compact-metrics reproduction confirms the checkpoint family is valid outside the custom evaluator.",
            ],
        },
        {
            "topic": "reset/init semantics",
            "original_compact_metrics_path": "im_eval player sets recovery/fall probabilities to 0, zero_out_far False, cycle_motion False, and may adjust reset bodies to eval track bodies; the player owns reset/done scheduling.",
            "stage1b_custom_evaluator": "Config forces stateInit=Start, recovery/fall=0, getup_schedule=False, force_deterministic_reset_after_init=True, and stop_sequence_on_reset=True.",
            "risk": "medium_high",
            "evidence": [
                "im_amp_players.py __init__ mutates humanoid_env flags under flags.im_eval.",
                "eval_full_heldout_config.json explicitly overrides state/init/reset flags.",
            ],
        },
        {
            "topic": "motion batching",
            "original_compact_metrics_path": "Original run evaluates the full motion file with env.num_envs=128 and PHC motion-lib batching/current motion ids.",
            "stage1b_custom_evaluator": "Custom evaluator slices one motion into a temporary pkl, so sampled_motion_id is 0 in each child and env.num_envs=1.",
            "risk": "medium",
            "evidence": [
                "Stage 1D traces show sampled_motion_id 0 after one-motion temporary pkl files.",
                "run_phc_badminton_dataset_metrics.sh defaults NUM_ENVS=128.",
            ],
        },
        {
            "topic": "metric aggregation",
            "original_compact_metrics_path": "Compact metrics count up to frame_count-1, track termination_frame, and aggregate per-sequence means in _dump_compact_metrics().",
            "stage1b_custom_evaluator": "Custom evaluator samples metrics after each task.step() and reports frame-weighted validity-trace values plus stop-on-reset sequence summaries.",
            "risk": "medium",
            "evidence": [
                "im_amp_players.py _update_compact_metrics() and _dump_compact_metrics().",
                "evaluate.py compute_body_tracking_metrics() and child JSON aggregation.",
            ],
        },
    ]

    ranked_suspects = [
        {
            "rank": 1,
            "suspect": "Manual frozen actor forward is not RL-Games-player-compatible.",
            "why": "It bypasses the restored model wrapper, RunningMeanStd module path, deterministic action dict, action clipping, and action rescale. This can change the 3D MCP weights fed to HumanoidImMCP.step().",
        },
        {
            "rank": 2,
            "suspect": "Custom per-sequence reset/game loop is not the original im_eval player loop.",
            "why": "Original player manages env_reset/done/forward_motion_samples and compact metric bookkeeping over batched motions. Stage 1B manually resets and stops on reset.",
        },
        {
            "rank": 3,
            "suspect": "One-motion temporary pkl + env.num_envs=1 may alter motion-lib batching semantics.",
            "why": "The original reproduction uses the full motion file and 128 envs; Stage 1B children all see sampled_motion_id 0.",
        },
        {
            "rank": 4,
            "suspect": "Reset/init overrides differ from player-driven im_eval semantics.",
            "why": "Stage 1B forces deterministic reset and Start init; original im_eval player mutates a different set of flags in IMAMPPlayerContinuous.",
        },
        {
            "rank": 5,
            "suspect": "Metric sampling/termination aggregation differs.",
            "why": "This can change reported metrics, but cannot alone explain large physical divergence if the same state trajectory were used.",
        },
    ]

    user_command = f"""cd {WORKSPACE_ROOT}
nvidia-smi
phc_baseline/envs/phc_isaac/bin/python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
print("torch_cuda", torch.version.cuda)
PY
phc_baseline/envs/phc_isaac/bin/python -m py_compile \\
  phc_baseline/analyze/frozen_body_virtual_racket_stage1b/audit_original_player_semantics_alignment.py \\
  phc_baseline/analyze/frozen_body_virtual_racket_stage1b/audit_body_rollout_mismatch_stage1d.py
bash phc_baseline/analyze/frozen_body_virtual_racket_stage1b/run_user_stage1d_body_only_repro.sh
python3 -m json.tool \\
  phc_baseline/reports/racket_calibration/frozen_body_head_integration/stage1d_user_run_original_baseline_repro_full.json >/dev/null
sed -n '1,180p' \\
  phc_baseline/reports/racket_calibration/frozen_body_head_integration/body_player_semantics_alignment_diff.md
"""

    return {
        "status": "completed_cpu_only_static_audit",
        "no_training": True,
        "no_optimizer": True,
        "no_backward": True,
        "no_ppo_rl": True,
        "no_reward_update": True,
        "no_gpu_eval_run_by_codex": True,
        "confirmed_contracts": {
            "body_checkpoint": "humenv/data_preparation/PHC/output/HumanoidIm/phc_comp_3/Humanoid.pth",
            "body_obs_dim": 934,
            "body_action_dim": 3,
            "racket_head_input_dim": 15,
            "racket_head_output_dim": 6,
            "combined_action_dim": 9,
        },
        "stage1d_reproduction": {
            "available": bool(original_summary),
            "full_dataset_mpjpe": original_summary.get("dataset_mean_mpjpe"),
            "full_dataset_root_error": original_summary.get("dataset_mean_root_error"),
            "matched_heldout_clip_count": stage1d_repro.get("matched_stage1b_sequences"),
            "matched_heldout_completed": stage1d_repro.get("completed_count_for_matched"),
            "matched_heldout_mpjpe": stage1d_repro.get("mean_mpjpe_for_matched"),
            "matched_heldout_root_error": stage1d_repro.get("mean_root_error_for_matched"),
        },
        "stage1b_custom_body_metrics": {
            "world_mpjpe": (same_run.get("frame_weighted") or {}).get("world_mpjpe_mean"),
            "root_error": (same_run.get("frame_weighted") or {}).get("root_error_mean"),
            "root_aligned_mpjpe": (same_run.get("frame_weighted") or {}).get("root_aligned_mpjpe_mean"),
            "completed": same_run.get("completed_count"),
            "terminated": same_run.get("terminated_count"),
        },
        "config_runtime_diff": diff_rows,
        "ranked_suspected_divergence_sources": ranked_suspects,
        "outcome": {
            "classification": "E4",
            "label": "custom evaluator has identified semantic bug/mismatch",
            "reason": "Original compact-metrics player path reproduces low body metrics, while Stage 1B manual actor/eval path diverges on the same clips.",
        },
        "body_only_minimal_reproduction_plan": [
            "First keep the original compact-metrics reproduction as the reference oracle for body-only parity.",
            "Implement a body-only Stage 1B parity path that uses the RL Games player/model get_action semantics, not FrozenPHCBodyActor raw actor_mlp+mu.",
            "Run the same 40 held-out clips without racket branch and require MPJPE/root close to 0.068215/0.062302 m and completed 40/40.",
            "Only after body-only parity passes, add a virtual-racket branch at the env boundary while preserving the original body player action route.",
        ],
        "stage1b_integration_path_decision": {
            "recommended": "Use an RL-Games-player-compatible wrapper/integration path for body actions.",
            "architecture": [
                "original player/full restored model: original body obs [934] -> MCP weights [3]",
                "virtual task/head route: Live Goal V2 [9] + realized feedback [6] -> virtual action [6]",
                "env boundary: [3] + [6] -> [9]",
            ],
            "do_not_do_yet": [
                "Do not run full racket integration until body-only player-compatible parity is proven.",
                "Do not add reward/coupling objective.",
                "Do not change Stage 1A head checkpoint.",
            ],
        },
        "user_run_body_only_reference_command": user_command,
        "source_checks": {
            "rl_games_player_get_action_has_clip_rescale": contains(RL_GAMES_PLAYER, "torch.clamp(current_action, -1.0, 1.0)") and contains(RL_GAMES_PLAYER, "rescale_actions"),
            "im_amp_player_compact_metrics": contains(IM_AMP_PLAYER, "_update_compact_metrics") and contains(IM_AMP_PLAYER, "PHC_COMPACT_METRICS_PATH"),
            "humanoid_im_mcp_step_combines_pnn_actions": contains(HUMANOID_IM_MCP, "torch.sum(weights[:, :, None] * x_all, dim=1)"),
            "stage1b_manual_actor_forward_present": contains(STAGE1B_EVALUATOR, "class FrozenPHCBodyActor") and contains(STAGE1B_EVALUATOR, "return actor(normalized)"),
        },
    }


def write_markdown(audit: dict[str, Any]) -> None:
    lines = [
        "# Body Player Semantics Alignment Diff",
        "",
        "Scope: CPU-only static/provenance audit. No GPU evaluation, training, optimizer, backward pass, PPO/RL, reward update, checkpoint modification, racket-head change, physical racket, shuttle, collision, or hitting reward was run.",
        "",
        "## Outcome",
        "",
        f"- classification: `{audit['outcome']['classification']} - {audit['outcome']['label']}`",
        f"- reason: {audit['outcome']['reason']}",
        "",
        "## Reproduction Evidence",
        "",
        f"- original compact-metrics full dataset MPJPE/root: `{audit['stage1d_reproduction']['full_dataset_mpjpe']}` / `{audit['stage1d_reproduction']['full_dataset_root_error']}`",
        f"- matched Stage 1B held-out clips: `{audit['stage1d_reproduction']['matched_heldout_clip_count']}`",
        f"- matched held-out completed: `{audit['stage1d_reproduction']['matched_heldout_completed']}`",
        f"- matched held-out MPJPE/root: `{audit['stage1d_reproduction']['matched_heldout_mpjpe']}` / `{audit['stage1d_reproduction']['matched_heldout_root_error']}`",
        f"- Stage 1B custom body MPJPE/root: `{audit['stage1b_custom_body_metrics']['world_mpjpe']}` / `{audit['stage1b_custom_body_metrics']['root_error']}`",
        "",
        "## Config / Runtime Diff",
        "",
    ]
    for row in audit["config_runtime_diff"]:
        lines.extend(
            [
                f"### {row['topic']}",
                "",
                f"- original compact-metrics path: {row['original_compact_metrics_path']}",
                f"- Stage 1B custom evaluator: {row['stage1b_custom_evaluator']}",
                f"- risk: `{row['risk']}`",
                "- evidence:",
            ]
        )
        lines.extend([f"  - {item}" for item in row["evidence"]])
        lines.append("")

    lines.extend(["## Ranked Suspected Divergence Sources", ""])
    for item in audit["ranked_suspected_divergence_sources"]:
        lines.append(f"{item['rank']}. {item['suspect']} {item['why']}")
    lines.append("")

    lines.extend(["## Body-Only Minimal Reproduction Plan", ""])
    lines.extend([f"- {item}" for item in audit["body_only_minimal_reproduction_plan"]])
    lines.append("")

    decision = audit["stage1b_integration_path_decision"]
    lines.extend(
        [
            "## Stage 1B Integration Path Decision",
            "",
            f"- recommended: {decision['recommended']}",
            "- architecture:",
        ]
    )
    lines.extend([f"  - {item}" for item in decision["architecture"]])
    lines.append("- blocked until:")
    lines.append("  - body-only player-compatible parity on the same 40 clips approaches the compact-metrics result.")
    lines.append("- do not do yet:")
    lines.extend([f"  - {item}" for item in decision["do_not_do_yet"]])
    lines.append("")

    lines.extend(
        [
            "## User-Run Body-Only Reference Command",
            "",
            "This command re-runs the known-good original compact-metrics path. It is a reference check, not a Stage 1B custom evaluator fix.",
            "",
            "```bash",
            audit["user_run_body_only_reference_command"].strip(),
            "```",
            "",
        ]
    )

    (REPORT_DIR / "body_player_semantics_alignment_diff.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    audit = build_audit()
    write_json(REPORT_DIR / "body_player_semantics_alignment_diff.json", audit)
    write_markdown(audit)
    print(json.dumps({"classification": audit["outcome"]["classification"], "label": audit["outcome"]["label"]}, indent=2))


if __name__ == "__main__":
    main()
