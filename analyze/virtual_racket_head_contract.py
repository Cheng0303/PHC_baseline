#!/usr/bin/env python3
"""Separate virtual racket head interface contract.

This is a shape/metadata scaffold only. It performs no optimizer update,
backward pass, training loop, checkpoint save, or policy performance eval.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_OUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "phc_baseline"
    / "reports"
    / "racket_calibration"
    / "controller_interface"
)


@dataclass(frozen=True)
class VirtualRacketHeadContract:
    goal_dim: int = 9
    realized_state_dim: int = 6
    input_dim: int = 15
    action_dim: int = 6
    goal_version: str = "v2_sim_root_projected_world_target"
    realized_state_version: str = "v2_sim_root_projected_world_state"
    action_version: str = "v2_heading_local_velocity"


def pack_head_input(live_goal_v2: np.ndarray, realized_state_v2: np.ndarray) -> np.ndarray:
    live_goal_v2 = np.asarray(live_goal_v2, dtype=np.float32)
    realized_state_v2 = np.asarray(realized_state_v2, dtype=np.float32)
    if live_goal_v2.shape[-1] != 9:
        raise ValueError(f"expected live goal dim 9, got {live_goal_v2.shape[-1]}")
    if realized_state_v2.shape[-1] != 6:
        raise ValueError(f"expected realized state dim 6, got {realized_state_v2.shape[-1]}")
    packed = np.concatenate([live_goal_v2, realized_state_v2], axis=-1)
    if not np.isfinite(packed).all():
        raise ValueError("non-finite virtual racket head input")
    return packed


def validate_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 6:
        raise ValueError(f"expected virtual racket action dim 6, got {action.shape[-1]}")
    if not np.isfinite(action).all():
        raise ValueError("non-finite virtual racket action")
    return action


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract = VirtualRacketHeadContract()

    dummy_goal = np.zeros((4, contract.goal_dim), dtype=np.float32)
    dummy_goal[:, 3] = 1.0
    dummy_goal[:, 8] = 1.0
    dummy_state = np.zeros((4, contract.realized_state_dim), dtype=np.float32)
    dummy_state[:, 5] = 1.0
    packed = pack_head_input(dummy_goal, dummy_state)
    zero_action = validate_action(np.zeros((4, contract.action_dim), dtype=np.float32))

    summary = {
        "scope": "separate virtual racket head shape/metadata scaffold only",
        "training_run": False,
        "optimizer_update": False,
        "backward_pass": False,
        "policy_inference_with_original_checkpoint": False,
        "contract": contract.__dict__,
        "stage_1_data_flow": {
            "frozen_body_policy_input": "original PHC body observation",
            "frozen_body_policy_output": "MCP body action [4]",
            "new_racket_head_input": "Live Goal V2 [9] + realized state V2 [6]",
            "new_racket_head_input_dim": contract.input_dim,
            "new_racket_head_output": "virtual racket action [6]",
            "combined_env_action": "body action [4] + virtual action [6] = [10]",
        },
        "dummy_shape_check": {
            "input_shape": list(packed.shape),
            "action_shape": list(zero_action.shape),
            "finite": bool(np.isfinite(packed).all() and np.isfinite(zero_action).all()),
        },
        "checkpoint_guard_expectation": {
            "original_phc_checkpoint_compatible_with_racket_head": False,
            "new_racket_head_checkpoint_metadata_required": [
                "goal_version",
                "realized_state_version",
                "action_version",
                "input_dim",
                "action_dim",
                "train_session_split",
            ],
        },
    }
    (args.output_dir / "separate_virtual_racket_head_contract_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
