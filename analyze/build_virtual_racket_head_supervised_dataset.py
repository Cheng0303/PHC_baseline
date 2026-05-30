#!/usr/bin/env python3
"""Build Stage 1A supervised oracle-action dataset.

No optimizer, backward pass, PHC actor import, or policy rollout is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from virtual_racket_head_stage1a_utils import (
    DATASET_DIR,
    DT,
    REPORT_DIR,
    TEST_GROUPS,
    TRAIN_GROUPS,
    VAL_GROUPS,
    PerturbConfig,
    build_sequence_samples,
    concat_sample_dicts,
    load_manifest_rows,
    reconstruction_errors,
    save_npz,
    session_split,
    summarize,
)


def _norm_stats(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "x_mean": x.mean(axis=0).astype(np.float32),
        "x_std": np.maximum(x.std(axis=0), 1e-6).astype(np.float32),
        "y_mean": y.mean(axis=0).astype(np.float32),
        "y_std": np.maximum(y.std(axis=0), 1e-6).astype(np.float32),
    }


def _split_doc(split_rows: dict[str, list[dict[str, str]]]) -> dict[str, object]:
    return {
        name: {
            "groups": sorted({row["session_group"] for row in rows}),
            "clip_count": len(rows),
            "sequences": [row["sequence"] for row in rows],
        }
        for name, rows in split_rows.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--report_dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_variants_per_transition", type=int, default=2)
    parser.add_argument("--eval_variants_per_transition", type=int, default=1)
    parser.add_argument("--handle_offset_bound_m", type=float, default=0.035)
    parser.add_argument("--axis_rot_bound_deg", type=float, default=7.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    config = PerturbConfig(
        seed=args.seed,
        train_variants_per_transition=args.train_variants_per_transition,
        eval_variants_per_transition=args.eval_variants_per_transition,
        handle_offset_bound_m=args.handle_offset_bound_m,
        axis_rot_bound_deg=args.axis_rot_bound_deg,
    )
    rows = load_manifest_rows()
    split_rows = session_split(rows)
    split_info = _split_doc(split_rows)
    expected_counts = {"train": 160, "validation": 40, "test": 40}
    count_passed = all(split_info[name]["clip_count"] == expected_counts[name] for name in expected_counts)
    if not count_passed:
        raise RuntimeError(f"split counts mismatch: {split_info}")

    train_clean_items, train_aug_items = [], []
    val_clean_items, test_clean_items, test_pert_items = [], [], []
    for split_name, split_list in split_rows.items():
        for i, row in enumerate(split_list):
            if split_name == "train":
                train_clean_items.append(build_sequence_samples(row, i, split_name, variants=1, perturbed=False, config=config))
                train_aug_items.append(build_sequence_samples(row, i, split_name, variants=config.train_variants_per_transition, perturbed=True, config=config))
            elif split_name == "validation":
                val_clean_items.append(build_sequence_samples(row, i, split_name, variants=1, perturbed=False, config=config))
            elif split_name == "test":
                test_clean_items.append(build_sequence_samples(row, i, split_name, variants=1, perturbed=False, config=config))
                test_pert_items.append(build_sequence_samples(row, i, split_name, variants=config.eval_variants_per_transition, perturbed=True, config=config))

    train_clean = concat_sample_dicts(train_clean_items)
    train_aug = concat_sample_dicts(train_aug_items)
    train = concat_sample_dicts([train_clean, train_aug])
    validation = concat_sample_dicts(val_clean_items)
    test_clean = concat_sample_dicts(test_clean_items)
    test_pert = concat_sample_dicts(test_pert_items)
    norm = _norm_stats(train["x"], train["y"])

    for name, samples in [("train", train), ("validation", validation), ("test_clean", test_clean), ("test_perturbed", test_pert)]:
        if not np.isfinite(samples["x"]).all() or not np.isfinite(samples["y"]).all():
            raise RuntimeError(f"{name} contains non-finite x/y")
        errors = reconstruction_errors(samples)
        if max(float(errors["handle"].max()), float(errors["tip"].max())) > 1e-5 or float(errors["axis_deg"].max()) > 5e-2:
            raise RuntimeError(f"{name} oracle reconstruction failed")

    save_npz(args.output_dir / "train_transitions.npz", train, norm)
    save_npz(args.output_dir / "validation_transitions.npz", validation, norm)
    save_npz(args.output_dir / "test_transitions_clean.npz", test_clean, norm)
    save_npz(args.output_dir / "test_transitions_perturbed.npz", test_pert, norm)

    recon = {name: {metric: summarize(values) for metric, values in reconstruction_errors(samples).items()} for name, samples in [("train", train), ("validation", validation), ("test_clean", test_clean), ("test_perturbed", test_pert)]}
    action_stats = {
        name: {
            "full_action_magnitude": summarize(np.linalg.norm(samples["y"], axis=-1)),
            "handle_speed_m_per_s": summarize(np.linalg.norm(samples["y"][:, :3], axis=-1)),
            "axis_omega_rad_per_s": summarize(np.linalg.norm(samples["y"][:, 3:6], axis=-1)),
        }
        for name, samples in [("train", train), ("train_clean", train_clean), ("train_augmented", train_aug), ("validation", validation), ("test_clean", test_clean), ("test_perturbed", test_pert)]
    }
    summary = {
        "scope": "offline no-physics separate-head supervised oracle-action dataset",
        "training_run": False,
        "dt": DT,
        "split": split_info,
        "expected_clip_counts_passed": count_passed,
        "groups": {"train": TRAIN_GROUPS, "validation": VAL_GROUPS, "test": TEST_GROUPS},
        "transition_counts": {
            "train_clean": int(len(train_clean["x"])),
            "train_augmented": int(len(train_aug["x"])),
            "train_total": int(len(train["x"])),
            "validation_clean": int(len(validation["x"])),
            "test_clean": int(len(test_clean["x"])),
            "test_perturbed": int(len(test_pert["x"])),
        },
        "input_dim": int(train["x"].shape[1]),
        "output_dim": int(train["y"].shape[1]),
        "normalization_source": "train_total only",
        "perturbation_config": config.__dict__,
        "oracle_one_step_reconstruction": recon,
        "action_distribution": action_stats,
        "invalid_nonfinite_count": 0,
        "passed": True,
    }
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.report_dir / "supervised_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.report_dir / "dataset_split.json").write_text(json.dumps({"split": split_info, "groups": summary["groups"]}, indent=2), encoding="utf-8")
    (args.report_dir / "temporal_semantics_preflight.json").write_text(
        json.dumps(
            {
                "actual_env_goal_time": "next frame: (progress_buf + 1) * dt + motion_start_times + motion_start_times_offset",
                "actual_realized_state_feedback_time": "current/pre-action realized state at step t",
                "dataset_indexing": "input = [goal_{t+1}, state_t], label = oracle_action_t",
                "root_heading_time": "current root at step t",
                "oracle_one_step_reconstruction": recon,
                "passed": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    (args.report_dir / "dataset_split.md").write_text(
        f"""# Stage 1A Dataset Split

No frame-level random split is used. Session groups are fixed.

- train groups: `{', '.join(TRAIN_GROUPS)}` ({split_info['train']['clip_count']} clips)
- validation groups: `{', '.join(VAL_GROUPS)}` ({split_info['validation']['clip_count']} clips)
- test groups: `{', '.join(TEST_GROUPS)}` ({split_info['test']['clip_count']} clips)

Normalization statistics are fit on train transitions only. Test groups are not used for model selection, normalization, early stopping, or hyperparameter selection.
""",
        encoding="utf-8",
    )
    (args.report_dir / "temporal_semantics_preflight.md").write_text(
        f"""# Temporal Semantics Preflight

This gate ran before any optimizer/backward command.

- input goal time: target at `t+1`
- realized-state feedback time: current/pre-action state at `t`
- root/heading frame time: current root heading at `t`
- label: oracle action that advances current realized state to target world state at `t+1`
- pass: `True`

Max oracle reconstruction errors:

- train tip: `{recon['train']['tip']['max']:.9e}` m
- validation tip: `{recon['validation']['tip']['max']:.9e}` m
- test clean tip: `{recon['test_clean']['tip']['max']:.9e}` m
- test perturbed tip: `{recon['test_perturbed']['tip']['max']:.9e}` m
- test perturbed axis: `{recon['test_perturbed']['axis_deg']['max']:.9e}` deg

The supervised dataset uses `[goal_{{t+1}}, state_t] -> action_t`, matching the runtime hook semantics.
""",
        encoding="utf-8",
    )
    (args.report_dir / "supervised_dataset_summary.md").write_text(
        f"""# Supervised Oracle-Action Dataset Summary

Offline no-physics separate-head dataset only. No PHC body rollout and no policy training was run by this builder.

- train clean transitions: `{summary['transition_counts']['train_clean']}`
- train augmented transitions: `{summary['transition_counts']['train_augmented']}`
- train total transitions: `{summary['transition_counts']['train_total']}`
- validation clean transitions: `{summary['transition_counts']['validation_clean']}`
- test clean transitions: `{summary['transition_counts']['test_clean']}`
- test perturbed transitions: `{summary['transition_counts']['test_perturbed']}`
- input/output dims: `{summary['input_dim']} -> {summary['output_dim']}`
- perturbation: handle +/- `{config.handle_offset_bound_m}` m, axis +/- `{config.axis_rot_bound_deg}` deg, seed `{config.seed}`
- train action magnitude mean/p95/max: `{action_stats['train']['full_action_magnitude']['mean']:.6f}` / `{action_stats['train']['full_action_magnitude']['p95']:.6f}` / `{action_stats['train']['full_action_magnitude']['max']:.6f}`
- invalid/non-finite count: `0`
""",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
