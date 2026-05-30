#!/usr/bin/env python3
"""Train Stage 1A separate virtual racket heads from one JSON config.

This script trains only new virtual racket head MLPs. It never imports,
loads, modifies, or fine-tunes the original PHC body actor/checkpoint.
Evaluation is offline no-physics virtual dynamics only.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


ANALYZE_DIR = Path(__file__).resolve().parents[1]
if str(ANALYZE_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYZE_DIR))

from virtual_racket_head_stage1a_utils import (  # noqa: E402
    DT,
    axis_angle_error_deg,
    derive_oracle_action,
    pack_input,
    step_dynamics,
    summarize,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int = 6, activation: str = "relu"):
        super().__init__()
        act = nn.ReLU if activation == "relu" else nn.SiLU
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), act()])
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _select_x(samples: dict[str, np.ndarray], model_cfg: dict[str, object]) -> np.ndarray:
    start = int(model_cfg["input_start"])
    dim = int(model_cfg["input_dim"])
    return samples["x"][:, start : start + dim].astype(np.float32)


def _norm_x(x: np.ndarray, train: dict[str, np.ndarray], model_cfg: dict[str, object]) -> np.ndarray:
    start = int(model_cfg["input_start"])
    dim = int(model_cfg["input_dim"])
    return ((x - train["x_mean"][start : start + dim]) / train["x_std"][start : start + dim]).astype(np.float32)


def _norm_y(y: np.ndarray, train: dict[str, np.ndarray]) -> np.ndarray:
    return ((y - train["y_mean"]) / train["y_std"]).astype(np.float32)


def _denorm_y(y_norm: np.ndarray, train: dict[str, np.ndarray]) -> np.ndarray:
    return (y_norm * train["y_std"] + train["y_mean"]).astype(np.float32)


def _predict_actions(model: nn.Module, samples: dict[str, np.ndarray], train: dict[str, np.ndarray], model_cfg: dict[str, object], device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    x = _norm_x(_select_x(samples, model_cfg), train, model_cfg)
    outs = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[i : i + batch_size], device=device)
            outs.append(model(xb).cpu().numpy())
    return _denorm_y(np.concatenate(outs, axis=0), train)


def _transition_metrics(samples: dict[str, np.ndarray], actions: np.ndarray) -> dict[str, np.ndarray]:
    h_next, axis_next = step_dynamics(
        samples["current_handle_world"].astype(np.float64),
        samples["current_axis_world"].astype(np.float64),
        actions.astype(np.float64),
        samples["root_rot_t_world"].astype(np.float64),
    )
    tip_next = h_next + samples["racket_length"].astype(np.float64)[:, None] * axis_next
    return {
        "action_mse": np.mean((actions - samples["y"]) ** 2, axis=-1),
        "action_mae": np.mean(np.abs(actions - samples["y"]), axis=-1),
        "handle_error_m": np.linalg.norm(h_next - samples["target_handle_next_world"], axis=-1),
        "tip_error_m": np.linalg.norm(tip_next - samples["target_tip_next_world"], axis=-1),
        "axis_error_deg": axis_angle_error_deg(axis_next, samples["target_axis_next_world"]),
        "action_magnitude": np.linalg.norm(actions, axis=-1),
    }


def _train_one(name: str, model_cfg: dict[str, object], train: dict[str, np.ndarray], validation: dict[str, np.ndarray], cfg: dict[str, object], root: Path, device: torch.device) -> tuple[nn.Module, dict[str, object], list[dict[str, object]]]:
    train_cfg = cfg["training"]
    model = MLPHead(int(model_cfg["input_dim"]), list(model_cfg["hidden_dims"]), 6, str(model_cfg["activation"])).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    loss_fn = nn.MSELoss()
    x_train = _norm_x(_select_x(train, model_cfg), train, model_cfg)
    y_train = _norm_y(train["y"].astype(np.float32), train)
    x_val = _norm_x(_select_x(validation, model_cfg), train, model_cfg)
    y_val = _norm_y(validation["y"].astype(np.float32), train)
    dataset = TensorDataset(torch.as_tensor(x_train), torch.as_tensor(y_train))
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=bool(train_cfg["shuffle_train"]),
        num_workers=int(train_cfg["num_workers"]),
    )
    best_state = None
    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    for epoch in range(1, int(train_cfg["max_epochs"]) + 1):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name} non-finite training loss at epoch {epoch}")
            loss.backward()
            clip = float(train_cfg.get("gradient_clip_norm", 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(0, len(x_val), int(train_cfg["batch_size"])):
                xb = torch.as_tensor(x_val[i : i + int(train_cfg["batch_size"])], device=device)
                yb = torch.as_tensor(y_val[i : i + int(train_cfg["batch_size"])], device=device)
                val_losses.append(float(loss_fn(model(xb), yb).cpu().item()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history.append({"model": name, "epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= int(train_cfg["patience"]):
            break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a valid checkpoint")
    model.load_state_dict(best_state)
    metadata = {
        "model_name": name,
        "display_name": model_cfg["display_name"],
        "model_type": "MLPHead",
        "hidden_dims": list(model_cfg["hidden_dims"]),
        "input_dim": int(model_cfg["input_dim"]),
        "output_dim": 6,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "epochs_run": len(history),
    }
    model_dir = _resolve(root, cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": best_state, "metadata": metadata}, model_dir / str(model_cfg["checkpoint_name"]))
    return model, metadata, history


def _closed_loop_for_sequence(samples_clean: dict[str, np.ndarray], samples_pert: dict[str, np.ndarray], sequence: str, policy_name: str, action_fn) -> dict[str, object]:
    seq_mask = samples_clean["sequence"] == sequence
    idx = np.where(seq_mask)[0]
    order = idx[np.argsort(samples_clean["frame_index_t"][idx])]
    pert_idx = np.where(samples_pert["sequence"] == sequence)[0]
    pert_order = pert_idx[np.argsort(samples_pert["frame_index_t"][pert_idx])]
    rows = []
    for mode in ["clean", "perturbed_initial"]:
        current_handle = samples_clean["current_handle_world"][order[0]].astype(np.float64).copy()
        current_axis = samples_clean["current_axis_world"][order[0]].astype(np.float64).copy()
        if mode == "perturbed_initial" and len(pert_order):
            current_handle = samples_pert["current_handle_world"][pert_order[0]].astype(np.float64).copy()
            current_axis = samples_pert["current_axis_world"][pert_order[0]].astype(np.float64).copy()
        prev_action = None
        handle_errors, tip_errors, axis_errors, action_mags, smooth = [], [], [], [], []
        target_tip_path, pred_tip_path = [], []
        for sample_i in order:
            sample = {key: value[sample_i : sample_i + 1] for key, value in samples_clean.items()}
            x = pack_input(
                sample["target_handle_next_world"].astype(np.float64),
                sample["target_tip_next_world"].astype(np.float64),
                sample["target_axis_next_world"].astype(np.float64),
                current_handle[None, :],
                current_axis[None, :],
                sample["root_pos_t_world"].astype(np.float64),
                sample["root_rot_t_world"].astype(np.float64),
            )
            action = action_fn(x, current_handle[None, :], current_axis[None, :], sample)[0].astype(np.float64)
            next_handle, next_axis = step_dynamics(current_handle[None, :], current_axis[None, :], action[None, :], sample["root_rot_t_world"].astype(np.float64))
            current_handle = next_handle[0]
            current_axis = next_axis[0]
            pred_tip = current_handle + float(sample["racket_length"][0]) * current_axis
            handle_errors.append(float(np.linalg.norm(current_handle - sample["target_handle_next_world"][0])))
            tip_errors.append(float(np.linalg.norm(pred_tip - sample["target_tip_next_world"][0])))
            axis_errors.append(float(axis_angle_error_deg(current_axis[None, :], sample["target_axis_next_world"].astype(np.float64))[0]))
            action_mags.append(float(np.linalg.norm(action)))
            if prev_action is not None:
                smooth.append(float(np.linalg.norm(action - prev_action)))
            prev_action = action
            target_tip_path.append(sample["target_tip_next_world"][0].astype(np.float64))
            pred_tip_path.append(pred_tip)
        rows.append(
            {
                "sequence": sequence,
                "mode": mode,
                "policy": policy_name,
                "frames": len(order),
                "handle_error_m": summarize(np.asarray(handle_errors)),
                "tip_error_m": summarize(np.asarray(tip_errors)),
                "axis_error_deg": summarize(np.asarray(axis_errors)),
                "action_magnitude": summarize(np.asarray(action_mags)),
                "action_smoothness": summarize(np.asarray(smooth if smooth else [0.0])),
                "target_tip_path": np.asarray(target_tip_path),
                "pred_tip_path": np.asarray(pred_tip_path),
            }
        )
    return {"rows": rows}


def _write_loss_plot(history: list[dict[str, object]], plot_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    for name in sorted({row["model"] for row in history}):
        rows = [row for row in history if row["model"] == name]
        plt.plot([r["epoch"] for r in rows], [r["train_loss"] for r in rows], label=f"{name} train")
        plt.plot([r["epoch"] for r in rows], [r["validation_loss"] for r in rows], "--", label=f"{name} val")
    plt.xlabel("epoch")
    plt.ylabel("normalized action MSE")
    plt.title("offline no-physics separate-head evaluation only\nnot PHC body rollout racket accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "training_validation_loss_curve.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = _repo_root()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    _set_seed(int(cfg["seed"]))
    device = torch.device("cuda" if cfg["device"] == "auto" and torch.cuda.is_available() else "cpu")
    dataset_dir = _resolve(root, cfg["dataset_dir"])
    report_dir = _resolve(root, cfg["report_dir"])
    model_dir = _resolve(root, cfg["model_dir"])
    plot_dir = report_dir / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    train = _load_npz(dataset_dir / cfg["train_file"])
    validation = _load_npz(dataset_dir / cfg["validation_file"])
    test_clean = _load_npz(dataset_dir / cfg["test_clean_file"])
    test_pert = _load_npz(dataset_dir / cfg["test_perturbed_file"])

    models: dict[str, nn.Module] = {}
    model_meta: dict[str, dict[str, object]] = {}
    all_history: list[dict[str, object]] = []
    for name, model_cfg in cfg["models"].items():
        model, meta, history = _train_one(name, model_cfg, train, validation, cfg, root, device)
        models[name] = model
        model_meta[name] = meta
        all_history.extend(history)

    with (model_dir / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(all_history)
    _write_loss_plot(all_history, plot_dir)

    batch_size = int(cfg["training"]["batch_size"])
    action_sets = {
        "null": {
            "test_clean": np.zeros_like(test_clean["y"], dtype=np.float32),
            "test_perturbed": np.zeros_like(test_pert["y"], dtype=np.float32),
        },
        "oracle": {"test_clean": test_clean["y"].astype(np.float32), "test_perturbed": test_pert["y"].astype(np.float32)},
    }
    for name, model in models.items():
        action_sets[name] = {
            "test_clean": _predict_actions(model, test_clean, train, cfg["models"][name], device, batch_size),
            "test_perturbed": _predict_actions(model, test_pert, train, cfg["models"][name], device, batch_size),
        }

    eval_rows = []
    eval_summary: dict[str, object] = {}
    for policy, split_actions in action_sets.items():
        eval_summary[policy] = {}
        for split_name, actions in split_actions.items():
            samples = test_clean if split_name == "test_clean" else test_pert
            metrics = _transition_metrics(samples, actions)
            eval_summary[policy][split_name] = {key: summarize(value) for key, value in metrics.items()}
            eval_rows.append(
                {
                    "eval_type": "one_step",
                    "split": split_name,
                    "policy": policy,
                    "action_mse_mean": eval_summary[policy][split_name]["action_mse"]["mean"],
                    "action_mae_mean": eval_summary[policy][split_name]["action_mae"]["mean"],
                    "handle_error_mean_m": eval_summary[policy][split_name]["handle_error_m"]["mean"],
                    "tip_error_mean_m": eval_summary[policy][split_name]["tip_error_m"]["mean"],
                    "axis_error_mean_deg": eval_summary[policy][split_name]["axis_error_deg"]["mean"],
                    "tip_error_p90_m": eval_summary[policy][split_name]["tip_error_m"]["p90"],
                    "axis_error_p90_deg": eval_summary[policy][split_name]["axis_error_deg"]["p90"],
                }
            )

    def make_model_action_fn(name: str):
        model = models[name]
        model_cfg = cfg["models"][name]

        def _fn(x: np.ndarray, _current_handle, _current_axis, _sample) -> np.ndarray:
            tmp = {"x": x.astype(np.float32)}
            return _predict_actions(model, tmp, train, model_cfg, device, batch_size=8192)

        return _fn

    def null_fn(x, current_handle, current_axis, sample):
        return np.zeros((len(x), 6), dtype=np.float32)

    def oracle_fn(x, current_handle, current_axis, sample):
        return derive_oracle_action(
            current_handle.astype(np.float64),
            current_axis.astype(np.float64),
            sample["target_handle_next_world"].astype(np.float64),
            sample["target_axis_next_world"].astype(np.float64),
            sample["root_rot_t_world"].astype(np.float64),
            DT,
        ).astype(np.float32)

    closed_loop_rows = []
    per_seq_rows = []
    sequences = sorted(set(test_clean["sequence"].tolist()))
    action_fns = {"null": null_fn, "oracle": oracle_fn}
    action_fns.update({name: make_model_action_fn(name) for name in models.keys()})
    representative_paths = {}
    for seq in sequences:
        for policy, action_fn in action_fns.items():
            result = _closed_loop_for_sequence(test_clean, test_pert, seq, policy, action_fn)
            for row in result["rows"]:
                compact = {k: v for k, v in row.items() if k not in {"target_tip_path", "pred_tip_path"}}
                per_seq_rows.append(
                    {
                        "sequence": compact["sequence"],
                        "mode": compact["mode"],
                        "policy": compact["policy"],
                        "frames": compact["frames"],
                        "handle_error_mean_m": compact["handle_error_m"]["mean"],
                        "handle_error_p90_m": compact["handle_error_m"]["p90"],
                        "handle_error_max_m": compact["handle_error_m"]["max"],
                        "tip_error_mean_m": compact["tip_error_m"]["mean"],
                        "tip_error_p90_m": compact["tip_error_m"]["p90"],
                        "tip_error_max_m": compact["tip_error_m"]["max"],
                        "axis_error_mean_deg": compact["axis_error_deg"]["mean"],
                        "axis_error_p90_deg": compact["axis_error_deg"]["p90"],
                        "axis_error_max_deg": compact["axis_error_deg"]["max"],
                        "action_magnitude_mean": compact["action_magnitude"]["mean"],
                        "action_smoothness_mean": compact["action_smoothness"]["mean"],
                    }
                )
                closed_loop_rows.append(compact)
                if seq == cfg["evaluation"]["representative_sequence"] and compact["mode"] == "clean":
                    representative_paths[policy] = row

    closed_summary: dict[str, object] = {}
    for policy in action_fns.keys():
        closed_summary[policy] = {}
        for mode in ["clean", "perturbed_initial"]:
            rows = [r for r in per_seq_rows if r["policy"] == policy and r["mode"] == mode]
            closed_summary[policy][mode] = {
                "handle_error_m": summarize(np.asarray([r["handle_error_mean_m"] for r in rows])),
                "tip_error_m": summarize(np.asarray([r["tip_error_mean_m"] for r in rows])),
                "axis_error_deg": summarize(np.asarray([r["axis_error_mean_deg"] for r in rows])),
                "action_magnitude": summarize(np.asarray([r["action_magnitude_mean"] for r in rows])),
                "action_smoothness": summarize(np.asarray([r["action_smoothness_mean"] for r in rows])),
            }

    with (report_dir / "stage1a_per_sequence_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_seq_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_seq_rows)
    with (report_dir / "stage1a_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)

    # Comparison plots.
    for split_name, title_suffix in [("test_clean", "clean test"), ("test_perturbed", "perturbed recovery test")]:
        policies = ["null", "goal_only", "goal_state", "oracle"]
        tip = [eval_summary[p][split_name]["tip_error_m"]["mean"] for p in policies]
        axis = [eval_summary[p][split_name]["axis_error_deg"]["mean"] for p in policies]
        fig, ax1 = plt.subplots(figsize=(8, 4))
        xloc = np.arange(len(policies))
        ax1.bar(xloc - 0.18, tip, width=0.36, label="tip mean (m)")
        ax2 = ax1.twinx()
        ax2.bar(xloc + 0.18, axis, width=0.36, color="tab:orange", label="axis mean (deg)")
        ax1.set_xticks(xloc)
        ax1.set_xticklabels(policies, rotation=15)
        ax1.set_title(f"{title_suffix}\noffline no-physics separate-head evaluation only; not PHC body rollout racket accuracy")
        ax1.set_ylabel("tip error (m)")
        ax2.set_ylabel("axis error (deg)")
        fig.tight_layout()
        fig.savefig(plot_dir / f"{split_name}_tip_axis_comparison.png", dpi=160)
        plt.close(fig)

    if representative_paths:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        target = next(iter(representative_paths.values()))["target_tip_path"]
        ax.plot(target[:, 0], target[:, 1], target[:, 2], label="target")
        for policy, row in representative_paths.items():
            if policy in {"goal_only", "goal_state", "oracle", "null"}:
                pred = row["pred_tip_path"]
                ax.plot(pred[:, 0], pred[:, 1], pred[:, 2], label=policy)
        ax.set_title("representative held-out tip trajectory\noffline no-physics; not PHC body rollout racket accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_dir / "closed_loop_representative_tip_trajectory.png", dpi=160)
        plt.close(fig)

    plt.figure(figsize=(7, 4))
    for policy in ["goal_only", "goal_state", "oracle"]:
        plt.hist(np.linalg.norm(action_sets[policy]["test_clean"], axis=-1), bins=80, alpha=0.45, label=policy)
    plt.xlabel("action magnitude")
    plt.title("clean test action magnitude distribution\noffline no-physics separate-head evaluation only")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "action_magnitude_distribution.png", dpi=160)
    plt.close()

    training_summary = {
        "scope": "Stage 1A separate virtual racket head supervised oracle-action warm start",
        "training_executed": True,
        "training_command_contract": "train.py --config <json>; all train settings loaded from JSON",
        "policy_scope": "separate virtual racket head only",
        "original_phc_checkpoint_loaded": False,
        "original_phc_body_weights_modified": False,
        "ppo_or_rl": False,
        "reward_training": False,
        "device": str(device),
        "config": cfg,
        "models": model_meta,
    }
    evaluation_summary = {
        "scope": "offline no-physics separate-head virtual dynamics evaluation only; not PHC body rollout racket accuracy",
        "one_step": eval_summary,
        "closed_loop": closed_summary,
        "body_path_safety": {
            "original_phc_body_checkpoint_untouched": True,
            "body_completion_mpjpe_recomputed": False,
            "coupled_body_racket_behavior_claimed": False,
        },
    }
    metadata = {
        **cfg["metadata"],
        "dt": cfg["dt"],
        "models": model_meta,
        "normalization": {
            "x_mean": train["x_mean"].astype(float).tolist(),
            "x_std": train["x_std"].astype(float).tolist(),
            "y_mean": train["y_mean"].astype(float).tolist(),
            "y_std": train["y_std"].astype(float).tolist(),
        },
        "training": cfg["training"],
        "best_primary_model": "goal_state",
    }
    (model_dir / "checkpoint_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / "training_summary.json").write_text(json.dumps(training_summary, indent=2), encoding="utf-8")
    (report_dir / "stage1a_evaluation_summary.json").write_text(json.dumps(evaluation_summary, indent=2), encoding="utf-8")

    b_clean = eval_summary["goal_state"]["test_clean"]
    b_pert = eval_summary["goal_state"]["test_perturbed"]
    a_pert = eval_summary["goal_only"]["test_perturbed"]
    report = f"""# Stage 1A Separate Virtual Racket Head Evaluation

This is offline no-physics separate-head virtual dynamics evaluation only. It is not PHC body rollout racket accuracy, not physical racket performance, and not a trained coupled PHC racket-aware controller.

## Training Scope

- entrypoint: `train.py --config {args.config}`
- settings source: one JSON config
- original PHC checkpoint loaded: `False`
- original PHC body weights modified: `False`
- PPO/RL/reward training: `False`

## Models

- Model A `goal_only`: Live Goal V2 `[9]` -> virtual action `[6]`
- Model B `goal_state`: Live Goal V2 `[9]` + realized state `[6]` -> virtual action `[6]`
- hidden dims: `{cfg['models']['goal_state']['hidden_dims']}`
- seed: `{cfg['seed']}`

## Best Validation

- Model A best epoch/loss: `{model_meta['goal_only']['best_epoch']}` / `{model_meta['goal_only']['best_validation_loss']:.9e}`
- Model B best epoch/loss: `{model_meta['goal_state']['best_epoch']}` / `{model_meta['goal_state']['best_validation_loss']:.9e}`

## One-Step Test Metrics

Clean test, Model B:

- tip mean/p90/max: `{b_clean['tip_error_m']['mean']:.6f}` / `{b_clean['tip_error_m']['p90']:.6f}` / `{b_clean['tip_error_m']['max']:.6f}` m
- axis mean/p90/max: `{b_clean['axis_error_deg']['mean']:.6f}` / `{b_clean['axis_error_deg']['p90']:.6f}` / `{b_clean['axis_error_deg']['max']:.6f}` deg

Perturbed test:

- Model A tip mean / axis mean: `{a_pert['tip_error_m']['mean']:.6f}` m / `{a_pert['axis_error_deg']['mean']:.6f}` deg
- Model B tip mean / axis mean: `{b_pert['tip_error_m']['mean']:.6f}` m / `{b_pert['axis_error_deg']['mean']:.6f}` deg

## Closed-Loop Head-Only Test

Clean initialization:

- Model B tip mean over held-out sequences: `{closed_summary['goal_state']['clean']['tip_error_m']['mean']:.6f}` m
- Model B axis mean over held-out sequences: `{closed_summary['goal_state']['clean']['axis_error_deg']['mean']:.6f}` deg

Perturbed initialization:

- Model A tip/axis mean: `{closed_summary['goal_only']['perturbed_initial']['tip_error_m']['mean']:.6f}` m / `{closed_summary['goal_only']['perturbed_initial']['axis_error_deg']['mean']:.6f}` deg
- Model B tip/axis mean: `{closed_summary['goal_state']['perturbed_initial']['tip_error_m']['mean']:.6f}` m / `{closed_summary['goal_state']['perturbed_initial']['axis_error_deg']['mean']:.6f}` deg

## Interpretation

This validates the first separate-head supervised warm start only. Body/racket coupled behavior remains untested because the frozen PHC body policy was not executed.
"""
    (report_dir / "stage1a_evaluation_report.md").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
