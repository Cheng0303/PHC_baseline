"""Utilities for Stage 1A separate virtual racket head data/eval.

This module is offline/no-physics. It never imports PHC actors or checkpoints.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "phc_baseline/racket_calibration/racket_aware_reference_task_cross_session_dataset/manifest.csv"
DATASET_DIR = REPO_ROOT / "phc_baseline/racket_calibration/separate_virtual_racket_head_dataset"
REPORT_DIR = REPO_ROOT / "phc_baseline/reports/racket_calibration/separate_head_training"
MODEL_DIR = REPO_ROOT / "phc_baseline/models/separate_virtual_racket_head_stage1a"
DT = 1.0 / 30.0
EPS = 1e-8

TRAIN_GROUPS = ["241217_1", "241224_4", "241224_3", "241226_2", "250108_2", "241217_4", "241224_2", "241217_3"]
VAL_GROUPS = ["241224_1", "250108_1"]
TEST_GROUPS = ["241217_2", "241226_1"]


@dataclass(frozen=True)
class PerturbConfig:
    seed: int = 42
    train_variants_per_transition: int = 2
    eval_variants_per_transition: int = 1
    handle_offset_bound_m: float = 0.035
    axis_rot_bound_deg: float = 7.5


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(norm < EPS):
        raise ValueError("near-zero vector")
    return v / norm


def summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def axis_angle_error_deg(axis_a: np.ndarray, axis_b: np.ndarray) -> np.ndarray:
    a = normalize(axis_a)
    b = normalize(axis_b)
    dot = np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def yaw_matrix(yaw_rad: np.ndarray) -> np.ndarray:
    c = np.cos(yaw_rad)
    s = np.sin(yaw_rad)
    z = np.zeros_like(c)
    o = np.ones_like(c)
    return np.stack(
        [
            np.stack([c, -s, z], axis=-1),
            np.stack([s, c, z], axis=-1),
            np.stack([z, z, o], axis=-1),
        ],
        axis=-2,
    )


def heading_matrix_from_root_rot(root_rot: np.ndarray) -> np.ndarray:
    x_axis = root_rot[..., :, 0]
    yaw = np.arctan2(x_axis[..., 1], x_axis[..., 0])
    return yaw_matrix(yaw)


def project_heading_local(vec_or_point: np.ndarray, root_pos: np.ndarray | None, root_rot: np.ndarray) -> np.ndarray:
    heading = heading_matrix_from_root_rot(root_rot)
    values = vec_or_point - root_pos if root_pos is not None else vec_or_point
    return np.einsum("...ji,...j->...i", heading, values)


def heading_local_to_world(vec_local: np.ndarray, root_rot: np.ndarray) -> np.ndarray:
    heading = heading_matrix_from_root_rot(root_rot)
    return np.einsum("...ij,...j->...i", heading, vec_local)


def rotate_axis(axis: np.ndarray, omega_world: np.ndarray, dt: float = DT) -> np.ndarray:
    angle_vec = omega_world * dt
    angle = np.linalg.norm(angle_vec, axis=-1, keepdims=True)
    rot_axis = np.zeros_like(angle_vec)
    nonzero = angle[..., 0] > EPS
    rot_axis[nonzero] = angle_vec[nonzero] / angle[nonzero]
    cos = np.cos(angle)
    sin = np.sin(angle)
    cross = np.cross(rot_axis, axis)
    dot = np.sum(rot_axis * axis, axis=-1, keepdims=True)
    rotated = axis * cos + cross * sin + rot_axis * dot * (1.0 - cos)
    rotated[~nonzero] = axis[~nonzero]
    return normalize(rotated)


def derive_oracle_action(current_handle: np.ndarray, current_axis: np.ndarray, target_handle_next: np.ndarray, target_axis_next: np.ndarray, root_rot_t: np.ndarray, dt: float = DT) -> np.ndarray:
    v_world = (target_handle_next - current_handle) / dt
    a = normalize(current_axis)
    b = normalize(target_axis_next)
    cross = np.cross(a, b)
    sin = np.linalg.norm(cross, axis=-1, keepdims=True)
    cos = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)
    angle = np.arctan2(sin, cos)
    rot_axis = np.zeros_like(a)
    nonzero = sin[..., 0] > EPS
    rot_axis[nonzero] = cross[nonzero] / sin[nonzero]
    omega_world = rot_axis * angle / dt
    v_local = project_heading_local(v_world, None, root_rot_t)
    omega_local = project_heading_local(omega_world, None, root_rot_t)
    return np.concatenate([v_local, omega_local], axis=-1)


def step_dynamics(handle_world: np.ndarray, axis_world: np.ndarray, action_local: np.ndarray, root_rot_t: np.ndarray, dt: float = DT) -> tuple[np.ndarray, np.ndarray]:
    v_world = heading_local_to_world(action_local[..., :3], root_rot_t)
    omega_world = heading_local_to_world(action_local[..., 3:6], root_rot_t)
    handle_next = handle_world + dt * v_world
    axis_next = rotate_axis(axis_world, omega_world, dt)
    return handle_next, axis_next


def pack_input(goal_handle_next: np.ndarray, goal_tip_next: np.ndarray, goal_axis_next: np.ndarray, current_handle: np.ndarray, current_axis: np.ndarray, root_pos_t: np.ndarray, root_rot_t: np.ndarray) -> np.ndarray:
    goal = np.concatenate(
        [
            project_heading_local(goal_handle_next, root_pos_t, root_rot_t),
            project_heading_local(goal_tip_next, root_pos_t, root_rot_t),
            normalize(project_heading_local(goal_axis_next, None, root_rot_t)),
        ],
        axis=-1,
    )
    state = np.concatenate(
        [
            project_heading_local(current_handle, root_pos_t, root_rot_t),
            normalize(project_heading_local(current_axis, None, root_rot_t)),
        ],
        axis=-1,
    )
    return np.concatenate([goal, state], axis=-1)


def rotate_vectors_by_rotvec(vecs: np.ndarray, rotvecs: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvecs, axis=-1, keepdims=True)
    axis = np.zeros_like(rotvecs)
    nonzero = angle[..., 0] > EPS
    axis[nonzero] = rotvecs[nonzero] / angle[nonzero]
    cos = np.cos(angle)
    sin = np.sin(angle)
    cross = np.cross(axis, vecs)
    dot = np.sum(axis * vecs, axis=-1, keepdims=True)
    out = vecs * cos + cross * sin + axis * dot * (1.0 - cos)
    out[~nonzero] = vecs[~nonzero]
    return normalize(out)


def load_manifest_rows(manifest: Path = MANIFEST) -> list[dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        row
        for row in rows
        if row.get("task_export_passed") == "True"
        and row.get("integrity_check_passed") == "True"
        and row.get("dynamic_replay_passed") == "True"
    ]


def session_split(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    split = {"train": [], "validation": [], "test": []}
    for row in rows:
        group = row["session_group"]
        if group in TRAIN_GROUPS:
            split["train"].append(row)
        elif group in VAL_GROUPS:
            split["validation"].append(row)
        elif group in TEST_GROUPS:
            split["test"].append(row)
        else:
            raise ValueError(f"unexpected session group {group}")
    return split


def load_task_npz(row: dict[str, str]) -> dict[str, np.ndarray]:
    data = np.load(row["npz_path"], allow_pickle=True)
    handle = np.asarray(data["racket_handle_phc_world"], dtype=np.float64)
    tip = np.asarray(data["racket_tip_phc_world"], dtype=np.float64)
    axis = normalize(np.asarray(data["racket_long_axis_phc_world"], dtype=np.float64))
    root_pos = np.asarray(data["root_position_phc_world"], dtype=np.float64)
    root_rot = np.asarray(data["root_rotation_phc_world_matrix"], dtype=np.float64)
    length = np.linalg.norm(tip - handle, axis=-1)
    return {"handle": handle, "tip": tip, "axis": axis, "root_pos": root_pos, "root_rot": root_rot, "length": length}


def sample_perturbations(n: int, variants: int, config: PerturbConfig, sequence_index: int, split_name: str) -> tuple[np.ndarray, np.ndarray]:
    seed_offset = {"train": 0, "validation": 100000, "test": 200000}.get(split_name, 300000)
    rng = np.random.default_rng(config.seed + seed_offset + sequence_index * 9973)
    offsets = rng.uniform(-config.handle_offset_bound_m, config.handle_offset_bound_m, size=(variants, n, 3))
    rotvecs = rng.uniform(-np.deg2rad(config.axis_rot_bound_deg), np.deg2rad(config.axis_rot_bound_deg), size=(variants, n, 3))
    return offsets, rotvecs


def build_sequence_samples(row: dict[str, str], sequence_index: int, split_name: str, *, variants: int, perturbed: bool, config: PerturbConfig) -> dict[str, np.ndarray]:
    task = load_task_npz(row)
    n = len(task["handle"]) - 1
    h_t = task["handle"][:-1]
    axis_t = task["axis"][:-1]
    h_next = task["handle"][1:]
    tip_next = task["tip"][1:]
    axis_next = task["axis"][1:]
    root_pos_t = task["root_pos"][:-1]
    root_rot_t = task["root_rot"][:-1]
    length_t = task["length"][:-1]
    if not perturbed:
        variants = 1
    offsets_local, rotvecs_local = sample_perturbations(n, variants, config, sequence_index, split_name)
    xs, ys, meta_variant = [], [], []
    cur_handles, cur_axes, target_handles, target_axes, target_tips, roots_pos, roots_rot, lengths = [], [], [], [], [], [], [], []
    for variant in range(variants):
        if perturbed:
            offset_world = heading_local_to_world(offsets_local[variant], root_rot_t)
            current_handle = h_t + offset_world
            axis_local = normalize(project_heading_local(axis_t, None, root_rot_t))
            axis_local_pert = rotate_vectors_by_rotvec(axis_local, rotvecs_local[variant])
            current_axis = normalize(heading_local_to_world(axis_local_pert, root_rot_t))
        else:
            current_handle = h_t
            current_axis = axis_t
        x = pack_input(h_next, tip_next, axis_next, current_handle, current_axis, root_pos_t, root_rot_t)
        y = derive_oracle_action(current_handle, current_axis, h_next, axis_next, root_rot_t)
        xs.append(x)
        ys.append(y)
        meta_variant.append(np.full((n,), variant, dtype=np.int32))
        cur_handles.append(current_handle)
        cur_axes.append(current_axis)
        target_handles.append(h_next)
        target_axes.append(axis_next)
        target_tips.append(tip_next)
        roots_pos.append(root_pos_t)
        roots_rot.append(root_rot_t)
        lengths.append(length_t)
    return {
        "x": np.concatenate(xs).astype(np.float32),
        "y": np.concatenate(ys).astype(np.float32),
        "current_handle_world": np.concatenate(cur_handles).astype(np.float32),
        "current_axis_world": np.concatenate(cur_axes).astype(np.float32),
        "target_handle_next_world": np.concatenate(target_handles).astype(np.float32),
        "target_tip_next_world": np.concatenate(target_tips).astype(np.float32),
        "target_axis_next_world": np.concatenate(target_axes).astype(np.float32),
        "root_pos_t_world": np.concatenate(roots_pos).astype(np.float32),
        "root_rot_t_world": np.concatenate(roots_rot).astype(np.float32),
        "racket_length": np.concatenate(lengths).astype(np.float32),
        "sequence": np.asarray([row["sequence"]] * (n * variants)),
        "session_group": np.asarray([row["session_group"]] * (n * variants)),
        "frame_index_t": np.tile(np.arange(n, dtype=np.int32), variants),
        "variant_index": np.concatenate(meta_variant),
    }


def concat_sample_dicts(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = items[0].keys()
    return {key: np.concatenate([item[key] for item in items], axis=0) for key in keys}


def reconstruction_errors(samples: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    h_next, axis_next = step_dynamics(
        samples["current_handle_world"].astype(np.float64),
        samples["current_axis_world"].astype(np.float64),
        samples["y"].astype(np.float64),
        samples["root_rot_t_world"].astype(np.float64),
    )
    tip_next = h_next + samples["racket_length"].astype(np.float64)[:, None] * axis_next
    return {
        "handle": np.linalg.norm(h_next - samples["target_handle_next_world"], axis=-1),
        "tip": np.linalg.norm(tip_next - samples["target_tip_next_world"], axis=-1),
        "axis_deg": axis_angle_error_deg(axis_next, samples["target_axis_next_world"]),
    }


def save_npz(path: Path, samples: dict[str, np.ndarray], norm: dict[str, np.ndarray] | None = None) -> None:
    payload = dict(samples)
    if norm:
        for key, value in norm.items():
            payload[key] = value
    np.savez_compressed(path, **payload)
