#!/usr/bin/env python3
"""Render a visual-only passive racket attached to PHC body rollout diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2", "L_Ankle",
    "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar", "R_Collar",
    "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist",
    "R_Wrist", "L_Hand", "R_Hand",
]

EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def load_series(path: Path, sequence: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = []
    ref = []
    steps = []
    for rec in payload["records"]:
        keys = rec.get("motion_keys", [])
        if keys and keys[0] != sequence:
            continue
        body.append(np.asarray(rec["body_pos"][0], dtype=np.float64))
        ref.append(np.asarray(rec["ref_body_pos"][0], dtype=np.float64))
        steps.append(int(rec["step"]))
    if not body:
        raise ValueError(f"no records for {sequence} in {path}")
    return np.stack(body), np.stack(ref), steps


def joint_index(name: str) -> int:
    if name not in JOINT_NAMES:
        raise KeyError(f"unknown joint/body name {name}; available: {JOINT_NAMES}")
    return JOINT_NAMES.index(name)


def normalize(vec: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vec))
    if length < 1e-8:
        return np.asarray([1.0, 0.0, 0.0])
    return vec / length


def racket_geometry(joints: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hand_idx = joint_index(cfg.get("attached_body", "R_Hand"))
    wrist_idx = joint_index(cfg.get("wrist_body", "R_Wrist"))
    base = joints[hand_idx] + np.asarray(cfg.get("local_position_offset", [0.0, 0.0, 0.0]), dtype=np.float64)
    direction = normalize(joints[hand_idx] - joints[wrist_idx])
    if np.linalg.norm(joints[hand_idx] - joints[wrist_idx]) < 1e-8:
        direction = normalize(joints[hand_idx] - joints[19])
    tip = base + direction * float(cfg.get("handle_length_m", 0.18) + cfg.get("shaft_length_m", 0.50))

    up = np.asarray([0.0, 0.0, 1.0])
    side = np.cross(direction, up)
    if np.linalg.norm(side) < 1e-8:
        side = np.asarray([0.0, 1.0, 0.0])
    side = normalize(side)
    radius = float(cfg.get("head_radius_m", 0.12))
    theta = np.linspace(0, 2 * np.pi, 32)
    ring = tip + radius * (np.cos(theta)[:, None] * side + np.sin(theta)[:, None] * direction)
    return base, tip, ring


def set_equal_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=(0, 1))
    maxs = pts.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    radius = max(span * 0.58, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius * 0.6), center[2] + radius * 0.9)


def draw_skeleton(ax, joints: np.ndarray, color: str, label: str, alpha: float) -> None:
    first = True
    for a, b in EDGES:
        ax.plot(
            [joints[a, 0], joints[b, 0]],
            [joints[a, 1], joints[b, 1]],
            [joints[a, 2], joints[b, 2]],
            color=color,
            linewidth=2.0,
            alpha=alpha,
            label=label if first else None,
        )
        first = False
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=10, alpha=alpha)


def draw_racket(ax, joints: np.ndarray, cfg: dict, color: str, label: str, alpha: float) -> None:
    base, tip, ring = racket_geometry(joints, cfg)
    ax.plot([base[0], tip[0]], [base[1], tip[1]], [base[2], tip[2]], color=color, linewidth=3.0, alpha=alpha, label=label)
    ax.plot(ring[:, 0], ring[:, 1], ring[:, 2], color=color, linewidth=2.0, alpha=alpha)
    ax.scatter([tip[0]], [tip[1]], [tip[2]], color=color, s=18, alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    body, ref, steps = load_series(args.diagnostic, args.sequence)
    body = body[:: args.stride]
    ref = ref[:: args.stride]
    steps = steps[:: args.stride]

    pts = np.concatenate([body, ref], axis=1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=args.fps, bitrate=2800)

    with writer.saving(fig, str(args.output), dpi=140):
        for i in range(len(body)):
            ax.clear()
            set_equal_axes(ax, pts)
            ax.view_init(elev=16, azim=-68)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(
                f"{args.sequence}\n"
                "Passive visual-only racket | no dynamics / no collision / no policy modification\n"
                f"PHC rollout=red | reference=blue | transform={cfg.get('mode')}\n"
                "racket tip error unavailable: no calibrated reference racket transform",
                fontsize=9,
            )
            draw_skeleton(ax, ref[i], "#4b7bec", "reference body", 0.42)
            draw_skeleton(ax, body[i], "#ff3b30", "PHC body", 0.95)
            draw_racket(ax, ref[i], cfg, "#00a8ff", "reference visual racket", 0.55)
            draw_racket(ax, body[i], cfg, "#ff9500", "PHC visual racket", 0.95)
            ax.text2D(0.02, 0.02, f"frame {steps[i]}", transform=ax.transAxes)
            ax.legend(loc="upper right", fontsize=8)
            writer.grab_frame()

    plt.close(fig)
    print(f"wrote passive racket video: {args.output}")


if __name__ == "__main__":
    main()
