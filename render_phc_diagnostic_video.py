#!/usr/bin/env python3
"""Render a PHC diagnostic JSON rollout into a simple comparison video."""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


SMPL_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (4, 7), (7, 10),
    (2, 5), (5, 8), (8, 11),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 13), (13, 16), (16, 18), (18, 20), (20, 22),
    (9, 14), (14, 17), (17, 19), (19, 21), (21, 23),
]


def _load_series(path: Path) -> tuple[np.ndarray, np.ndarray, list[int], list[bool]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    body = []
    ref = []
    steps = []
    term = []
    for rec in payload["records"]:
        if "body_pos" not in rec or "ref_body_pos" not in rec:
            raise KeyError(
                "diagnostic JSON does not contain full body_pos/ref_body_pos; "
                "rerun diagnostic after enabling full body recording"
            )
        body.append(np.asarray(rec["body_pos"][0], dtype=np.float32))
        ref.append(np.asarray(rec["ref_body_pos"][0], dtype=np.float32))
        steps.append(int(rec["step"]))
        term.append(bool(rec["terminate"][0]))

    return np.stack(body), np.stack(ref), steps, term


def _set_equal_axes(ax, pts: np.ndarray) -> None:
    mins = pts.min(axis=(0, 1))
    maxs = pts.max(axis=(0, 1))
    center = (mins + maxs) / 2.0
    span = float(np.max(maxs - mins))
    radius = max(span * 0.58, 1.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(max(0.0, center[2] - radius * 0.6), center[2] + radius * 0.9)


def _draw_skeleton(ax, joints: np.ndarray, color: str, label: str, alpha: float = 1.0) -> None:
    first = True
    for a, b in SMPL_EDGES:
        if a >= len(joints) or b >= len(joints):
            continue
        ax.plot(
            [joints[a, 0], joints[b, 0]],
            [joints[a, 1], joints[b, 1]],
            [joints[a, 2], joints[b, 2]],
            color=color,
            linewidth=2.2,
            alpha=alpha,
            label=label if first else None,
        )
        first = False
    ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], color=color, s=12, alpha=alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--title", default="PHC forced rollout")
    args = parser.parse_args()

    body, ref, steps, term = _load_series(args.diagnostic)
    body = body[:: args.stride]
    ref = ref[:: args.stride]
    steps = steps[:: args.stride]
    term = term[:: args.stride]

    pts = np.concatenate([body, ref], axis=1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8), dpi=140)
    ax = fig.add_subplot(111, projection="3d")
    writer = FFMpegWriter(fps=args.fps, bitrate=2800)

    with writer.saving(fig, str(args.output), dpi=140):
        for i in range(len(body)):
            ax.clear()
            _set_equal_axes(ax, pts)
            ax.view_init(elev=16, azim=-68)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.set_title(
                f"{args.title}\nstep {steps[i]} / {steps[-1]}"
                + ("  terminated" if term[i] else "")
            )
            _draw_skeleton(ax, ref[i], "#4b7bec", "reference", alpha=0.45)
            _draw_skeleton(ax, body[i], "#ff3b30", "PHC rollout", alpha=0.95)
            ax.plot(
                [pts[:, :, 0].min(), pts[:, :, 0].max()],
                [pts[:, :, 1].min(), pts[:, :, 1].min()],
                [0.0, 0.0],
                color="black",
                linewidth=1.0,
                alpha=0.25,
            )
            ax.legend(loc="upper right")
            writer.grab_frame()

    plt.close(fig)
    print(f"wrote video: {args.output}")


if __name__ == "__main__":
    main()
