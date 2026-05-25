#!/usr/bin/env python3
"""Inspect racket OBJ geometry and define local racket markers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--racket_obj", required=True, type=Path)
    parser.add_argument("--output_config", required=True, type=Path)
    parser.add_argument("--output_plot", required=True, type=Path)
    args = parser.parse_args()

    mesh = trimesh.load(args.racket_obj, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    extent = bbox_max - bbox_min
    long_axis = int(np.argmax(extent))

    # The OBJ is long on local x. In this asset x ~= 0 is the handle side
    # because process_racket_obj adds canonical wrist_position before applying
    # the racket transform. The far negative-x end is the racket head tip.
    x = vertices[:, long_axis]
    low_band = vertices[x <= np.percentile(x, 1.0)]
    head_band = vertices[x <= np.percentile(x, 20.0)]
    high_band = vertices[x >= np.percentile(x, 99.0)]
    tip_marker = low_band.mean(axis=0)
    head_center = head_band.mean(axis=0)
    handle_anchor = np.zeros(3, dtype=np.float64)
    handle_side_center = high_band.mean(axis=0)

    cfg = {
        "racket_obj_path": str(args.racket_obj),
        "coordinate_definition": "local OBJ coordinate before per-frame racket_transform; original mesh transform uses R @ (local_vertex + canonical_wrist_position) + t",
        "handle_anchor_definition": "local [0,0,0], matching the original process_racket_obj wrist-position addition convention",
        "tip_marker_definition": "mean of vertices in the lowest 1 percent along the longest OBJ axis; for this OBJ that is the far negative local-x racket-head end",
        "head_center_definition": "mean of vertices in the lowest 20 percent along the longest OBJ axis",
        "tip_marker_local_xyz": tip_marker.tolist(),
        "head_center_local_xyz": head_center.tolist(),
        "handle_anchor_local_xyz": handle_anchor.tolist(),
        "handle_side_center_local_xyz": handle_side_center.tolist(),
        "obj_bbox_min": bbox_min.tolist(),
        "obj_bbox_max": bbox_max.tolist(),
        "obj_extent": extent.tolist(),
        "long_axis_index": long_axis,
        "validation_notes": "Please visually inspect racket_obj_marker_definition.png. Marker definition is geometric; no source metadata explicitly labels the racket tip.",
    }

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    args.output_plot.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9, 7), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    sample = vertices[:: max(1, len(vertices) // 6000)]
    ax.scatter(sample[:, 0], sample[:, 1], sample[:, 2], s=1, alpha=0.12, color="gray")
    markers = [
        ("handle_anchor", handle_anchor, "red"),
        ("tip_marker", tip_marker, "blue"),
        ("head_center", head_center, "green"),
        ("handle_side_center", handle_side_center, "orange"),
    ]
    for label, point, color in markers:
        ax.scatter([point[0]], [point[1]], [point[2]], s=50, color=color, label=label)
        ax.text(point[0], point[1], point[2], label, color=color)
    ax.set_title("Racket OBJ Marker Definition")
    ax.set_xlabel("local x")
    ax.set_ylabel("local y")
    ax.set_zlabel("local z")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.output_plot)
    plt.close(fig)
    print(json.dumps(cfg, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
