#!/usr/bin/env python3
"""Evaluate PHC rollout racket tip error after a passed calibration."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from fit_wrist_to_racket_transform import hand_frames


def load_diag(path: Path, sequence: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = []
    ref = []
    mpjpe = []
    root = []
    steps = []
    for rec in payload["records"]:
        keys = rec.get("motion_keys", [])
        if keys and keys[0] != sequence:
            continue
        body.append(np.asarray(rec["body_pos"][0], dtype=np.float64))
        ref.append(np.asarray(rec["ref_body_pos"][0], dtype=np.float64))
        mpjpe.append(float(rec["mpjpe"][0]))
        root.append(float(rec["root_error"][0]))
        steps.append(int(rec["step"]))
    return np.stack(body), np.stack(ref), np.asarray(mpjpe), np.asarray(root), steps


def reconstruct(body: np.ndarray, local: np.ndarray) -> np.ndarray:
    hand, rot = hand_frames(body)
    return hand + np.einsum("tij,j->ti", rot, local)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transform", required=True, type=Path)
    parser.add_argument("--reference_npz", nargs="+", required=True, type=Path)
    parser.add_argument("--diagnostic", nargs="+", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    args = parser.parse_args()

    transform = json.loads(args.transform.read_text(encoding="utf-8"))
    if not transform.get("calibration_passed", False):
        raise RuntimeError("Calibration did not pass; refusing to compute PHC racket tip accuracy.")
    local = np.asarray(transform["local_tip_offset"], dtype=np.float64)

    diag_by_name = {p.stem: p for p in args.diagnostic}
    frame_rows = []
    summaries = []
    for ref_path in args.reference_npz:
        ref_data = np.load(ref_path, allow_pickle=True)
        sequence = str(ref_data["sequence"].item())
        safe = sequence.replace("/", "_")
        diag_path = None
        for p in args.diagnostic:
            if safe in p.name:
                diag_path = p
                break
        if diag_path is None:
            raise KeyError(sequence)
        rollout_body, _, body_mpjpe, root_error, steps = load_diag(diag_path, sequence)
        ref_tip = np.asarray(ref_data["reference_racket_tip_phc_world"], dtype=np.float64)
        n = min(len(ref_tip), len(rollout_body))
        pred_tip = reconstruct(rollout_body[:n], local)
        err = np.linalg.norm(pred_tip - ref_tip[:n], axis=1)
        for i in range(n):
            frame_rows.append({
                "sequence": sequence,
                "frame_idx": steps[i],
                "reference_racket_tip_phc_world_x": ref_tip[i, 0],
                "reference_racket_tip_phc_world_y": ref_tip[i, 1],
                "reference_racket_tip_phc_world_z": ref_tip[i, 2],
                "rollout_racket_tip_phc_world_x": pred_tip[i, 0],
                "rollout_racket_tip_phc_world_y": pred_tip[i, 1],
                "rollout_racket_tip_phc_world_z": pred_tip[i, 2],
                "racket_tip_error_m": err[i],
                "body_mpjpe_m": body_mpjpe[i],
                "root_error_m": root_error[i],
                "completed": True,
            })
        summaries.append({
            "sequence": sequence,
            "mean_racket_tip_error": float(np.mean(err)),
            "median_racket_tip_error": float(np.median(err)),
            "p90_racket_tip_error": float(np.percentile(err, 90)),
            "max_racket_tip_error": float(np.max(err)),
            "mean_body_mpjpe": float(np.mean(body_mpjpe[:n])),
            "completed": True,
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)
    payload = {"summary": summaries, "frame_metrics_count": len(frame_rows)}
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Success Racket Tip Evaluation", "", "| sequence | mean | median | p90 | max | mean body MPJPE |", "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"| `{row['sequence']}` | {row['mean_racket_tip_error']:.6f} | {row['median_racket_tip_error']:.6f} | {row['p90_racket_tip_error']:.6f} | {row['max_racket_tip_error']:.6f} | {row['mean_body_mpjpe']:.6f} |")
    args.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
