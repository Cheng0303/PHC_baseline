#!/usr/bin/env python3
"""Export a session-balanced racket-aware reference task dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from export_racket_aware_reference_task_batch import (  # noqa: E402
    boolish,
    completed_sequences,
    export_task_for_sequence,
    integrity_check,
    load_custom_smpl,
    load_phc_skeleton,
    replay_check,
    safe_name,
)


def group_of(sequence: str) -> str:
    return sequence.split("/")[0]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def existing_task_counts(paths: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*_racket_aware_reference_task.npz"):
            try:
                data = np.load(path, allow_pickle=True)
                seq = str(data["sequence"].item())
                counts[group_of(seq)] += 1
            except Exception:
                continue
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_csv", required=True, type=Path)
    parser.add_argument("--converted_motion", required=True, type=Path)
    parser.add_argument("--phc_root", required=True, type=Path)
    parser.add_argument("--source_pth_root", required=True, type=Path)
    parser.add_argument("--custom_smpl_model_path", required=True, type=Path)
    parser.add_argument("--smpl_model_path", required=True, type=Path)
    parser.add_argument("--racket_markers_json", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--audit_dir", required=True, type=Path)
    parser.add_argument("--existing_task_dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--max_groups", type=int, default=12)
    parser.add_argument("--clips_per_group", type=int, default=20)
    parser.add_argument("--min_groups", type=int, default=8)
    parser.add_argument("--min_eligible_clips", type=int, default=120)
    parser.add_argument("--gender", default="male")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    completed = completed_sequences(args.metrics_csv)
    by_group: dict[str, list[str]] = defaultdict(list)
    for seq in completed:
        by_group[group_of(seq)].append(seq)
    for seqs in by_group.values():
        seqs.sort()

    existing_counts = existing_task_counts(args.existing_task_dirs)
    counts_rows = []
    for group in sorted(by_group):
        seqs = by_group[group]
        source_available = sum((args.source_pth_root / f"{seq}.pth").exists() for seq in seqs)
        counts_rows.append(
            {
                "session_group": group,
                "completed_sequence_count": len(seqs),
                "source_pth_available_count": source_available,
                "existing_exported_task_count": existing_counts.get(group, 0),
            }
        )

    selected_groups = [
        row["session_group"]
        for row in counts_rows
        if int(row["source_pth_available_count"]) > 0
    ][: args.max_groups]
    selected_sequences = []
    for group in selected_groups:
        selected_sequences.extend(by_group[group][: args.clips_per_group])

    marker_cfg = json.loads(args.racket_markers_json.read_text(encoding="utf-8"))
    marker_local = np.asarray(
        [
            marker_cfg["handle_anchor_local_xyz"],
            marker_cfg["tip_marker_local_xyz"],
            marker_cfg["head_center_local_xyz"],
        ],
        dtype=np.float64,
    )
    motion = joblib.load(args.converted_motion)
    skeleton_state_cls, skeleton_tree = load_phc_skeleton(args.phc_root)
    smplx = load_custom_smpl(args.custom_smpl_model_path)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    tasks_dir = args.output_dir / "tasks"
    summaries_dir = args.output_dir / "summaries"
    rows: list[dict[str, Any]] = []
    smpl_model_cache: dict[int, Any] = {}
    group_ranks: Counter[str] = Counter()
    for seq in selected_sequences:
        group = group_of(seq)
        group_ranks[group] += 1
        source_pth = args.source_pth_root / f"{seq}.pth"
        out_npz = tasks_dir / f"{safe_name(seq)}_racket_aware_reference_task.npz"
        out_summary = summaries_dir / f"{safe_name(seq)}_racket_aware_reference_task_summary.json"
        row: dict[str, Any] = {
            "sequence": seq,
            "session_group": group,
            "selection_rank_within_group": group_ranks[group],
            "frame_count": 0,
            "completed_body_baseline": True,
            "source_pth_exists": source_pth.exists(),
            "task_export_passed": False,
            "integrity_check_passed": False,
            "dynamic_replay_checked": False,
            "dynamic_replay_passed": False,
            "npz_path": str(out_npz),
            "summary_json_path": str(out_summary),
            "failure_reason": "",
        }
        try:
            if not source_pth.exists():
                raise FileNotFoundError(f"source .pth missing: {source_pth}")
            if seq not in motion:
                raise KeyError(f"sequence missing from converted motion: {seq}")
            source_meta = torch.load(source_pth, map_location="cpu")
            raw_frames = int(source_meta["trans"].shape[0])
            mask_meta = source_meta.get("mask", torch.ones(raw_frames))
            valid_frames = int(mask_meta.detach().cpu().numpy().astype(bool).sum())
            if valid_frames not in smpl_model_cache:
                smpl_model_cache[valid_frames] = smplx.SMPL(
                    model_path=str(args.smpl_model_path.resolve()),
                    gender=args.gender,
                    batch_size=valid_frames,
                ).to(device).eval()
            export_task_for_sequence(
                sequence=seq,
                source_pth=source_pth,
                entry=motion[seq],
                smpl_model=smpl_model_cache[valid_frames],
                smpl_device=device,
                skeleton_state_cls=skeleton_state_cls,
                skeleton_tree=skeleton_tree,
                marker_local=marker_local,
                output_npz=out_npz,
                output_summary_json=out_summary,
            )
            row["task_export_passed"] = True
            ok, reason, stats = integrity_check(out_npz)
            row["integrity_check_passed"] = ok
            row["frame_count"] = int(stats.get("frame_count", 0))
            if not ok:
                raise ValueError(f"integrity check failed: {reason}")
            replay_ok, _ = replay_check(out_npz)
            row["dynamic_replay_checked"] = True
            row["dynamic_replay_passed"] = replay_ok
            if not replay_ok:
                raise ValueError("dynamic replay failed")
        except Exception as exc:  # noqa: BLE001 - keep per-sequence failure in manifest
            row["failure_reason"] = str(exc)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "manifest.csv", rows)
    (args.output_dir / "manifest.json").write_text(json.dumps({"rows": rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.audit_dir / "cross_session_completed_sequence_counts.csv", counts_rows)

    eligible_rows = [row for row in rows if boolish(row["integrity_check_passed"]) and boolish(row["dynamic_replay_passed"])]
    selected_group_counts = Counter(row["session_group"] for row in rows)
    eligible_group_counts = Counter(row["session_group"] for row in eligible_rows)
    failure_reasons = Counter(row["failure_reason"] for row in rows if row["failure_reason"])
    criteria_met = len(eligible_group_counts) >= args.min_groups and len(eligible_rows) >= args.min_eligible_clips
    audit = {
        "total_completed_sequences": len(completed),
        "distinct_session_group_count": len(by_group),
        "selected_groups": selected_groups,
        "selected_clips_per_group": dict(selected_group_counts),
        "eligible_clips_per_group": dict(eligible_group_counts),
        "attempted_clip_count": len(rows),
        "task_export_passed_count": sum(boolish(row["task_export_passed"]) for row in rows),
        "integrity_passed_count": sum(boolish(row["integrity_check_passed"]) for row in rows),
        "dynamic_replay_checked_count": sum(boolish(row["dynamic_replay_checked"]) for row in rows),
        "dynamic_replay_passed_count": sum(boolish(row["dynamic_replay_passed"]) for row in rows),
        "eligible_clip_count": len(eligible_rows),
        "eligible_distinct_session_group_count": len(eligible_group_counts),
        "criteria_met_for_cross_session_diagnostic": criteria_met,
        "failure_reasons": dict(failure_reasons),
        "selection_rule": (
            "completed=True sequences grouped by sequence.split('/')[0]; groups sorted by name; "
            f"up to {args.clips_per_group} sorted clips per group; first {args.max_groups} available groups"
        ),
        "counts_csv": str(args.audit_dir / "cross_session_completed_sequence_counts.csv"),
    }
    args.audit_dir.mkdir(parents=True, exist_ok=True)
    (args.audit_dir / "cross_session_dataset_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Cross-Session Dataset Audit",
        "",
        f"- Total completed sequences: `{len(completed)}`",
        f"- Distinct session groups: `{len(by_group)}`",
        f"- Selected groups: `{', '.join(selected_groups)}`",
        f"- Attempted / eligible clips: `{len(rows)}` / `{len(eligible_rows)}`",
        f"- Eligible groups: `{len(eligible_group_counts)}`",
        f"- Dynamic replay checked / passed: `{audit['dynamic_replay_checked_count']}` / `{audit['dynamic_replay_passed_count']}`",
        f"- Criteria met for cross-session diagnostic: `{criteria_met}`",
        "",
        "Eligible clips per selected group:",
    ]
    for group in selected_groups:
        lines.append(f"- `{group}`: {eligible_group_counts.get(group, 0)} / {selected_group_counts.get(group, 0)}")
    lines += ["", "Failure reasons:"]
    if failure_reasons:
        for reason, count in failure_reasons.most_common():
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- none")
    lines += ["", "Scope: offline reference-level diagnostic dataset only; not PHC simulated rollout accuracy."]
    (args.audit_dir / "cross_session_dataset_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
