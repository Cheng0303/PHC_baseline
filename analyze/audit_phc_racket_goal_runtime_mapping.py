#!/usr/bin/env python3
"""Audit PHC motion rows against racket-aware task rows.

The goal is to decide whether runtime motion frame indices should index task
rows directly or use source-frame labels.  This script is read-only and does
not run PHC policy inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def resolve_path(root: Path, path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return root / path


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def audit_row(repo_root: Path, motion_data: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    sequence = row["sequence"]
    out: dict[str, Any] = {
        "sequence": sequence,
        "phc_motion_key_found": sequence in motion_data,
        "phc_motion_num_frames": "",
        "phc_motion_fps": "",
        "phc_motion_dt": "",
        "task_npz_num_rows": "",
        "source_frame_idx_contiguous": "",
        "source_frame_first": "",
        "source_frame_last": "",
        "frame_count_match": False,
        "row_root_alignment_max_error": "",
        "source_label_root_alignment_max_error": "",
        "mapping_convention_determined": "missing",
        "eligible_for_runtime_time_adapter": False,
        "failure_reason": "",
    }
    if not boolish(row.get("task_export_passed")) or not boolish(row.get("integrity_check_passed")):
        out["failure_reason"] = "task export or integrity check did not pass"
        return out
    if sequence not in motion_data:
        out["failure_reason"] = "sequence key missing from PHC motion file"
        return out

    npz_path = resolve_path(repo_root, row["npz_path"])
    if not npz_path.exists():
        out["mapping_convention_determined"] = "missing"
        out["failure_reason"] = f"task npz missing: {npz_path}"
        return out
    task = np.load(npz_path, allow_pickle=True)
    entry = motion_data[sequence]
    root = to_numpy(entry["root_trans_offset"]).astype(np.float64)
    fps = float(entry.get("fps", 30))
    task_root = np.asarray(task["root_position_phc_world"], dtype=np.float64)
    source_idx = np.asarray(task["source_frame_idx"], dtype=np.int64)

    phc_n = len(root)
    task_n = len(source_idx)
    out.update(
        {
            "phc_motion_num_frames": phc_n,
            "phc_motion_fps": fps,
            "phc_motion_dt": 1.0 / fps,
            "task_npz_num_rows": task_n,
            "source_frame_idx_contiguous": bool(np.all(np.diff(source_idx) == 1)) if task_n > 1 else True,
            "source_frame_first": int(source_idx[0]),
            "source_frame_last": int(source_idx[-1]),
            "frame_count_match": phc_n == task_n,
        }
    )

    if phc_n == task_n and len(task_root) == task_n:
        row_err = np.linalg.norm(root - task_root, axis=1)
        out["row_root_alignment_max_error"] = float(row_err.max()) if len(row_err) else 0.0
    else:
        row_err = np.asarray([np.inf])
        out["row_root_alignment_max_error"] = "not_tested_length_mismatch"

    source_label_err = np.asarray([np.inf])
    if len(task_root) == task_n and source_idx.min(initial=0) >= 0 and source_idx.max(initial=-1) < phc_n:
        source_label_err = np.linalg.norm(root[source_idx] - task_root, axis=1)
        out["source_label_root_alignment_max_error"] = float(source_label_err.max()) if len(source_label_err) else 0.0
    else:
        out["source_label_root_alignment_max_error"] = "not_applicable_source_idx_out_of_phc_range"

    row_ok = bool(phc_n == task_n and np.isfinite(row_err).all() and row_err.max(initial=0.0) < 1e-6)
    source_label_ok = bool(
        np.isfinite(source_label_err).all()
        and source_label_err.max(initial=0.0) < 1e-6
        and len(source_label_err) == task_n
    )

    if row_ok:
        out["mapping_convention_determined"] = "row_index_aligned"
        out["eligible_for_runtime_time_adapter"] = True
        if source_label_ok and not out["source_frame_idx_contiguous"]:
            out["failure_reason"] = "row and source-label alignment both match; row alignment retained because PHC frame count equals compressed task rows"
    elif source_label_ok:
        out["mapping_convention_determined"] = "source_frame_label_aligned"
        out["eligible_for_runtime_time_adapter"] = True
    else:
        out["mapping_convention_determined"] = "ambiguous"
        out["failure_reason"] = "neither row-index root alignment nor source-frame-label root alignment matched"
    return out


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Runtime Mapping Audit",
        "",
        "Scope: PHC motion-to-racket-task frame mapping audit only. No policy inference or training was run.",
        "",
        "## Summary",
        "",
        f"- Clips checked: `{summary['clips_checked']}`",
        f"- PHC motion keys found: `{summary['phc_motion_keys_found']}`",
        f"- Eligible for runtime-time adapter: `{summary['eligible_for_runtime_time_adapter']}`",
        f"- Excluded / ambiguous: `{summary['excluded_or_ambiguous']}`",
        f"- Continuous / non-contiguous `source_frame_idx`: `{summary['continuous_source_frame_sequences']}` / `{summary['non_contiguous_source_frame_sequences']}`",
        f"- Mapping conventions: `{summary['mapping_convention_counts']}`",
        "",
        "## Conclusion",
        "",
        summary["conclusion"],
        "",
        "## Non-Contiguous Clips",
        "",
        "Non-contiguous clips are still eligible when PHC motion frame rows align to task NPZ rows. In this dataset, the root-position row comparison shows that PHC motion rows are already compressed/aligned with task rows; `source_frame_idx` remains a provenance label, not the runtime PHC frame index.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest_csv", required=True, type=Path)
    parser.add_argument("--phc_motion_file", required=True, type=Path)
    parser.add_argument("--output_json", required=True, type=Path)
    parser.add_argument("--output_report", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path.cwd()
    rows = [
        row for row in load_manifest(args.manifest_csv)
        if boolish(row.get("task_export_passed")) and boolish(row.get("integrity_check_passed")) and boolish(row.get("dynamic_replay_passed"))
    ]
    motion_data = joblib.load(args.phc_motion_file)
    audited = [audit_row(repo_root, motion_data, row) for row in rows]
    convention_counts = Counter(row["mapping_convention_determined"] for row in audited)
    eligible = sum(1 for row in audited if row["eligible_for_runtime_time_adapter"])
    continuous = sum(1 for row in audited if row["source_frame_idx_contiguous"] is True)
    non_contiguous = sum(1 for row in audited if row["source_frame_idx_contiguous"] is False)
    summary = {
        "scope": "runtime mapping audit; no PHC policy execution",
        "manifest_csv": str(args.manifest_csv),
        "phc_motion_file": str(args.phc_motion_file),
        "clips_checked": len(audited),
        "phc_motion_keys_found": sum(1 for row in audited if row["phc_motion_key_found"]),
        "eligible_for_runtime_time_adapter": eligible,
        "excluded_or_ambiguous": len(audited) - eligible,
        "continuous_source_frame_sequences": continuous,
        "non_contiguous_source_frame_sequences": non_contiguous,
        "mapping_convention_counts": dict(convention_counts),
        "all_eligible": eligible == len(audited),
        "conclusion": (
            "All checked cross-session task clips are row-index aligned with the PHC motion pkl. "
            "For non-contiguous source_frame_idx clips, PHC motion frame rows match task rows and root positions exactly; "
            "source_frame_idx should be treated as provenance/source-label metadata, not as the PHC runtime frame index."
            if eligible == len(audited) and convention_counts.get("row_index_aligned", 0) == len(audited)
            else "Some clips are missing or ambiguous; inspect runtime_mapping_per_sequence.csv before using a runtime-time adapter."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(args.output_csv, audited)
    write_report(args.output_report, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
