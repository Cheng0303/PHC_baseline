#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--md", required=True)
    args = parser.parse_args()

    in_path = Path(args.input)
    csv_path = Path(args.csv)
    md_path = Path(args.md)
    payload = json.loads(in_path.read_text())
    summary = payload.get("summary", {})
    rows_all = payload.get("per_sequence", [])
    rows = [row for row in rows_all if row.get("valid_export", True)]
    invalid_rows = [row for row in rows_all if not row.get("valid_export", True)]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_name",
        "frame_count",
        "count",
        "completed",
        "termination_frame",
        "mean_mpjpe",
        "max_mpjpe",
        "mean_root_error",
        "max_root_error",
        "mean_body_error",
        "max_body_error",
        "valid_export",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows_all, key=lambda x: x.get("sequence_name", "")):
            writer.writerow({key: row.get(key) for key in fieldnames})

    top_mpjpe = sorted(rows, key=lambda x: x.get("max_mpjpe", 0.0), reverse=True)[:10]
    top_root = sorted(rows, key=lambda x: x.get("max_root_error", 0.0), reverse=True)[:10]

    def table(items, metric):
        lines = ["| sequence | completed | termination | mean MPJPE | max MPJPE | mean root | max root |",
                 "|---|---:|---:|---:|---:|---:|---:|"]
        for item in items:
            term = item.get("termination_frame")
            lines.append(
                "| {sequence_name} | {completed} | {term} | {mean_mpjpe:.6f} | {max_mpjpe:.6f} | {mean_root_error:.6f} | {max_root_error:.6f} |".format(
                    term="" if term is None else term,
                    **item,
                )
            )
        return "\n".join(lines)

    md = [
        "# PHC NewRacket Dataset Metrics",
        "",
        f"- Motion file: `{summary.get('motion_file', '')}`",
        f"- Recorded sequences: {summary.get('num_sequences_recorded')} / {summary.get('num_sequences_total')}",
        f"- Valid exported rows: {len(rows)}",
        f"- Invalid exported rows excluded from summary: {len(invalid_rows)}",
        f"- Completed: {summary.get('num_completed')}",
        f"- Terminated: {summary.get('num_terminated')}",
        f"- Dataset mean MPJPE: {summary.get('dataset_mean_mpjpe')}",
        f"- Dataset max MPJPE: {summary.get('dataset_max_mpjpe')}",
        f"- Dataset mean root error: {summary.get('dataset_mean_root_error')}",
        f"- Dataset max root error: {summary.get('dataset_max_root_error')}",
        "",
        "## Top 10 Max MPJPE",
        "",
        table(top_mpjpe, "max_mpjpe"),
        "",
        "## Top 10 Max Root Error",
        "",
        table(top_root, "max_root_error"),
        "",
        f"CSV: `{csv_path}`",
        f"JSON: `{in_path}`",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
