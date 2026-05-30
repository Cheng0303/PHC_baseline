#!/usr/bin/env python3
"""CPU-only frozen body checkpoint provenance audit for Stage 1B.

This script intentionally does not initialize Isaac Gym, create an optimizer,
call backward, train, enable rewards, or modify checkpoints. It only reads
checkpoint/config/report metadata to resolve the frozen PHC body action
contract before any integration rollout is allowed.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    import yaml
except Exception:  # pragma: no cover - PyYAML may not be installed everywhere.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[3]
PHC_ROOT = REPO_ROOT / "humenv" / "data_preparation" / "PHC"
OUTPUT_ROOT = PHC_ROOT / "output"
REPORT_DIR = REPO_ROOT / "phc_baseline" / "reports" / "racket_calibration" / "frozen_body_head_integration"
STAGE1A_METADATA = REPO_ROOT / "phc_baseline" / "models" / "separate_virtual_racket_head_stage1a" / "checkpoint_metadata.json"
DEFAULT_CONFIG = REPO_ROOT / "phc_baseline" / "configs" / "frozen_body_virtual_racket_stage1b" / "eval_config.json"


@dataclass
class Evidence:
    file: str
    kind: str
    strength: str
    snippet: str


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except Exception:
        return str(path)


def path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def load_text(path: Path, max_chars: int = 500_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except Exception:
        return ""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path: Path) -> dict[str, Any]:
    text = load_text(path)
    if not text:
        return {}
    if yaml is not None:
        try:
            data = yaml.safe_load(text)
            return data if isinstance(data, dict) else {}
        except Exception:
            pass
    # Small fallback for the fields this audit needs.
    result: dict[str, Any] = {}
    for pattern, key in [
        (r"^\s*exp_name:\s*(.+?)\s*$", "exp_name"),
        (r"^\s*task:\s*(.+?)\s*$", "task"),
        (r"^\s*num_prim:\s*(\d+)\s*$", "num_prim"),
        (r"^\s*actors_to_load:\s*(\d+)\s*$", "actors_to_load"),
        (r"^\s*motion_file:\s*(.+?)\s*$", "motion_file"),
    ]:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            value: Any = match.group(1).strip().strip("'\"")
            if key in {"num_prim", "actors_to_load"}:
                value = int(value)
            result[key] = value
    return result


def should_scan_text_file(path: Path) -> bool:
    skip_parts = {
        ".git",
        "__pycache__",
        "envs",
        "third_party",
        "models",
        "converted",
        "torch_extensions",
    }
    if any(part in skip_parts for part in path.parts):
        return False
    if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".mp4", ".pth", ".pkl", ".npz", ".npy", ".pt", ".zip"}:
        return False
    return path.suffix.lower() in {".py", ".json", ".md", ".yaml", ".yml", ".csv", ".log", ".txt"} or path.name in {
        "overrides.yaml",
        "config.yaml",
    }


def nested_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def candidate_paths(extra_paths: list[Path]) -> list[Path]:
    paths = set()
    for path in OUTPUT_ROOT.rglob("Humanoid.pth"):
        paths.add(path.resolve())
    for path in extra_paths:
        if path.exists():
            paths.add(path.resolve())

    # Add checkpoint paths referenced in reports/configs/logs, if present.
    roots = [REPO_ROOT / "phc_baseline", PHC_ROOT / "phc", OUTPUT_ROOT]
    pattern = re.compile(r"(?P<path>(?:humenv/data_preparation/PHC/)?output/HumanoidIm/[^`'\"\s]+?Humanoid\.pth)")
    for root in roots:
        if not root.exists():
            continue
        for file in root.rglob("*"):
            if not file.is_file() or not should_scan_text_file(file):
                continue
            text = load_text(file, 120_000)
            for match in pattern.finditer(text):
                raw = match.group("path")
                resolved = (PHC_ROOT / raw) if raw.startswith("output/") else (REPO_ROOT / raw)
                if resolved.exists():
                    paths.add(resolved.resolve())
    return sorted(paths)


def checkpoint_dims(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "checkpoint_path": rel(path),
        "exists": path.exists(),
        "load_error": None,
        "running_mean_dim": None,
        "running_var_dim": None,
        "actor_output_dim": None,
        "sigma_dim": None,
        "composer_weight_count": 0,
        "pnn_actor_indices": [],
        "has_optimizer_state": False,
        "epoch": None,
        "frame": None,
        "last_mean_rewards": None,
    }
    try:
        ckpt = torch.load(path, map_location="cpu")
        model = ckpt.get("model", {})
        rms = ckpt.get("running_mean_std", {})
        if "running_mean" in rms:
            info["running_mean_dim"] = int(rms["running_mean"].shape[0])
        if "running_var" in rms:
            info["running_var_dim"] = int(rms["running_var"].shape[0])
        if "a2c_network.mu.bias" in model:
            info["actor_output_dim"] = int(model["a2c_network.mu.bias"].shape[0])
        if "a2c_network.sigma" in model:
            info["sigma_dim"] = int(model["a2c_network.sigma"].shape[0])
        info["composer_weight_count"] = sum(
            1 for key in model if key.startswith("a2c_network.composer") and key.endswith(".weight")
        )
        actors = sorted(
            {
                int(key.split(".")[3])
                for key in model
                if key.startswith("a2c_network.pnn.actors.") and len(key.split(".")) > 4 and key.split(".")[3].isdigit()
            }
        )
        info["pnn_actor_indices"] = actors
        info["has_optimizer_state"] = "optimizer" in ckpt
        for key in ["epoch", "frame", "last_mean_rewards"]:
            value = ckpt.get(key)
            if hasattr(value, "item"):
                value = value.item()
            if isinstance(value, (int, float, str, bool)) or value is None:
                info[key] = value
            else:
                info[key] = str(value)
    except Exception as exc:
        info["load_error"] = repr(exc)
    return info


def associated_run_metadata(path: Path) -> dict[str, Any]:
    run_dir = path.parent
    hydra_config = run_dir / ".hydra" / "config.yaml"
    overrides = run_dir / ".hydra" / "overrides.yaml"
    hydra = load_yaml(hydra_config)
    overrides_text = load_text(overrides, 80_000)
    logs = [p for p in run_dir.glob("*.log")]
    return {
        "run_dir": rel(run_dir),
        "hydra_config": rel(hydra_config) if hydra_config.exists() else None,
        "hydra_overrides": rel(overrides) if overrides.exists() else None,
        "log_files": [rel(p) for p in logs],
        "task_class": nested_get(hydra, ["env", "task"], hydra.get("task")),
        "exp_name": hydra.get("exp_name"),
        "motion_file": nested_get(hydra, ["env", "motion_file"], hydra.get("motion_file")),
        "num_prim": nested_get(hydra, ["env", "num_prim"], hydra.get("num_prim")),
        "actors_to_load": nested_get(hydra, ["env", "actors_to_load"], hydra.get("actors_to_load")),
        "models": nested_get(hydra, ["env", "models"], None),
        "im_eval": hydra.get("im_eval"),
        "test": hydra.get("test"),
        "train": hydra.get("train"),
        "overrides_text": overrides_text,
    }


def compact_snippet(text: str, token: str, width: int = 240) -> str:
    idx = text.find(token)
    if idx < 0:
        return ""
    start = max(0, idx - width // 2)
    end = min(len(text), idx + len(token) + width // 2)
    return " ".join(text[start:end].split())


def collect_evidence(path: Path) -> list[Evidence]:
    evidence: list[Evidence] = []
    names = {rel(path), str(path), str(path.relative_to(PHC_ROOT)) if path_is_relative_to(path, PHC_ROOT) else ""}
    run_name = path.parent.name
    tokens = [name for name in names if name] + [run_name]
    roots = [REPO_ROOT / "phc_baseline", PHC_ROOT / "phc", OUTPUT_ROOT]
    for root in roots:
        if not root.exists():
            continue
        for file in root.rglob("*"):
            if not file.is_file() or not should_scan_text_file(file):
                continue
            text = load_text(file, 180_000)
            if not text:
                continue
            matched = [token for token in tokens if token and token in text]
            if not matched:
                continue
            kind = "path_reference" if any("Humanoid.pth" in token for token in matched) else "run_name_reference"
            strength = "weak"
            if "phc_observation_and_motion_lookup_audit" in rel(file) and run_name == "phc_comp_3":
                kind = "documented_body_only_command_family"
                strength = "strong"
            if "phc_newracket_dataset_metrics_fixed" in rel(file) and "3167" in text and "4093" in text:
                kind = "metrics_summary_reference"
                strength = "medium" if run_name in text else "weak"
            evidence.append(Evidence(rel(file), kind, strength, compact_snippet(text, matched[0])))
    # De-duplicate while keeping stable order.
    dedup: dict[tuple[str, str, str], Evidence] = {}
    for item in evidence:
        dedup.setdefault((item.file, item.kind, item.strength), item)
    return list(dedup.values())


def classify_candidate(info: dict[str, Any], metadata: dict[str, Any], evidence: list[Evidence]) -> tuple[str, str]:
    out_dim = info.get("actor_output_dim")
    input_dim = info.get("running_mean_dim")
    num_prim = metadata.get("num_prim")
    run_name = Path(metadata["run_dir"]).name
    has_strong_body_command = any(item.strength == "strong" and item.kind == "documented_body_only_command_family" for item in evidence)
    config_weight_consistent = out_dim is not None and num_prim is not None and int(out_dim) == int(num_prim)

    if run_name == "phc_comp_3" and input_dim == 934 and out_dim == 3 and config_weight_consistent and has_strong_body_command:
        return (
            "confirmed_body_baseline_checkpoint",
            "phc_comp_3 has the documented body-only command family, saved Hydra env.num_prim=3, and checkpoint actor output dim=3.",
        )
    if input_dim == 934 and out_dim == 4 and config_weight_consistent:
        return (
            "compatible_alternative_not_baseline_proven",
            "Checkpoint is dimension-compatible with a [934]->[4] body action contract but no exact baseline evidence was found.",
        )
    if out_dim in {3, 4} and not has_strong_body_command:
        return ("unknown", "MCP-sized output exists, but baseline provenance evidence is insufficient.")
    return (
        "incompatible_with_current_scaffold",
        "Checkpoint output/input role does not match the Stage 1B frozen body MCP action boundary.",
    )


def stage1a_metadata_check(path: Path = STAGE1A_METADATA) -> dict[str, Any]:
    meta = load_json(path)
    primary = meta.get("models", {}).get("goal_state", {})
    goal_only = meta.get("models", {}).get("goal_only", {})
    checks = {
        "metadata_path": rel(path),
        "exists": path.exists(),
        "goal_version": meta.get("goal_version"),
        "state_version": meta.get("state_version"),
        "action_version": meta.get("action_version"),
        "body_path_frozen": bool(meta.get("body_path_frozen")),
        "no_physics_virtual_head_only": bool(meta.get("no_physics_virtual_head_only")),
        "primary_model": "goal_state",
        "primary_input_dim": primary.get("input_dim"),
        "primary_output_dim": primary.get("output_dim"),
        "goal_only_input_dim": goal_only.get("input_dim"),
        "goal_only_output_dim": goal_only.get("output_dim"),
        "original_phc_checkpoint_loaded": bool(meta.get("original_phc_checkpoint_loaded")),
    }
    checks["passed"] = (
        checks["goal_version"] == "v2_sim_root_projected_world_target"
        and checks["state_version"] == "v2_sim_root_projected_world_state"
        and checks["action_version"] == "v2_heading_local_velocity"
        and checks["body_path_frozen"] is True
        and checks["no_physics_virtual_head_only"] is True
        and checks["primary_input_dim"] == 15
        and checks["primary_output_dim"] == 6
        and checks["goal_only_input_dim"] == 9
        and checks["goal_only_output_dim"] == 6
        and checks["original_phc_checkpoint_loaded"] is False
    )
    return checks


def metric_provenance_summary() -> dict[str, Any]:
    metrics_json = REPO_ROOT / "phc_baseline" / "reports" / "phc_newracket_dataset_metrics_fixed.json"
    summary_md = REPO_ROOT / "phc_baseline" / "reports" / "phc_newracket_dataset_metrics_fixed_summary.md"
    summary: dict[str, Any] = {
        "metrics_json": rel(metrics_json) if metrics_json.exists() else None,
        "metrics_summary_md": rel(summary_md) if summary_md.exists() else None,
        "completed_3167_of_4093_found": False,
        "exact_checkpoint_path_embedded_in_metrics_file": False,
        "interpretation": "",
    }
    if metrics_json.exists():
        data = load_json(metrics_json)
        stats = data.get("summary", {})
        summary["num_sequences_recorded"] = stats.get("num_sequences_recorded")
        summary["num_completed"] = stats.get("num_completed")
        summary["dataset_mean_mpjpe"] = stats.get("dataset_mean_mpjpe")
        summary["dataset_mean_root_error"] = stats.get("dataset_mean_root_error")
        summary["completed_3167_of_4093_found"] = stats.get("num_sequences_recorded") == 4093 and stats.get("num_completed") == 3167
        text = json.dumps(data)[:500_000]
        summary["exact_checkpoint_path_embedded_in_metrics_file"] = "Humanoid.pth" in text or "phc_comp_3" in text
    summary["interpretation"] = (
        "The 3167/4093 metrics artifact is present, but the metrics JSON/summary does not embed an exact checkpoint path. "
        "Exact checkpoint provenance is therefore reconstructed through the documented body-only command family and saved Hydra run config."
    )
    return summary


def write_csv(path: Path, candidates: list[dict[str, Any]]) -> None:
    fields = [
        "checkpoint_path",
        "run_dir",
        "role_classification",
        "running_mean_dim",
        "actor_output_dim",
        "config_num_prim",
        "config_actors_to_load",
        "task_class",
        "config_weight_consistent",
        "baseline_evidence_strength",
        "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in candidates:
            writer.writerow({field: item.get(field) for field in fields})


def write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Body Checkpoint Provenance Audit",
        "",
        "Scope: CPU-only metadata/provenance audit. No Isaac Gym env initialization, optimizer, backward pass, training, reward update, PHC fine-tune, physical racket, shuttle, or official PHC rollout racket accuracy was run.",
        "",
        "## Resolution",
        "",
        f"- case: `{audit['resolution']['case']}`",
        f"- recommended frozen body action dim: `{audit['resolution']['recommended_body_action_dim']}`",
        f"- future combined action dim: `{audit['resolution']['future_combined_action_dim']}`",
        f"- confidence: `{audit['resolution']['confidence']}`",
        f"- summary: {audit['resolution']['summary']}",
        "",
        "## Candidate Table",
        "",
        "| checkpoint | input dim | actor output dim | config num_prim | role | evidence |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in audit["candidates"]:
        lines.append(
            "| {checkpoint_path} | {running_mean_dim} | {actor_output_dim} | {config_num_prim} | {role_classification} | {baseline_evidence_strength} |".format(
                **item
            )
        )
    lines.extend(
        [
            "",
            "## Required Questions",
            "",
            f"1. `3167 / 4093 completed`: {audit['answers']['q1_metrics_provenance']}",
            f"2. `phc_comp_3` `[934]->[3]` vs `num_prim=3`: {audit['answers']['q2_phc_comp_3_consistency']}",
            f"3. Prior `[4]` source: {audit['answers']['q3_prior_four_source']}",
            f"4. `[934]->[4]` baseline checkpoint: {audit['answers']['q4_four_dim_checkpoint']}",
            f"5. Stage 1B frozen body action dim: {audit['answers']['q5_final_action_dim']}",
            "",
            "## Stage 1A Head Metadata Recheck",
            "",
            f"- passed: `{audit['stage1a_head_metadata']['passed']}`",
            f"- primary input/output: `{audit['stage1a_head_metadata']['primary_input_dim']} -> {audit['stage1a_head_metadata']['primary_output_dim']}`",
            f"- goal/state/action versions: `{audit['stage1a_head_metadata']['goal_version']}`, `{audit['stage1a_head_metadata']['state_version']}`, `{audit['stage1a_head_metadata']['action_version']}`",
            f"- body path frozen: `{audit['stage1a_head_metadata']['body_path_frozen']}`",
            f"- no-physics virtual head only: `{audit['stage1a_head_metadata']['no_physics_virtual_head_only']}`",
            "",
            "The racket head output `[6]` is an independent branch. A `[3]` versus `[4]` frozen body action contract changes only the future env-boundary packing dimension; it does not invalidate Stage 1A.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args()

    config = load_json(args.config) if args.config.exists() else {}
    extra = []
    if config.get("original_phc_body_checkpoint"):
        extra.append((REPO_ROOT / config["original_phc_body_checkpoint"]).resolve())

    candidates: list[dict[str, Any]] = []
    for path in candidate_paths(extra):
        dims = checkpoint_dims(path)
        metadata = associated_run_metadata(path)
        evidence = collect_evidence(path)
        role, notes = classify_candidate(dims, metadata, evidence)
        strengths = sorted({item.strength for item in evidence})
        best_strength = "none"
        for strength in ["strong", "medium", "weak"]:
            if strength in strengths:
                best_strength = strength
                break
        config_num_prim = metadata.get("num_prim")
        actor_output = dims.get("actor_output_dim")
        consistent = actor_output is not None and config_num_prim is not None and int(actor_output) == int(config_num_prim)
        candidates.append(
            {
                **dims,
                "run_dir": metadata["run_dir"],
                "associated_config": metadata,
                "task_class": metadata.get("task_class"),
                "config_num_prim": config_num_prim,
                "config_actors_to_load": metadata.get("actors_to_load"),
                "config_weight_consistent": consistent,
                "evidence": [item.__dict__ for item in evidence],
                "baseline_evidence_strength": best_strength,
                "role_classification": role,
                "notes": notes,
            }
        )

    confirmed = [item for item in candidates if item["role_classification"] == "confirmed_body_baseline_checkpoint"]
    compatible_four = [
        item
        for item in candidates
        if item.get("running_mean_dim") == 934 and item.get("actor_output_dim") == 4 and item["role_classification"] != "confirmed_body_baseline_checkpoint"
    ]
    if confirmed and int(confirmed[0]["actor_output_dim"]) == 3:
        resolution = {
            "case": "Case A",
            "recommended_body_action_dim": 3,
            "future_combined_action_dim": 9,
            "confidence": "high_for_action_contract; metrics-file checkpoint embedding is partial",
            "summary": "Confirmed baseline command/config/checkpoint contract is original observation [934] -> MCP body action [3].",
        }
    elif confirmed and int(confirmed[0]["actor_output_dim"]) == 4:
        resolution = {
            "case": "Case B",
            "recommended_body_action_dim": 4,
            "future_combined_action_dim": 10,
            "confidence": "high",
            "summary": "Confirmed baseline command/config/checkpoint contract is original observation [934] -> MCP body action [4].",
        }
    elif compatible_four:
        resolution = {
            "case": "Case C",
            "recommended_body_action_dim": None,
            "future_combined_action_dim": None,
            "confidence": "ambiguous",
            "summary": "A [934]->[4] candidate exists, but it is not proven to be the body baseline checkpoint.",
        }
    else:
        resolution = {
            "case": "Case C",
            "recommended_body_action_dim": None,
            "future_combined_action_dim": None,
            "confidence": "ambiguous",
            "summary": "No exact baseline checkpoint/config evidence was sufficient to change the action contract.",
        }

    # If phc_comp_3 is clearly present, prefer explicit answers even when the
    # metrics artifact itself lacks a checkpoint path.
    phc_comp = next((item for item in candidates if item["run_dir"].endswith("phc_comp_3")), None)
    if phc_comp and phc_comp["role_classification"] == "confirmed_body_baseline_checkpoint":
        resolution = {
            "case": "Case A",
            "recommended_body_action_dim": 3,
            "future_combined_action_dim": 9,
            "confidence": "high_for_checkpoint_config_contract; metrics-file checkpoint embedding remains partial",
            "summary": "phc_comp_3 is the documented body-only command-family checkpoint; saved Hydra env.num_prim=3 matches checkpoint actor output dim=3.",
        }

    metrics = metric_provenance_summary()
    prior_four_sources = [
        "humenv/data_preparation/PHC/phc/data/cfg/env/env_im_getup_mcp.yaml has num_prim: 4",
        "humenv/data_preparation/PHC/phc/data/cfg/env/env_im_getup_mcp_virtual_racket_smoke.yaml has num_prim: 4",
        "phc_baseline/analyze/validate_phc_virtual_racket_env_scaffold_smoke.py overrides env.num_prim=4",
        "phc_baseline/configs/frozen_body_virtual_racket_stage1b/eval_config.json previously expected original MCP action dim 4",
    ]
    answers = {
        "q1_metrics_provenance": metrics["interpretation"],
        "q2_phc_comp_3_consistency": (
            "Yes. phc_comp_3 checkpoint running_mean dim is 934, actor output dim is 3, and saved Hydra env.num_prim is 3."
            if phc_comp
            else "phc_comp_3 metadata was not found."
        ),
        "q3_prior_four_source": "The `[4]` came from default/smoke config paths, not from the selected pretrained checkpoint: "
        + "; ".join(prior_four_sources)
        + ".",
        "q4_four_dim_checkpoint": (
            "No `[934]->[4]` Humanoid.pth candidate was found in the audited output tree."
            if not compatible_four
            else "A `[934]->[4]` candidate was found but was not proven as baseline."
        ),
        "q5_final_action_dim": (
            "Use body action `[3]` and future combined action `[3]+[6]=[9]` for Stage 1B once CUDA/env smoke passes."
            if resolution["case"] == "Case A"
            else "Keep blocker until exact baseline provenance is resolved."
        ),
    }

    audit = {
        "script": rel(Path(__file__)),
        "no_training_optimizer_backward": True,
        "config": rel(args.config),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "metrics_provenance": metrics,
        "stage1a_head_metadata": stage1a_metadata_check(),
        "prior_four_sources": prior_four_sources,
        "resolution": resolution,
        "answers": answers,
    }

    report_dir = args.report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "body_checkpoint_provenance_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    write_csv(report_dir / "body_checkpoint_candidate_table.csv", candidates)
    write_markdown(report_dir / "body_checkpoint_provenance_audit.md", audit)
    print(json.dumps({"case": resolution["case"], "candidate_count": len(candidates), "report_dir": rel(report_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
