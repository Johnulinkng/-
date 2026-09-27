#!/usr/bin/env python3
"""Build an evidence-bounded CRA-v2 paper report from explicit result files.

The tool never discovers campaign files recursively.  Every input is named in
the manifest, and target-final rows require both a dedicated normalized schema
and ``--allow-final``.  This keeps development, public/external, and untouched
final evidence separate while still producing one set of paper tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Sequence


SCHEMA = "cra_v2_paper_report_manifest_v1"
NORMALIZED_SCHEMA = "cra_v2_normalized_results_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TIERS = ("development", "external", "final")
FAMILY_LABELS = {
    "same_condition_cross_sensor": "same-condition cross-sensor",
    "same_sensor_cross_condition": "condition-only",
    "cross_condition_same_sensor": "condition-only",
    "condition_shift": "condition-only",
    "cross_sensor_and_condition": "double-cross",
}
PLOT_FILES = (
    "figure_clean_macro_f1.png",
    "figure_paired_macro_f1_delta.png",
    "figure_noise_robustness.png",
)


class ReportError(RuntimeError):
    """Raised when an input would blur an evidence boundary or is malformed."""


@dataclass(frozen=True)
class LoadedInputs:
    clean_rows: tuple[dict[str, Any], ...]
    noise_rows: tuple[dict[str, Any], ...]
    gates: tuple[dict[str, Any], ...]
    source_rows: tuple[dict[str, Any], ...]
    input_hashes: tuple[dict[str, str], ...]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReportError(f"cannot read JSON: {path}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReportError(f"{name} must be finite")
    return result


def _metric(value: Any, name: str) -> float:
    result = _finite_number(value, name)
    if not 0.0 <= result <= 1.0:
        raise ReportError(f"{name} must be in [0, 1]")
    return result


def _target_tuple(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or tuple(value) not in ((0,), (1,), (2,), (0, 1, 2)):
        raise ReportError("target must be one physical slot or [0,1,2]")
    return tuple(int(item) for item in value)


def _target_view(target: Sequence[int]) -> str:
    return "all3" if tuple(target) == (0, 1, 2) else f"slot{target[0]}"


def _channel_setting(target: Sequence[int]) -> str:
    return "3-to-3" if len(target) == 3 else "3-to-1"


def _scenario(family: str, target: Sequence[int]) -> str:
    label = FAMILY_LABELS.get(family, family)
    return f"{_channel_setting(target)} / {label}"


def _zero_recall_count(value: Any) -> int:
    if not isinstance(value, list) or not value:
        raise ReportError("per_class_recall must be a nonempty list")
    recalls = [_metric(item, "per_class_recall") for item in value]
    return sum(item <= 1e-12 for item in recalls)


def _resolve_path(base: Path, raw: Any) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ReportError("source path must be a nonempty string")
    path = Path(raw)
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise ReportError(f"source path does not exist: {path}") from error


def _source_fields(source: dict[str, Any]) -> tuple[str, str, str]:
    source_id = source.get("source_id")
    tier = source.get("evidence_tier")
    role = source.get("dataset_role")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ReportError("each source needs a nonempty source_id")
    if tier not in TIERS:
        raise ReportError(f"invalid evidence_tier for {source_id}: {tier}")
    if not isinstance(role, str) or not role.strip():
        raise ReportError(f"dataset_role is missing for {source_id}")
    return source_id, tier, role


def _clean_row(
    raw: dict[str, Any], source_id: str, tier: str, role: str, split: str
) -> dict[str, Any]:
    arm = raw.get("arm")
    if not isinstance(arm, dict):
        raise ReportError(f"clean row in {source_id} has no arm")
    target = _target_tuple(arm.get("target"))
    dataset = arm.get("dataset")
    task = arm.get("task")
    family = arm.get("family")
    variant = arm.get("variant")
    seed = arm.get("seed")
    if not all(isinstance(item, str) and item for item in (dataset, task, family, variant)):
        raise ReportError(f"clean row in {source_id} has incomplete arm identity")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReportError(f"clean row in {source_id} has invalid seed")
    recalls = raw.get("per_class_recall")
    zero_count = _zero_recall_count(recalls)
    support = raw.get("support_per_class", [])
    if support and (
        not isinstance(support, list)
        or len(support) != len(recalls)
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in support)
    ):
        raise ReportError(f"clean row in {source_id} has invalid support_per_class")
    return {
        "source_id": source_id,
        "evidence_tier": tier,
        "dataset_role": role,
        "evaluated_split": split,
        "dataset": dataset,
        "task": task,
        "family": family,
        "family_label": FAMILY_LABELS.get(family, family),
        "scenario": _scenario(family, target),
        "channel_setting": _channel_setting(target),
        "target_view": _target_view(target),
        "target_slots": ",".join(str(item) for item in target),
        "variant": variant,
        "seed": seed,
        "accuracy": _metric(raw.get("accuracy"), "accuracy"),
        "macro_f1": _metric(raw.get("macro_f1"), "macro_f1"),
        "zero_recall_count": zero_count,
        "num_classes": len(recalls),
        "n_samples": sum(support) if support else raw.get("n_samples", ""),
    }


def _noise_rows(
    raw: dict[str, Any], source_id: str, tier: str, role: str
) -> list[dict[str, Any]]:
    if raw.get("target_final_evaluated") is not False:
        raise ReportError(f"noise source {source_id} does not preserve target-final")
    if raw.get("evaluated_split") != "target_dev_only":
        raise ReportError(f"noise source {source_id} is not target_dev_only")
    arm = raw.get("arm")
    if not isinstance(arm, dict):
        raise ReportError(f"noise source {source_id} has no arm")
    target = _target_tuple(arm.get("target"))
    dataset, task, family, variant, seed = (
        arm.get("dataset"), arm.get("task"), arm.get("family"), arm.get("variant"), arm.get("seed")
    )
    if not all(isinstance(item, str) and item for item in (dataset, task, family, variant)):
        raise ReportError(f"noise source {source_id} has incomplete arm identity")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ReportError(f"noise source {source_id} has invalid model seed")
    slot_results = raw.get("slot_results")
    if not isinstance(slot_results, dict) or set(slot_results) != {_target_view(target)}:
        raise ReportError(f"noise source {source_id} target view differs from arm")
    conditions = slot_results[_target_view(target)].get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise ReportError(f"noise source {source_id} has no conditions")
    result = []
    for condition in conditions:
        snr = condition.get("snr_db")
        if snr == "clean":
            snr_key, noise_seed = "clean", ""
        else:
            if isinstance(snr, bool) or not isinstance(snr, (int, float)) or not math.isfinite(float(snr)):
                raise ReportError(f"noise source {source_id} has invalid SNR")
            snr_key = str(int(snr)) if float(snr).is_integer() else str(float(snr))
            noise_seed = condition.get("noise_seed")
            if isinstance(noise_seed, bool) or not isinstance(noise_seed, int):
                raise ReportError(f"noise source {source_id} has invalid noise seed")
        recalls = condition.get("per_class_recall")
        result.append(
            {
                "source_id": source_id,
                "evidence_tier": tier,
                "dataset_role": role,
                "evaluated_split": "target_dev_only",
                "dataset": dataset,
                "task": task,
                "family": family,
                "family_label": FAMILY_LABELS.get(family, family),
                "scenario": _scenario(family, target),
                "channel_setting": _channel_setting(target),
                "target_view": _target_view(target),
                "target_slots": ",".join(str(item) for item in target),
                "variant": variant,
                "model_seed": seed,
                "condition_id": condition.get("condition_id", ""),
                "snr_db": snr_key,
                "noise_seed": noise_seed,
                "accuracy": _metric(condition.get("accuracy"), "noise accuracy"),
                "macro_f1": _metric(condition.get("macro_f1"), "noise macro_f1"),
                "zero_recall_count": _zero_recall_count(recalls),
                "num_classes": len(recalls),
            }
        )
    return result


def _gate_rows(summary: dict[str, Any], source_id: str, tier: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    candidates = []
    if isinstance(summary.get("screen_gate"), dict):
        candidates.append(("screen_gate", summary["screen_gate"]))
    for name in ("causal_comparison", "formal_comparison"):
        block = summary.get(name)
        if not isinstance(block, dict):
            continue
        for gate_name in ("expansion_gate", "historical_threshold_diagnostic"):
            if isinstance(block.get(gate_name), dict):
                candidates.append((f"{name}.{gate_name}", block[gate_name]))
    for gate_name, gate in candidates:
        result.append(
            {
                "source_id": source_id,
                "evidence_tier": tier,
                "gate": gate_name,
                "passed": gate.get("passed"),
                "jointly_improved_conditions": gate.get("jointly_improved_conditions", ""),
                "mean_accuracy_delta": gate.get("mean_accuracy_delta", ""),
                "mean_macro_f1_delta": gate.get("mean_macro_f1_delta", ""),
                "double_cross_macro_f1_delta": gate.get("double_cross_macro_f1_delta", ""),
            }
        )
    return result


def load_inputs(manifest_path: Path, *, allow_final: bool = False) -> LoadedInputs:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise ReportError(f"manifest schema must be {SCHEMA}")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ReportError("manifest sources must be a nonempty list")
    source_ids: set[str] = set()
    clean: list[dict[str, Any]] = []
    noise: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    hashes: list[dict[str, str]] = [
        {"source_id": "manifest", "path": str(manifest_path), "sha256": _sha256(manifest_path)}
    ]
    for source in sources:
        if not isinstance(source, dict):
            raise ReportError("each source must be an object")
        source_id, tier, role = _source_fields(source)
        if source_id in source_ids:
            raise ReportError(f"duplicate source_id: {source_id}")
        source_ids.add(source_id)
        kind = source.get("kind")
        path = _resolve_path(manifest_path.parent, source.get("path"))
        if tier == "final" and not allow_final:
            raise ReportError("final evidence requires explicit --allow-final")
        start_clean, start_noise = len(clean), len(noise)
        split = ""
        if kind == "campaign_dev":
            if tier == "final":
                raise ReportError("campaign_dev cannot be labeled final; use normalized_results")
            if not path.is_dir():
                raise ReportError(f"campaign_dev path must be a directory: {path}")
            named = {
                name: path / name
                for name in (
                    "plan.json", "plan.sha256", "fit_results.json", "predev_seal.json",
                    "dev_results.json", "summary.json", "pipeline_state.json",
                )
            }
            if any(not item.is_file() for item in named.values()):
                raise ReportError(f"campaign_dev source is incomplete: {path}")
            expected_plan_sha = named["plan.sha256"].read_text(encoding="ascii").strip()
            if (
                len(expected_plan_sha) != 64
                or any(character not in "0123456789abcdef" for character in expected_plan_sha)
                or _sha256(named["plan.json"]) != expected_plan_sha
            ):
                raise ReportError(f"campaign {source_id} plan SHA does not match")
            dev_rows, summary, pipeline = (
                _read_json(named["dev_results.json"]),
                _read_json(named["summary.json"]),
                _read_json(named["pipeline_state.json"]),
            )
            if (
                not isinstance(dev_rows, list)
                or summary.get("target_final_evaluated") is not False
                or pipeline.get("status") != "completed"
                or pipeline.get("target_dev_evaluated") is not True
                or pipeline.get("target_final_evaluated") is not False
            ):
                raise ReportError(f"campaign {source_id} violates the development boundary")
            seal = _read_json(named["predev_seal.json"])
            if not isinstance(seal, dict) or seal.get("target_final_evaluated") is not False:
                raise ReportError(f"campaign {source_id} lacks a final-safe pre-development seal")
            audit_path = _resolve_path(manifest_path.parent, source.get("audit_path"))
            if not audit_path.is_file():
                raise ReportError(f"campaign {source_id} independent audit must be a file")
            audit = _read_json(audit_path)
            audit_inputs = audit.get("input_sha256")
            if (
                audit.get("schema")
                not in {
                    "channel_robust_completed_campaign_audit_v1",
                    "channel_robust_v2_completed_campaign_audit_v1",
                }
                or audit.get("status") != "PASS"
                or audit.get("campaign_plan_sha256") != expected_plan_sha
                or audit.get("target_final_evaluated") is not False
                or not isinstance(audit_inputs, dict)
                or any(
                    audit_inputs.get(name) != _sha256(named[name])
                    for name in (
                        "plan.json", "fit_results.json", "predev_seal.json",
                        "dev_results.json", "summary.json", "pipeline_state.json",
                    )
                )
            ):
                raise ReportError(f"campaign {source_id} independent audit differs")
            split = "target_dev_only"
            clean.extend(_clean_row(row, source_id, tier, role, split) for row in dev_rows)
            gates.extend(_gate_rows(summary, source_id, tier))
            for name, item in named.items():
                hashes.append({"source_id": source_id, "path": str(item), "sha256": _sha256(item)})
            hashes.append(
                {"source_id": source_id, "path": str(audit_path), "sha256": _sha256(audit_path)}
            )
        elif kind == "noise_dev":
            if tier == "final":
                raise ReportError("noise_dev cannot be labeled final")
            if not path.is_file():
                raise ReportError(f"noise_dev path must be a file: {path}")
            raw = _read_json(path)
            noise.extend(_noise_rows(raw, source_id, tier, role))
            split = "target_dev_only"
            hashes.append({"source_id": source_id, "path": str(path), "sha256": _sha256(path)})
        elif kind == "normalized_results":
            if not path.is_file():
                raise ReportError(f"normalized_results path must be a file: {path}")
            raw = _read_json(path)
            if not isinstance(raw, dict) or raw.get("schema") != NORMALIZED_SCHEMA:
                raise ReportError(f"normalized source {source_id} has wrong schema")
            split = raw.get("evaluated_split")
            target_final = raw.get("target_final_evaluated")
            if tier == "final":
                if split != "target_final_only" or target_final is not True:
                    raise ReportError(f"final source {source_id} lacks an explicit final receipt")
            elif target_final is not False or split not in ("target_dev_only", "external_test"):
                raise ReportError(f"non-final source {source_id} crosses the target-final boundary")
            rows = raw.get("rows")
            if not isinstance(rows, list) or not rows:
                raise ReportError(f"normalized source {source_id} has no rows")
            clean.extend(_clean_row(row, source_id, tier, role, split) for row in rows)
            hashes.append({"source_id": source_id, "path": str(path), "sha256": _sha256(path)})
        else:
            raise ReportError(f"unsupported source kind for {source_id}: {kind}")
        source_rows.append(
            {
                "source_id": source_id,
                "kind": kind,
                "evidence_tier": tier,
                "dataset_role": role,
                "evaluated_split": split,
                "clean_rows": len(clean) - start_clean,
                "noise_rows": len(noise) - start_noise,
                "path": str(path),
            }
        )

    clean_keys: set[tuple[Any, ...]] = set()
    for row in clean:
        key = tuple(row[name] for name in (
            "evidence_tier", "dataset", "task", "target_slots", "variant", "seed"
        ))
        if key in clean_keys:
            raise ReportError(f"duplicate clean result identity: {key}")
        clean_keys.add(key)
    noise_keys: set[tuple[Any, ...]] = set()
    for row in noise:
        key = tuple(row[name] for name in (
            "evidence_tier", "dataset", "task", "target_slots", "variant",
            "model_seed", "snr_db", "noise_seed",
        ))
        if key in noise_keys:
            raise ReportError(f"duplicate noise result identity: {key}")
        noise_keys.add(key)
    return LoadedInputs(tuple(clean), tuple(noise), tuple(gates), tuple(source_rows), tuple(hashes))


def _sample_sd(values: Sequence[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def _summaries(rows: Sequence[dict[str, Any]], *, noise: bool = False) -> list[dict[str, Any]]:
    fields = [
        "evidence_tier", "dataset_role", "dataset", "task", "family",
        "family_label", "scenario", "channel_setting", "target_view", "target_slots", "variant",
    ]
    if noise:
        fields.append("snr_db")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in fields)].append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        if noise:
            # Noise realizations are repeated measurements within a model seed.
            # First average those realizations, then estimate between-model-seed
            # variability.  This avoids presenting noise-seed scatter as if it
            # were independent training-seed uncertainty.
            by_model_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for row in values:
                by_model_seed[row["model_seed"]].append(row)
            accuracy = [
                statistics.fmean(row["accuracy"] for row in seed_rows)
                for seed_rows in by_model_seed.values()
            ]
            macro_f1 = [
                statistics.fmean(row["macro_f1"] for row in seed_rows)
                for seed_rows in by_model_seed.values()
            ]
        else:
            accuracy = [row["accuracy"] for row in values]
            macro_f1 = [row["macro_f1"] for row in values]
        record = dict(zip(fields, key))
        record.update(
            {
                "n_observations": len(values),
                "n_model_seeds": len({row.get("seed", row.get("model_seed")) for row in values}),
                "accuracy_mean": statistics.fmean(accuracy),
                "accuracy_sd": _sample_sd(accuracy),
                "macro_f1_mean": statistics.fmean(macro_f1),
                "macro_f1_sd": _sample_sd(macro_f1),
                "runs_with_zero_recall": sum(row["zero_recall_count"] > 0 for row in values),
                "zero_recall_classes_mean": statistics.fmean(row["zero_recall_count"] for row in values),
                "zero_recall_classes_max": max(row["zero_recall_count"] for row in values),
            }
        )
        if noise:
            record["n_noise_seeds"] = len({row["noise_seed"] for row in values if row["noise_seed"] != ""})
            pooled_accuracy = [row["accuracy"] for row in values]
            pooled_macro_f1 = [row["macro_f1"] for row in values]
            record["accuracy_pooled_observation_sd"] = _sample_sd(pooled_accuracy)
            record["macro_f1_pooled_observation_sd"] = _sample_sd(pooled_macro_f1)
        result.append(record)
    return result


def _comparison_config(manifest_path: Path) -> list[dict[str, str]]:
    manifest = _read_json(manifest_path.resolve(strict=True))
    values = manifest.get("comparisons", [])
    if not isinstance(values, list):
        raise ReportError("comparisons must be a list")
    result = []
    names: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ReportError("comparison entries must be objects")
        name, candidate, control = value.get("name"), value.get("candidate"), value.get("control")
        if not all(isinstance(item, str) and item for item in (name, candidate, control)):
            raise ReportError("comparison name/candidate/control must be nonempty strings")
        if name in names or candidate == control:
            raise ReportError("comparison names must be unique and variants must differ")
        names.add(name)
        result.append({"name": name, "candidate": candidate, "control": control})
    return result


def _paired(rows: Sequence[dict[str, Any]], comparisons: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    identity = (
        "evidence_tier", "dataset_role", "dataset", "task", "family", "family_label",
        "scenario", "channel_setting", "target_view", "target_slots", "seed",
    )
    index = {(tuple(row[name] for name in identity), row["variant"]): row for row in rows}
    result = []
    for comparison in comparisons:
        for (key, variant), candidate in sorted(index.items(), key=lambda item: str(item[0])):
            if variant != comparison["candidate"]:
                continue
            control = index.get((key, comparison["control"]))
            if control is None:
                continue
            record = dict(zip(identity, key))
            record.update(
                {
                    "comparison": comparison["name"],
                    "candidate": comparison["candidate"],
                    "control": comparison["control"],
                    "candidate_accuracy": candidate["accuracy"],
                    "control_accuracy": control["accuracy"],
                    "accuracy_delta": candidate["accuracy"] - control["accuracy"],
                    "candidate_macro_f1": candidate["macro_f1"],
                    "control_macro_f1": control["macro_f1"],
                    "macro_f1_delta": candidate["macro_f1"] - control["macro_f1"],
                    "jointly_improved": (
                        candidate["accuracy"] > control["accuracy"]
                        and candidate["macro_f1"] > control["macro_f1"]
                    ),
                    "new_zero_recall": (
                        candidate["zero_recall_count"] > 0
                        and control["zero_recall_count"] == 0
                    ),
                }
            )
            result.append(record)
    return result


def _paired_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("evidence_tier", "dataset_role", "comparison", "channel_setting", "family_label")
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in fields)].append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        acc = [row["accuracy_delta"] for row in values]
        f1 = [row["macro_f1_delta"] for row in values]
        record = dict(zip(fields, key))
        record.update(
            {
                "n_pairs": len(values),
                "n_model_seeds": len({row["seed"] for row in values}),
                "accuracy_delta_mean": statistics.fmean(acc),
                "accuracy_delta_sd": _sample_sd(acc),
                "macro_f1_delta_mean": statistics.fmean(f1),
                "macro_f1_delta_sd": _sample_sd(f1),
                "jointly_improved_pairs": sum(row["jointly_improved"] for row in values),
                "new_zero_recall_pairs": sum(row["new_zero_recall"] for row in values),
            }
        )
        result.append(record)
    return result


def _coverage(clean: Sequence[dict[str, Any]], noise: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for tier in TIERS:
        scoped = [row for row in clean if row["evidence_tier"] == tier]
        noisy = [row for row in noise if row["evidence_tier"] == tier]
        checks = {
            "3-to-3 same-condition": any(
                row["channel_setting"] == "3-to-3" and row["family_label"] == "same-condition cross-sensor"
                for row in scoped
            ),
            "3-to-3 double-cross": any(
                row["channel_setting"] == "3-to-3" and row["family_label"] == "double-cross"
                for row in scoped
            ),
            **{
                f"3-to-1 slot{slot}": any(row["target_view"] == f"slot{slot}" for row in scoped)
                for slot in (0, 1, 2)
            },
            **{
                f"noise SNR {snr}": any(row["snr_db"] == snr for row in noisy)
                for snr in ("clean", "20", "10", "0")
            },
        }
        rows.extend(
            {"evidence_tier": tier, "requirement": name, "covered": value}
            for name, value in checks.items()
        )
    return rows


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        if not fields:
            stream.write("")
            return
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows({key: _csv_value(row.get(key)) for key in fields} for row in rows)


def _pct(value: Any, *, delta: bool = False) -> str:
    if value in (None, ""):
        return "NA"
    number = float(value) * 100.0
    return f"{number:+.2f}" if delta else f"{number:.2f}"


def _mean_sd(mean: float, sd: float | None, n: int) -> str:
    if sd is None:
        return f"{100*mean:.2f} (n={n}; SD NA)"
    return f"{100*mean:.2f} ± {100*sd:.2f}"


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    result = ["|" + "|".join(headers) + "|", "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        result.append("|" + "|".join(str(value).replace("|", "\\|") for value in row) + "|")
    return result


def _write_markdown(
    path: Path,
    title: str,
    inputs: LoadedInputs,
    clean_summary: Sequence[dict[str, Any]],
    paired_summary: Sequence[dict[str, Any]],
    noise_summary: Sequence[dict[str, Any]],
    coverage: Sequence[dict[str, Any]],
) -> None:
    tiers_present = {row["evidence_tier"] for row in inputs.clean_rows + inputs.noise_rows}
    final_present = "final" in tiers_present
    registered_failures = [
        row for row in inputs.gates
        if row["gate"] in {"screen_gate", "causal_comparison.expansion_gate"}
        and row["passed"] is False
    ]
    lines = [f"# {title}", ""]
    if registered_failures:
        lines += [
            "> **NOT PROMOTED — the registered development performance gate failed.**",
            "> Integrity checks passed, but this report does not support a superiority or high-performance claim.",
            "",
        ]
    lines += ["## Evidence boundary", ""]
    lines.extend(
        _markdown_table(
            ("Source", "Tier", "Role", "Split", "Clean rows", "Noise rows"),
            (
                (
                    row["source_id"], row["evidence_tier"], row["dataset_role"],
                    row["evaluated_split"], row["clean_rows"], row["noise_rows"],
                )
                for row in inputs.source_rows
            ),
        )
    )
    lines += [
        "",
        "- Development rows are model-development evidence and must not be described as untouched confirmation.",
        "- External rows document performance on a separately named public/external dataset; they do not replace an internal untouched final set.",
        (
            "- Untouched final evidence is present and was read only through the explicit final-results gate."
            if final_present
            else "- No target-final evidence was read; final superiority or deployment claims are unsupported by this report."
        ),
        "- A blank SD means that only one independent model seed was available; it is not reported as zero variability.",
        "",
        "## Coverage", "",
    ]
    lines.extend(
        _markdown_table(
            ("Tier", "Required cell", "Covered"),
            ((row["evidence_tier"], row["requirement"], row["covered"]) for row in coverage),
        )
    )
    lines += ["", "## Clean Accuracy and Macro-F1", ""]
    lines.extend(
        _markdown_table(
            ("Tier", "Dataset", "Task", "Scenario", "View", "Variant", "Accuracy %, mean ± SD", "Macro-F1 %, mean ± SD", "Zero-recall runs"),
            (
                (
                    row["evidence_tier"], row["dataset"], row["task"], row["scenario"],
                    row["target_view"], row["variant"],
                    _mean_sd(row["accuracy_mean"], row["accuracy_sd"], row["n_model_seeds"]),
                    _mean_sd(row["macro_f1_mean"], row["macro_f1_sd"], row["n_model_seeds"]),
                    f"{row['runs_with_zero_recall']}/{row['n_observations']}",
                )
                for row in clean_summary
            ),
        )
    )
    lines += ["", "## Paired candidate-control changes", ""]
    if paired_summary:
        lines.extend(
            _markdown_table(
                ("Tier", "Comparison", "Channel", "Family", "Pairs", "Accuracy delta pp", "Macro-F1 delta pp", "Joint gains", "New zero-recall"),
                (
                    (
                        row["evidence_tier"], row["comparison"], row["channel_setting"],
                        row["family_label"], row["n_pairs"], _pct(row["accuracy_delta_mean"], delta=True),
                        _pct(row["macro_f1_delta_mean"], delta=True), row["jointly_improved_pairs"],
                        row["new_zero_recall_pairs"],
                    )
                    for row in paired_summary
                ),
            )
        )
    else:
        lines.append("No fully matched candidate-control pairs were available.")
    lines += ["", "## AWGN robustness", ""]
    if noise_summary:
        lines.extend(
            _markdown_table(
                ("Tier", "Dataset", "Task", "View", "Variant", "SNR dB", "Accuracy %, mean ± SD", "Macro-F1 %, mean ± SD", "Zero-recall runs"),
                (
                    (
                        row["evidence_tier"], row["dataset"], row["task"], row["target_view"],
                        row["variant"], row["snr_db"],
                        _mean_sd(row["accuracy_mean"], row["accuracy_sd"], row["n_model_seeds"]),
                        _mean_sd(row["macro_f1_mean"], row["macro_f1_sd"], row["n_model_seeds"]),
                        f"{row['runs_with_zero_recall']}/{row['n_observations']}",
                    )
                    for row in noise_summary
                ),
            )
        )
    else:
        lines.append("Noise results are pending; the noise figure is a placeholder.")
    if inputs.gates:
        lines += ["", "## Frozen campaign gates", ""]
        lines.extend(
            _markdown_table(
                ("Source", "Gate", "Passed", "Joint", "Mean Acc delta pp", "Mean F1 delta pp", "Double-cross F1 delta pp"),
                (
                    (
                        row["source_id"], row["gate"], row["passed"], row["jointly_improved_conditions"],
                        _pct(row["mean_accuracy_delta"], delta=True),
                        _pct(row["mean_macro_f1_delta"], delta=True),
                        _pct(row["double_cross_macro_f1_delta"], delta=True),
                    )
                    for row in inputs.gates
                ),
            )
        )
    lines += [
        "", "## Figure files", "",
        *[f"- `{name}`" for name in PLOT_FILES], "",
        "All numeric source rows, group summaries, pairwise deltas, coverage checks, and hashes are available in the adjacent CSV and receipt files.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ReportError("PNG generation requires matplotlib") from error
    return plt


def _plot_clean(path: Path, rows: Sequence[dict[str, Any]], *, gate_failed: bool = False) -> None:
    plt = _pyplot()
    cells = sorted(
        {
            (row["evidence_tier"], row["dataset"], row["task"], row["target_view"])
            for row in rows
        }
    )
    variants = sorted({row["variant"] for row in rows})
    index = {
        (row["evidence_tier"], row["dataset"], row["task"], row["target_view"], row["variant"]): row
        for row in rows
    }
    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(cells)), 5.5), constrained_layout=True)
    if not rows:
        ax.text(0.5, 0.5, "Clean results pending", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        width = 0.8 / max(1, len(variants))
        palette = ("#35618f", "#d66b36", "#4f8d5b", "#8d63a8", "#c49a31")
        for variant_index, variant in enumerate(variants):
            positions, values, errors = [], [], []
            for cell_index, cell in enumerate(cells):
                row = index.get((*cell, variant))
                positions.append(cell_index - 0.4 + width / 2 + variant_index * width)
                values.append(math.nan if row is None else 100 * row["macro_f1_mean"])
                errors.append(math.nan if row is None or row["macro_f1_sd"] is None else 100 * row["macro_f1_sd"])
            ax.bar(
                positions, values, width=width,
                yerr=errors if any(math.isfinite(error) and error > 0 for error in errors) else None,
                capsize=2,
                color=palette[variant_index % len(palette)], edgecolor="white", label=variant,
            )
        labels = [
            f"{tier[0].upper()}:{dataset}\n{task}/{view}"
            for tier, dataset, task, view in cells
        ]
        ax.set_xticks(range(len(cells)), labels, rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Macro-F1 (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        prefix = "DEVELOPMENT — GATE FAILED\n" if gate_failed else ""
        ax.set_title(prefix + "Clean transfer results (mean and sample SD across model seeds)")
        ax.legend(fontsize=7, ncol=min(3, len(variants)))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_delta(
    path: Path, rows: Sequence[dict[str, Any]], primary: str | None, *, gate_failed: bool = False
) -> None:
    plt = _pyplot()
    selected = [row for row in rows if primary is None or row["comparison"] == primary]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in selected:
        grouped[(row["task"], row["target_view"])].append(row["macro_f1_delta"])
    tasks = sorted({key[0] for key in grouped})
    views = [view for view in ("all3", "slot0", "slot1", "slot2") if any(key[1] == view for key in grouped)]
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(views)), max(3.5, 0.75 * len(tasks))), constrained_layout=True)
    if not tasks or not views:
        ax.text(0.5, 0.5, "Paired comparison pending", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        import numpy as np

        matrix = np.full((len(tasks), len(views)), np.nan)
        for i, task in enumerate(tasks):
            for j, view in enumerate(views):
                values = grouped.get((task, view), [])
                if values:
                    matrix[i, j] = 100 * statistics.fmean(values)
        bound = max(1.0, float(np.nanmax(np.abs(matrix))))
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        for i in range(len(tasks)):
            for j in range(len(views)):
                if math.isfinite(float(matrix[i, j])):
                    ax.text(j, i, f"{matrix[i, j]:+.1f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(views)), views)
        ax.set_yticks(range(len(tasks)), tasks)
        prefix = "DEVELOPMENT — GATE FAILED\n" if gate_failed else ""
        ax.set_title(prefix + f"Macro-F1 paired delta (pp): {primary or 'configured comparison'}")
        fig.colorbar(image, ax=ax, label="Candidate - control (pp)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_noise(path: Path, rows: Sequence[dict[str, Any]], *, gate_failed: bool = False) -> None:
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, constrained_layout=True)
    order = ("clean", "20", "10", "0")
    for axis, channel in zip(axes, ("3-to-3", "3-to-1")):
        scoped = [row for row in rows if row["channel_setting"] == channel]
        variants = sorted({row["variant"] for row in scoped})
        if not scoped:
            axis.text(0.5, 0.5, f"{channel} noise results pending", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            continue
        for variant in variants:
            means, errors = [], []
            for snr in order:
                values = [row for row in scoped if row["variant"] == variant and row["snr_db"] == snr]
                per_model: dict[int, list[float]] = defaultdict(list)
                for row in values:
                    per_model[row["model_seed"]].append(row["macro_f1"])
                model_means = [statistics.fmean(group) for group in per_model.values()]
                means.append(100 * statistics.fmean(model_means) if model_means else math.nan)
                sd = _sample_sd(model_means)
                errors.append(math.nan if sd is None else 100 * sd)
            yerr = errors if any(math.isfinite(error) and error > 0 for error in errors) else None
            axis.errorbar(range(len(order)), means, yerr=yerr, marker="o", capsize=3, label=variant)
        axis.set_xticks(range(len(order)), order)
        axis.set_xlabel("SNR (dB; clean at left)")
        axis.set_title(channel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Macro-F1 (%)")
    axes[0].set_ylim(0, 100)
    prefix = "DEVELOPMENT — GATE FAILED\n" if gate_failed else ""
    fig.suptitle(prefix + "AWGN robustness (noise means per model seed; sample SD across model seeds)")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_report(
    manifest_path: Path,
    output_dir: Path,
    *,
    allow_final: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = _read_json(manifest_path)
    output = output_dir.resolve()
    try:
        output.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ReportError(f"report output must stay under {PROJECT_ROOT}") from error
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise ReportError("output directory is not empty; pass --overwrite to refresh report files")
    output.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(manifest_path, allow_final=allow_final)
    comparisons = _comparison_config(manifest_path)
    clean_summary = _summaries(inputs.clean_rows)
    noise_summary = _summaries(inputs.noise_rows, noise=True)
    paired = _paired(inputs.clean_rows, comparisons)
    paired_summary = _paired_summary(paired)
    coverage = _coverage(inputs.clean_rows, inputs.noise_rows)
    gate_failed = any(
        row["gate"] in {"screen_gate", "causal_comparison.expansion_gate"}
        and row["passed"] is False
        for row in inputs.gates
    )
    zero_rows = [
        {
            "result_kind": "clean",
            **{key: row[key] for key in (
                "source_id", "evidence_tier", "dataset", "task", "scenario", "target_view",
                "variant", "seed", "zero_recall_count", "num_classes",
            )},
            "snr_db": "clean",
            "noise_seed": "",
        }
        for row in inputs.clean_rows
    ] + [
        {
            "result_kind": "noise",
            **{key: row[key] for key in (
                "source_id", "evidence_tier", "dataset", "task", "scenario", "target_view",
                "variant", "zero_recall_count", "num_classes", "snr_db", "noise_seed",
            )},
            "seed": row["model_seed"],
        }
        for row in inputs.noise_rows
    ]

    csv_outputs = {
        "evidence_sources.csv": inputs.source_rows,
        "clean_results_long.csv": inputs.clean_rows,
        "clean_results_summary.csv": clean_summary,
        "paired_deltas_long.csv": paired,
        "paired_deltas_summary.csv": paired_summary,
        "noise_results_long.csv": inputs.noise_rows,
        "noise_results_summary.csv": noise_summary,
        "zero_recall_audit.csv": zero_rows,
        "coverage_matrix.csv": coverage,
        "campaign_gates.csv": inputs.gates,
    }
    for name, rows in csv_outputs.items():
        _write_csv(output / name, rows)
    title = manifest.get("report_title", "CRA-v2 paper evidence report")
    if not isinstance(title, str) or not title.strip():
        raise ReportError("report_title must be a nonempty string")
    _write_markdown(
        output / "REPORT.md", title, inputs, clean_summary, paired_summary, noise_summary, coverage
    )
    _plot_clean(output / PLOT_FILES[0], clean_summary, gate_failed=gate_failed)
    primary = comparisons[0]["name"] if comparisons else None
    _plot_delta(output / PLOT_FILES[1], paired, primary, gate_failed=gate_failed)
    _plot_noise(output / PLOT_FILES[2], inputs.noise_rows, gate_failed=gate_failed)

    output_names = ["REPORT.md", *csv_outputs, *PLOT_FILES]
    receipt = {
        "schema": "cra_v2_paper_report_receipt_v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "allow_final": allow_final,
        "target_final_evidence_read": any(
            row["evidence_tier"] == "final" for row in inputs.clean_rows
        ),
        "integrity_status": "PASS",
        "performance_gate_status": "FAILED" if gate_failed else "PASSED_OR_NOT_APPLICABLE",
        "promotion_status": "NOT_PROMOTED" if gate_failed else "ELIGIBLE_FOR_REVIEW",
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "python": sys.version,
            "platform": platform.platform(),
            "matplotlib": importlib.metadata.version("matplotlib"),
        },
        "counts": {
            "clean_rows": len(inputs.clean_rows),
            "noise_rows": len(inputs.noise_rows),
            "paired_rows": len(paired),
            "model_seeds": sorted({row["seed"] for row in inputs.clean_rows}),
        },
        "input_files": list(inputs.input_hashes),
        "output_files": [
            {"path": name, "sha256": _sha256(output / name)} for name in output_names
        ],
    }
    (output / "report_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-final", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_report(
        args.manifest, args.output, allow_final=args.allow_final, overwrite=args.overwrite
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output.resolve()),
                "counts": receipt["counts"],
                "target_final_evidence_read": receipt["target_final_evidence_read"],
            },
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
