"""Build a portable, self-verifying delivery from a completed CRA-v2 campaign.

The builder does not open signal arrays, label files, or target-final data.  It
first runs the independent completed-campaign auditor, then copies frozen code,
checkpoints, development receipts, documentation, and a frozen v4 comparison
into a new directory.  Exact membership and SHA-256 hashes are verified before
the ZIP archive is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import zipfile


PROJECT = Path(__file__).resolve().parents[1]
AUDITOR_SOURCE = PROJECT / "audit" / "verify_completed_channel_robust_v2_campaign.py"
VERIFIER_SOURCE = Path(__file__).resolve().with_name("verify_channel_robust_v2_delivery.py")
PREDICTOR_SOURCE = Path(__file__).resolve().with_name("predict_channel_robust_v2.py")
CLASS_MAPPING_SOURCE = Path(__file__).resolve().with_name("channel_robust_class_mappings.json")
NOISE_SOURCE = PROJECT / "protocol_next" / "evaluate_channel_robust_v2_noise.py"
DEVICE_HASH_FIX_SOURCE = PROJECT / "audit" / "seal_channel_robust_v2_device_hash_fix.py"
PAPER_REPORT_GENERATOR_SOURCE = PROJECT / "analysis" / "generate_cra_v2_paper_report.py"
PAPER_REPORT_LOCK_SOURCE = PROJECT / "analysis" / "requirements-paper-report.lock.txt"
MECHANISM_ASSETS = (
    (
        PROJECT / "audit" / "channel_robust_v2a_mechanism_diagnostic_20260925.md",
        Path("evidence/channel_robust_v2a_mechanism_diagnostic_20260925.md"),
    ),
    (
        PROJECT / "audit" / "channel_robust_v2a_mechanism_diagnostic_20260925.json",
        Path("evidence/channel_robust_v2a_mechanism_diagnostic_20260925.json"),
    ),
    (
        PROJECT / "audit" / "channel_robust_v2a_screen_v5_mechanism_review_20260925.md",
        Path("evidence/channel_robust_v2a_screen_v5_mechanism_review_20260925.md"),
    ),
    (
        PROJECT / "audit" / "diagnose_channel_robust_v2a_20260925.py",
        Path("tools/diagnose_channel_robust_v2a_20260925.py"),
    ),
)
CAMPAIGN_KIND = "channel_robust_v2a_u_fixedk_campaign_v1"
MECHANISM_REQUIRED_PLAN_SHA = "8bf146d8412b4aaef104e2ae39769a42494597880090d1d36b072e1b85f6fac9"
EXPERIMENT_FILES = (
    "plan.json",
    "plan.sha256",
    "fit_state.json",
    "fit_results.json",
    "predev_fit_audit.json",
    "predev_seal.json",
    "dev_state.json",
    "dev_results.json",
    "pipeline_state.json",
    "summary.json",
    "REPORT.md",
)
REFERENCE_FILES = (
    "plan.json",
    "plan.sha256",
    "fit_results.json",
    "predev_seal.json",
    "dev_results.json",
    "summary.json",
    "REPORT.md",
)
RUN_FILES = ("best_model.pth", "run_metadata.json", "run_state.json", "epoch_metrics.jsonl")
DOC_ASSETS = (
    ("delivery_tools/V2_DELIVERY_README.md", "docs/V2_DELIVERY_README.md"),
    (
        "delivery_tools/V2_THIRD_PARTY_AND_RIGHTS.md",
        "docs/V2_THIRD_PARTY_AND_RIGHTS.md",
    ),
    ("audit/channel_robust_v2_design_decision_20260925.md", "docs/v2_design_decision.md"),
    ("audit/channel_robust_v2_validation_protocol_20260925.md", "docs/v2_validation_protocol.md"),
    ("audit/channel_robust_v2_validation_protocol_20260925.json", "docs/v2_validation_protocol.json"),
    (
        "audit/channel_robust_v2a_staged_validation_addendum_20260925.md",
        "docs/v2a_staged_validation_addendum.md",
    ),
    ("audit/channel_robust_v2_code_review_20260925.md", "docs/v2_code_review.md"),
    (
        "audit/channel_robust_delivery_readiness_20260925.md",
        "docs/delivery_readiness_preimplementation_audit.md",
    ),
    (
        "audit/channel_robust_v2_delivery_implementation_20260925.md",
        "docs/v2_delivery_implementation_receipt.md",
    ),
    ("论文实验执行标准.md", "docs/论文实验执行标准.md"),
    ("执行资源约定.md", "docs/执行资源约定.md"),
)
CACHE_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".mypy_cache", ".ruff_cache"
)


if str(AUDITOR_SOURCE.parent) not in sys.path:
    sys.path.insert(0, str(AUDITOR_SOURCE.parent))
from verify_completed_channel_robust_v2_campaign import audit_completed_campaign

if str(VERIFIER_SOURCE.parent) not in sys.path:
    sys.path.insert(0, str(VERIFIER_SOURCE.parent))
from verify_channel_robust_v2_delivery import verify as verify_delivery


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def runtime_path(value) -> Path:
    if isinstance(value, dict):
        key = "windows" if os.name == "nt" else "wsl"
        value = value.get(key)
        if value is None:
            raise ValueError(f"path record lacks {key}")
    text = str(value)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 7:
        text = f"{text[5].upper()}:/{text[7:]}"
    return Path(text).resolve(strict=True)


def copy_file(source: Path, target: Path) -> None:
    source = Path(source).resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"source is not a file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    shutil.copy2(source, target)
    if sha256(source) != sha256(target):
        raise IOError(f"copy hash differs: {source}")


def package_versions() -> dict:
    result = {}
    for name in ("torch", "numpy", "scipy", "scikit-learn", "pytest"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def model_destination(role: str, identifier: str) -> Path:
    family = {
        "source_anchor": "anchors",
        "common_anchor_control": "controls",
        "v2a_candidate": "candidates",
    }[role]
    return Path("models") / family / identifier


def copy_run(
    campaign: Path,
    output: Path,
    row: dict,
    *,
    identifier: str,
    role: str,
) -> tuple[str, dict]:
    run = campaign / "runs" / Path(str(row.get("run"))).name
    destination = model_destination(role, identifier)
    hashes = {}
    for name in RUN_FILES:
        copy_file(run / name, output / destination / name)
        hashes[name] = sha256(output / destination / name)
    if hashes["best_model.pth"] != row.get("checkpoint_sha256"):
        raise ValueError("copied checkpoint differs from registered fit")
    return (destination / "best_model.pth").as_posix(), hashes


def validate_noise_result(path: Path, *, plan_sha: str, candidate_by_id: dict) -> dict:
    value = read_json(Path(path).resolve(strict=True))
    arm = value.get("arm")
    identifier = arm.get("id") if isinstance(arm, dict) else None
    expected = candidate_by_id.get(identifier)
    slot_results = value.get("slot_results")
    receipt = value.get("receipt")
    if (
        value.get("kind") != "channel_robust_alignment_awgn_dev_v2"
        or value.get("campaign_plan_sha256") != plan_sha
        or value.get("evaluated_split") != "target_dev_only"
        or value.get("target_final_evaluated") is not False
        or expected is None
        or arm != expected["arm"]
        or value.get("checkpoint_sha256") != expected["checkpoint_sha256"]
        or not isinstance(value.get("n_samples"), int)
        or value["n_samples"] < 1
        or not isinstance(slot_results, dict)
        or len(slot_results) != 1
        or not isinstance(receipt, dict)
        or receipt.get("private_origin_metadata_opened") is not False
        or receipt.get("target_final_indices_accessed") is not False
        or receipt.get("final_signal_or_label_rows_decoded") is not False
    ):
        raise ValueError("AWGN result identity or target-final boundary differs")
    conditions = next(iter(slot_results.values())).get("conditions")
    expected_conditions = [("clean", "clean", None)] + [
        (f"snr{snr}_seed{seed}", snr, seed)
        for snr in (20, 10, 0)
        for seed in (1701, 1702, 1703)
    ]
    actual_conditions = [
        (row.get("condition_id"), row.get("snr_db"), row.get("noise_seed"))
        for row in conditions
    ] if isinstance(conditions, list) else None
    if actual_conditions != expected_conditions:
        raise ValueError("AWGN result condition matrix differs")
    affected = value.get("target_view", {}).get("affected_physical_slots")
    if not isinstance(affected, list) or not affected:
        raise ValueError("AWGN result affected-slot identity is missing")
    return value


def validate_paper_report(
    report_root: Path,
    report_manifest: Path,
    *,
    campaign: Path,
    campaign_audit: dict,
    gate_passed: bool,
) -> dict:
    report_root = Path(report_root).resolve(strict=True)
    report_manifest = Path(report_manifest).resolve(strict=True)
    if not report_root.is_dir() or not report_manifest.is_file():
        raise ValueError("paper report root/manifest differs")
    receipt = read_json(report_root / "report_receipt.json")
    expected_gate = "PASSED" if gate_passed else "FAILED"
    expected_promotion = "PROMOTED" if gate_passed else "NOT_PROMOTED"
    if (
        receipt.get("schema") != "cra_v2_paper_report_receipt_v1"
        or receipt.get("target_final_evidence_read") is not False
        or receipt.get("integrity_status") != "PASS"
        or receipt.get("performance_gate_status") != expected_gate
        or receipt.get("promotion_status") != expected_promotion
        or receipt.get("manifest_sha256") != sha256(report_manifest)
        or receipt.get("generator", {}).get("sha256") != sha256(PAPER_REPORT_GENERATOR_SOURCE)
    ):
        raise ValueError("paper report receipt boundary differs")
    outputs = receipt.get("output_files")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("paper report output registry is missing")
    registered = set()
    for row in outputs:
        relative = row.get("path")
        digest = row.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in registered
            or sha256(report_root / relative) != digest
        ):
            raise ValueError("paper report output registry differs")
        registered.add(relative)
    actual = {
        path.relative_to(report_root).as_posix()
        for path in report_root.rglob("*")
        if path.is_file() and path.name != "report_receipt.json"
    }
    if registered != actual:
        raise ValueError("paper report membership differs from receipt")
    inputs = receipt.get("input_files")
    if not isinstance(inputs, list):
        raise ValueError("paper report input registry is missing")
    plan_sha = sha256(campaign / "plan.json")
    if not any(row.get("sha256") == plan_sha for row in inputs):
        raise ValueError("paper report is not bound to this campaign plan")
    audit_rows = []
    for row in inputs:
        raw_path = row.get("path")
        if not isinstance(raw_path, str) or "completed_audit" not in Path(raw_path).name:
            continue
        source = Path(raw_path).resolve(strict=True)
        saved_audit = read_json(source)
        comparable_saved = {key: value for key, value in saved_audit.items() if key != "created_unix"}
        comparable_live = {key: value for key, value in campaign_audit.items() if key != "created_unix"}
        if sha256(source) == row.get("sha256") and comparable_saved == comparable_live:
            audit_rows.append(row)
    if not audit_rows:
        raise ValueError("paper report is not bound to the independent campaign audit")
    if campaign_audit.get("status") != "PASS":
        raise ValueError("campaign audit status differs")
    return receipt


def build(
    campaign: Path,
    output: Path,
    archive: Path,
    *,
    noise_results=(),
    paper_report: Path | None = None,
    paper_report_manifest: Path | None = None,
) -> dict:
    campaign = Path(campaign).resolve(strict=True)
    output = Path(output).resolve()
    archive = Path(archive).resolve()
    sidecar = Path(str(archive) + ".sha256")
    if output.exists() or archive.exists() or sidecar.exists():
        raise FileExistsError("fresh delivery directory, archive, and sidecar are required")
    if output == campaign or output.is_relative_to(campaign) or campaign.is_relative_to(output):
        raise ValueError("delivery and campaign directories must be disjoint")
    audit = audit_completed_campaign(campaign)
    if audit.get("status") != "PASS" or audit.get("target_final_evaluated") is not False:
        raise ValueError("completed campaign did not pass the independent audit")
    plan = read_json(campaign / "plan.json")
    plan_sha = sha256(campaign / "plan.json")
    fits = read_json(campaign / "fit_results.json")
    dev_rows = read_json(campaign / "dev_results.json")
    summary = read_json(campaign / "summary.json")
    pipeline = read_json(campaign / "pipeline_state.json")
    if plan.get("kind") != CAMPAIGN_KIND or plan.get("output_name") != campaign.name:
        raise ValueError("campaign plan identity differs")
    mechanism_evidence_included = plan_sha == MECHANISM_REQUIRED_PLAN_SHA
    if len(fits.get("anchors", [])) != 4 or len(fits.get("arms", [])) != 32:
        raise ValueError("campaign fit matrix differs")
    gate_passed = summary["causal_comparison"]["expansion_gate"]["passed"]
    if gate_passed is not pipeline.get("screen_gate_passed"):
        raise ValueError("campaign gate differs between summary and pipeline")
    if noise_results and gate_passed is not True:
        raise ValueError("AWGN results cannot be packaged before the registered causal gate passes")
    if (paper_report is None) is not (paper_report_manifest is None):
        raise ValueError("paper report and its manifest must be supplied together")
    paper_receipt = None
    if paper_report is not None:
        paper_receipt = validate_paper_report(
            paper_report,
            paper_report_manifest,
            campaign=campaign,
            campaign_audit=audit,
            gate_passed=gate_passed,
        )

    output.mkdir(parents=True)
    packaged_campaign = output / "experiment" / campaign.name
    for name in EXPERIMENT_FILES:
        copy_file(campaign / name, packaged_campaign / name)
    seal = read_json(campaign / "predev_seal.json")
    correction_sha = seal.get("device_hash_correction_receipt_sha256")
    if correction_sha is not None:
        for name in (
            "device_hash_seal_correction_receipt.json",
            "device_hash_seal_execution_receipt.json",
        ):
            copy_file(campaign / name, packaged_campaign / name)
        copy_file(
            DEVICE_HASH_FIX_SOURCE,
            output / "audit" / "seal_channel_robust_v2_device_hash_fix.py",
        )
    for arm in plan["arms"]:
        source = campaign / "dev" / f"{arm['id']}.json"
        copy_file(source, output / "scores" / f"{arm['id']}.json")
        copy_file(source, packaged_campaign / "dev" / f"{arm['id']}.json")

    shutil.copytree(campaign / "code", output / "code", ignore=CACHE_IGNORE)
    for relative, digest in plan["code_sha256"].items():
        if sha256(output / "code" / relative) != digest:
            raise ValueError("copied frozen campaign code differs: " + relative)

    models = []
    for row in fits["anchors"]:
        item = row["anchor"]
        checkpoint, artifacts = copy_run(
            campaign,
            output,
            row,
            identifier=item["id"],
            role="source_anchor",
        )
        models.append(
            {
                "id": item["id"],
                "role": "source_anchor",
                "task_id": item["task"],
                "dataset": item["dataset"],
                "target_slots": None,
                "checkpoint": checkpoint,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "artifacts_sha256": artifacts,
                "target_final_evaluated": False,
            }
        )
    score_by_id = {row["arm"]["id"]: row for row in dev_rows}
    for row in fits["arms"]:
        arm = row["arm"]
        role = (
            "common_anchor_control"
            if arm["variant"] == "common_anchor_control"
            else "v2a_candidate"
        )
        checkpoint, artifacts = copy_run(
            campaign,
            output,
            row,
            identifier=arm["id"],
            role=role,
        )
        score = score_by_id[arm["id"]]
        models.append(
            {
                "id": arm["id"],
                "role": role,
                "variant": arm["variant"],
                "task_id": arm["task"],
                "dataset": arm["dataset"],
                "family": arm["family"],
                "target_slots": arm["target"],
                "checkpoint": checkpoint,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "artifacts_sha256": artifacts,
                "development_score": f"scores/{arm['id']}.json",
                "development_result_sha256": score["result_sha256"],
                "promoted_for_followup": bool(gate_passed and role == "v2a_candidate"),
                "target_final_evaluated": False,
            }
        )
    write_json(
        output / "model_registry.json",
        {
            "schema": "channel_robust_v2_model_registry_v1",
            "architecture_contract": {
                "shared_model_structure": True,
                "target_channel_training_policy": (
                    "one common architecture; separate training and checkpoint per target view"
                ),
                "universal_weight_model_delivered": False,
                "universal_weight_model_status": "reserved for a later extension",
                "model_code_sha256": plan["code_sha256"],
            },
            "models": models,
            "recommended_model_id": None,
            "target_final_evaluated": False,
        },
    )

    candidate_by_id = {
        row["arm"]["id"]: row
        for row in fits["arms"]
        if row["arm"]["variant"] == "channel_robust_alignment_v2a"
    }
    packaged_noise = []
    seen_noise = set()
    for index, source in enumerate(noise_results):
        source = Path(source).resolve(strict=True)
        value = validate_noise_result(
            source,
            plan_sha=audit["campaign_plan_sha256"],
            candidate_by_id=candidate_by_id,
        )
        key = (
            value["arm"]["id"],
            tuple(value["target_view"]["affected_physical_slots"]),
        )
        if key in seen_noise:
            raise ValueError("duplicate AWGN arm/affected-slot result")
        seen_noise.add(key)
        destination = Path("robustness") / "results" / f"{index:02d}_{value['arm']['id']}.json"
        copy_file(source, output / destination)
        packaged_noise.append(
            {
                "arm_id": value["arm"]["id"],
                "affected_physical_slots": value["target_view"]["affected_physical_slots"],
                "path": destination.as_posix(),
                "sha256": sha256(output / destination),
                "evaluated_split": "target_dev_only",
                "target_final_evaluated": False,
            }
        )
    write_json(
        output / "robustness" / "result_registry.json",
        {
            "schema": "channel_robust_v2_awgn_result_registry_v1",
            "results": packaged_noise,
            "target_final_evaluated": False,
        },
    )

    write_json(output / "audit" / "completed_v2_campaign_audit.json", audit)
    copy_file(AUDITOR_SOURCE, output / "tools" / AUDITOR_SOURCE.name)
    copy_file(Path(__file__).resolve(), output / "tools" / Path(__file__).name)
    copy_file(VERIFIER_SOURCE, output / "tools" / VERIFIER_SOURCE.name)
    copy_file(PREDICTOR_SOURCE, output / "inference" / PREDICTOR_SOURCE.name)
    copy_file(CLASS_MAPPING_SOURCE, output / "inference" / CLASS_MAPPING_SOURCE.name)
    copy_file(CLASS_MAPPING_SOURCE, output / "config" / CLASS_MAPPING_SOURCE.name)
    copy_file(NOISE_SOURCE, output / "robustness" / NOISE_SOURCE.name)
    if mechanism_evidence_included:
        for source, destination in MECHANISM_ASSETS:
            copy_file(source, output / destination)

    if paper_receipt is not None:
        paper_root = Path(paper_report).resolve(strict=True)
        for source in sorted(paper_root.rglob("*")):
            if source.is_file():
                copy_file(source, output / "paper_report" / source.relative_to(paper_root))
        copy_file(
            Path(paper_report_manifest),
            output / "paper_report" / "report_manifest.json",
        )
        copy_file(
            PAPER_REPORT_GENERATOR_SOURCE,
            output / "tools" / PAPER_REPORT_GENERATOR_SOURCE.name,
        )
        copy_file(
            PAPER_REPORT_LOCK_SOURCE,
            output / "environment" / PAPER_REPORT_LOCK_SOURCE.name,
        )
        report_audit_source = None
        for row in paper_receipt.get("input_files", []):
            raw_path = row.get("path")
            if not isinstance(raw_path, str) or "completed_audit" not in Path(raw_path).name:
                continue
            source = Path(raw_path).resolve(strict=True)
            value = read_json(source)
            if (
                value.get("campaign_plan_sha256") == audit["campaign_plan_sha256"]
                and value.get("status") == "PASS"
                and sha256(source) == row.get("sha256")
            ):
                report_audit_source = source
                break
        if report_audit_source is None:
            raise ValueError("paper report campaign audit input cannot be packaged")
        copy_file(
            report_audit_source,
            output / "audit" / "paper_report_campaign_audit.json",
        )

    task_registry = {
        "schema": "channel_robust_v2_task_registry_v1",
        "window_samples": 2048,
        "source_slots": [0, 1, 2],
        "sampling_rate_hz": {"waterpump": 24000, "gearbox": 24000},
        "tasks": {
            task: {
                "metadata": plan["task_metadata"][task],
                "data_metadata_sha256": plan["data"][task]["sha256"],
                "runtime_path_required": True,
                "target_final_included": False,
            }
            for task in plan["data"]
        },
    }
    write_json(output / "config" / "task_registry.json", task_registry)
    write_json(
        output / "config" / "paths.local.example.json",
        {
            "schema": "channel_robust_v2_local_paths_v1",
            "note": "Copy this file outside the verified delivery or rename it before filling local paths.",
            "tasks": {
                task: {
                    "public_root_windows": "<drive>:\\path\\to\\public_root",
                    "public_root_wsl": "/mnt/<drive>/path/to/public_root",
                    "private_audit_root_windows": "<drive>:\\path\\to\\private_audit_root",
                    "private_audit_root_wsl": "/mnt/<drive>/path/to/private_audit_root",
                }
                for task in plan["data"]
            },
        },
    )

    copy_file(PROJECT / "workbench" / "requirements-cpu.lock.txt", output / "environment" / "requirements-cpu.lock.txt")
    copy_file(PROJECT / "workbench" / "requirements-gpu.lock.txt", output / "environment" / "requirements-gpu.lock.txt")
    write_json(
        output / "environment" / "environment.json",
        {
            "schema": "channel_robust_v2_build_environment_v1",
            "created_unix": time.time(),
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
            "training_runtime": plan.get("training"),
            "portable_paths": ["Windows", "WSL"],
        },
    )
    for source, destination in DOC_ASSETS:
        copy_file(PROJECT / source, output / destination)

    reference = plan["strong_v4_reference"]
    reference_root = runtime_path(reference["campaign_root"])
    for name in REFERENCE_FILES:
        copy_file(reference_root / name, output / "reference" / "v4" / name)
    receipt_source = runtime_path(reference["completed_audit_receipt"]["path"])
    copy_file(receipt_source, output / "reference" / "v4" / "completed_audit_receipt.json")
    reference_project = runtime_path(reference["project_root"])
    reference_controls = []
    for item in reference["controls"]:
        source = reference_project / item["checkpoint_relative"]
        destination = Path("reference") / "v4" / "checkpoints" / f"{item['arm']['id']}.pth"
        copy_file(source, output / destination)
        if sha256(output / destination) != item["checkpoint_sha256"]:
            raise ValueError("copied v4 control checkpoint differs")
        reference_controls.append(
            {
                "arm": item["arm"],
                "checkpoint": destination.as_posix(),
                "checkpoint_sha256": item["checkpoint_sha256"],
            }
        )
    write_json(
        output / "reference" / "v4" / "control_registry.json",
        {
            "schema": "channel_robust_v2_v4_reference_registry_v1",
            "historical_threshold_baseline": True,
            "decides_expansion": False,
            "controls": reference_controls,
        },
    )

    status = {
        "schema": "channel_robust_v2_delivery_status_v1",
        "campaign_completed": True,
        "campaign_name": campaign.name,
        "campaign_directory": f"experiment/{campaign.name}",
        "campaign_plan_sha256": audit["campaign_plan_sha256"],
        "source_anchor_count": 4,
        "branch_checkpoint_count": 32,
        "control_checkpoint_count": 16,
        "candidate_checkpoint_count": 16,
        "shared_model_structure": True,
        "separate_checkpoint_per_target_view": True,
        "screen_gate_passed": gate_passed,
        "integrity_status": "PASS",
        "performance_gate_status": "PASSED" if gate_passed else "FAILED",
        "recommended_model_id": None,
        "intended_use": "research_evaluation_only" if not gate_passed else "registered_followup_evaluation",
        "promotion_status": (
            "registered_development_gate_passed"
            if gate_passed
            else "not_promoted_beyond_completed_system_evaluation"
        ),
        "target_dev_evaluated": True,
        "target_final_evaluated": False,
        "awgn_evaluated": bool(packaged_noise),
        "awgn_result_count": len(packaged_noise),
        "awgn_scope": "target_dev_followup" if packaged_noise else "not_run",
        "paper_report_included": paper_receipt is not None,
        "mechanism_evidence_included": mechanism_evidence_included,
        "paper_report": (
            {
                "path": "paper_report/REPORT.md",
                "receipt": "paper_report/report_receipt.json",
                "receipt_sha256": sha256(output / "paper_report" / "report_receipt.json"),
                "manifest": "paper_report/report_manifest.json",
                "manifest_sha256": sha256(output / "paper_report" / "report_manifest.json"),
                "integrity_status": paper_receipt["integrity_status"],
                "performance_gate_status": paper_receipt["performance_gate_status"],
                "promotion_status": paper_receipt["promotion_status"],
            }
            if paper_receipt is not None
            else None
        ),
        "completed_campaign_audit": {
            "path": "audit/completed_v2_campaign_audit.json",
            "status": "PASS",
            "audit_code_sha256": audit["audit_code_sha256"],
        },
    }
    write_json(output / "STATUS.json", status)
    write_json(
        output / "robustness" / "README.json",
        {
            "schema": "channel_robust_v2_awgn_usage_v1",
            "allowed_after_screen_gate": True,
            "evaluated_split": "target_dev_only",
            "snr_db": [20, 10, 0],
            "noise_seeds": [1701, 1702, 1703],
            "target_final_evaluated": False,
        },
    )
    (output / "robustness" / "README.md").write_text(
        "# AWGN target-development evaluation\n\n"
        "The evaluator is registered for candidate arms only and requires the causal screen gate to pass. "
        f"Supply `--campaign <delivery>/experiment/{campaign.name}`, `--code-root <delivery>/code`, the explicit candidate "
        "checkpoint, and local public/private audit roots. It evaluates target-dev only; target-final stays sealed.\n",
        encoding="utf-8",
    )
    (output / "README_CN.md").write_text(
        "# CRA-v2A-U-fixedK 可验证交付包\n\n"
        "本包保存一套共用模型结构，以及每个任务、每种目标通道视图分别训练的 checkpoint。"
        "`model_registry.json` 是模型入口；不存在声称可直接覆盖任意目标通道的通用权重。\n\n"
        "先运行 `python tools/verify_channel_robust_v2_delivery.py .`。推理入口为 "
        "`inference/predict_channel_robust_v2.py`，输入必须是 float32、形状 `[N,C,2048]` 的原始窗口。"
        "实验室水泵/齿轮箱使用 24 kHz 合同；程序不会自动重采样或识别传感器。\n\n"
        f"完整性审计：`PASS`。性能门：`{'PASSED' if gate_passed else 'FAILED'}`。"
        f"本次注册开发门为 `{gate_passed}`，推荐模型 ID 为 `null`，用途限定为 `research_evaluation_only`。"
        "性能门失败意味着候选方法未达到预注册提升标准，不能作为高性能或已推广模型交付。"
        "目标最终分区从未评估。AWGN 是开发集后续验证，"
        "不得写成独立最终测试结果。完整状态见 `STATUS.json`，结果与限制以 `experiment/REPORT.md` "
        "和 `docs/` 为准。\n",
        encoding="utf-8",
    )

    pre_sum_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name not in {"MANIFEST.json", "SHA256SUMS.txt"}
    )
    sums = [f"{sha256(path)}  {path.relative_to(output).as_posix()}" for path in pre_sum_files]
    (output / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    manifest_files = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "MANIFEST.json"
    }
    write_json(
        output / "MANIFEST.json",
        {
            "schema": "channel_robust_v2_delivery_manifest_v1",
            "created_unix": time.time(),
            "campaign_plan_sha256": audit["campaign_plan_sha256"],
            "target_final_evaluated": False,
            "files": manifest_files,
        },
    )
    verification = verify_delivery(output)

    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                bundle.write(path, (Path(output.name) / path.relative_to(output)).as_posix())
    with zipfile.ZipFile(archive, "r") as bundle:
        if bundle.testzip() is not None:
            raise IOError("delivery ZIP failed CRC verification")
    archive_sha = sha256(archive)
    sidecar.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    return {
        "schema": "channel_robust_v2_delivery_build_v1",
        "status": "PASS",
        "delivery": str(output),
        "archive": str(archive),
        "archive_sha256": archive_sha,
        "archive_sidecar": str(sidecar),
        "manifest_sha256": verification["manifest_sha256"],
        "files_verified": verification["files_verified"],
        "models_verified": verification["models_verified"],
        "awgn_results_packaged": len(packaged_noise),
        "paper_report_packaged": paper_receipt is not None,
        "screen_gate_passed": gate_passed,
        "target_final_evaluated": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument(
        "--noise-result",
        type=Path,
        action="append",
        default=[],
        help="optional registered v2 target-dev AWGN result; repeat for multiple arms/views",
    )
    parser.add_argument("--paper-report", type=Path)
    parser.add_argument("--paper-report-manifest", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "not_executed",
                    "campaign": str(args.campaign),
                    "output": str(args.output),
                    "archive": str(args.archive),
                    "target_final_evaluated": False,
                },
                ensure_ascii=False,
            )
        )
        return 0
    result = build(
        args.campaign,
        args.output,
        args.archive,
        noise_results=args.noise_result,
        paper_report=args.paper_report,
        paper_report_manifest=args.paper_report_manifest,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
