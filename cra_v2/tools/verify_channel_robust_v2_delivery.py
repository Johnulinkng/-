"""Offline, fail-closed integrity check for a CRA-v2 delivery directory.

The verifier needs neither the training data nor private labels.  It validates
exact package membership, SHA-256 receipts, the common-architecture/per-target
checkpoint registry, the completed-campaign audit receipt, and the frozen v4
reference identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


MANIFEST_SCHEMA = "channel_robust_v2_delivery_manifest_v1"
STATUS_SCHEMA = "channel_robust_v2_delivery_status_v1"
REGISTRY_SCHEMA = "channel_robust_v2_model_registry_v1"
AUDIT_SCHEMA = "channel_robust_v2_completed_campaign_audit_v1"
CAMPAIGN_KIND = "channel_robust_v2a_u_fixedk_campaign_v1"
MECHANISM_REQUIRED_PLAN_SHA = "8bf146d8412b4aaef104e2ae39769a42494597880090d1d36b072e1b85f6fac9"
FORBIDDEN_DIRS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo"}
REQUIRED_FILES = {
    "README_CN.md",
    "STATUS.json",
    "SHA256SUMS.txt",
    "model_registry.json",
    "config/channel_robust_class_mappings.json",
    "config/task_registry.json",
    "config/paths.local.example.json",
    "environment/environment.json",
    "environment/requirements-cpu.lock.txt",
    "environment/requirements-gpu.lock.txt",
    "audit/completed_v2_campaign_audit.json",
    "inference/predict_channel_robust_v2.py",
    "inference/channel_robust_class_mappings.json",
    "robustness/evaluate_channel_robust_v2_noise.py",
    "robustness/README.md",
    "robustness/result_registry.json",
    "docs/V2_DELIVERY_README.md",
    "docs/V2_THIRD_PARTY_AND_RIGHTS.md",
    "tools/verify_completed_channel_robust_v2_campaign.py",
    "tools/build_channel_robust_v2_delivery.py",
    "tools/verify_channel_robust_v2_delivery.py",
}


def verify_device_hash_correction(
    root: Path,
    campaign_root: Path,
    files: dict[str, str],
    plan_sha: str,
    seal: dict,
) -> bool:
    expected = seal.get("device_hash_correction_receipt_sha256")
    if expected is None:
        return False
    campaign_relative = campaign_root.relative_to(root).as_posix()
    receipt_relative = f"{campaign_relative}/device_hash_seal_correction_receipt.json"
    execution_relative = f"{campaign_relative}/device_hash_seal_execution_receipt.json"
    tool_relative = "audit/seal_channel_robust_v2_device_hash_fix.py"
    if not {receipt_relative, execution_relative, tool_relative}.issubset(files):
        raise ValueError("device-hash correction evidence is incomplete")
    if files[receipt_relative] != expected:
        raise ValueError("device-hash correction receipt is not bound into predev seal")
    receipt = read_json(root / receipt_relative)
    execution = read_json(root / execution_relative)
    fits = read_json(campaign_root / "fit_results.json")
    expected_anchors = {
        (row["anchor"]["id"], row["checkpoint_sha256"])
        for row in fits.get("anchors", [])
    }
    expected_branches = {
        (row["arm"]["id"], row["checkpoint_sha256"])
        for row in fits.get("arms", [])
    }
    actual_anchors = {
        (row.get("anchor_id"), row.get("checkpoint_sha256"))
        for row in receipt.get("anchor_diagnostics", [])
        if row.get("all_components_match") is True
        and row.get("stored_replay_identity") == row.get("cuda_recomputed_replay_identity")
    }
    actual_branches = {
        (row.get("id"), row.get("sha256"))
        for row in receipt.get("branch_checkpoints", [])
    }
    if (
        receipt.get("schema") != "channel_robust_v2_device_hash_seal_correction_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("plan_sha256") != plan_sha
        or receipt.get("fit_results_sha256") != sha256(campaign_root / "fit_results.json")
        or receipt.get("predev_fit_audit_sha256") != sha256(campaign_root / "predev_fit_audit.json")
        or receipt.get("correction_tool_relative")
        != "audit/seal_channel_robust_v2_device_hash_fix.py"
        or receipt.get("correction_tool_sha256") != files[tool_relative]
        or receipt.get("checkpoints_rewritten") is not False
        or receipt.get("frozen_code_modified") is not False
        or receipt.get("target_dev_predictions_or_labels_read") is not False
        or receipt.get("target_final_evaluated") is not False
        or actual_anchors != expected_anchors
        or actual_branches != expected_branches
    ):
        raise ValueError("device-hash correction receipt differs")
    if (
        execution.get("schema") != "channel_robust_v2_device_hash_seal_execution_v1"
        or execution.get("status") != "PASS"
        or execution.get("device_hash_correction_receipt_sha256") != expected
        or execution.get("predev_seal_sha256") != sha256(campaign_root / "predev_seal.json")
        or execution.get("target_dev_predictions_or_labels_read") is not False
        or execution.get("target_final_evaluated") is not False
    ):
        raise ValueError("device-hash correction execution receipt differs")
    return True


def verify_paper_report(
    root: Path,
    files: dict[str, str],
    status: dict,
    *,
    gate_passed: bool,
    plan_sha: str,
) -> bool:
    included = status.get("paper_report_included")
    record = status.get("paper_report")
    if included is False and record is None:
        return False
    if included is not True or not isinstance(record, dict):
        raise ValueError("paper report status differs")
    required = {
        "paper_report/REPORT.md",
        "paper_report/report_receipt.json",
        "paper_report/report_manifest.json",
        "tools/generate_cra_v2_paper_report.py",
        "environment/requirements-paper-report.lock.txt",
        "audit/paper_report_campaign_audit.json",
    }
    if not required.issubset(files):
        raise ValueError("paper report evidence is incomplete")
    receipt = read_json(root / "paper_report" / "report_receipt.json")
    expected_gate = "PASSED" if gate_passed else "FAILED"
    expected_promotion = "PROMOTED" if gate_passed else "NOT_PROMOTED"
    if (
        record.get("path") != "paper_report/REPORT.md"
        or record.get("receipt") != "paper_report/report_receipt.json"
        or record.get("receipt_sha256") != files["paper_report/report_receipt.json"]
        or record.get("manifest") != "paper_report/report_manifest.json"
        or record.get("manifest_sha256") != files["paper_report/report_manifest.json"]
        or record.get("integrity_status") != "PASS"
        or record.get("performance_gate_status") != expected_gate
        or record.get("promotion_status") != expected_promotion
        or receipt.get("schema") != "cra_v2_paper_report_receipt_v1"
        or receipt.get("manifest_sha256") != files["paper_report/report_manifest.json"]
        or receipt.get("target_final_evidence_read") is not False
        or receipt.get("integrity_status") != "PASS"
        or receipt.get("performance_gate_status") != expected_gate
        or receipt.get("promotion_status") != expected_promotion
        or receipt.get("generator", {}).get("sha256")
        != files["tools/generate_cra_v2_paper_report.py"]
    ):
        raise ValueError("paper report receipt/status differs")
    output_rows = receipt.get("output_files")
    if not isinstance(output_rows, list) or not output_rows:
        raise ValueError("paper report output registry is missing")
    registered = {}
    for row in output_rows:
        relative = row.get("path")
        packaged = f"paper_report/{relative}" if isinstance(relative, str) else None
        if (
            not valid_relative_file(relative)
            or packaged in registered
            or files.get(packaged) != row.get("sha256")
        ):
            raise ValueError("paper report output registry differs")
        registered[packaged] = row.get("sha256")
    actual = {
        name
        for name in files
        if name.startswith("paper_report/")
        and name not in {"paper_report/report_receipt.json", "paper_report/report_manifest.json"}
    }
    if set(registered) != actual:
        raise ValueError("paper report package membership differs")
    inputs = receipt.get("input_files")
    if not isinstance(inputs, list) or not any(row.get("sha256") == plan_sha for row in inputs):
        raise ValueError("paper report is not bound to packaged campaign")
    packaged_audit_sha = files["audit/paper_report_campaign_audit.json"]
    packaged_audit = read_json(root / "audit" / "paper_report_campaign_audit.json")
    if (
        not any(row.get("sha256") == packaged_audit_sha for row in inputs)
        or packaged_audit.get("status") != "PASS"
        or packaged_audit.get("campaign_plan_sha256") != plan_sha
        or packaged_audit.get("target_final_evaluated") is not False
    ):
        raise ValueError("paper report independent audit input differs")
    return True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read package JSON: {path}") from error


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def valid_sha(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def valid_relative_file(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and path.name not in ("", ".")


def parse_sums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError("cannot read SHA256SUMS.txt") from error
    for line in lines:
        if not line or "  " not in line:
            raise ValueError("malformed SHA256SUMS.txt line")
        digest, relative = line.split("  ", 1)
        if not valid_sha(digest) or not valid_relative_file(relative) or relative in result:
            raise ValueError("invalid or duplicate SHA256SUMS.txt entry")
        result[relative] = digest
    return result


def forbidden(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    lowered = {part.lower() for part in relative.parts}
    return bool(lowered & FORBIDDEN_DIRS) or (
        path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    )


def verify(root: Path) -> dict:
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("delivery root must be a directory")
    manifest = read_json(root / "MANIFEST.json")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("unknown delivery manifest schema")
    files = manifest.get("files")
    if (
        not isinstance(files, dict)
        or not files
        or any(not valid_relative_file(name) or not valid_sha(digest) for name, digest in files.items())
    ):
        raise ValueError("delivery manifest file map is invalid")
    forbidden_paths = sorted(
        path.relative_to(root).as_posix() for path in root.rglob("*") if forbidden(path, root)
    )
    if forbidden_paths:
        raise ValueError("delivery contains cache artifacts: " + repr(forbidden_paths))
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "MANIFEST.json"
    }
    if set(files) != actual:
        raise ValueError(
            "manifest membership differs; missing="
            + repr(sorted(set(files) - actual))
            + ", extra="
            + repr(sorted(actual - set(files)))
        )
    if not REQUIRED_FILES.issubset(files):
        raise ValueError("delivery lacks required files: " + repr(sorted(REQUIRED_FILES - set(files))))
    for relative, expected in files.items():
        if sha256(root / relative) != expected:
            raise ValueError("manifest SHA256 differs: " + relative)
    sums = parse_sums(root / "SHA256SUMS.txt")
    expected_sums = {name: digest for name, digest in files.items() if name != "SHA256SUMS.txt"}
    if sums != expected_sums:
        raise ValueError("SHA256SUMS.txt differs from the manifest")

    status = read_json(root / "STATUS.json")
    campaign_relative = status.get("campaign_directory")
    if not valid_relative_file(campaign_relative):
        raise ValueError("delivery campaign directory is invalid")
    campaign_parts = PurePosixPath(campaign_relative).parts
    if len(campaign_parts) != 2 or campaign_parts[0] != "experiment":
        raise ValueError("delivery campaign must be one named directory below experiment")
    campaign_root = (root / campaign_relative).resolve(strict=True)
    if not campaign_root.is_relative_to((root / "experiment").resolve(strict=True)):
        raise ValueError("delivery campaign directory escapes experiment")
    campaign_required = {
        f"{campaign_relative}/{name}"
        for name in (
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
    }
    if not campaign_required.issubset(files):
        raise ValueError("delivery lacks completed campaign files")
    plan = read_json(campaign_root / "plan.json")
    plan_sha = sha256(campaign_root / "plan.json")
    if (
        plan.get("kind") != CAMPAIGN_KIND
        or plan.get("output_name") != campaign_root.name
        or (campaign_root / "plan.sha256").read_text(encoding="ascii").strip() != plan_sha
        or plan.get("counts")
        != {
            "source_anchors": 4,
            "second_stage_control": 16,
            "second_stage_v2a": 16,
            "second_stage_total": 32,
            "optimizer_runs_total": 36,
        }
    ):
        raise ValueError("frozen campaign plan identity or counts differ")
    for relative, digest in plan.get("code_sha256", {}).items():
        packaged = f"code/{relative}"
        if packaged not in files or files[packaged] != digest:
            raise ValueError("packaged frozen code differs from plan: " + relative)

    summary = read_json(campaign_root / "summary.json")
    pipeline = read_json(campaign_root / "pipeline_state.json")
    seal = read_json(campaign_root / "predev_seal.json")
    causal_gate = summary.get("causal_comparison", {}).get("expansion_gate", {})
    gate_passed = causal_gate.get("passed")
    if (
        summary.get("method") != "v2A-U-fixedK"
        or summary.get("target_final_evaluated") is not False
        or gate_passed not in (True, False)
        or pipeline.get("status") != "completed"
        or pipeline.get("screen_gate_passed") is not gate_passed
        or pipeline.get("target_final_evaluated") is not False
    ):
        raise ValueError("completed campaign summary/pipeline boundary differs")
    mechanism_required = plan_sha == MECHANISM_REQUIRED_PLAN_SHA
    mechanism_files = {
        "tools/diagnose_channel_robust_v2a_20260925.py",
        "evidence/channel_robust_v2a_mechanism_diagnostic_20260925.md",
        "evidence/channel_robust_v2a_mechanism_diagnostic_20260925.json",
        "evidence/channel_robust_v2a_screen_v5_mechanism_review_20260925.md",
    }
    if mechanism_required and not mechanism_files.issubset(files):
        raise ValueError("delivery lacks required mechanism evidence")
    if status.get("mechanism_evidence_included") is not mechanism_required:
        raise ValueError("mechanism evidence status differs")
    if mechanism_required:
        mechanism = read_json(
            root / "evidence" / "channel_robust_v2a_mechanism_diagnostic_20260925.json"
        )
        mechanism_pairs = mechanism.get("pairs")
        mechanism_aggregate = mechanism.get("aggregate", {})
        if (
            mechanism.get("plan_sha256") != plan_sha
            or mechanism.get("dev_results_sha256") != sha256(campaign_root / "dev_results.json")
            or mechanism.get("existing_artifacts_only") is not True
            or mechanism.get("new_training") is not False
            or mechanism.get("target_train_labels_opened_by_diagnostic") is not False
            or mechanism.get("target_dev_labels_opened_by_diagnostic") is not False
            or mechanism.get("target_final_opened_by_diagnostic") is not False
            or not isinstance(mechanism_pairs, list)
            or len(mechanism_pairs) != 16
            or mechanism_aggregate.get("joint_improved")
            != causal_gate.get("jointly_improved_conditions")
        ):
            raise ValueError("mechanism diagnostic evidence differs from packaged campaign")

    registry = read_json(root / "model_registry.json")
    models = registry.get("models")
    if (
        status.get("schema") != STATUS_SCHEMA
        or registry.get("schema") != REGISTRY_SCHEMA
        or status.get("campaign_plan_sha256") != plan_sha
        or status.get("campaign_completed") is not True
        or status.get("screen_gate_passed") is not gate_passed
        or status.get("target_final_evaluated") is not False
        or status.get("source_anchor_count") != 4
        or status.get("branch_checkpoint_count") != 32
        or not isinstance(models, list)
        or len(models) != 36
    ):
        raise ValueError("delivery status/model registry differs")
    expected_gate_status = "PASSED" if gate_passed else "FAILED"
    expected_use = "registered_followup_evaluation" if gate_passed else "research_evaluation_only"
    if (
        status.get("integrity_status") != "PASS"
        or status.get("performance_gate_status") != expected_gate_status
        or status.get("recommended_model_id") is not None
        or status.get("intended_use") != expected_use
        or registry.get("recommended_model_id") is not None
    ):
        raise ValueError("delivery claim boundary differs")
    architecture = registry.get("architecture_contract", {})
    if (
        architecture.get("shared_model_structure") is not True
        or architecture.get("target_channel_training_policy")
        != "one common architecture; separate training and checkpoint per target view"
        or architecture.get("universal_weight_model_delivered") is not False
        or architecture.get("model_code_sha256") != plan.get("code_sha256")
    ):
        raise ValueError("common-architecture checkpoint policy differs")

    identifiers = set()
    role_counts = {"source_anchor": 0, "common_anchor_control": 0, "v2a_candidate": 0}
    branch_matrix = set()
    for model in models:
        identifier = model.get("id")
        relative = model.get("checkpoint")
        digest = model.get("checkpoint_sha256")
        role = model.get("role")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or role not in role_counts
            or not valid_relative_file(relative)
            or relative not in files
            or files[relative] != digest
            or model.get("target_final_evaluated") is not False
        ):
            raise ValueError("invalid model registry row")
        identifiers.add(identifier)
        role_counts[role] += 1
        artifacts = model.get("artifacts_sha256")
        parent = PurePosixPath(relative).parent
        if (
            not isinstance(artifacts, dict)
            or set(artifacts)
            != {"best_model.pth", "run_metadata.json", "run_state.json", "epoch_metrics.jsonl"}
        ):
            raise ValueError("model artifact registry differs: " + identifier)
        for name, expected in artifacts.items():
            artifact = (parent / name).as_posix()
            if files.get(artifact) != expected:
                raise ValueError("model artifact SHA differs: " + identifier + "/" + name)
        if role != "source_anchor":
            target = model.get("target_slots")
            if target not in ([0], [1], [2], [0, 1, 2]):
                raise ValueError("branch target-slot view differs")
            score_path = model.get("development_score")
            if (
                not valid_relative_file(score_path)
                or files.get(score_path) != model.get("development_result_sha256")
                or files.get(f"{campaign_relative}/dev/{identifier}.json")
                != model.get("development_result_sha256")
            ):
                raise ValueError("branch development score receipt differs")
            branch_matrix.add((model.get("task_id"), tuple(target), role))
        elif model.get("target_slots") is not None:
            raise ValueError("source anchor unexpectedly has a target view")
    if role_counts != {"source_anchor": 4, "common_anchor_control": 16, "v2a_candidate": 16}:
        raise ValueError("registry model role counts differ")
    expected_matrix = {
        (task, target, role)
        for task in ("WP-S0", "WP-D1", "PG-S1", "PG-D1")
        for target in ((0, 1, 2), (0,), (1,), (2,))
        for role in ("common_anchor_control", "v2a_candidate")
    }
    if branch_matrix != expected_matrix:
        raise ValueError("registry does not contain one checkpoint for every target view and role")
    expected_promotion = (
        "registered_development_gate_passed"
        if gate_passed
        else "not_promoted_beyond_completed_system_evaluation"
    )
    if status.get("promotion_status") != expected_promotion:
        raise ValueError("delivery promotion status differs from the registered gate")
    correction_verified = verify_device_hash_correction(
        root, campaign_root, files, plan_sha, seal
    )
    paper_report_verified = verify_paper_report(
        root, files, status, gate_passed=gate_passed, plan_sha=plan_sha
    )
    noise_registry = read_json(root / "robustness" / "result_registry.json")
    noise_rows = noise_registry.get("results")
    if (
        noise_registry.get("schema") != "channel_robust_v2_awgn_result_registry_v1"
        or noise_registry.get("target_final_evaluated") is not False
        or not isinstance(noise_rows, list)
        or status.get("awgn_evaluated") is not bool(noise_rows)
        or status.get("awgn_result_count") != len(noise_rows)
        or status.get("awgn_scope")
        != ("target_dev_followup" if noise_rows else "not_run")
    ):
        raise ValueError("AWGN result registry/status differs")
    noise_keys = set()
    model_by_id = {row["id"]: row for row in models}
    for row in noise_rows:
        key = (row.get("arm_id"), tuple(row.get("affected_physical_slots", ())))
        path = row.get("path")
        model = model_by_id.get(row.get("arm_id"))
        if (
            key in noise_keys
            or model is None
            or model.get("role") != "v2a_candidate"
            or not valid_relative_file(path)
            or files.get(path) != row.get("sha256")
            or row.get("evaluated_split") != "target_dev_only"
            or row.get("target_final_evaluated") is not False
        ):
            raise ValueError("AWGN result registry row differs")
        noise_keys.add(key)
        raw_noise = read_json(root / path)
        noise_receipt = raw_noise.get("receipt", {})
        if (
            raw_noise.get("kind") != "channel_robust_alignment_awgn_dev_v2"
            or raw_noise.get("campaign_plan_sha256") != plan_sha
            or raw_noise.get("checkpoint_sha256") != model.get("checkpoint_sha256")
            or raw_noise.get("evaluated_split") != "target_dev_only"
            or raw_noise.get("target_final_evaluated") is not False
            or noise_receipt.get("private_origin_metadata_opened") is not False
            or noise_receipt.get("target_final_indices_accessed") is not False
            or noise_receipt.get("final_signal_or_label_rows_decoded") is not False
        ):
            raise ValueError("packaged AWGN result identity differs")

    audit = read_json(root / "audit" / "completed_v2_campaign_audit.json")
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("status") != "PASS"
        or audit.get("campaign_plan_sha256") != plan_sha
        or audit.get("anchors_verified") != 4
        or audit.get("branches_verified") != 32
        or audit.get("development_results_verified") != 32
        or audit.get("target_final_evaluated") is not False
        or audit.get("screen_gate_passed") is not gate_passed
        or audit.get("reconstructed_dev_results_sha256")
        != canonical_sha256(read_json(campaign_root / "dev_results.json"))
        or audit.get("reconstructed_summary_sha256") != canonical_sha256(summary)
        or audit.get("audit_code_sha256")
        != sha256(root / "tools" / "verify_completed_channel_robust_v2_campaign.py")
    ):
        raise ValueError("independent completed-campaign audit receipt differs")
    for name, digest in audit.get("input_sha256", {}).items():
        if not valid_relative_file(name) or sha256(campaign_root / name) != digest:
            raise ValueError("audit input receipt differs: " + str(name))

    data_identity = read_json(root / "config" / "task_registry.json")
    if data_identity.get("schema") != "channel_robust_v2_task_registry_v1":
        raise ValueError("task registry schema differs")
    tasks = data_identity.get("tasks")
    if not isinstance(tasks, dict) or set(tasks) != set(plan.get("data", {})):
        raise ValueError("task registry task set differs")
    for task, record in tasks.items():
        if (
            record.get("metadata") != plan["task_metadata"][task]
            or record.get("data_metadata_sha256") != plan["data"][task]["sha256"]
            or "root" in record
            or record.get("target_final_included") is not False
        ):
            raise ValueError("task registry identity differs: " + task)
    mapping_config = root / "config" / "channel_robust_class_mappings.json"
    mapping_inference = root / "inference" / "channel_robust_class_mappings.json"
    mapping = read_json(mapping_config)
    if (
        sha256(mapping_config) != sha256(mapping_inference)
        or mapping.get("schema") != "channel_robust_class_mappings_v1"
        or mapping.get("index_base") != 0
    ):
        raise ValueError("packaged class mapping copies or schema differ")

    reference = plan.get("strong_v4_reference", {})
    reference_names = {
        "plan.json": reference.get("plan_sha256"),
        "fit_results.json": reference.get("fit_results_sha256"),
        "predev_seal.json": reference.get("predev_seal_sha256"),
        "dev_results.json": reference.get("dev_results_sha256"),
        "summary.json": reference.get("summary_sha256"),
        "REPORT.md": reference.get("report_sha256"),
    }
    for name, digest in reference_names.items():
        relative = f"reference/v4/{name}"
        if not valid_sha(digest) or files.get(relative) != digest:
            raise ValueError("frozen v4 reference differs: " + name)
    reference_receipt = reference.get("completed_audit_receipt", {})
    if (
        reference_receipt.get("schema") != "channel_robust_completed_campaign_audit_v1"
        or reference_receipt.get("status") != "PASS"
        or files.get("reference/v4/completed_audit_receipt.json")
        != reference_receipt.get("sha256")
    ):
        raise ValueError("frozen v4 completed-audit receipt differs")
    reference_registry = read_json(root / "reference" / "v4" / "control_registry.json")
    controls = reference_registry.get("controls")
    if (
        reference_registry.get("schema") != "channel_robust_v2_v4_reference_registry_v1"
        or not isinstance(controls, list)
        or len(controls) != 16
    ):
        raise ValueError("v4 control registry differs")
    expected_controls = {
        (row["arm"]["id"], row["checkpoint_sha256"]) for row in reference.get("controls", [])
    }
    actual_controls = set()
    for row in controls:
        relative = row.get("checkpoint")
        actual_controls.add((row.get("arm", {}).get("id"), row.get("checkpoint_sha256")))
        if not valid_relative_file(relative) or files.get(relative) != row.get("checkpoint_sha256"):
            raise ValueError("packaged v4 control checkpoint differs")
    if actual_controls != expected_controls:
        raise ValueError("packaged v4 control identities differ from plan")

    return {
        "schema": "channel_robust_v2_delivery_verification_v1",
        "status": "PASS",
        "delivery": root.name,
        "files_verified": len(files),
        "models_verified": len(models),
        "branch_checkpoints_verified": 32,
        "screen_gate_passed": gate_passed,
        "integrity_status": "PASS",
        "performance_gate_status": expected_gate_status,
        "device_hash_correction_verified": correction_verified,
        "paper_report_verified": paper_report_verified,
        "mechanism_evidence_verified": mechanism_required,
        "target_final_evaluated": False,
        "manifest_sha256": sha256(root / "MANIFEST.json"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = verify(args.delivery)
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError("verification output already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
