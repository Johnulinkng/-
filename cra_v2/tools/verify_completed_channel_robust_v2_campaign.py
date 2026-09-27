"""Independent, fail-closed audit of a completed CRA-v2A campaign.

The audit reads frozen checkpoints and target-development result JSON files. It
never opens a private label file, raw signal bank, or target-final partition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time


CAMPAIGN_KIND = "channel_robust_v2a_u_fixedk_campaign_v1"
AUDIT_SCHEMA = "channel_robust_v2_completed_campaign_audit_v1"
DEVICE_HASH_COMPAT_PLAN_SHA256 = (
    "8bf146d8412b4aaef104e2ae39769a42494597880090d1d36b072e1b85f6fac9"
)
VARIANTS = ("common_anchor_control", "channel_robust_alignment_v2a")
VARIANT_CONTRACTS = {
    "common_anchor_control": {"adaptation": False, "shared_anchor": True},
    "channel_robust_alignment_v2a": {"adaptation": True, "shared_anchor": True},
}
LOSS_CONTRACT = {
    "fusion_residual_strength": 0.5,
    "masked_ce_weight": 0.5,
    "consistency_weight": 0.1,
    "attention_balance_weight": 0.01,
    "channel_alignment_weight": 0.5,
    "alignment_weight": 0.02,
    "alignment_gradient_rule": "apply only when raw feature-gradient dot source-gradient >= 0",
    "mmd": "equal-row unbiased 5-kernel Gaussian MMD",
    "channel_draw": "independent domain RNG; random offset then balanced physical-slot cycle",
}
PAIRED_BRANCH_CONTRACT = (
    "same anchor model+optimizer+scheduler+all RNG; identical 20-epoch source batches/masks; "
    "candidate differs only by gated fixed-kernel alignment"
)
TARGETS = ([0, 1, 2], [0], [1], [2])
TASKS = ("WP-S0", "WP-D1", "PG-S1", "PG-D1")
HEX = frozenset("0123456789abcdef")


class CampaignAuditError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def runner_json_sha256(value) -> str:
    """Reproduce the campaign runner's sealed JSON-value digest exactly."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignAuditError(f"cannot read JSON: {path}") from error


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise CampaignAuditError(f"cannot read JSONL: {path}") from error
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise CampaignAuditError(f"invalid JSONL at {path}:{number}") from error
        if not isinstance(value, dict):
            raise CampaignAuditError(f"JSONL row is not a mapping at {path}:{number}")
        rows.append(value)
    return rows


def valid_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= HEX


def runtime_path(value) -> Path:
    if isinstance(value, dict):
        key = "windows" if os.name == "nt" else "wsl"
        if key not in value:
            raise CampaignAuditError(f"cross-platform path lacks {key}")
        value = value[key]
    text = str(value)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 7:
        text = f"{text[5].upper()}:/{text[7:]}"
    return Path(text).resolve(strict=True)


def expected_run_name(item: dict, device: str, *, anchor: bool) -> str:
    if anchor:
        return f"{item['task']}_source012_common_anchor_seed{item['seed']}_{device}"
    target = "all3" if item["target"] == [0, 1, 2] else f"slot{item['target'][0]}"
    return (
        f"{item['task']}_source012_target_{target}_{item['variant']}_"
        f"seed{item['seed']}_{device}"
    )


def local_run(root: Path, recorded, expected_name: str) -> Path:
    if Path(str(recorded)).name != expected_name:
        raise CampaignAuditError("recorded run basename differs from the frozen arm")
    path = (root / "runs" / expected_name).resolve(strict=True)
    if not path.is_relative_to((root / "runs").resolve(strict=True)):
        raise CampaignAuditError("run escapes campaign/runs")
    return path


def finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def validate_metric_row(row: dict, classes: int) -> None:
    recalls = row.get("per_class_recall")
    confusion = row.get("confusion_matrix")
    support = row.get("support_per_class")
    if (
        not finite_number(row.get("accuracy"))
        or not 0 <= row["accuracy"] <= 1
        or not finite_number(row.get("macro_f1"))
        or not 0 <= row["macro_f1"] <= 1
        or not isinstance(recalls, list)
        or len(recalls) != classes
        or any(not finite_number(value) or not 0 <= value <= 1 for value in recalls)
        or not isinstance(confusion, list)
        or len(confusion) != classes
        or any(
            not isinstance(line, list)
            or len(line) != classes
            or any(type(value) is not int or value < 0 for value in line)
            for line in confusion
        )
        or not isinstance(support, list)
        or len(support) != classes
        or any(type(value) is not int or value < 0 for value in support)
        or support != [sum(line) for line in confusion]
    ):
        raise CampaignAuditError("development metric structure differs")


def pairs(rows: list[dict], control: str) -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for row in rows:
        arm = row["arm"]
        key = (arm["task"], tuple(arm["target"]), arm["seed"])
        bucket = by_key.setdefault(key, {})
        if arm["variant"] in bucket:
            raise CampaignAuditError("duplicate variant in paired results")
        bucket[arm["variant"]] = row
    result = []
    for key, values in sorted(by_key.items()):
        if set(values) != {control, "channel_robust_alignment_v2a"}:
            raise CampaignAuditError("paired result variant set differs")
        candidate, baseline = values["channel_robust_alignment_v2a"], values[control]
        result.append(
            {
                "task": key[0],
                "target": list(key[1]),
                "seed": key[2],
                "family": candidate["arm"]["family"],
                "candidate_accuracy": candidate["accuracy"],
                "control_accuracy": baseline["accuracy"],
                "accuracy_delta": candidate["accuracy"] - baseline["accuracy"],
                "candidate_macro_f1": candidate["macro_f1"],
                "control_macro_f1": baseline["macro_f1"],
                "macro_f1_delta": candidate["macro_f1"] - baseline["macro_f1"],
                "jointly_improved": candidate["accuracy"] > baseline["accuracy"]
                and candidate["macro_f1"] > baseline["macro_f1"],
                "new_zero_recall": any(
                    new == 0 and old > 0
                    for new, old in zip(
                        candidate["per_class_recall"], baseline["per_class_recall"]
                    )
                ),
            }
        )
    if len(result) != 16:
        raise CampaignAuditError("expected exactly 16 paired comparisons")
    return result


def gate(rows: list[dict]) -> dict:
    joint = sum(row["jointly_improved"] for row in rows)
    mean_acc = statistics.fmean(row["accuracy_delta"] for row in rows)
    mean_f1 = statistics.fmean(row["macro_f1_delta"] for row in rows)
    double = statistics.fmean(
        row["macro_f1_delta"]
        for row in rows
        if row["family"] == "cross_sensor_and_condition"
    )
    slot_positive = {
        task: sum(
            row["jointly_improved"]
            for row in rows
            if row["task"] == task and len(row["target"]) == 1
        )
        for task in TASKS
    }
    singles = [row for row in rows if len(row["target"]) == 1]
    weakest = min(singles, key=lambda row: row["control_macro_f1"])
    checks = {
        "joint_10_of_16": joint >= 10,
        "mean_accuracy_plus_1pp": mean_acc >= 0.01,
        "mean_macro_f1_plus_2pp": mean_f1 >= 0.02,
        "double_cross_macro_f1_plus_2pp": double >= 0.02,
        "two_positive_slots_each_task": all(value >= 2 for value in slot_positive.values()),
        "weakest_3to1_improves": weakest["macro_f1_delta"] > 0,
        "no_new_zero_recall": not any(row["new_zero_recall"] for row in rows),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "jointly_improved_conditions": joint,
        "mean_accuracy_delta": mean_acc,
        "mean_macro_f1_delta": mean_f1,
        "double_cross_macro_f1_delta": double,
        "positive_slots_by_task": slot_positive,
        "weakest_3to1": weakest,
    }


def validate_checkpoint_common(
    checkpoint: dict,
    expected: dict,
    code_sha: dict,
    *,
    classes: int,
    training: dict,
) -> None:
    if not isinstance(checkpoint, dict):
        raise CampaignAuditError("checkpoint payload is not a mapping")
    if (
        checkpoint.get("task_id") != expected["task"]
        or checkpoint.get("dataset") != expected["dataset"]
        or checkpoint.get("seed") != expected["seed"]
        or checkpoint.get("window_size") != 2048
        or checkpoint.get("num_classes") != classes
        or checkpoint.get("loss_contract") != LOSS_CONTRACT
        or checkpoint.get("code_sha256") != {
            Path(name).name: digest
            for name, digest in code_sha.items()
            if Path(name).name
            in {
                "train_channel_robust_alignment_v2.py",
                "public_loader.py",
                "channel_robust_alignment_v2.py",
                "channel_robust_alignment.py",
                "balanced_unbiased_mmd.py",
                "spectral_grid_pilot.py",
                "spectral_shared.py",
                "experiment_protocol.py",
            }
        }
        or checkpoint.get("target_dev_or_final_loader_constructed") is not False
        or checkpoint.get("target_dev_evaluated") is not False
        or checkpoint.get("target_final_evaluated") is not False
        or checkpoint.get("synchronous_pairs_used") is not False
    ):
        raise CampaignAuditError("checkpoint common provenance differs")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise CampaignAuditError("checkpoint lacks model state")
    try:
        import torch

        if any(not isinstance(value, torch.Tensor) or not torch.isfinite(value).all() for value in state.values()):
            raise CampaignAuditError("checkpoint model state is non-finite or malformed")
        required_buffers = {
            "source_center",
            "source_scale",
            "target_center",
            "target_scale",
            "source_calibrated",
            "target_calibrated",
        }
        if (
            not required_buffers.issubset(state)
            or state["source_calibrated"].numel() != 1
            or not bool(state["source_calibrated"].item())
            or torch.any(state["source_scale"] <= 0)
            or torch.any(state["target_scale"] <= 0)
        ):
            raise CampaignAuditError("checkpoint calibration buffers are incomplete or invalid")
    except ImportError as error:
        raise CampaignAuditError("torch is required to audit v2 checkpoints") from error
    config = checkpoint.get("model_config")
    details = checkpoint.get("model_details")
    if (
        not isinstance(config, dict)
        or config.get("input_channels") != 3
        or config.get("spectral_grid") != training.get("spectral_grid")
        or config.get("encoder_width") != training.get("encoder_width")
        or config.get("fusion") != "bounded_attention"
        or config.get("fusion_residual_strength") != 0.5
        or not isinstance(details, dict)
        or details.get("identity") != "channel_robust_alignment_v2a"
        or details.get("spectral_grid") != training.get("spectral_grid")
        or details.get("fusion_mode") != "bounded_attention"
        or details.get("fusion_residual_strength") != 0.5
        or details.get("adaptation_selection")
        != "fixed 20-epoch endpoint; no target-label selection"
    ):
        raise CampaignAuditError("checkpoint v2 model configuration or architecture differs")


def load_checkpoint(path: Path):
    try:
        import torch

        return torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise CampaignAuditError(f"cannot load checkpoint safely: {path}") from error


def validate_kernel(kernel: dict) -> None:
    if (
        not isinstance(kernel, dict)
        or kernel.get("kind") != "source_anchor_fixed_five_rbf"
        or kernel.get("multipliers") != [0.25, 0.5, 1.0, 2.0, 4.0]
        or kernel.get("kernel_weights") != [1.0] * 5
        or kernel.get("target_inputs_used") is not False
        or kernel.get("fixed_before_branch_training") is not True
    ):
        raise CampaignAuditError("fixed-kernel contract differs")
    for name in ("pool_local_indices_sha256", "source_feature_pool_sha256"):
        if not valid_sha(kernel.get(name)):
            raise CampaignAuditError("kernel source-pool identity is invalid")
    for level in ("fused", "channel"):
        item = kernel.get(level)
        base = item.get("median_squared_distance") if isinstance(item, dict) else None
        bands = item.get("bandwidths") if isinstance(item, dict) else None
        if (
            not finite_number(base)
            or base <= 0
            or not isinstance(bands, list)
            or bands != [float(base) * value for value in [0.25, 0.5, 1.0, 2.0, 4.0]]
        ):
            raise CampaignAuditError("kernel bandwidth payload differs")


def validate_device_hash_correction(
    root: Path, expected_plan_sha: str, fits: dict, seal: dict
) -> None:
    """Require the bound compatibility audit for the affected frozen plan."""
    if expected_plan_sha != DEVICE_HASH_COMPAT_PLAN_SHA256:
        return
    receipt_path = root / "device_hash_seal_correction_receipt.json"
    execution_path = root / "device_hash_seal_execution_receipt.json"
    if not receipt_path.is_file() or not execution_path.is_file():
        raise CampaignAuditError("device-hash correction receipt is missing")
    receipt_sha = sha256(receipt_path)
    if seal.get("device_hash_correction_receipt_sha256") != receipt_sha:
        raise CampaignAuditError("device-hash correction is not bound into the seal")
    receipt = read_json(receipt_path)
    relative = receipt.get("correction_tool_relative")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise CampaignAuditError("device-hash correction tool path is invalid")
    project_root = root.parents[1].resolve()
    tool_path = (project_root / relative).resolve()
    try:
        tool_path.relative_to(project_root)
    except ValueError as error:
        raise CampaignAuditError("device-hash correction tool escapes project") from error
    expected_anchors = {
        row["anchor"]["id"]: row for row in fits["anchors"]
    }
    diagnostics = receipt.get("anchor_diagnostics")
    diagnostic_by_id = {
        row.get("anchor_id"): row for row in diagnostics
    } if isinstance(diagnostics, list) else {}
    if (
        receipt.get("schema") != "channel_robust_v2_device_hash_seal_correction_v1"
        or receipt.get("status") != "PASS"
        or receipt.get("plan_sha256") != expected_plan_sha
        or receipt.get("fit_results_sha256") != sha256(root / "fit_results.json")
        or receipt.get("predev_fit_audit_sha256") != sha256(root / "predev_fit_audit.json")
        or receipt.get("correction_tool_sha256") != sha256(tool_path)
        or receipt.get("frozen_runner_sha256")
        != sha256(root / "code" / "protocol_next" / "run_channel_robust_v2_campaign.py")
        or receipt.get("frozen_trainer_sha256")
        != sha256(root / "code" / "protocol_next" / "train_channel_robust_alignment_v2.py")
        or receipt.get("anchors_verified") != 4
        or receipt.get("branches_verified_by_frozen_auditor") != 32
        or receipt.get("checkpoints_rewritten") is not False
        or receipt.get("frozen_code_modified") is not False
        or receipt.get("target_dev_predictions_or_labels_read") is not False
        or receipt.get("target_final_evaluated") is not False
        or len(diagnostic_by_id) != 4
        or receipt.get("branch_checkpoints")
        != [
            {"id": row["arm"]["id"], "sha256": row["checkpoint_sha256"]}
            for row in fits["arms"]
        ]
    ):
        raise CampaignAuditError("device-hash correction receipt differs")
    for anchor_id, fit in expected_anchors.items():
        row = diagnostic_by_id.get(anchor_id)
        if (
            not isinstance(row, dict)
            or row.get("checkpoint_sha256") != fit["checkpoint_sha256"]
            or row.get("stored_replay_identity") != fit["anchor_replay_identity"]
            or row.get("cuda_recomputed_replay_identity") != fit["anchor_replay_identity"]
            or row.get("all_components_match") is not True
        ):
            raise CampaignAuditError(f"device-hash anchor diagnostic differs: {anchor_id}")
    execution = read_json(execution_path)
    if (
        execution.get("schema") != "channel_robust_v2_device_hash_seal_execution_v1"
        or execution.get("status") != "PASS"
        or execution.get("device_hash_correction_receipt_sha256") != receipt_sha
        or execution.get("predev_seal_sha256") != sha256(root / "predev_seal.json")
        or execution.get("target_dev_predictions_or_labels_read") is not False
        or execution.get("target_final_evaluated") is not False
    ):
        raise CampaignAuditError("device-hash seal execution receipt differs")


def validate_strong_reference(plan: dict) -> list[dict]:
    reference = plan.get("strong_v4_reference")
    if not isinstance(reference, dict) or reference.get("dev_results_read_during_prepare") is not False:
        raise CampaignAuditError("strong v4 reference is missing or violates pre-dev boundary")
    project = runtime_path(reference.get("project_root"))
    campaign = runtime_path(reference.get("campaign_root"))
    try:
        relative = campaign.relative_to(project).as_posix()
    except ValueError as error:
        raise CampaignAuditError("strong v4 campaign escapes its project root") from error
    if relative != reference.get("campaign_relative"):
        raise CampaignAuditError("strong v4 relative identity differs")
    names = {
        "plan.json": "plan_sha256",
        "fit_results.json": "fit_results_sha256",
        "predev_seal.json": "predev_seal_sha256",
        "dev_results.json": "dev_results_sha256",
        "summary.json": "summary_sha256",
        "REPORT.md": "report_sha256",
    }
    for name, field in names.items():
        if not valid_sha(reference.get(field)) or sha256(campaign / name) != reference[field]:
            raise CampaignAuditError(f"strong v4 reference changed: {name}")
    receipt = reference.get("completed_audit_receipt")
    if (
        not isinstance(receipt, dict)
        or receipt.get("schema") != "channel_robust_completed_campaign_audit_v1"
        or receipt.get("status") != "PASS"
        or not valid_sha(receipt.get("sha256"))
        or sha256(runtime_path(receipt.get("path"))) != receipt["sha256"]
    ):
        raise CampaignAuditError("strong v4 completed-audit receipt differs")
    rows = read_json(campaign / "dev_results.json")
    controls = [
        row
        for row in rows
        if row.get("arm", {}).get("variant") == "matched_control"
        and row.get("arm", {}).get("seed") == 42
    ]
    expected = {
        item["arm"]["id"]: item["checkpoint_sha256"]
        for item in reference.get("controls", [])
    }
    if (
        len(controls) != 16
        or len(expected) != 16
        or any(expected.get(row["arm"]["id"]) != row.get("checkpoint_sha256") for row in controls)
    ):
        raise CampaignAuditError("strong v4 control registry differs")
    return controls


def audit_completed_campaign(campaign: Path) -> dict:
    root = Path(campaign).resolve(strict=True)
    if not root.is_dir():
        raise CampaignAuditError("campaign root is not a directory")
    required = (
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
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise CampaignAuditError("completed campaign is missing: " + ", ".join(missing))
    expected_plan_sha = (root / "plan.sha256").read_text(encoding="ascii").strip()
    if not valid_sha(expected_plan_sha) or sha256(root / "plan.json") != expected_plan_sha:
        raise CampaignAuditError("campaign plan SHA differs")
    plan = read_json(root / "plan.json")
    if (
        plan.get("kind") != CAMPAIGN_KIND
        or plan.get("output_name") != root.name
        or plan.get("counts")
        != {
            "source_anchors": 4,
            "second_stage_control": 16,
            "second_stage_v2a": 16,
            "second_stage_total": 32,
            "optimizer_runs_total": 36,
        }
        or plan.get("information_boundary", {}).get("target_final_evaluated") is not False
    ):
        raise CampaignAuditError("campaign plan kind, counts, or final boundary differs")
    anchors, arms = plan.get("anchors"), plan.get("arms")
    if (
        not isinstance(anchors, list)
        or len(anchors) != 4
        or not isinstance(arms, list)
        or len(arms) != 32
        or len({item.get("id") for item in anchors}) != 4
        or len({item.get("id") for item in arms}) != 32
    ):
        raise CampaignAuditError("frozen anchor/branch matrix differs")
    expected_order = [
        (task, target, variant)
        for task in TASKS
        for target in TARGETS
        for variant in VARIANTS
    ]
    actual_order = [(arm.get("task"), arm.get("target"), arm.get("variant")) for arm in arms]
    if actual_order != expected_order:
        raise CampaignAuditError("branch matrix order or members differ")

    code_sha = plan.get("code_sha256")
    if not isinstance(code_sha, dict) or not code_sha:
        raise CampaignAuditError("frozen code receipt is absent")
    for name, digest in code_sha.items():
        if not valid_sha(digest) or sha256(root / "code" / name) != digest:
            raise CampaignAuditError(f"frozen code differs: {name}")
    for task, record in plan.get("data", {}).items():
        data_root = runtime_path(record.get("root"))
        for name, digest in record.get("sha256", {}).items():
            if not valid_sha(digest) or sha256(data_root / name) != digest:
                raise CampaignAuditError(f"public data metadata differs: {task}/{name}")

    fit_state = read_json(root / "fit_state.json")
    dev_state = read_json(root / "dev_state.json")
    pipeline = read_json(root / "pipeline_state.json")
    fits = read_json(root / "fit_results.json")
    dev_rows = read_json(root / "dev_results.json")
    if (
        fit_state.get("status") != "completed"
        or fit_state.get("completed_anchors") != 4
        or fit_state.get("completed_arms") != 32
        or fit_state.get("active") is not None
        or dev_state.get("status") != "completed"
        or dev_state.get("completed_arms") != 32
        or dev_state.get("active_arm") is not None
        or pipeline.get("status") != "completed"
        or pipeline.get("fit_anchors") != 4
        or pipeline.get("fit_arms") != 32
        or pipeline.get("dev_arms") != 32
        or pipeline.get("target_dev_evaluated") is not True
        or pipeline.get("target_final_evaluated") is not False
        or not isinstance(fits, dict)
        or len(fits.get("anchors", [])) != 4
        or len(fits.get("arms", [])) != 32
        or not isinstance(dev_rows, list)
        or len(dev_rows) != 32
    ):
        raise CampaignAuditError("completed state files disagree")

    device = plan.get("training", {}).get("device")
    anchor_sha_by_task = {}
    anchor_receipts = []
    for item, row in zip(anchors, fits["anchors"]):
        if row.get("anchor") != item:
            raise CampaignAuditError("anchor fit order differs")
        run = local_run(root, row.get("run"), expected_run_name(item, device, anchor=True))
        checkpoint_path = run / "best_model.pth"
        if row.get("checkpoint") and Path(str(row["checkpoint"])).name != "best_model.pth":
            raise CampaignAuditError("anchor checkpoint basename differs")
        if not valid_sha(row.get("checkpoint_sha256")) or sha256(checkpoint_path) != row["checkpoint_sha256"]:
            raise CampaignAuditError("anchor checkpoint SHA differs")
        metadata, state = read_json(run / "run_metadata.json"), read_json(run / "run_state.json")
        records = read_jsonl(run / "epoch_metrics.jsonl")
        if len(records) != 60 or [entry.get("epoch") for entry in records] != list(range(60)):
            raise CampaignAuditError("anchor epoch log must cover 0..59")
        try:
            selected = max(
                records,
                key=lambda entry: (
                    entry["source_val"]["accuracy"],
                    -entry["source_val"]["loss"],
                    -entry["epoch"],
                ),
            )
        except (KeyError, TypeError) as error:
            raise CampaignAuditError("anchor source-val selection log is malformed") from error
        optimizer_totals = [entry.get("optimizer_steps_total") for entry in records]
        if (
            any(type(value) is not int or value <= 0 for value in optimizer_totals)
            or any(right <= left for left, right in zip(optimizer_totals, optimizer_totals[1:]))
        ):
            raise CampaignAuditError("anchor optimizer-step log differs")
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint_common(
            checkpoint,
            item,
            code_sha,
            classes=plan["task_metadata"][item["task"]]["classes"],
            training=plan["training"],
        )
        validate_kernel(checkpoint.get("kernel_contract"))
        if (
            metadata.get("stage") != "source_anchor"
            or metadata.get("variant") != "source_anchor_v2a"
            or metadata.get("target_train_fault_labels") != "not_loaded"
            or checkpoint.get("stage") != "source_anchor"
            or checkpoint.get("variant") != "source_anchor_v2a"
            or bool(checkpoint["model_state_dict"]["target_calibrated"].item())
            or checkpoint.get("selected_epoch") != selected["epoch"]
            or checkpoint.get("source_val") != selected["source_val"]
            or state.get("status") != "completed"
            or state.get("completed_pretrain_epochs") != 60
            or state.get("selected_epoch") != selected["epoch"]
            or checkpoint.get("source_pretrain_optimizer_steps_at_selection")
            != selected["optimizer_steps_total"]
            or state.get("optimizer_steps") != optimizer_totals[-1]
            or row.get("optimizer_steps") != optimizer_totals[-1]
            or state.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or state.get("target_train_waveforms_read") is not False
            or state.get("target_final_evaluated") is not False
            or checkpoint.get("anchor_replay_identity") != row.get("anchor_replay_identity")
            or checkpoint.get("kernel_contract") != row.get("kernel_contract")
            or checkpoint.get("channels") != {"source": [0, 1, 2], "target": None}
        ):
            raise CampaignAuditError("anchor metadata, selection, or replay identity differs")
        if item["task"] in anchor_sha_by_task:
            raise CampaignAuditError("task has more than one common anchor")
        anchor_sha_by_task[item["task"]] = row["checkpoint_sha256"]
        anchor_receipts.append(
            {
                "id": item["id"],
                "epochs": 60,
                "selected_epoch": selected["epoch"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "epoch_metrics_sha256": sha256(run / "epoch_metrics.jsonl"),
            }
        )

    branch_receipts = []
    reconstructed_dev = []
    for index, (arm, row, dev) in enumerate(zip(arms, fits["arms"], dev_rows)):
        if row.get("arm") != arm or dev.get("arm") != arm:
            raise CampaignAuditError("branch fit/development order differs")
        run = local_run(root, row.get("run"), expected_run_name(arm, device, anchor=False))
        checkpoint_path = run / "best_model.pth"
        if not valid_sha(row.get("checkpoint_sha256")) or sha256(checkpoint_path) != row["checkpoint_sha256"]:
            raise CampaignAuditError("branch checkpoint SHA differs")
        metadata, state = read_json(run / "run_metadata.json"), read_json(run / "run_state.json")
        records = read_jsonl(run / "epoch_metrics.jsonl")
        if len(records) != 20 or [entry.get("branch_epoch") for entry in records] != list(range(20)):
            raise CampaignAuditError("branch epoch log must cover 0..19")
        optimizer_totals = [entry.get("optimizer_steps_total") for entry in records]
        alignment_totals = [entry.get("alignment_steps_total") for entry in records]
        if (
            any(type(value) is not int or value <= 0 for value in optimizer_totals)
            or any(right <= left for left, right in zip(optimizer_totals, optimizer_totals[1:]))
            or any(type(value) is not int or value < 0 for value in alignment_totals)
            or any(right < left for left, right in zip(alignment_totals, alignment_totals[1:]))
        ):
            raise CampaignAuditError("branch optimizer/alignment step log differs")
        checkpoint = load_checkpoint(checkpoint_path)
        validate_checkpoint_common(
            checkpoint,
            arm,
            code_sha,
            classes=plan["task_metadata"][arm["task"]]["classes"],
            training=plan["training"],
        )
        validate_kernel(checkpoint.get("kernel_contract"))
        expected_alignment = arm["variant"] == "channel_robust_alignment_v2a"
        steps = checkpoint.get("alignment_steps")
        applied = checkpoint.get("applied_alignment_steps")
        suppressed = checkpoint.get("suppressed_alignment_steps")
        if (
            metadata.get("variant") != arm["variant"]
            or checkpoint.get("variant") != arm["variant"]
            or checkpoint.get("variant_contract") != VARIANT_CONTRACTS[arm["variant"]]
            or checkpoint.get("channels") != {"source": [0, 1, 2], "target": arm["target"]}
            or checkpoint.get("target_train_fault_labels") != "required_minus_one"
            or not bool(checkpoint["model_state_dict"]["target_calibrated"].item())
            or checkpoint.get("selected_epoch") != 19
            or checkpoint.get("adaptation_endpoint") is not True
            or checkpoint.get("adaptation_epochs") != 20
            or checkpoint.get("selection")
            != "fixed branch epoch 19; no target-label or source-val checkpoint choice"
            or checkpoint.get("paired_branch_contract") != PAIRED_BRANCH_CONTRACT
            or checkpoint.get("alignment_enabled") is not expected_alignment
            or checkpoint.get("anchor_checkpoint_sha256") != anchor_sha_by_task[arm["task"]]
            or row.get("anchor_checkpoint_sha256") != anchor_sha_by_task[arm["task"]]
            or checkpoint.get("branch_start_identity") != row.get("branch_start_identity")
            or checkpoint.get("branch_source_indices_sha256") != row.get("branch_source_indices_sha256")
            or checkpoint.get("branch_source_masks_sha256") != row.get("branch_source_masks_sha256")
            or any(
                entry.get("alignment_weight") != (0.02 if expected_alignment else 0.0)
                for entry in records
            )
            or type(steps) is not int
            or type(applied) is not int
            or type(suppressed) is not int
            or min(steps, applied, suppressed) < 0
            or applied + suppressed != steps
            or (steps > 0) is not expected_alignment
            or checkpoint.get("adapt_optimizer_steps") != optimizer_totals[-1]
            or steps != alignment_totals[-1]
            or state.get("status") != "completed"
            or state.get("completed_adaptation_epochs") != 20
            or state.get("selected_epoch") != 19
            or state.get("optimizer_steps") != optimizer_totals[-1]
            or row.get("optimizer_steps") != optimizer_totals[-1]
            or state.get("alignment_steps") != steps
            or row.get("alignment_steps") != steps
            or state.get("applied_alignment_steps") != applied
            or row.get("applied_alignment_steps") != applied
            or state.get("suppressed_alignment_steps") != suppressed
            or row.get("suppressed_alignment_steps") != suppressed
            or state.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or state.get("target_final_evaluated") is not False
        ):
            raise CampaignAuditError("branch endpoint, alignment, or provenance differs")
        result_path = root / "dev" / f"{arm['id']}.json"
        raw = read_json(result_path)
        classes = plan["task_metadata"][arm["task"]]["classes"]
        validate_metric_row(raw, classes)
        if (
            not valid_sha(dev.get("result_sha256"))
            or sha256(result_path) != dev["result_sha256"]
            or raw.get("checkpoint_sha256") != row["checkpoint_sha256"]
            or raw.get("target_final_evaluated") is not False
            or raw.get("checkpoint_identity", {}).get("variant") != arm["variant"]
            or raw.get("checkpoint_identity", {}).get("channels", {}).get("target") != arm["target"]
        ):
            raise CampaignAuditError("branch development result identity differs")
        rebuilt = {
            "arm": arm,
            "result_sha256": dev["result_sha256"],
            **{
                key: raw[key]
                for key in (
                    "checkpoint_sha256",
                    "accuracy",
                    "macro_f1",
                    "per_class_recall",
                    "confusion_matrix",
                    "support_per_class",
                )
            },
        }
        if rebuilt != dev:
            raise CampaignAuditError("development registry row differs from hashed result")
        reconstructed_dev.append(rebuilt)
        branch_receipts.append(
            {
                "id": arm["id"],
                "epochs": 20,
                "endpoint": 19,
                "checkpoint_sha256": row["checkpoint_sha256"],
                "epoch_metrics_sha256": sha256(run / "epoch_metrics.jsonl"),
            }
        )
        if index % 2 == 1:
            control, candidate = fits["arms"][index - 1], fits["arms"][index]
            for field in (
                "anchor_checkpoint_sha256",
                "branch_start_identity",
                "branch_source_indices_sha256",
                "branch_source_masks_sha256",
            ):
                if control.get(field) != candidate.get(field):
                    raise CampaignAuditError(f"paired causal replay differs: {field}")

    predev_fit = read_json(root / "predev_fit_audit.json")
    seal = read_json(root / "predev_seal.json")
    validate_device_hash_correction(root, expected_plan_sha, fits, seal)
    if (
        predev_fit.get("schema") != "channel_robust_v2_predev_fit_audit_v1"
        or predev_fit.get("status") != "PASS"
        or predev_fit.get("target_dev_read") is not False
        or predev_fit.get("target_final_evaluated") is not False
        or predev_fit.get("fit_results_sha256") != sha256(root / "fit_results.json")
        or predev_fit.get("plan_sha256") != expected_plan_sha
        or predev_fit.get("unique_anchor_sha256_by_task") != anchor_sha_by_task
        or predev_fit.get("anchors") != anchor_receipts
        or predev_fit.get("arms") != branch_receipts
        or seal.get("plan_sha256") != expected_plan_sha
        or seal.get("fit_results_sha256") != sha256(root / "fit_results.json")
        or seal.get("predev_fit_audit_sha256") != sha256(root / "predev_fit_audit.json")
        or seal.get("strong_v4_reference_sha256") != runner_json_sha256(plan["strong_v4_reference"])
        or seal.get("target_dev_predictions_or_labels_read") is not False
        or seal.get("target_final_evaluated") is not False
        or seal.get("anchor_checkpoints")
        != [{"id": row["anchor"]["id"], "sha256": row["checkpoint_sha256"]} for row in fits["anchors"]]
        or seal.get("branch_checkpoints")
        != [{"id": row["arm"]["id"], "sha256": row["checkpoint_sha256"]} for row in fits["arms"]]
    ):
        raise CampaignAuditError("pre-development fit audit or seal differs")

    strong = validate_strong_reference(plan)
    causal_pairs = pairs(reconstructed_dev, "common_anchor_control")
    candidates = [
        row
        for row in reconstructed_dev
        if row["arm"]["variant"] == "channel_robust_alignment_v2a"
    ]
    formal_pairs = pairs(candidates + strong, "matched_control")
    rebuilt_summary = {
        "method": "v2A-U-fixedK",
        "development_arms": 32,
        "causal_comparison": {
            "control": "common_anchor_control",
            "pairs": causal_pairs,
            "expansion_gate": gate(causal_pairs),
        },
        "formal_comparison": {
            "control": "frozen_v4_matched_control",
            "pairs": formal_pairs,
            "historical_threshold_diagnostic": gate(formal_pairs),
            "decides_expansion": False,
        },
        "target_dev_history_boundary": plan["target_dev_history_boundary"],
        "target_final_evaluated": False,
    }
    if read_json(root / "summary.json") != rebuilt_summary:
        raise CampaignAuditError("reported summary or gate differs from independent reconstruction")
    if pipeline.get("screen_gate_passed") != rebuilt_summary["causal_comparison"]["expansion_gate"]["passed"]:
        raise CampaignAuditError("pipeline gate flag differs from reconstructed causal gate")

    input_names = required
    result = {
        "schema": AUDIT_SCHEMA,
        "status": "PASS",
        "created_unix": time.time(),
        "campaign": root.name,
        "campaign_plan_sha256": expected_plan_sha,
        "anchors_verified": 4,
        "branches_verified": 32,
        "development_results_verified": 32,
        "anchor_receipts": anchor_receipts,
        "branch_receipts": branch_receipts,
        "paired_replay_verified": 16,
        "screen_gate_passed": rebuilt_summary["causal_comparison"]["expansion_gate"]["passed"],
        "target_dev_evaluated": True,
        "target_final_evaluated": False,
        "reconstructed_dev_results_sha256": canonical_sha256(reconstructed_dev),
        "reconstructed_summary_sha256": canonical_sha256(rebuilt_summary),
        "input_sha256": {name: sha256(root / name) for name in input_names},
        "audit_code_sha256": sha256(Path(__file__).resolve()),
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = audit_completed_campaign(args.campaign)
    if args.output is not None:
        output = args.output.resolve()
        if output.exists():
            raise FileExistsError("audit output already exists")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
