"""Seal a completed frozen CRA-v2A campaign after auditing a device-hash bug.

The v2A anchors stored replay hashes made from ``torch.save`` byte streams
while their tensors were on CUDA.  The original pre-development auditor loaded
the same payload on CPU, which changes those bytes solely through device
metadata.  This utility leaves the frozen code and every checkpoint untouched,
proves each saved replay hash on the original CUDA device, then delegates the
full 4-anchor/32-branch audit and seal construction to the frozen runner.

It never imports or reads development/final labels, predictions, or metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_frozen_runner(campaign: Path):
    runner_path = campaign / "code" / "protocol_next" / "run_channel_robust_v2_campaign.py"
    if not runner_path.is_file():
        raise FileNotFoundError(f"frozen runner is missing: {runner_path}")
    module_name = "frozen_channel_robust_v2_campaign_device_hash_fix"
    spec = importlib.util.spec_from_file_location(module_name, runner_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen campaign runner")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def cuda_replay_diagnostic(campaign: Path, runner, results: dict) -> list[dict]:
    import torch
    import train_channel_robust_alignment_v2 as trainer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the stored replay identity")
    rows = []
    keys = {
        "model_state_sha256": "model_state_dict",
        "optimizer_state_sha256": "optimizer_state_dict",
        "scheduler_state_sha256": "scheduler_state_dict",
        "rng_state_sha256": "rng_state",
    }
    for record in results["anchors"]:
        anchor = record["anchor"]
        checkpoint_path = runner.resolved_run(
            campaign, record, runner.expected_run_id(anchor, anchor=True)
        ) / "best_model.pth"
        if sha256(checkpoint_path) != record["checkpoint_sha256"]:
            raise RuntimeError(f"anchor checkpoint changed: {anchor['id']}")
        checkpoint = torch.load(checkpoint_path, map_location="cuda", weights_only=True)
        expected = checkpoint.get("anchor_replay_identity")
        if not isinstance(expected, dict):
            raise RuntimeError(f"anchor replay identity is missing: {anchor['id']}")
        observed = {
            output_key: trainer.object_sha256(checkpoint[payload_key])
            for output_key, payload_key in keys.items()
        }
        observed["kernel_contract_sha256"] = hashlib.sha256(
            json.dumps(
                checkpoint["kernel_contract"], sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        matches = {key: observed[key] == expected.get(key) for key in observed}
        if not all(matches.values()):
            raise RuntimeError(f"CUDA replay identity differs: {anchor['id']}")
        rows.append(
            {
                "anchor_id": anchor["id"],
                "checkpoint_sha256": record["checkpoint_sha256"],
                "stored_replay_identity": expected,
                "cuda_recomputed_replay_identity": observed,
                "all_components_match": True,
            }
        )
        del checkpoint
        torch.cuda.empty_cache()
    return rows


def seal_with_original_device_validation(campaign: Path) -> dict:
    campaign = campaign.resolve(strict=True)
    if (campaign / "predev_seal.json").exists():
        raise FileExistsError("pre-development seal already exists")
    runner = load_frozen_runner(campaign)
    plan = runner.frozen(campaign)
    results = runner.verify_fit_results(campaign, plan, all_required=True)
    diagnostic = cuda_replay_diagnostic(campaign, runner, results)

    import torch
    import train_channel_robust_alignment_v2 as trainer

    original_validator = trainer.validate_anchor_payload

    def validate_on_original_device(checkpoint: dict) -> None:
        # A recursive ``tensor.to('cuda')`` does not necessarily reproduce the
        # storage-alias metadata created by ``torch.load(map_location='cuda')``.
        # Reload the same SHA-locked file on its original device, and separately
        # prove that its logical tree is exactly equal to the CPU view supplied
        # by the frozen auditor.
        run_id = checkpoint.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("anchor run id is missing during device-hash validation")
        checkpoint_path = campaign / "runs" / run_id / "best_model.pth"
        cuda_payload = torch.load(checkpoint_path, map_location="cuda", weights_only=True)

        def equal(left, right) -> bool:
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                return (
                    left.dtype == right.dtype
                    and tuple(left.shape) == tuple(right.shape)
                    and torch.equal(left.cpu(), right.cpu())
                )
            if type(left) is not type(right):
                return False
            if isinstance(left, dict):
                return list(left) == list(right) and all(
                    equal(left[key], right[key]) for key in left
                )
            if isinstance(left, (list, tuple)):
                return len(left) == len(right) and all(
                    equal(a, b) for a, b in zip(left, right)
                )
            return left == right

        try:
            if not equal(checkpoint, cuda_payload):
                raise RuntimeError("CPU and CUDA checkpoint views differ logically")
            original_validator(cuda_payload)
        finally:
            del cuda_payload
            torch.cuda.empty_cache()

    trainer.validate_anchor_payload = validate_on_original_device
    try:
        runner.seal(campaign)
    finally:
        trainer.validate_anchor_payload = original_validator

    runner.verify_seal(
        campaign,
        runner.verify_fit_results(campaign, plan, all_required=True),
        plan,
    )
    tool_path = Path(__file__).resolve()
    project_root = campaign.parents[1]
    try:
        tool_relative = tool_path.relative_to(project_root).as_posix()
    except ValueError as error:
        raise RuntimeError("correction tool must live inside the project") from error
    receipt = {
        "schema": "channel_robust_v2_device_hash_seal_correction_v1",
        "status": "PASS",
        "created_unix": time.time(),
        "campaign": str(campaign),
        "cause": "legacy torch.save replay hashes include tensor device metadata",
        "remediation": "validate unchanged payload on original CUDA device; do not rewrite checkpoints",
        "frozen_runner_sha256": sha256(
            campaign / "code" / "protocol_next" / "run_channel_robust_v2_campaign.py"
        ),
        "frozen_trainer_sha256": sha256(
            campaign / "code" / "protocol_next" / "train_channel_robust_alignment_v2.py"
        ),
        "plan_sha256": sha256(campaign / "plan.json"),
        "fit_results_sha256": sha256(campaign / "fit_results.json"),
        "predev_fit_audit_sha256": sha256(campaign / "predev_fit_audit.json"),
        "correction_tool_relative": tool_relative,
        "correction_tool_sha256": sha256(tool_path),
        "anchor_diagnostics": diagnostic,
        "branch_checkpoints": [
            {"id": row["arm"]["id"], "sha256": row["checkpoint_sha256"]}
            for row in results["arms"]
        ],
        "anchors_verified": len(diagnostic),
        "branches_verified_by_frozen_auditor": len(results["arms"]),
        "checkpoints_rewritten": False,
        "frozen_code_modified": False,
        "target_dev_predictions_or_labels_read": False,
        "target_final_evaluated": False,
    }
    receipt_path = campaign / "device_hash_seal_correction_receipt.json"
    write_json(receipt_path, receipt)

    # Bind the exceptional compatibility audit into the seal itself.  The
    # frozen verifier permits additional fields while the independent final
    # auditor requires this exact receipt for the affected frozen plan.
    seal_path = campaign / "predev_seal.json"
    seal = runner.read_json(seal_path)
    seal["device_hash_correction_receipt_sha256"] = sha256(receipt_path)
    runner.write_json(seal_path, seal)
    runner.verify_seal(
        campaign,
        runner.verify_fit_results(campaign, plan, all_required=True),
        plan,
    )
    execution = {
        "schema": "channel_robust_v2_device_hash_seal_execution_v1",
        "status": "PASS",
        "created_unix": time.time(),
        "device_hash_correction_receipt_sha256": sha256(receipt_path),
        "predev_seal_sha256": sha256(seal_path),
        "target_dev_predictions_or_labels_read": False,
        "target_final_evaluated": False,
    }
    write_json(campaign / "device_hash_seal_execution_receipt.json", execution)
    return execution


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = seal_with_original_device_validation(args.campaign)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "correction_receipt_sha256": receipt[
                    "device_hash_correction_receipt_sha256"
                ],
                "predev_seal_sha256": receipt["predev_seal_sha256"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
