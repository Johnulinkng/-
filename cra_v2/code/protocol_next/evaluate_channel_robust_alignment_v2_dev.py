"""Score one SHA-locked CRA-v2 branch on target development only.

The evaluator accepts neither target-final nor training labels.  It validates
the fixed 20-epoch endpoint and all v2A provenance before the private evaluator
decodes development labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
WORKBENCH = ROOT.parent / "workbench"
if str(WORKBENCH) not in sys.path:
    sys.path.insert(0, str(WORKBENCH))

import numpy as np
import torch

from antipair_builder import file_sha256
from private_evaluator import verify_private_and_evaluate
from public_loader import PublicProtocolError, load_public_task
from models.channel_robust_alignment_v2 import ChannelRobustAlignmentV2
from train_channel_robust_alignment_v2 import (
    ADAPT_EPOCHS, ARM_VARIANTS, KERNEL_MULTIPLIERS, LOSS_CONTRACT, code_identity, parse_slots,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--target-slots", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.source_channels = [0, 1, 2]
    args.target_channels = parse_slots(args.target_slots)
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch size and CPU threads must be positive")
    if len(args.checkpoint_sha256) != 64 or any(c not in "0123456789abcdef" for c in args.checkpoint_sha256):
        parser.error("checkpoint SHA256 must be lowercase hexadecimal")
    return args


class LockedPredictor:
    def __init__(self, path: Path, expected_sha: str, task, channels: dict):
        self.path = Path(path).resolve(strict=True)
        if file_sha256(self.path) != expected_sha:
            raise PublicProtocolError("locked checkpoint SHA mismatch")
        try:
            checkpoint = torch.load(self.path, map_location="cpu", weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise PublicProtocolError("cannot load locked v2 checkpoint") from error
        if not isinstance(checkpoint, dict):
            raise PublicProtocolError("checkpoint payload must be a mapping")
        expected_channels = {key: list(value) for key, value in channels.items()}
        variant = checkpoint.get("variant")
        start = checkpoint.get("branch_start_identity")
        hex_fields = ("model_state_sha256", "optimizer_state_sha256", "scheduler_state_sha256",
                      "rng_state_sha256", "anchor_checkpoint_sha256")
        valid_start = isinstance(start, dict) and all(
            isinstance(start.get(key), str) and len(start[key]) == 64
            and all(c in "0123456789abcdef" for c in start[key]) for key in hex_fields
        ) and isinstance(start.get("source_slot_cycle"), dict) and isinstance(start.get("target_slot_cycle"), dict)
        source_digests_valid = all(
            isinstance(checkpoint.get(key), str) and len(checkpoint[key]) == 64
            and all(c in "0123456789abcdef" for c in checkpoint[key])
            for key in ("branch_source_indices_sha256", "branch_source_masks_sha256")
        )
        kernel = checkpoint.get("kernel_contract")
        kernel_valid = (
            isinstance(kernel, dict)
            and kernel.get("kind") == "source_anchor_fixed_five_rbf"
            and kernel.get("multipliers") == list(KERNEL_MULTIPLIERS)
            and kernel.get("kernel_weights") == [1.0] * 5
            and kernel.get("target_inputs_used") is False
            and kernel.get("fixed_before_branch_training") is True
            and all(
                isinstance(kernel.get(level), dict)
                and isinstance(kernel[level].get("bandwidths"), list)
                and len(kernel[level]["bandwidths"]) == 5
                and all(isinstance(value, (int, float)) and np.isfinite(value) and value > 0
                        for value in kernel[level]["bandwidths"])
                for level in ("fused", "channel")
            )
        )
        steps = checkpoint.get("alignment_steps")
        applied, suppressed = checkpoint.get("applied_alignment_steps"), checkpoint.get("suppressed_alignment_steps")
        step_contract_valid = (
            type(steps) is int and type(applied) is int and type(suppressed) is int
            and steps >= 0 and applied >= 0 and suppressed >= 0
            and applied + suppressed == steps
            and ((variant == "common_anchor_control" and steps == 0)
                 or (variant == "channel_robust_alignment_v2a" and steps > 0))
        )
        if (
            checkpoint.get("task_id") != task.task_id
            or checkpoint.get("dataset") != task.dataset
            or checkpoint.get("public_manifest_sha256") != task.manifest_sha256
            or checkpoint.get("public_plan_sha256") != task.plan_sha256
            or checkpoint.get("channels") != expected_channels
            or checkpoint.get("target_statistics") != "zscore_train_only"
            or checkpoint.get("target_train_fault_labels") != "required_minus_one"
            or checkpoint.get("target_dev_or_final_loader_constructed") is not False
            or checkpoint.get("target_dev_evaluated") is not False
            or checkpoint.get("target_final_evaluated") is not False
            or checkpoint.get("synchronous_pairs_used") is not False
            or variant not in ARM_VARIANTS
            or checkpoint.get("variant_contract") != ARM_VARIANTS[variant]
            or checkpoint.get("loss_contract") != LOSS_CONTRACT
            or checkpoint.get("code_sha256") != code_identity()
            or checkpoint.get("selected_epoch") != ADAPT_EPOCHS - 1
            or checkpoint.get("adaptation_endpoint") is not True
            or checkpoint.get("anchor_checkpoint_sha256") is None
            or checkpoint.get("adapt_optimizer_steps", 0) <= 0
            or checkpoint.get("paired_branch_contract") != (
                "same anchor model+optimizer+scheduler+all RNG; identical 20-epoch source batches/masks; "
                "candidate differs only by gated fixed-kernel alignment"
            )
            or not valid_start
            or not source_digests_valid
            or not kernel_valid
            or not step_contract_valid
        ):
            raise PublicProtocolError("v2 checkpoint provenance or information boundary differs")
        config = checkpoint.get("model_config")
        if not isinstance(config, dict) or config.get("input_channels") != 3:
            raise PublicProtocolError("v2 model configuration differs")
        model = ChannelRobustAlignmentV2(
            3, task.num_classes,
            spectral_grid=config["spectral_grid"], encoder_width=config["encoder_width"],
            fusion=config["fusion"],
            fusion_residual_strength=config["fusion_residual_strength"],
        ).cpu()
        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            raise PublicProtocolError("checkpoint weights do not fit v2 model") from error
        if not model.source_calibrated.item() or not model.target_calibrated.item():
            raise PublicProtocolError("training-only calibration buffers are not frozen")
        if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
            raise PublicProtocolError("checkpoint contains non-finite state")
        model.eval()
        self.model, self.checkpoint, self.channels = model, checkpoint, expected_channels

    def __call__(self, checkpoint_path, waveforms):
        if Path(checkpoint_path).resolve(strict=True) != self.path:
            raise PublicProtocolError("predictor invoked with a different checkpoint")
        batch = np.asarray(waveforms)
        expected = (len(self.channels["target"]), 2048)
        if batch.ndim != 3 or batch.shape[1:] != expected or batch.dtype != np.float32:
            raise PublicProtocolError("target-development waveform shape or dtype differs")
        with torch.inference_mode():
            logits = self.model.predict_target(torch.from_numpy(np.array(batch, copy=True)))
        if not torch.isfinite(logits).all():
            raise PublicProtocolError("target-development logits are non-finite")
        return logits.argmax(dim=1).numpy().astype(np.int64)


def evaluate(args):
    if not args.execute:
        raise ValueError("explicit --execute is required")
    output = args.output_json.resolve()
    if output.exists():
        raise FileExistsError("evaluation output already exists")
    torch.set_num_threads(args.cpu_threads)
    channels = {"source": args.source_channels, "target": args.target_channels}
    task = load_public_task(args.public_root.resolve(), args.task_id, channels, role="evaluate_dev")
    predictor = LockedPredictor(args.checkpoint, args.checkpoint_sha256, task, channels)
    result = verify_private_and_evaluate(
        args.public_root.resolve(), args.private_root.resolve(), args.task_id, channels,
        predictor.path, args.checkpoint_sha256, predictor, batch_size=args.batch_size,
    )
    result.update(
        checkpoint_identity={key: predictor.checkpoint[key] for key in (
            "run_id", "task_id", "dataset", "channels", "variant", "selected_epoch",
            "source_val", "anchor_checkpoint_sha256", "branch_start_identity",
            "branch_source_indices_sha256", "branch_source_masks_sha256",
            "public_manifest_sha256", "public_plan_sha256", "code_sha256",
        )},
        target_final_evaluated=False,
        evaluation_code_sha256={name: file_sha256(ROOT / name) for name in (
            "evaluate_channel_robust_alignment_v2_dev.py", "private_evaluator.py", "public_loader.py",
        )},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "not_executed", "split": "target_dev_only"}))
        return 0
    result = evaluate(args)
    print(json.dumps({"status": "completed", "accuracy": result["accuracy"], "macro_f1": result["macro_f1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
