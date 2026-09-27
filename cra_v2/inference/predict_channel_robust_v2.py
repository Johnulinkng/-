"""Fail-closed inference for frozen CRA-v2A branch checkpoints.

The input is a raw float32 NPY array with shape ``[N, C, 2048]``.  ``C``
must match the checkpoint's frozen target-slot contract.  This program never
opens a task dataset, label file, target-development split, or target-final
split; target-train calibration statistics are already frozen in the
checkpoint buffers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
_WORKBENCH_CANDIDATES = (ROOT / "workbench", ROOT / "code" / "workbench")
_PROTOCOL_CANDIDATES = (ROOT / "protocol_next", ROOT / "code" / "protocol_next")
_REQUIRED_MODEL_FILES = (
    "models/channel_robust_alignment_v2.py",
    "models/channel_robust_alignment.py",
    "models/spectral_grid_pilot.py",
    "models/spectral_shared.py",
    "loss/balanced_unbiased_mmd.py",
)
_REQUIRED_PROTOCOL_FILES = (
    "train_channel_robust_alignment_v2.py",
    "public_loader.py",
)
_valid_workbenches = [
    candidate
    for candidate in _WORKBENCH_CANDIDATES
    if all((candidate / relative).is_file() for relative in _REQUIRED_MODEL_FILES)
]
if len(_valid_workbenches) != 1:
    raise RuntimeError(
        "expected exactly one complete model-code layout: ROOT/workbench (workspace) "
        "or ROOT/code/workbench (delivery)"
    )
WORKBENCH = _valid_workbenches[0].resolve(strict=True)
_valid_protocols = [
    candidate
    for candidate in _PROTOCOL_CANDIDATES
    if all((candidate / relative).is_file() for relative in _REQUIRED_PROTOCOL_FILES)
]
if len(_valid_protocols) != 1:
    raise RuntimeError(
        "expected exactly one complete protocol-code layout: ROOT/protocol_next (workspace) "
        "or ROOT/code/protocol_next (delivery)"
    )
PROTOCOL = _valid_protocols[0].resolve(strict=True)
if str(WORKBENCH) not in sys.path:
    sys.path.insert(0, str(WORKBENCH))

from models.channel_robust_alignment_v2 import ChannelRobustAlignmentV2
from models.spectral_grid_pilot import GRIDS


WINDOW_SIZE = 2048
CLASS_MAPPING_SCHEMA = "channel_robust_class_mappings_v1"
CLASS_MAPPING_PATH = Path(__file__).resolve().with_name("channel_robust_class_mappings.json")
DATASET_SAMPLE_RATES = {
    "waterpump": 24000,
    "gearbox": 24000,
    "seu_gearbox": 5120,
}
DATASET_TASK_PREFIXES = {
    "waterpump": "WP-",
    "gearbox": "PG-",
    "seu_gearbox": "SEU-",
}
PUBLIC_VARIANTS = {
    "control": "common_anchor_control",
    "candidate": "channel_robust_alignment_v2a",
}
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
MODEL_CODE_PATHS = {
    "train_channel_robust_alignment_v2.py": PROTOCOL / "train_channel_robust_alignment_v2.py",
    "public_loader.py": PROTOCOL / "public_loader.py",
    "channel_robust_alignment_v2.py": WORKBENCH / "models" / "channel_robust_alignment_v2.py",
    "channel_robust_alignment.py": WORKBENCH / "models" / "channel_robust_alignment.py",
    "balanced_unbiased_mmd.py": WORKBENCH / "loss" / "balanced_unbiased_mmd.py",
    "spectral_grid_pilot.py": WORKBENCH / "models" / "spectral_grid_pilot.py",
    "spectral_shared.py": WORKBENCH / "models" / "spectral_shared.py",
    "experiment_protocol.py": WORKBENCH / "experiment_protocol.py",
}
ADAPT_EPOCHS = 20
KERNEL_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]
PAIRED_BRANCH_CONTRACT = (
    "same anchor model+optimizer+scheduler+all RNG; identical 20-epoch source batches/masks; "
    "candidate differs only by gated fixed-kernel alignment"
)


class InferenceContractError(ValueError):
    """The supplied artifact or invocation violates the frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def class_mapping_for_dataset(
    dataset: str,
    num_classes: int,
    mapping_path: Path = CLASS_MAPPING_PATH,
) -> dict:
    """Load and validate the delivery's authoritative zero-based class map."""
    try:
        path = Path(mapping_path).resolve(strict=True)
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise InferenceContractError("cannot read the packaged class mapping") from error
    datasets = payload.get("datasets") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != CLASS_MAPPING_SCHEMA
        or payload.get("index_base") != 0
        or not isinstance(datasets, dict)
        or set(datasets) != set(DATASET_SAMPLE_RATES)
    ):
        raise InferenceContractError("class mapping schema or dataset set differs")
    if dataset not in datasets:
        raise InferenceContractError("checkpoint dataset has no authoritative class mapping")
    record = datasets[dataset]
    classes = record.get("classes") if isinstance(record, dict) else None
    if (
        not isinstance(record, dict)
        or isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or record.get("num_classes") != num_classes
        or not isinstance(classes, list)
        or len(classes) != num_classes
    ):
        raise InferenceContractError("class mapping count differs from checkpoint")
    for expected_index, item in enumerate(classes):
        if (
            not isinstance(item, dict)
            or item.get("index") != expected_index
            or not isinstance(item.get("display_name"), str)
            or not item["display_name"].strip()
            or (
                item.get("chinese_name") is not None
                and (
                    not isinstance(item["chinese_name"], str)
                    or not item["chinese_name"].strip()
                )
            )
            or not isinstance(item.get("original_names"), list)
            or not item["original_names"]
            or any(not isinstance(name, str) or not name.strip() for name in item["original_names"])
            or not isinstance(item.get("translation_status"), str)
            or not item["translation_status"].strip()
        ):
            raise InferenceContractError("class mapping entry is malformed or out of order")
    evidence = record.get("evidence_files")
    limitations = record.get("limitations")
    if (
        not isinstance(evidence, list)
        or not evidence
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("path"), str)
            or not item["path"].strip()
            or not _valid_sha(item.get("sha256"))
            or not isinstance(item.get("relevance"), str)
            or not item["relevance"].strip()
            for item in evidence
        )
        or not isinstance(limitations, list)
        or any(not isinstance(value, str) or not value.strip() for value in limitations)
    ):
        raise InferenceContractError("class mapping evidence or limitations are malformed")
    return {
        "schema": CLASS_MAPPING_SCHEMA,
        "mapping_file": path.name,
        "mapping_file_sha256": sha256(path),
        "dataset": dataset,
        "index_base": 0,
        "num_classes": num_classes,
        "classes": classes,
        "evidence_files": evidence,
        "limitations": limitations,
    }


def _target_slots(values) -> list[int]:
    slots = list(values)
    if slots not in ([0], [1], [2], [0, 1, 2]):
        raise InferenceContractError(
            "target slots must be one frozen slot (0, 1, or 2) or ordered slots 0 1 2"
        )
    return slots


def _require_checkpoint_metadata(
    checkpoint: dict,
    *,
    task_id: str,
    target_slots: list[int],
    public_variant: str,
    sampling_rate_hz: int,
) -> tuple[dict, dict]:
    internal_variant = PUBLIC_VARIANTS[public_variant]
    channels = {"source": [0, 1, 2], "target": target_slots}
    dataset = checkpoint.get("dataset")
    if dataset not in DATASET_SAMPLE_RATES:
        raise InferenceContractError("checkpoint dataset is unsupported")
    if (
        checkpoint.get("task_id") != task_id
        or not task_id.startswith(DATASET_TASK_PREFIXES[dataset])
        or checkpoint.get("channels") != channels
        or checkpoint.get("window_size") != WINDOW_SIZE
        or checkpoint.get("variant") != internal_variant
        or checkpoint.get("variant_contract") != VARIANT_CONTRACTS[internal_variant]
        or checkpoint.get("loss_contract") != LOSS_CONTRACT
    ):
        raise InferenceContractError("task, target slots, window, or variant differs from checkpoint")
    if sampling_rate_hz != DATASET_SAMPLE_RATES[dataset]:
        raise InferenceContractError(
            f"sampling rate {sampling_rate_hz} Hz differs from the frozen {dataset} boundary "
            f"({DATASET_SAMPLE_RATES[dataset]} Hz)"
        )
    feature_physics = checkpoint.get("feature_physics")
    if feature_physics is not None and feature_physics.get("sampling_rate_hz") != sampling_rate_hz:
        raise InferenceContractError("checkpoint feature-physics sampling rate differs")
    num_classes = checkpoint.get("num_classes")
    selected_epoch = checkpoint.get("selected_epoch")
    if (
        isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 2
        or isinstance(selected_epoch, bool)
        or not isinstance(selected_epoch, int)
        or selected_epoch != ADAPT_EPOCHS - 1
        or checkpoint.get("adaptation_endpoint") is not True
        or checkpoint.get("adaptation_epochs") != ADAPT_EPOCHS
        or checkpoint.get("selection")
        != "fixed branch epoch 19; no target-label or source-val checkpoint choice"
    ):
        raise InferenceContractError("class count or fixed v2 branch endpoint is invalid")
    if (
        checkpoint.get("target_statistics") != "zscore_train_only"
        or checkpoint.get("target_train_fault_labels") != "required_minus_one"
        or checkpoint.get("target_dev_or_final_loader_constructed") is not False
        or checkpoint.get("synchronous_pairs_used") is not False
        or checkpoint.get("target_dev_evaluated") is not False
        or checkpoint.get("target_final_evaluated") is not False
        or checkpoint.get("paired_branch_contract") != PAIRED_BRANCH_CONTRACT
    ):
        raise InferenceContractError("checkpoint target-label or held-out-data boundary differs")
    for field in (
        "public_manifest_sha256",
        "public_plan_sha256",
        "anchor_checkpoint_sha256",
        "branch_source_indices_sha256",
        "branch_source_masks_sha256",
    ):
        if not _valid_sha(checkpoint.get(field)):
            raise InferenceContractError(f"checkpoint {field} is invalid")
    start = checkpoint.get("branch_start_identity")
    if not isinstance(start, dict):
        raise InferenceContractError("checkpoint lacks the paired branch start identity")
    for field in (
        "model_state_sha256",
        "optimizer_state_sha256",
        "scheduler_state_sha256",
        "rng_state_sha256",
        "anchor_checkpoint_sha256",
    ):
        if not _valid_sha(start.get(field)):
            raise InferenceContractError(f"branch start identity is invalid: {field}")
    if (
        start.get("anchor_checkpoint_sha256") != checkpoint["anchor_checkpoint_sha256"]
        or not isinstance(start.get("source_slot_cycle"), dict)
        or not isinstance(start.get("target_slot_cycle"), dict)
    ):
        raise InferenceContractError("branch start identity does not match the anchor")
    kernel = checkpoint.get("kernel_contract")
    if (
        not isinstance(kernel, dict)
        or kernel.get("kind") != "source_anchor_fixed_five_rbf"
        or kernel.get("multipliers") != KERNEL_MULTIPLIERS
        or kernel.get("kernel_weights") != [1.0] * 5
        or kernel.get("target_inputs_used") is not False
        or kernel.get("fixed_before_branch_training") is not True
    ):
        raise InferenceContractError("checkpoint fixed-kernel contract differs")
    for level in ("fused", "channel"):
        record = kernel.get(level)
        bandwidths = record.get("bandwidths") if isinstance(record, dict) else None
        if (
            not isinstance(bandwidths, list)
            or len(bandwidths) != 5
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
                for value in bandwidths
            )
        ):
            raise InferenceContractError(f"checkpoint {level} kernel bandwidths differ")
    steps = checkpoint.get("alignment_steps")
    applied = checkpoint.get("applied_alignment_steps")
    suppressed = checkpoint.get("suppressed_alignment_steps")
    optimizer_steps = checkpoint.get("adapt_optimizer_steps")
    if (
        any(type(value) is not int or value < 0 for value in (steps, applied, suppressed))
        or type(optimizer_steps) is not int
        or optimizer_steps <= 0
        or applied + suppressed != steps
        or (
            internal_variant == "common_anchor_control"
            and (steps != 0 or checkpoint.get("alignment_enabled") is not False)
        )
        or (
            internal_variant == "channel_robust_alignment_v2a"
            and (steps <= 0 or checkpoint.get("alignment_enabled") is not True)
        )
    ):
        raise InferenceContractError("checkpoint v2 alignment-step contract differs")
    code_identity = checkpoint.get("code_sha256")
    if not isinstance(code_identity, dict):
        raise InferenceContractError("checkpoint lacks frozen code identity")
    for name, path in MODEL_CODE_PATHS.items():
        if code_identity.get(name) != sha256(path):
            raise InferenceContractError(f"model implementation differs from checkpoint: {name}")
    config = checkpoint.get("model_config")
    if not isinstance(config, dict):
        raise InferenceContractError("checkpoint model configuration is invalid")
    if (
        config.get("input_channels") != 3
        or config.get("spectral_grid") not in GRIDS
        or config.get("encoder_width") not in (64, 128)
        or config.get("fusion") != "bounded_attention"
        or not isinstance(config.get("fusion_residual_strength"), (int, float))
        or isinstance(config.get("fusion_residual_strength"), bool)
        or not math.isfinite(float(config["fusion_residual_strength"]))
        or float(config["fusion_residual_strength"]) != LOSS_CONTRACT["fusion_residual_strength"]
    ):
        raise InferenceContractError("checkpoint model configuration differs from the frozen variant")
    return config, {
        "dataset": dataset,
        "num_classes": num_classes,
        "selected_epoch": selected_epoch,
        "internal_variant": internal_variant,
        "channels": channels,
    }


def load_checkpoint(
    checkpoint_path: Path,
    expected_sha256: str,
    *,
    task_id: str,
    target_slots,
    variant: str,
    sampling_rate_hz: int,
    device: str = "cpu",
) -> dict:
    """Verify identity and construct an evaluation-only model."""
    path = Path(checkpoint_path).resolve(strict=True)
    if not _valid_sha(expected_sha256):
        raise InferenceContractError("checkpoint SHA256 must be lowercase hexadecimal")
    actual_sha = sha256(path)
    if actual_sha != expected_sha256:
        raise InferenceContractError("checkpoint SHA256 mismatch")
    if variant not in PUBLIC_VARIANTS:
        raise InferenceContractError("variant must be control or candidate")
    slots = _target_slots(target_slots)
    if isinstance(sampling_rate_hz, bool) or not isinstance(sampling_rate_hz, int):
        raise InferenceContractError("sampling rate must be an integer in Hz")
    if device not in ("cpu", "cuda"):
        raise InferenceContractError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise InferenceContractError("CUDA was requested but is unavailable")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise InferenceContractError("cannot load checkpoint with weights-only deserialization") from error
    if sha256(path) != actual_sha:
        raise InferenceContractError("checkpoint changed while it was being loaded")
    if not isinstance(checkpoint, dict):
        raise InferenceContractError("checkpoint payload must be a mapping")
    config, identity = _require_checkpoint_metadata(
        checkpoint,
        task_id=task_id,
        target_slots=slots,
        public_variant=variant,
        sampling_rate_hz=sampling_rate_hz,
    )
    class_mapping = class_mapping_for_dataset(identity["dataset"], identity["num_classes"])
    model = ChannelRobustAlignmentV2(
        3,
        identity["num_classes"],
        spectral_grid=config["spectral_grid"],
        encoder_width=config["encoder_width"],
        fusion=config["fusion"],
        fusion_residual_strength=float(config["fusion_residual_strength"]),
    )
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (KeyError, RuntimeError, TypeError, ValueError) as error:
        raise InferenceContractError("checkpoint weights do not fit the declared model") from error
    details = checkpoint.get("model_details")
    actual_details = model.architecture_metadata()
    if not isinstance(details, dict):
        raise InferenceContractError("checkpoint architecture metadata is invalid")
    for key in (
        "identity",
        "spectral_grid",
        "encoder_width",
        "fusion_mode",
        "fusion_parameter_count",
        "fusion_initialization",
        "fusion_residual_strength",
        "single_sensor_fusion",
        "parameter_count",
        "calibration_lifecycle",
        "adaptation_selection",
    ):
        if details.get(key) != actual_details.get(key):
            raise InferenceContractError(f"checkpoint architecture metadata differs: {key}")
    state = model.state_dict()
    if not model.source_calibrated.item() or not model.target_calibrated.item():
        raise InferenceContractError("training-only calibration buffers are not frozen")
    if torch.any(model.source_scale <= 0) or torch.any(model.target_scale <= 0):
        raise InferenceContractError("checkpoint calibration scales must be positive")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise InferenceContractError("checkpoint contains non-finite state")
    model.eval().to(torch.device(device))
    positive_bins, pool_width = GRIDS[config["spectral_grid"]]
    frequency_spacing = sampling_rate_hz / WINDOW_SIZE
    return {
        "model": model,
        "checkpoint": checkpoint,
        "checkpoint_path": path,
        "checkpoint_sha256": actual_sha,
        "task_id": task_id,
        "dataset": identity["dataset"],
        "target_slots": slots,
        "num_classes": identity["num_classes"],
        "selected_epoch": identity["selected_epoch"],
        "variant": variant,
        "internal_variant": identity["internal_variant"],
        "sampling_rate_hz": sampling_rate_hz,
        "device": device,
        "spectral_grid": config["spectral_grid"],
        "frequency_boundary": {
            "fft_spacing_hz": frequency_spacing,
            "first_positive_frequency_hz": frequency_spacing,
            "last_retained_frequency_hz": positive_bins * frequency_spacing,
            "positive_frequency_bins_retained": positive_bins,
            "power_pool_width": pool_width,
            "output_spectral_bins": positive_bins // pool_width,
        },
        "class_mapping": class_mapping,
    }


def normalize_raw_batch(raw: np.ndarray) -> np.ndarray:
    """Reproduce ``public_loader.SplitDataset`` preprocessing exactly."""
    if not isinstance(raw, np.ndarray) or raw.dtype != np.float32 or raw.ndim != 3:
        raise InferenceContractError("raw batch must be a float32 [N,C,2048] ndarray")
    if raw.shape[0] < 1 or raw.shape[1] < 1 or raw.shape[2] != WINDOW_SIZE:
        raise InferenceContractError("raw batch must be nonempty with exactly 2048 samples")
    if not np.isfinite(raw).all():
        raise InferenceContractError("raw input contains NaN or infinity")
    signal = np.array(raw, dtype=np.float32, copy=True, order="C")
    mean = signal.mean(axis=-1, keepdims=True)
    std = signal.std(axis=-1, keepdims=True)
    return ((signal - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)


def _open_raw(path: Path, channels: int) -> np.memmap:
    resolved = Path(path).resolve(strict=True)
    if resolved.suffix.lower() != ".npy":
        raise InferenceContractError("input must be a .npy file")
    try:
        array = np.load(resolved, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise InferenceContractError("input is not a valid non-pickled NPY array") from error
    if (
        not isinstance(array, np.memmap)
        or array.dtype != np.dtype(np.float32)
        or array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1:] != (channels, WINDOW_SIZE)
    ):
        raise InferenceContractError(
            f"input must be nonempty raw float32 [N,{channels},{WINDOW_SIZE}]"
        )
    expected_size = array.offset + array.size * array.dtype.itemsize
    if resolved.stat().st_size != expected_size:
        raise InferenceContractError("NPY file size differs from its header")
    return array


def predict(loaded: dict, raw: np.memmap, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise InferenceContractError("batch size must be a positive integer")
    predictions = []
    logits_blocks = []
    model = loaded["model"]
    device = torch.device(loaded["device"])
    with torch.inference_mode():
        for start in range(0, len(raw), batch_size):
            normalized = normalize_raw_batch(np.asarray(raw[start : start + batch_size]))
            logits = model.predict_target(torch.from_numpy(normalized).to(device))
            if logits.shape != (len(normalized), loaded["num_classes"]):
                raise InferenceContractError("model returned an unexpected logits shape")
            if not torch.isfinite(logits).all().item():
                raise InferenceContractError("model returned non-finite logits")
            logits_cpu = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            logits_blocks.append(logits_cpu)
            predictions.append(logits_cpu.argmax(axis=1).astype(np.int64))
    return np.concatenate(predictions), np.concatenate(logits_blocks)


def run(
    *,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    task_id: str,
    target_slots,
    variant: str,
    sampling_rate_hz: int,
    input_path: Path,
    output_path: Path,
    batch_size: int = 128,
    device: str = "cpu",
    cpu_threads: int = 1,
) -> dict:
    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise InferenceContractError("CPU thread count must be a positive integer")
    torch.set_num_threads(cpu_threads)
    loaded = load_checkpoint(
        checkpoint_path,
        checkpoint_sha256,
        task_id=task_id,
        target_slots=target_slots,
        variant=variant,
        sampling_rate_hz=sampling_rate_hz,
        device=device,
    )
    input_resolved = Path(input_path).resolve(strict=True)
    input_sha_before = sha256(input_resolved)
    raw = _open_raw(input_resolved, len(loaded["target_slots"]))
    input_shape = list(raw.shape)
    predicted_class, logits = predict(loaded, raw, batch_size)
    names = [item["display_name"] for item in loaded["class_mapping"]["classes"]]
    longest_name = max(map(len, names))
    predicted_label = np.asarray(
        [names[int(index)] for index in predicted_class], dtype=f"<U{longest_name}"
    )
    del raw
    if sha256(input_resolved) != input_sha_before:
        raise InferenceContractError("input changed while inference was running")
    output = Path(output_path).resolve()
    if output.suffix.lower() != ".npz" or output.exists():
        raise InferenceContractError("output must be a new .npz path")
    metadata = {
        "schema": "channel_robust_alignment_inference_v2",
        "task_id": loaded["task_id"],
        "dataset": loaded["dataset"],
        "variant": loaded["variant"],
        "internal_variant": loaded["internal_variant"],
        "target_slots": loaded["target_slots"],
        "sampling_rate_hz": loaded["sampling_rate_hz"],
        "frequency_boundary": loaded["frequency_boundary"],
        "spectral_grid": loaded["spectral_grid"],
        "window_samples": WINDOW_SIZE,
        "input_shape": input_shape,
        "input_dtype": "float32",
        "input_sha256": input_sha_before,
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "checkpoint_selected_epoch": loaded["selected_epoch"],
        "public_manifest_sha256": loaded["checkpoint"]["public_manifest_sha256"],
        "public_plan_sha256": loaded["checkpoint"]["public_plan_sha256"],
        "num_classes": loaded["num_classes"],
        "class_output": "zero_based_training_class_index",
        "class_mapping": loaded["class_mapping"],
        "preprocessing": "per-window per-channel population zscore, then frozen model spectrum/calibration",
        "batch_size": batch_size,
        "device": loaded["device"],
        "cpu_threads": cpu_threads,
        "labels_read": False,
        "task_dataset_opened": False,
        "target_dev_read": False,
        "target_final_read": False,
        "test_time_tuning": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream,
            predicted_class=predicted_class,
            predicted_label=predicted_label,
            logits=logits,
            metadata_json=np.array(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
    return {
        "status": "completed",
        "predictions": len(predicted_class),
        "classes": loaded["num_classes"],
        "class_mapping_sha256": loaded["class_mapping"]["mapping_file_sha256"],
        "task_id": loaded["task_id"],
        "variant": loaded["variant"],
        "target_slots": loaded["target_slots"],
        "output": str(output),
        "checkpoint_sha256": loaded["checkpoint_sha256"],
        "labels_read": False,
        "target_dev_read": False,
        "target_final_read": False,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--variant", choices=tuple(PUBLIC_VARIANTS), required=True)
    parser.add_argument("--target-slots", type=int, nargs="+", required=True)
    parser.add_argument("--sampling-rate-hz", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    result = run(
        checkpoint_path=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        task_id=args.task_id,
        target_slots=args.target_slots,
        variant=args.variant,
        sampling_rate_hz=args.sampling_rate_hz,
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
