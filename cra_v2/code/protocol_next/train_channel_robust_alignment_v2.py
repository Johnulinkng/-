"""Train CRA-v2A without target labels or target evaluation data.

Phases are explicit. ``anchor`` trains one shared 60-epoch source model per
task/seed. ``control`` and ``adapt`` load that exact anchor, add identical
train-only target z-score statistics, and each perform exactly 20 source-update
epochs.  The candidate's only extra operation is gated alignment. Development
and final paths are not accepted.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parent
WORKBENCH = ROOT.parent / "workbench"
if str(WORKBENCH) not in sys.path:
    sys.path.insert(0, str(WORKBENCH))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from experiment_protocol import classification_metrics
from loss.balanced_unbiased_mmd import (
    BalancedPhysicalSlotCycle,
    unbiased_multi_kernel_mmd,
)
from models.channel_robust_alignment_v2 import ChannelRobustAlignmentV2
from models.spectral_grid_pilot import GRIDS
from models.spectral_shared import nonempty_channel_mask


ARM_VARIANTS = {
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
PRETRAIN_EPOCHS = 60
ADAPT_EPOCHS = 20
KERNEL_POOL_SIZE = 256
KERNEL_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_slots(text: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in text.split(",")]
    except (AttributeError, TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("slots must be comma-separated integers") from error
    if (
        values not in ([0], [1], [2], [0, 1, 2])
        or len(set(values)) != len(values)
    ):
        raise argparse.ArgumentTypeError("target slots must be 0, 1, 2, or 0,1,2")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--phase", choices=("anchor", "control", "adapt"), required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--target-slots")
    parser.add_argument("--anchor-checkpoint", type=Path)
    parser.add_argument("--anchor-sha256")
    parser.add_argument("--spectral-grid", choices=tuple(GRIDS), default="low3k_pool2")
    parser.add_argument("--encoder-width", type=int, choices=(64, 128), default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-seed", type=int, required=True)
    parser.add_argument("--source-draw-seed", type=int, required=True)
    parser.add_argument("--target-draw-seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.30)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--strict-cuda", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    args.source_channels = [0, 1, 2]
    args.target_channels = parse_slots(args.target_slots) if args.target_slots else None
    if args.phase == "anchor":
        if args.target_channels is not None or args.anchor_checkpoint or args.anchor_sha256:
            parser.error("anchor phase does not accept target slots or anchor identity")
    elif (
        args.target_channels is None
        or args.anchor_checkpoint is None
        or args.anchor_sha256 is None
        or len(args.anchor_sha256) != 64
        or any(char not in "0123456789abcdef" for char in args.anchor_sha256)
    ):
        parser.error("control/adapt require target slots and lowercase anchor SHA256")
    seeds = (args.seed, args.mask_seed, args.source_draw_seed, args.target_draw_seed)
    if (
        any(isinstance(value, bool) or value < 0 or value >= 2**63 for value in seeds)
        or args.seed >= 2**32
        or args.cpu_threads < 1
        or args.batch_size < 2
        or not 0 < args.gpu_memory_fraction <= 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
        or not math.isfinite(args.weight_decay)
        or args.weight_decay < 0
        or not math.isfinite(args.max_grad_norm)
        or args.max_grad_norm <= 0
        or args.max_seconds < 1
    ):
        parser.error("invalid seed, optimizer, or resource setting")
    return args


def run_id(args) -> str:
    if args.phase == "anchor":
        return f"{args.task_id}_source012_common_anchor_seed{args.seed}_{args.device}"
    target = "all3" if len(args.target_channels) == 3 else f"slot{args.target_channels[0]}"
    variant = "common_anchor_control" if args.phase == "control" else "channel_robust_alignment_v2a"
    return f"{args.task_id}_source012_target_{target}_{variant}_seed{args.seed}_{args.device}"


class HiddenTargetDataset(Dataset):
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        waveform, label = self.base[index]
        if int(label) != -1:
            raise RuntimeError("target-train labels must be hidden as -1")
        return waveform, -1


class IndexedSourceDataset(Dataset):
    """Expose only the public dataset's local row number for replay auditing."""
    def __init__(self, base):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        waveform, label = self.base[index]
        return waveform, label, int(index)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def object_sha256(value) -> str:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def capture_rng(generators: dict[str, torch.Generator]) -> dict:
    numpy_state = np.random.get_state()
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": repr(random.getstate()),
        "numpy": {
            "name": numpy_state[0],
            # PyTorch 2.1 (the frozen WSL runtime) cannot construct a tensor
            # directly from NumPy uint32.  MT19937 keys fit losslessly in
            # int64 and are cast back to uint32 by restore_rng.
            "keys": torch.from_numpy(numpy_state[1].astype(np.int64, copy=True)),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "explicit_generators": {
            name: generator.get_state() for name, generator in sorted(generators.items())
        },
    }


def restore_rng(value: dict, generators: dict[str, torch.Generator]) -> None:
    torch.set_rng_state(value["torch_cpu"].cpu())
    if torch.cuda.is_available() and value.get("torch_cuda"):
        # torch.load(..., map_location=cuda) also moves serialized RNG byte
        # tensors to CUDA, while torch.cuda.set_rng_state_all requires CPU
        # ByteTensors on the PyTorch 2.1 runtime.
        cuda_states = [
            state.detach().to(device="cpu", dtype=torch.uint8)
            for state in value["torch_cuda"]
        ]
        torch.cuda.set_rng_state_all(cuda_states)
    random.setstate(ast.literal_eval(value["python"]))
    numpy_state = value["numpy"]
    np.random.set_state((
        numpy_state["name"], numpy_state["keys"].cpu().numpy().astype(np.uint32),
        int(numpy_state["position"]), int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    states = value.get("explicit_generators")
    if not isinstance(states, dict) or set(states) != set(generators):
        raise ValueError("anchor explicit-generator state set differs")
    for name, generator in generators.items():
        generator.set_state(states[name].cpu())


def code_identity() -> dict[str, str]:
    files = {
        "train_channel_robust_alignment_v2.py": ROOT / "train_channel_robust_alignment_v2.py",
        "public_loader.py": ROOT / "public_loader.py",
        "channel_robust_alignment_v2.py": WORKBENCH / "models" / "channel_robust_alignment_v2.py",
        "channel_robust_alignment.py": WORKBENCH / "models" / "channel_robust_alignment.py",
        "balanced_unbiased_mmd.py": WORKBENCH / "loss" / "balanced_unbiased_mmd.py",
        "spectral_grid_pilot.py": WORKBENCH / "models" / "spectral_grid_pilot.py",
        "spectral_shared.py": WORKBENCH / "models" / "spectral_shared.py",
        "experiment_protocol.py": WORKBENCH / "experiment_protocol.py",
    }
    return {name: sha256(path) for name, path in files.items()}


def configure_runtime(args) -> torch.device:
    if not args.execute:
        raise ValueError("explicit --execute is required")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.set_num_threads(args.cpu_threads)
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        if args.strict_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=False)
    return device


def make_model(args, classes: int) -> ChannelRobustAlignmentV2:
    return ChannelRobustAlignmentV2(
        3,
        classes,
        spectral_grid=args.spectral_grid,
        encoder_width=args.encoder_width,
        fusion="bounded_attention",
        fusion_residual_strength=LOSS_CONTRACT["fusion_residual_strength"],
    )


def source_objective(model, source_x, source_y, mask_generator):
    sensors = model.sensor_features(source_x, "source")
    fused = model.fuse_sensors(sensors)
    logits = model.classifier(fused)
    source_ce = nn.functional.cross_entropy(logits, source_y)
    mask = nonempty_channel_mask(
        len(source_x), sensors.shape[1], sensors.device, mask_generator
    )
    masked_logits = model.classifier(model.fuse_sensors(sensors, mask))
    masked_ce = nn.functional.cross_entropy(masked_logits, source_y)
    consistency = nn.functional.kl_div(
        nn.functional.log_softmax(masked_logits, dim=1),
        nn.functional.softmax(logits.detach(), dim=1),
        reduction="batchmean",
    )
    balance = model.attention_balance_loss(sensors)
    total = (
        source_ce
        + LOSS_CONTRACT["masked_ce_weight"] * masked_ce
        + LOSS_CONTRACT["consistency_weight"] * consistency
        + LOSS_CONTRACT["attention_balance_weight"] * balance
    )
    return {
        "sensors": sensors,
        "fused": fused,
        "logits": logits,
        "source_ce": source_ce,
        "masked_ce": masked_ce,
        "consistency": consistency,
        "balance": balance,
        "total": total,
        "mask": mask,
    }


@torch.no_grad()
def evaluate_source(model, loader, device, classes: int) -> dict:
    model.eval()
    confusion = np.zeros((classes, classes), dtype=np.int64)
    total_loss = 0.0
    count = 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(device), labels.to(device)
        logits = model.predict_source_branch(waveforms)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("source validation logits are non-finite")
        loss = nn.functional.cross_entropy(logits, labels)
        truth = labels.cpu().numpy()
        prediction = logits.argmax(dim=1).cpu().numpy()
        np.add.at(confusion, (truth, prediction), 1)
        total_loss += float(loss.item()) * len(labels)
        count += len(labels)
    result = classification_metrics(confusion.tolist())
    result["loss"] = total_loss / count
    return result


def _save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _base_metadata(args, task, model, *, stage: str, variant: str, calibration: dict) -> dict:
    channels = {"source": [0, 1, 2], "target": args.target_channels}
    return {
        "run_id": run_id(args),
        "stage": stage,
        "task_id": task.task_id,
        "dataset": task.dataset,
        "public_root": str(args.public_root.resolve()),
        "public_manifest_sha256": task.manifest_sha256,
        "public_plan_sha256": task.plan_sha256,
        "channels": channels,
        "window_size": task.window_size,
        "num_classes": task.num_classes,
        "variant": variant,
        "variant_contract": ARM_VARIANTS.get(variant),
        "loss_contract": LOSS_CONTRACT,
        "model_config": {
            "input_channels": 3,
            "spectral_grid": args.spectral_grid,
            "encoder_width": args.encoder_width,
            "fusion": "bounded_attention",
            "fusion_residual_strength": LOSS_CONTRACT["fusion_residual_strength"],
        },
        "model_details": model.architecture_metadata(),
        "calibration": calibration,
        "target_statistics": None if stage == "source_anchor" else "zscore_train_only",
        "target_train_fault_labels": "not_loaded" if stage == "source_anchor" else "required_minus_one",
        "target_dev_or_final_loader_constructed": False,
        "synchronous_pairs_used": False,
        "code_sha256": code_identity(),
        "seed": args.seed,
        "mask_seed": args.mask_seed,
        "source_draw_seed": args.source_draw_seed,
        "target_draw_seed": args.target_draw_seed,
        "target_dev_evaluated": False,
        "target_final_evaluated": False,
    }


def _load_task(args, load_public_task, *, source_only: bool = False):
    target = args.target_channels if args.target_channels is not None else [0, 1, 2]
    task = load_public_task(
        args.public_root.resolve(),
        args.task_id,
        {"source": [0, 1, 2], "target": target},
        role="source_only" if source_only else "train",
    )
    if task.window_size != 2048:
        raise ValueError("v2A requires 2048-sample windows")
    return task


def _generators(args, device):
    return {
        "source_loader": torch.Generator().manual_seed(args.seed),
        "target_loader": torch.Generator().manual_seed(args.seed + 10000),
        "mask": torch.Generator(device=device.type).manual_seed(args.mask_seed),
        "source_draw": torch.Generator(device=device.type).manual_seed(args.source_draw_seed),
        "target_draw": torch.Generator(device=device.type).manual_seed(args.target_draw_seed),
    }


@torch.no_grad()
def freeze_source_anchor_kernels(model, source_train, args, device) -> dict:
    pool_size = min(KERNEL_POOL_SIZE, len(source_train))
    if pool_size < 2:
        raise ValueError("source kernel pool requires at least two rows")
    rng = np.random.default_rng(args.seed + 30000)
    indices = np.sort(rng.choice(len(source_train), size=pool_size, replace=False)).astype("<i8")
    loader = DataLoader(
        Subset(source_train, indices.tolist()), batch_size=args.batch_size,
        shuffle=False, drop_last=False, num_workers=0,
    )
    model.eval()
    sensors, fused = [], []
    for waveforms, _labels in loader:
        values = model.sensor_features(waveforms.to(device), "source")
        sensors.append(values.cpu())
        fused.append(model.fuse_sensors(values).cpu())
    sensor_pool = torch.cat(sensors, dim=0).flatten(0, 1).float()
    fused_pool = torch.cat(fused, dim=0).float()

    def contract(values):
        squared = torch.pdist(values, p=2).square()
        base = float(squared.median())
        if not math.isfinite(base) or base <= 0:
            raise FloatingPointError("source-anchor median squared distance is not positive finite")
        bandwidths = [base * multiplier for multiplier in KERNEL_MULTIPLIERS]
        if any(not math.isfinite(value) or value <= 0 for value in bandwidths):
            raise FloatingPointError("source-anchor bandwidths are not positive finite")
        return {"median_squared_distance": base, "bandwidths": bandwidths}

    return {
        "kind": "source_anchor_fixed_five_rbf",
        "pool_seed": args.seed + 30000,
        "pool_size": pool_size,
        "pool_local_indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "multipliers": list(KERNEL_MULTIPLIERS),
        "kernel_weights": [1.0] * 5,
        "fused": contract(fused_pool),
        "channel": contract(sensor_pool),
        "source_feature_pool_sha256": hashlib.sha256(
            fused_pool.contiguous().numpy().tobytes()
            + sensor_pool.contiguous().numpy().tobytes()
        ).hexdigest(),
        "target_inputs_used": False,
        "fixed_before_branch_training": True,
    }


def validate_anchor_payload(checkpoint: dict) -> None:
    """Validate every replay-critical internal anchor contract."""
    if not isinstance(checkpoint, dict):
        raise ValueError("anchor payload must be a mapping")
    contract = checkpoint.get("kernel_contract")
    replay = checkpoint.get("anchor_replay_identity")
    if not isinstance(contract, dict) or not isinstance(replay, dict):
        raise ValueError("anchor kernel/replay contract is missing")
    if (
        contract.get("kind") != "source_anchor_fixed_five_rbf"
        or contract.get("multipliers") != list(KERNEL_MULTIPLIERS)
        or contract.get("kernel_weights") != [1.0] * 5
        or contract.get("target_inputs_used") is not False
        or contract.get("fixed_before_branch_training") is not True
        or type(contract.get("pool_seed")) is not int
        or type(contract.get("pool_size")) is not int
        or contract["pool_size"] < 2
    ):
        raise ValueError("anchor kernel contract schema differs")
    for name in ("pool_local_indices_sha256", "source_feature_pool_sha256"):
        value = contract.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("anchor kernel identity is invalid")
    for level in ("fused", "channel"):
        item = contract.get(level)
        if not isinstance(item, dict):
            raise ValueError("anchor kernel level is missing")
        base, bandwidths = item.get("median_squared_distance"), item.get("bandwidths")
        if not isinstance(base, (int, float)) or not math.isfinite(base) or base <= 0:
            raise ValueError("anchor kernel median is invalid")
        expected = [float(base) * value for value in KERNEL_MULTIPLIERS]
        if not isinstance(bandwidths, list) or len(bandwidths) != 5 or bandwidths != expected:
            raise ValueError("anchor frozen bandwidths differ")
    expected_replay = {
        "model_state_sha256": object_sha256(checkpoint["model_state_dict"]),
        "optimizer_state_sha256": object_sha256(checkpoint["optimizer_state_dict"]),
        "scheduler_state_sha256": object_sha256(checkpoint["scheduler_state_dict"]),
        "rng_state_sha256": object_sha256(checkpoint["rng_state"]),
        "kernel_contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    if replay != expected_replay:
        raise ValueError("anchor replay hashes differ from payload")


def train_anchor(args, load_public_task):
    device = configure_runtime(args)
    task = _load_task(args, load_public_task, source_only=True)
    output = args.output_root.resolve() / run_id(args)
    if output.exists():
        raise FileExistsError("anchor run already exists")
    output.mkdir(parents=True)
    write_json(output / "run_state.json", {"status": "running", "target_dev_evaluated": False})
    model = make_model(args, task.num_classes).to(device)
    calibration = model.calibrate_source_anchor(task.source_train, batch_size=args.batch_size)
    generators = _generators(args, device)
    source_loader = DataLoader(
        IndexedSourceDataset(task.source_train), batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=0, generator=generators["source_loader"],
    )
    source_val_loader = DataLoader(
        task.source_val, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    metadata = _base_metadata(
        args, task, model, stage="source_anchor", variant="source_anchor_v2a", calibration=calibration
    )
    metadata.update(
        pretrain_epochs=PRETRAIN_EPOCHS,
        selection="source_val_accuracy_then_lower_CE_then_earliest_epoch_across_60_source_epochs",
        loader_role="source_only",
        target_train_dataset_constructed=False,
        target_waveform_samples_iterated=False,
    )
    write_json(output / "run_metadata.json", metadata)
    best = None
    optimizer_steps = 0
    start = time.monotonic()
    try:
        for epoch in range(PRETRAIN_EPOCHS):
            model.train()
            sums = {name: 0.0 for name in ("source_ce", "masked_ce", "consistency", "balance", "total")}
            rows = correct = 0
            for source_x, source_y, _local_indices in source_loader:
                if time.monotonic() - start > args.max_seconds:
                    raise TimeoutError("anchor training exceeded wall-clock budget")
                source_x, source_y = source_x.to(device), source_y.to(device)
                values = source_objective(model, source_x, source_y, generators["mask"])
                if not torch.isfinite(values["total"]):
                    raise FloatingPointError("source objective is non-finite")
                optimizer.zero_grad(set_to_none=True)
                values["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
                batch = len(source_y)
                rows += batch
                correct += int((values["logits"].argmax(1) == source_y).sum().item())
                for name in sums:
                    sums[name] += float(values[name].item()) * batch
            source_val = evaluate_source(model, source_val_loader, device, task.num_classes)
            scheduler.step()
            record = {
                "epoch": epoch,
                "source_train_accuracy": correct / rows,
                **{name: value / rows for name, value in sums.items()},
                "source_val": source_val,
                "optimizer_steps_total": optimizer_steps,
            }
            with (output / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            key = (source_val["accuracy"], -source_val["loss"], -epoch)
            if best is None or key > best[0]:
                checkpoint = {
                    **metadata,
                    "selected_epoch": epoch,
                    "source_val": source_val,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "rng_state": capture_rng(generators),
                    "source_pretrain_optimizer_steps_at_selection": optimizer_steps,
                }
                _save_checkpoint(output / "best_model.pth", checkpoint)
                best = (key, checkpoint)
        selected = torch.load(output / "best_model.pth", map_location=device, weights_only=True)
        model.load_state_dict(selected["model_state_dict"], strict=True)
        kernel_contract = freeze_source_anchor_kernels(
            model, task.source_train, args, device
        )
        selected["kernel_contract"] = kernel_contract
        selected["anchor_replay_identity"] = {
            "model_state_sha256": object_sha256(selected["model_state_dict"]),
            "optimizer_state_sha256": object_sha256(selected["optimizer_state_dict"]),
            "scheduler_state_sha256": object_sha256(selected["scheduler_state_dict"]),
            "rng_state_sha256": object_sha256(selected["rng_state"]),
            "kernel_contract_sha256": hashlib.sha256(
                json.dumps(kernel_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }
        _save_checkpoint(output / "best_model.pth", selected)
        best = (best[0], selected)
        state = {
            "status": "completed",
            "stage": "source_anchor",
            "run_id": run_id(args),
            "task_id": task.task_id, "dataset": task.dataset,
            "channels": {"source": [0, 1, 2], "target": None},
            "seed": args.seed, "mask_seed": args.mask_seed,
            "source_draw_seed": args.source_draw_seed, "target_draw_seed": args.target_draw_seed,
            "code_sha256": metadata["code_sha256"],
            "selected_epoch": best[1]["selected_epoch"],
            "source_val": best[1]["source_val"],
            "checkpoint_sha256": sha256(output / "best_model.pth"),
            "optimizer_steps": optimizer_steps,
            "completed_pretrain_epochs": PRETRAIN_EPOCHS,
            "anchor_replay_identity": selected["anchor_replay_identity"],
            "kernel_contract": kernel_contract,
            "training_seconds": time.monotonic() - start,
            "target_train_waveforms_read": False,
            "target_train_dataset_constructed": False,
            "target_dev_evaluated": False,
            "target_final_evaluated": False,
        }
        write_json(output / "run_state.json", state)
        return state
    except Exception as error:
        write_json(output / "run_state.json", {
            "status": "failed", "stage": "source_anchor", "run_id": run_id(args),
            "error_type": type(error).__name__, "error": str(error),
            "target_dev_evaluated": False, "target_final_evaluated": False,
        })
        raise


def load_anchor(args, task, device):
    path = args.anchor_checkpoint.resolve(strict=True)
    if sha256(path) != args.anchor_sha256:
        raise ValueError("anchor checkpoint SHA256 differs")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    validate_anchor_payload(checkpoint)
    if (
        checkpoint.get("stage") != "source_anchor"
        or checkpoint.get("variant") != "source_anchor_v2a"
        or checkpoint.get("task_id") != task.task_id
        or checkpoint.get("dataset") != task.dataset
        or checkpoint.get("public_manifest_sha256") != task.manifest_sha256
        or checkpoint.get("public_plan_sha256") != task.plan_sha256
        or checkpoint.get("seed") != args.seed
        or checkpoint.get("mask_seed") != args.mask_seed
        or checkpoint.get("source_draw_seed") != args.source_draw_seed
        or checkpoint.get("target_draw_seed") != args.target_draw_seed
        or checkpoint.get("target_dev_evaluated") is not False
        or checkpoint.get("target_final_evaluated") is not False
        or checkpoint.get("code_sha256") != code_identity()
    ):
        raise ValueError("anchor identity differs from this arm")
    model = make_model(args, task.num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.calibration_evidence = checkpoint.get("calibration")
    if not model.source_calibrated.item() or model.target_calibrated.item():
        raise ValueError("anchor calibration flags differ")
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise FloatingPointError("anchor contains non-finite state")
    return model, checkpoint


def materialize_control(args, load_public_task):
    return run_branch(args, load_public_task, alignment_enabled=False)


def agreement_backward(model, source_loss, alignment_loss, weight: float) -> dict:
    named = tuple((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)
    all_parameters = tuple(parameter for _, parameter in named)
    alignment_indices = [index for index, (name, _) in enumerate(named) if not name.startswith("classifier.")]
    encoder_indices = [index for index, (name, _) in enumerate(named) if name.startswith("encoder.")]
    if not encoder_indices:
        raise RuntimeError("encoder parameter set is empty")
    alignment_parameters = tuple(all_parameters[index] for index in alignment_indices)
    source_grads = torch.autograd.grad(source_loss, all_parameters, retain_graph=True, allow_unused=True)
    alignment_grads = torch.autograd.grad(alignment_loss, alignment_parameters, allow_unused=True)
    source_grads = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(all_parameters, source_grads))
    alignment_grads = tuple(torch.zeros_like(p) if g is None else g for p, g in zip(alignment_parameters, alignment_grads))
    alignment_by_index = dict(zip(alignment_indices, alignment_grads))
    source_vector = torch.cat([source_grads[i].detach().reshape(-1).double() for i in encoder_indices])
    alignment_vector = torch.cat([alignment_by_index[i].detach().reshape(-1).double() for i in encoder_indices])
    if not torch.isfinite(source_vector).all() or not torch.isfinite(alignment_vector).all():
        raise FloatingPointError("gradient vectors are non-finite")
    dot = torch.dot(source_vector, alignment_vector)
    norm_product = float(source_vector.norm() * alignment_vector.norm())
    apply_alignment = bool(dot >= 0)
    for parameter, gradient in zip(all_parameters, source_grads):
        parameter.grad = gradient.detach().clone()
    if apply_alignment:
        for index, gradient in zip(alignment_indices, alignment_grads):
            parameter = all_parameters[index]
            parameter.grad = (parameter.grad.double() + weight * gradient.detach().double()).to(parameter.dtype)
    return {
        "alignment_applied": apply_alignment,
        "dot": float(dot),
        "cosine": float(dot) / norm_product if norm_product > 0 else None,
        "weight": float(weight),
        "rule": "negative encoder-gradient dot suppresses the complete alignment gradient",
    }


def run_branch(args, load_public_task, *, alignment_enabled: bool):
    device = configure_runtime(args)
    task = _load_task(args, load_public_task)
    output = args.output_root.resolve() / run_id(args)
    if output.exists():
        raise FileExistsError("branch run already exists")
    output.mkdir(parents=True)
    write_json(output / "run_state.json", {"status": "running", "target_dev_evaluated": False})
    model, anchor = load_anchor(args, task, device)
    target_train = HiddenTargetDataset(task.target_train)
    calibration = model.calibrate_target_train(target_train, batch_size=args.batch_size)
    generators = _generators(args, device)
    source_loader = DataLoader(
        IndexedSourceDataset(task.source_train), batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=0, generator=generators["source_loader"],
    )
    target_loader = (
        DataLoader(
            target_train, batch_size=args.batch_size, shuffle=True, drop_last=True,
            num_workers=0, generator=generators["target_loader"],
        ) if alignment_enabled else None
    )
    source_val_loader = DataLoader(
        task.source_val, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0
    )
    if len(source_loader) < 1 or (alignment_enabled and len(target_loader) < 1):
        raise ValueError("both domains require a full minibatch")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    optimizer.load_state_dict(anchor["optimizer_state_dict"])
    scheduler.load_state_dict(anchor["scheduler_state_dict"])
    restore_rng(anchor["rng_state"], generators)
    source_slot_cycle = BalancedPhysicalSlotCycle(
        [0, 1, 2], generators["source_draw"], device=device
    )
    target_slot_cycle = BalancedPhysicalSlotCycle(
        args.target_channels, generators["target_draw"], device=device
    )
    branch_start_identity = {
        "model_state_sha256": object_sha256(model.state_dict()),
        "optimizer_state_sha256": object_sha256(optimizer.state_dict()),
        "scheduler_state_sha256": object_sha256(scheduler.state_dict()),
        "rng_state_sha256": object_sha256(capture_rng(generators)),
        "source_slot_cycle": source_slot_cycle.state_dict(),
        "target_slot_cycle": target_slot_cycle.state_dict(),
        "anchor_checkpoint_sha256": args.anchor_sha256,
    }
    variant = "channel_robust_alignment_v2a" if alignment_enabled else "common_anchor_control"
    stage = "adaptation" if alignment_enabled else "common_anchor_control"
    metadata = _base_metadata(
        args, task, model, stage=stage, variant=variant,
        calibration=calibration,
    )
    metadata.update(
        anchor_checkpoint_sha256=args.anchor_sha256,
        anchor_selected_epoch=anchor["selected_epoch"],
        anchor_source_val=anchor["source_val"],
        adaptation_epochs=ADAPT_EPOCHS,
        selection="fixed branch epoch 19; no target-label or source-val checkpoint choice",
        alignment_enabled=alignment_enabled,
        branch_start_identity=branch_start_identity,
        kernel_contract=anchor["kernel_contract"],
        paired_branch_contract=(
            "same anchor model+optimizer+scheduler+all RNG; identical 20-epoch source batches/masks; "
            "candidate differs only by gated fixed-kernel alignment"
        ),
    )
    write_json(output / "run_metadata.json", metadata)
    start = time.monotonic()
    optimizer_steps = alignment_steps = applied_steps = suppressed_steps = 0
    source_index_digest = hashlib.sha256()
    source_mask_digest = hashlib.sha256()
    try:
        for epoch in range(ADAPT_EPOCHS):
            model.train()
            target_iterator = iter(target_loader) if alignment_enabled else None
            lam = LOSS_CONTRACT["alignment_weight"] if alignment_enabled else 0.0
            sums = {name: 0.0 for name in (
                "source_ce", "masked_ce", "consistency", "balance", "total_source",
                "fused_mmd", "channel_mmd", "alignment",
            )}
            rows = correct = 0
            geometry = []
            fused_mmd_batches = []
            channel_mmd_batches = []
            source_slot_counts = [0, 0, 0]
            target_slot_counts = [0] * len(args.target_channels)
            for source_x, source_y, local_indices in source_loader:
                if time.monotonic() - start > args.max_seconds:
                    raise TimeoutError("branch training exceeded wall-clock budget")
                source_x, source_y = source_x.to(device), source_y.to(device)
                values = source_objective(model, source_x, source_y, generators["mask"])
                source_index_digest.update(local_indices.numpy().astype("<i8", copy=False).tobytes())
                source_mask_digest.update(values["mask"].detach().cpu().numpy().astype(np.uint8).tobytes())
                fused_mmd = values["total"].new_zeros(())
                channel_mmd = values["total"].new_zeros(())
                alignment = values["total"].new_zeros(())
                source_slots = target_slots = None
                if alignment_enabled:
                    try:
                        target_x, target_y = next(target_iterator)
                    except StopIteration:
                        target_iterator = iter(target_loader)
                        target_x, target_y = next(target_iterator)
                    if not torch.all(target_y == -1).item():
                        raise RuntimeError("target-train labels were exposed")
                    target_x = target_x.to(device)
                    target_sensors = model.sensor_features(target_x, "target")
                    target_fused = model.fuse_sensors(target_sensors)
                    fused_mmd = unbiased_multi_kernel_mmd(
                        values["fused"], target_fused,
                        bandwidths=anchor["kernel_contract"]["fused"]["bandwidths"],
                    )
                    source_sample, source_slots = source_slot_cycle.sample(values["sensors"])
                    target_sample, target_slots = target_slot_cycle.sample(target_sensors)
                    channel_mmd = unbiased_multi_kernel_mmd(
                        source_sample, target_sample,
                        bandwidths=anchor["kernel_contract"]["channel"]["bandwidths"],
                    )
                    alignment = fused_mmd + LOSS_CONTRACT["channel_alignment_weight"] * channel_mmd
                    fused_mmd_batches.append(float(fused_mmd.item()))
                    channel_mmd_batches.append(float(channel_mmd.item()))
                if not torch.isfinite(values["total"] + alignment):
                    raise FloatingPointError("training objective is non-finite")
                optimizer.zero_grad(set_to_none=True)
                if alignment_enabled:
                    diagnostic = agreement_backward(model, values["total"], alignment, lam)
                else:
                    values["total"].backward()
                    diagnostic = {"alignment_applied": False, "dot": None, "cosine": None, "weight": 0.0}
                geometry.append(diagnostic)
                if alignment_enabled:
                    if diagnostic["alignment_applied"]:
                        applied_steps += 1
                    else:
                        suppressed_steps += 1
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
                optimizer_steps += 1
                if alignment_enabled:
                    alignment_steps += 1
                    for index in source_slots.tolist():
                        source_slot_counts[index] += 1
                    target_count_index = {slot: index for index, slot in enumerate(args.target_channels)}
                    for physical_slot in target_slots.tolist():
                        target_slot_counts[target_count_index[physical_slot]] += 1
                batch = len(source_y)
                rows += batch
                correct += int((values["logits"].argmax(1) == source_y).sum().item())
                measured = {
                    "source_ce": values["source_ce"], "masked_ce": values["masked_ce"],
                    "consistency": values["consistency"], "balance": values["balance"],
                    "total_source": values["total"], "fused_mmd": fused_mmd,
                    "channel_mmd": channel_mmd, "alignment": alignment,
                }
                for name, value in measured.items():
                    sums[name] += float(value.item()) * batch
            source_val = evaluate_source(model, source_val_loader, device, task.num_classes)
            scheduler.step()
            record = {
                "branch_epoch": epoch,
                "alignment_weight": lam,
                "source_train_accuracy": correct / rows,
                **{name: value / rows for name, value in sums.items()},
                "source_val": source_val,
                "optimizer_steps_total": optimizer_steps,
                "alignment_steps_total": alignment_steps,
                "applied_alignment_steps_total": applied_steps,
                "suppressed_alignment_steps_total": suppressed_steps,
                "alignment_applied_fraction": (
                    sum(g["alignment_applied"] for g in geometry) / len(geometry)
                    if alignment_enabled else None
                ),
                "mean_alignment_source_cosine": (
                    float(np.mean([g["cosine"] for g in geometry if g["cosine"] is not None]))
                    if alignment_enabled else None
                ),
                "source_physical_slot_counts": source_slot_counts,
                "target_physical_slot_counts": target_slot_counts,
                "signed_mmd": {
                    name: (
                        {
                            "negative_fraction": sum(value < 0 for value in batch_values) / len(batch_values),
                            "mean": float(np.mean(batch_values)),
                            "std": float(np.std(batch_values)),
                            "p05": float(np.quantile(batch_values, 0.05)),
                            "p50": float(np.quantile(batch_values, 0.50)),
                            "p95": float(np.quantile(batch_values, 0.95)),
                        } if batch_values else None
                    )
                    for name, batch_values in (
                        ("fused", fused_mmd_batches), ("channel", channel_mmd_batches)
                    )
                },
            }
            with (output / "epoch_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        checkpoint = {
            **metadata,
            "selected_epoch": ADAPT_EPOCHS - 1,
            "adaptation_endpoint": True,
            "source_val": source_val,
            "model_state_dict": model.state_dict(),
            "adapt_optimizer_steps": optimizer_steps,
            "alignment_steps": alignment_steps,
            "applied_alignment_steps": applied_steps,
            "suppressed_alignment_steps": suppressed_steps,
            "branch_source_indices_sha256": source_index_digest.hexdigest(),
            "branch_source_masks_sha256": source_mask_digest.hexdigest(),
            "branch_final_optimizer_state_sha256": object_sha256(optimizer.state_dict()),
            "branch_final_scheduler_state_sha256": object_sha256(scheduler.state_dict()),
            "branch_final_rng_state": capture_rng(generators),
            "branch_final_slot_cycles": {
                "source": source_slot_cycle.state_dict(), "target": target_slot_cycle.state_dict()
            },
        }
        _save_checkpoint(output / "best_model.pth", checkpoint)
        state = {
            "status": "completed", "stage": stage, "variant": variant, "run_id": run_id(args),
            "task_id": task.task_id, "dataset": task.dataset,
            "channels": {"source": [0, 1, 2], "target": args.target_channels},
            "seed": args.seed, "mask_seed": args.mask_seed,
            "source_draw_seed": args.source_draw_seed, "target_draw_seed": args.target_draw_seed,
            "code_sha256": metadata["code_sha256"],
            "anchor_checkpoint_sha256": args.anchor_sha256,
            "checkpoint_sha256": sha256(output / "best_model.pth"),
            "selected_epoch": ADAPT_EPOCHS - 1, "source_val": source_val,
            "optimizer_steps": optimizer_steps, "alignment_steps": alignment_steps,
            "applied_alignment_steps": applied_steps, "suppressed_alignment_steps": suppressed_steps,
            "completed_adaptation_epochs": ADAPT_EPOCHS,
            "branch_start_identity": branch_start_identity,
            "branch_source_indices_sha256": source_index_digest.hexdigest(),
            "branch_source_masks_sha256": source_mask_digest.hexdigest(),
            "training_seconds": time.monotonic() - start,
            "target_dev_evaluated": False, "target_final_evaluated": False,
        }
        write_json(output / "run_state.json", state)
        return state
    except Exception as error:
        write_json(output / "run_state.json", {
            "status": "failed", "stage": stage, "run_id": run_id(args),
            "error_type": type(error).__name__, "error": str(error),
            "optimizer_steps": optimizer_steps, "target_dev_evaluated": False,
            "target_final_evaluated": False,
        })
        raise


def adapt(args, load_public_task):
    return run_branch(args, load_public_task, alignment_enabled=True)


def run_phase(args, load_public_task):
    if args.phase == "anchor":
        return train_anchor(args, load_public_task)
    if args.phase == "control":
        return materialize_control(args, load_public_task)
    return adapt(args, load_public_task)


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "not_executed", "phase": args.phase}))
        return 0
    from public_loader import load_public_task
    result = run_phase(args, load_public_task)
    print(json.dumps({"status": result["status"], "run_id": result["run_id"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
