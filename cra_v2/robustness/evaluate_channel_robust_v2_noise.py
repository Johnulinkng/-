"""Fail-closed AWGN robustness evaluation for a completed CRA-v2 campaign.

The evaluator accepts one arm from a completed, pre-development-sealed
campaign.  It evaluates clean target-development windows plus the frozen
20/10/0 dB x 1701/1702/1703 AWGN matrix.  Noise is added to the selected raw
target window before the same per-window normalization used by the public
loader.  The checkpoint's target-train spectral calibration remains frozen.

All condition predictions are fixed before private target-development labels
are decoded.  The sealed target label file is SHA-checked as opaque bytes;
private origin metadata is not opened and target-final rows are never indexed,
decoded, or scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent

import numpy as np
import torch

SNR_DB = (20, 10, 0)
NOISE_SEEDS = (1701, 1702, 1703)
REALIZED_SNR_TOLERANCE_DB = 0.05
CAMPAIGN_KIND = "channel_robust_v2a_u_fixedk_campaign_v1"
TASKS = ("WP-S0", "WP-D1", "PG-S1", "PG-D1")
TARGETS = ([0, 1, 2], [0], [1], [2])
VARIANTS = ("common_anchor_control", "channel_robust_alignment_v2a")


class NoiseEvaluationError(ValueError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NoiseEvaluationError(f"cannot read campaign JSON: {path}") from error


def _portable_path(value) -> Path:
    """Resolve a v2 ``{windows,wsl}`` path record on the current host."""
    if isinstance(value, dict):
        key = "windows" if os.name == "nt" else "wsl"
        if key not in value:
            raise NoiseEvaluationError(f"path record lacks {key}")
        value = value[key]
    text = str(value)
    if os.name == "nt" and text.startswith("/mnt/") and len(text) > 7:
        text = f"{text[5].upper()}:/{text[7:]}"
    return Path(text).resolve(strict=True)


def _inside(root: Path, value: str | Path) -> Path:
    path = _portable_path(value)
    if not path.is_relative_to(root):
        raise NoiseEvaluationError("campaign path escapes its frozen root")
    return path


def _valid_sha(value) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument(
        "--code-root",
        type=Path,
        help="frozen campaign code root; defaults to <campaign>/code",
    )
    parser.add_argument(
        "--affected-slots",
        default="all",
        help="'all' for every observed target slot, or comma-separated physical slots",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if not isinstance(args.arm_id, str) or not args.arm_id.strip():
        parser.error("arm id must be nonempty")
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch size and CPU threads must be positive")
    if args.affected_slots != "all":
        try:
            slots = tuple(int(item.strip()) for item in args.affected_slots.split(","))
        except ValueError:
            parser.error("affected slots must be 'all' or comma-separated integers")
        if (
            not slots
            or len(set(slots)) != len(slots)
            or any(slot not in (0, 1, 2) for slot in slots)
        ):
            parser.error("affected slots must be unique physical slots from 0,1,2")
        args.affected_slots = slots
    return args


def validate_completed_campaign(campaign: Path, arm_id: str, checkpoint: Path, code_root: Path | None) -> dict:
    """Return a v2 branch only after the frozen campaign completion gates pass."""
    root = Path(campaign).resolve(strict=True)
    if not root.is_dir():
        raise NoiseEvaluationError("campaign root must be a directory")
    expected_plan_sha = (root / "plan.sha256").read_text(encoding="ascii").strip()
    if not _valid_sha(expected_plan_sha) or file_sha256(root / "plan.json") != expected_plan_sha:
        raise NoiseEvaluationError("campaign plan SHA mismatch")
    plan = _json(root / "plan.json")
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
        raise NoiseEvaluationError("campaign kind or frozen output name differs")
    arms = plan.get("arms")
    if (
        not isinstance(arms, list)
        or len(arms) != 32
        or len({arm.get("id") for arm in arms if isinstance(arm, dict)}) != len(arms)
        or [
            (arm.get("task"), arm.get("target"), arm.get("variant"))
            for arm in arms
        ]
        != [
            (task, target, variant)
            for task in TASKS
            for target in TARGETS
            for variant in VARIANTS
        ]
    ):
        raise NoiseEvaluationError("campaign arm matrix is malformed or nonunique")

    frozen_code = (root / "code" if code_root is None else Path(code_root)).resolve(strict=True)
    code_sha = plan.get("code_sha256")
    if not isinstance(code_sha, dict) or not code_sha:
        raise NoiseEvaluationError("campaign frozen code receipt is missing")
    for relative, digest in code_sha.items():
        if not _valid_sha(digest) or file_sha256(frozen_code / relative) != digest:
            raise NoiseEvaluationError(f"frozen campaign code changed: {relative}")

    fits = _json(root / "fit_results.json")
    dev_rows = _json(root / "dev_results.json")
    fit_state = _json(root / "fit_state.json")
    dev_state = _json(root / "dev_state.json")
    pipeline = _json(root / "pipeline_state.json")
    if (
        fit_state.get("status") != "completed"
        or dev_state.get("status") != "completed"
        or pipeline.get("status") != "completed"
        or pipeline.get("target_dev_evaluated") is not True
        or pipeline.get("target_final_evaluated") is not False
        or not isinstance(fits, dict)
        or len(fits.get("anchors", [])) != 4
        or len(fits.get("arms", [])) != len(arms)
        or len(dev_rows) != len(arms)
        or fit_state.get("completed_anchors") != 4
        or fit_state.get("completed_arms") != len(arms)
        or dev_state.get("completed_arms") != len(arms)
    ):
        raise NoiseEvaluationError("noise evaluation requires a completed development campaign")
    summary = _json(root / "summary.json")
    causal = summary.get("causal_comparison", {})
    expansion_gate = causal.get("expansion_gate", {})
    if (
        summary.get("method") != "v2A-U-fixedK"
        or summary.get("development_arms") != len(arms)
        or summary.get("target_final_evaluated") is not False
        or causal.get("control") != "common_anchor_control"
        or expansion_gate.get("passed") is not True
        or pipeline.get("screen_gate_passed") is not True
    ):
        raise NoiseEvaluationError("campaign does not preserve the target-final boundary")

    seal = _json(root / "predev_seal.json")
    predev_fit = root / "predev_fit_audit.json"
    fit_audit = _json(predev_fit)
    if (
        seal.get("plan_sha256") != expected_plan_sha
        or seal.get("fit_results_sha256") != file_sha256(root / "fit_results.json")
        or seal.get("predev_fit_audit_sha256") != file_sha256(predev_fit)
        or fit_audit.get("schema") != "channel_robust_v2_predev_fit_audit_v1"
        or fit_audit.get("status") != "PASS"
        or fit_audit.get("target_dev_read") is not False
        or fit_audit.get("target_final_evaluated") is not False
        or len(seal.get("anchor_checkpoints", [])) != 4
        or len(seal.get("branch_checkpoints", [])) != 32
        or seal.get("anchor_checkpoints")
        != [
            {"id": row["anchor"]["id"], "sha256": row["checkpoint_sha256"]}
            for row in fits["anchors"]
        ]
        or seal.get("branch_checkpoints")
        != [
            {"id": row["arm"]["id"], "sha256": row["checkpoint_sha256"]}
            for row in fits["arms"]
        ]
        or seal.get("target_dev_predictions_or_labels_read") is not False
        or seal.get("target_final_evaluated") is not False
    ):
        raise NoiseEvaluationError("pre-development seal differs from completed fits")

    selected = None
    supplied_checkpoint = Path(checkpoint).resolve(strict=True)
    for arm, fit, dev in zip(arms, fits["arms"], dev_rows):
        if fit.get("arm") != arm or dev.get("arm") != arm:
            raise NoiseEvaluationError("fit/development order differs from frozen arms")
        checkpoint_sha = fit.get("checkpoint_sha256")
        if not _valid_sha(checkpoint_sha) or dev.get("checkpoint_sha256") != checkpoint_sha:
            raise NoiseEvaluationError("fit/development checkpoint identity differs")
        result_path = root / "dev" / f"{arm['id']}.json"
        if not _valid_sha(dev.get("result_sha256")) or file_sha256(result_path) != dev["result_sha256"]:
            raise NoiseEvaluationError("development result changed")
        raw_dev = _json(result_path)
        checkpoint_identity = raw_dev.get("checkpoint_identity", {})
        if (
            raw_dev.get("checkpoint_sha256") != checkpoint_sha
            or raw_dev.get("target_final_evaluated") is not False
            or checkpoint_identity.get("variant") != arm.get("variant")
            or checkpoint_identity.get("task_id") != arm.get("task")
            or checkpoint_identity.get("channels", {}).get("target") != arm.get("target")
            or any(
                dev.get(key) != raw_dev.get(key)
                for key in (
                    "checkpoint_sha256",
                    "accuracy",
                    "macro_f1",
                    "per_class_recall",
                    "confusion_matrix",
                    "support_per_class",
                )
            )
        ):
            raise NoiseEvaluationError("development result checkpoint/arm identity differs")
        if arm["id"] == arm_id:
            if arm.get("variant") != "channel_robust_alignment_v2a":
                raise NoiseEvaluationError("AWGN follow-up is restricted to a v2A candidate arm")
            if file_sha256(supplied_checkpoint) != checkpoint_sha:
                raise NoiseEvaluationError("supplied branch checkpoint SHA differs")
            selected = {
                "campaign": root,
                "plan": plan,
                "plan_sha256": expected_plan_sha,
                "arm": arm,
                "fit": fit,
                "dev": dev,
                "checkpoint": supplied_checkpoint,
                "checkpoint_sha256": checkpoint_sha,
                "predev_seal_sha256": file_sha256(root / "predev_seal.json"),
                "code_root": frozen_code,
            }
    if selected is None:
        raise NoiseEvaluationError("requested arm is not in the completed campaign")
    return selected


def resolve_affected_slots(requested, observed_slots) -> tuple[int, ...]:
    observed = tuple(observed_slots)
    if observed not in ((0,), (1,), (2,), (0, 1, 2)):
        raise NoiseEvaluationError("arm target slots violate the channel study contract")
    affected = observed if requested == "all" else tuple(requested)
    if not affected or any(slot not in observed for slot in affected):
        raise NoiseEvaluationError("affected slots must be a nonempty subset of observed target slots")
    return affected


def standardize_raw_windows(raw: np.ndarray) -> np.ndarray:
    """Exactly reproduce SplitDataset's per-window channel normalization."""
    value = np.asarray(raw)
    if value.ndim != 3 or value.dtype != np.float32 or not np.isfinite(value).all():
        raise NoiseEvaluationError("raw target windows must be finite float32 [N,C,T]")
    mean = value.mean(axis=-1, keepdims=True)
    std = value.std(axis=-1, keepdims=True)
    result = ((value - mean) / np.where(std > 0, std, 1.0)).astype(np.float32)
    if not np.isfinite(result).all():
        raise FloatingPointError("window normalization produced non-finite values")
    return result


def perturb_raw_waveforms(
    raw: np.ndarray,
    row_tokens,
    task_id: str,
    observed_slots,
    affected_slots,
    snr_db: int,
    noise_seed: int,
    *,
    add_awgn_fn,
) -> tuple[np.ndarray, float]:
    """Add deterministic AWGN to only the requested physical target slots."""
    value = np.asarray(raw)
    observed = tuple(observed_slots)
    affected = tuple(affected_slots)
    identities = tuple(row_tokens)
    if (
        value.ndim != 3
        or value.dtype != np.float32
        or value.shape[0] < 1
        or value.shape[1] != len(observed)
        or value.shape[2] < 2
        or not np.isfinite(value).all()
        or len(identities) != len(value)
        or len(set(identities)) != len(identities)
        or not isinstance(task_id, str)
        or not task_id
        or affected == ()
        or any(slot not in observed for slot in affected)
        or snr_db not in SNR_DB
        or noise_seed not in NOISE_SEEDS
        or not callable(add_awgn_fn)
    ):
        raise NoiseEvaluationError("noise request differs from the frozen observed-channel plan")
    result = np.array(value, copy=True)
    max_error = 0.0
    for physical_slot in affected:
        local_slot = observed.index(physical_slot)
        tokens = tuple(
            f"{task_id}:target_dev:row={token}:physical_slot={physical_slot}"
            for token in identities
        )
        perturbed = add_awgn_fn(
            value[:, local_slot : local_slot + 1],
            tokens,
            snr_db=snr_db,
            noise_seed=noise_seed,
            affected_channels=(0,),
        )
        result[:, local_slot : local_slot + 1] = perturbed
        clean64 = value[:, local_slot].astype(np.float64)
        noise64 = perturbed[:, 0].astype(np.float64) - clean64
        signal_power = np.mean(clean64 * clean64, axis=-1)
        noise_power = np.mean(noise64 * noise64, axis=-1)
        if not np.all((signal_power > 0) & (noise_power > 0)):
            raise FloatingPointError("realized raw signal/noise power must be positive")
        realized = 10.0 * np.log10(signal_power / noise_power)
        if not np.isfinite(realized).all():
            raise FloatingPointError("realized SNR is non-finite")
        max_error = max(max_error, float(np.max(np.abs(realized - snr_db))))
    if max_error > REALIZED_SNR_TOLERANCE_DB:
        raise FloatingPointError("realized raw-power SNR differs by more than 0.05 dB")
    return result, max_error


def _state_sha256(model: torch.nn.Module, names=None) -> str:
    digest = hashlib.sha256()
    state = model.state_dict()
    selected = sorted(state) if names is None else tuple(names)
    for name in selected:
        value = state.get(name)
        if value is None or not isinstance(value, torch.Tensor):
            raise NoiseEvaluationError(f"model state is missing: {name}")
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(json.dumps(list(cpu.shape)).encode("ascii"))
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _raw_dev_batch(inputs, manifest: dict, start: int, end: int):
    dataset = inputs.target_dev
    try:
        indices = tuple(dataset._indices)
        channels = tuple(dataset._channels)
        data = dataset._data
    except AttributeError as error:
        raise NoiseEvaluationError("public evaluation view cannot provide raw dev windows") from error
    expected_indices = tuple(manifest["splits"]["target_test"]["indices"])
    final_indices = set(manifest["splits"]["target_final_test"]["indices"])
    if indices != expected_indices or final_indices.intersection(indices):
        raise NoiseEvaluationError("target-dev view differs from the sealed split")
    rows = indices[start:end]
    raw = np.stack(
        [np.asarray(data[row, channels, :], dtype=np.float32).copy() for row in rows]
    )
    provenance = manifest.get("row_provenance", {}).get("target", ())
    tokens = tuple(provenance[row]["row_token"] for row in rows)
    if len(tokens) != len(set(tokens)):
        raise NoiseEvaluationError("target-dev row tokens are not unique")
    return raw, tokens


def _open_dev_label_bank(public_root: Path, private_root: Path, task_id: str, inputs):
    """Open the sealed target label bank without opening private origin metadata.

    The whole-file SHA is checked as opaque bytes.  Only target-development
    indices are ever decoded into integer labels; target-final indices are
    used solely as a public disjointness set and are never indexed.
    """
    public_root = Path(public_root).resolve(strict=True)
    private_root = Path(private_root).resolve(strict=True)
    if private_root != public_root.parent / f"{public_root.name}_private_audit":
        raise NoiseEvaluationError("expected sibling private audit directory")
    manifest = _json(public_root / "tasks" / task_id / "task_manifest.json")
    target = manifest.get("target")
    if (
        not isinstance(target, dict)
        or not isinstance(target.get("condition"), str)
        or not isinstance(target.get("group"), str)
        or any("/" in target[key] or "\\" in target[key] for key in ("condition", "group"))
    ):
        raise NoiseEvaluationError("public target bank identity is malformed")
    bank_id = f"{target['condition']}_{target['group']}_target"
    label_path = (private_root / "target_labels" / bank_id / "label.npy").resolve(strict=True)
    if not label_path.is_relative_to((private_root / "target_labels").resolve(strict=True)):
        raise NoiseEvaluationError("target label bank escapes the private target-label root")
    try:
        labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise NoiseEvaluationError("private target label NPY header is invalid") from error
    target_rows = int(inputs.target_dev._data.shape[0])
    expected_size = labels.offset + labels.size * labels.dtype.itemsize
    if (
        not isinstance(labels, np.memmap)
        or labels.dtype != np.dtype(np.int64)
        or labels.shape != (target_rows,)
        or label_path.stat().st_size != expected_size
    ):
        raise NoiseEvaluationError("private target label shape/dtype differs")
    expected_sha = manifest.get("target_label_sha256_for_sealed_evaluation")
    if not _valid_sha(expected_sha) or file_sha256(label_path) != expected_sha:
        raise NoiseEvaluationError("sealed target label file SHA differs")
    receipt = {
        "task_id": task_id,
        "plan_sha256": inputs.plan_sha256,
        "task_manifest_sha256": inputs.manifest_sha256,
        "sealed_target_label_file_sha256": expected_sha,
        "sealed_target_label_file_hashed_as_opaque_bytes": True,
        "private_origin_metadata_opened": False,
        "target_final_indices_accessed": False,
        "final_signal_or_label_rows_decoded": False,
    }
    return receipt, manifest, label_path, labels, (label_path.stat().st_size, label_path.stat().st_mtime_ns)


def _metrics(truth: np.ndarray, prediction, classes: int) -> dict:
    confusion = np.zeros((classes, classes), dtype=np.int64)
    for true, predicted in zip(truth, prediction):
        confusion[int(true), int(predicted)] += 1
    recall, f1 = [], []
    for label in range(classes):
        tp = int(confusion[label, label])
        support = int(confusion[label].sum())
        predicted = int(confusion[:, label].sum())
        recall.append(tp / support if support else 0.0)
        denominator = support + predicted
        f1.append(2 * tp / denominator if denominator else 0.0)
    return {
        "accuracy": float(np.trace(confusion) / len(truth)),
        "macro_f1": float(np.mean(f1)),
        "per_class_recall": recall,
        "confusion_matrix": confusion.tolist(),
        "support_per_class": confusion.sum(axis=1).tolist(),
    }


def evaluate(args):
    if not args.execute:
        raise ValueError("explicit --execute is required")
    output = args.output_json.resolve()
    if output.exists():
        raise FileExistsError("noise evaluation output already exists")
    selected = validate_completed_campaign(
        args.campaign, args.arm_id, args.checkpoint, args.code_root
    )
    if output.is_relative_to(selected["campaign"]):
        raise NoiseEvaluationError("noise output must not modify the frozen campaign directory")
    arm = selected["arm"]
    observed = tuple(arm["target"])
    affected = resolve_affected_slots(args.affected_slots, observed)
    plan = selected["plan"]
    public_root = args.public_root.resolve(strict=True)
    private_root = args.private_root.resolve(strict=True)
    for relative, digest in plan["data"][arm["task"]]["sha256"].items():
        if not _valid_sha(digest) or file_sha256(public_root / relative) != digest:
            raise NoiseEvaluationError(f"configured public data metadata differs: {relative}")
    if output.is_relative_to(public_root) or output.is_relative_to(private_root):
        raise NoiseEvaluationError("noise output must stay outside public/private data roots")
    channels = {"source": [0, 1, 2], "target": list(observed)}

    protocol = selected["code_root"] / "protocol_next"
    workbench = selected["code_root"] / "workbench"
    for path in (protocol, workbench):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from evaluate_channel_robust_alignment_v2_dev import LockedPredictor
        from noise_protocol import add_awgn
        from public_loader import load_public_task
    except (ImportError, OSError) as error:
        raise NoiseEvaluationError("cannot import the frozen v2 evaluation stack") from error

    torch.set_num_threads(args.cpu_threads)
    inputs = load_public_task(
        public_root, arm["task"], channels, role="evaluate_dev"
    )
    receipt, manifest, label_path, labels, label_stat = _open_dev_label_bank(
        public_root, private_root, arm["task"], inputs
    )
    predictor = LockedPredictor(
        selected["checkpoint"], selected["checkpoint_sha256"], inputs, channels
    )
    calibration_names = ("target_center", "target_scale", "target_calibrated")
    calibration_sha = _state_sha256(predictor.model, calibration_names)
    model_state_sha = _state_sha256(predictor.model)

    conditions = [("clean", None, None)] + [
        (f"snr{snr}_seed{seed}", snr, seed)
        for snr in SNR_DB
        for seed in NOISE_SEEDS
    ]
    predictions = {condition_id: [] for condition_id, _, _ in conditions}
    snr_error = {condition_id: 0.0 for condition_id, snr, _ in conditions if snr is not None}
    for start in range(0, len(inputs.target_dev), args.batch_size):
        end = min(start + args.batch_size, len(inputs.target_dev))
        raw, row_tokens = _raw_dev_batch(inputs, manifest, start, end)
        for condition_id, snr, seed in conditions:
            if snr is None:
                condition_raw = raw
            else:
                condition_raw, error = perturb_raw_waveforms(
                    raw, row_tokens, arm["task"], observed, affected, snr, seed,
                    add_awgn_fn=add_awgn,
                )
                snr_error[condition_id] = max(snr_error[condition_id], error)
            normalized = standardize_raw_windows(condition_raw)
            value = np.asarray(predictor(predictor.path, normalized))
            if (
                value.shape != (end - start,)
                or value.dtype.kind not in "iu"
                or np.any(value < 0)
                or np.any(value >= inputs.num_classes)
            ):
                raise NoiseEvaluationError("predictor returned invalid class IDs")
            predictions[condition_id].extend(value.tolist())

    # Every clean/noisy prediction is immutable before this first label decode.
    indices = manifest["splits"]["target_test"]["indices"]
    final_indices = set(manifest["splits"]["target_final_test"]["indices"])
    if final_indices.intersection(indices):
        raise NoiseEvaluationError("target-development/final index sets overlap")
    truth = np.asarray(labels[indices])
    plan_metadata = _json(public_root / "build_plan.json")
    expected_per_class = plan_metadata["quotas_per_class"]["target"]["dev"]
    if (
        len(truth) != len(inputs.target_dev)
        or np.any(truth < 0)
        or np.any(truth >= inputs.num_classes)
        or not np.array_equal(
            np.bincount(truth, minlength=inputs.num_classes),
            np.full(inputs.num_classes, expected_per_class),
        )
        or any(len(value) != len(truth) for value in predictions.values())
    ):
        raise NoiseEvaluationError("private target-development labels are invalid")
    if (label_path.stat().st_size, label_path.stat().st_mtime_ns) != label_stat:
        raise NoiseEvaluationError("private target label file changed during evaluation")
    if file_sha256(selected["checkpoint"]) != selected["checkpoint_sha256"]:
        raise NoiseEvaluationError("locked checkpoint changed during noise evaluation")
    if (
        _state_sha256(predictor.model, calibration_names) != calibration_sha
        or _state_sha256(predictor.model) != model_state_sha
    ):
        raise NoiseEvaluationError("model state or clean calibration changed during evaluation")

    condition_rows = []
    for condition_id, snr, seed in conditions:
        metrics = _metrics(truth, predictions[condition_id], inputs.num_classes)
        condition_rows.append(
            {
                "condition_id": condition_id,
                "snr_db": "clean" if snr is None else snr,
                "noise_seed": seed,
                "maximum_realized_raw_power_snr_error_db": (
                    None if snr is None else snr_error[condition_id]
                ),
                **metrics,
            }
        )
    target_tag = "all3" if len(observed) == 3 else f"slot{observed[0]}"
    result = {
        "kind": "channel_robust_alignment_awgn_dev_v2",
        "campaign": str(selected["campaign"]),
        "campaign_plan_sha256": selected["plan_sha256"],
        "predev_seal_sha256": selected["predev_seal_sha256"],
        "arm": arm,
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "checkpoint_model_state_sha256": model_state_sha,
        "clean_target_train_calibration_sha256": calibration_sha,
        "evaluated_split": "target_dev_only",
        "target_dev_history_boundary": plan.get("target_dev_history_boundary"),
        "target_final_evaluated": False,
        "n_samples": len(truth),
        "target_view": {
            "tag": target_tag,
            "observed_physical_slots": list(observed),
            "affected_physical_slots": list(affected),
            "affected_local_positions": [observed.index(slot) for slot in affected],
        },
        "noise_protocol": {
            "conditions": ["clean", 20, 10, 0],
            "noise_seeds": list(NOISE_SEEDS),
            "distribution": "zero-mean Gaussian, independently keyed by task/row/physical-slot/SNR/seed",
            "snr_definition": "10*log10(raw_window_mean_square/noise_mean_square)",
            "realized_snr_tolerance_db": REALIZED_SNR_TOLERANCE_DB,
            "operation_order": [
                "select target-development raw window and observed physical slots",
                "add AWGN to requested observed slots only",
                "apply training-identical per-window per-channel z-score",
                "apply checkpoint feature transform and frozen target-train calibration",
                "predict",
            ],
            "calibration_policy": "clean target-train calibration from checkpoint; never re-estimated per noise condition",
            "label_policy": "all condition predictions fixed before target-development labels are decoded",
        },
        "slot_results": {
            target_tag: {
                "observed_physical_slots": list(observed),
                "affected_physical_slots": list(affected),
                "conditions": condition_rows,
            }
        },
        "receipt": receipt,
        "evaluation_code_sha256": {
            name: file_sha256(protocol / name)
            for name in (
                "evaluate_channel_robust_alignment_v2_dev.py",
                "noise_protocol.py",
                "private_evaluator.py",
                "public_loader.py",
            )
        },
    }
    result["evaluation_code_sha256"]["evaluate_channel_robust_v2_noise.py"] = file_sha256(
        Path(__file__).resolve()
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main(argv=None):
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "not_executed",
                    "split": "target_dev_only",
                    "snr_db": ["clean", *SNR_DB],
                    "noise_seeds": list(NOISE_SEEDS),
                    "target_final_evaluated": False,
                },
                allow_nan=False,
            )
        )
        return 0
    result = evaluate(args)
    print(
        json.dumps(
            {
                "status": "completed",
                "arm_id": result["arm"]["id"],
                "target_view": result["target_view"]["tag"],
                "conditions": len(next(iter(result["slot_results"].values()))["conditions"]),
                "checkpoint_sha256": result["checkpoint_sha256"],
                "target_final_evaluated": False,
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
