from __future__ import annotations

import math
import io
import random
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "workbench"), str(ROOT / "protocol_next")]

from loss.balanced_unbiased_mmd import BalancedPhysicalSlotCycle, unbiased_multi_kernel_mmd
from train_channel_robust_alignment_v2 import (
    agreement_backward,
    capture_rng,
    object_sha256,
    restore_rng,
    validate_anchor_payload,
)
from run_channel_robust_v2_campaign import assert_checkpoint_finite


KERNELS = [0.25, 0.5, 1.0, 2.0, 4.0]


def test_rng_snapshot_uses_runtime_compatible_lossless_numpy_keys_and_roundtrips():
    random.seed(71)
    np.random.seed(72)
    torch.manual_seed(73)
    generators = {"probe": torch.Generator().manual_seed(74)}
    state = capture_rng(generators)
    assert state["numpy"]["keys"].dtype == torch.int64
    assert int(state["numpy"]["keys"].min()) >= 0
    assert int(state["numpy"]["keys"].max()) <= 2**32 - 1
    expected = (
        random.random(),
        int(np.random.randint(0, 2**31)),
        torch.rand(4),
        torch.rand(4, generator=generators["probe"]),
    )
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    generators["probe"].manual_seed(4)
    restore_rng(state, generators)
    observed = (
        random.random(),
        int(np.random.randint(0, 2**31)),
        torch.rand(4),
        torch.rand(4, generator=generators["probe"]),
    )
    assert observed[0:2] == expected[0:2]
    assert torch.equal(observed[2], expected[2])
    assert torch.equal(observed[3], expected[3])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA runtime required")
def test_rng_checkpoint_loaded_on_cuda_restores_cpu_byte_cuda_states_exactly():
    torch.cuda.manual_seed_all(81)
    generators = {"probe": torch.Generator(device="cuda").manual_seed(82)}
    state = capture_rng(generators)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, map_location="cuda", weights_only=True)
    assert loaded["torch_cuda"] and all(item.is_cuda for item in loaded["torch_cuda"])
    expected_global = torch.rand(8, device="cuda")
    expected_explicit = torch.rand(8, device="cuda", generator=generators["probe"])
    torch.cuda.manual_seed_all(1)
    generators["probe"].manual_seed(2)
    restore_rng(loaded, generators)
    observed_global = torch.rand(8, device="cuda")
    observed_explicit = torch.rand(8, device="cuda", generator=generators["probe"])
    assert torch.equal(observed_global, expected_global)
    assert torch.equal(observed_explicit, expected_explicit)


def test_physical_slot_cycle_has_prefix_balance_and_replays():
    def trace(slots, seed):
        generator = torch.Generator().manual_seed(seed)
        cycle = BalancedPhysicalSlotCycle(slots, generator, device="cpu")
        seen = []
        for _ in range(1000):
            features = torch.randn(32, len(slots), 4)
            _, physical = cycle.sample(features)
            seen.extend(physical.tolist())
            counts = [seen.count(slot) for slot in slots]
            assert max(counts) - min(counts) <= 1
        return seen, cycle.state_dict()
    first, state = trace([0, 1, 2], 913)
    second, replay = trace([0, 1, 2], 913)
    assert first == second and state == replay
    singleton, singleton_state = trace([2], 17)
    assert set(singleton) == {2}
    assert singleton_state["physical_slots"] == [2]


def test_domain_cycles_use_independent_rng_streams():
    source = BalancedPhysicalSlotCycle([0, 1, 2], torch.Generator().manual_seed(1), device="cpu")
    target = BalancedPhysicalSlotCycle([0, 1, 2], torch.Generator().manual_seed(2), device="cpu")
    before = target.state_dict().copy()
    source.sample(torch.randn(32, 3, 5))
    assert target.state_dict() == before


def test_unbiased_mmd_refuses_unequal_rows_and_keeps_negative_values():
    with pytest.raises(ValueError, match="identical"):
        unbiased_multi_kernel_mmd(torch.randn(32, 3), torch.randn(31, 3), bandwidths=KERNELS)
    observed = [
        float(unbiased_multi_kernel_mmd(
            torch.randn(16, 3, generator=torch.Generator().manual_seed(seed)),
            torch.randn(16, 3, generator=torch.Generator().manual_seed(seed + 10000)),
            bandwidths=KERNELS,
        ))
        for seed in range(200)
    ]
    assert any(value < 0 for value in observed)


def _null_mean(channels: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(32, channels, 4, generator=generator)
    target = torch.randn(32, channels, 4, generator=generator)
    source_cycle = BalancedPhysicalSlotCycle(range(channels), torch.Generator().manual_seed(seed + 1), device="cpu")
    target_cycle = BalancedPhysicalSlotCycle(range(channels), torch.Generator().manual_seed(seed + 2), device="cpu")
    x, _ = source_cycle.sample(source)
    y, _ = target_cycle.sample(target)
    return float(unbiased_multi_kernel_mmd(x, y, bandwidths=KERNELS))


def test_null_500_seed_ci_and_all3_single_slot_agreement():
    all3 = torch.tensor([_null_mean(3, seed) for seed in range(500)], dtype=torch.float64)
    slot = torch.tensor([_null_mean(1, seed + 2000) for seed in range(500)], dtype=torch.float64)
    for values in (all3, slot):
        se = values.std(unbiased=True) / math.sqrt(len(values))
        assert values.mean() - 1.96 * se <= 0 <= values.mean() + 1.96 * se
    pooled_se = torch.sqrt(all3.var(unbiased=True) / len(all3) + slot.var(unbiased=True) / len(slot))
    assert abs(all3.mean() - slot.mean()) <= 2 * pooled_se


def test_mean_shift_repeated_mmd_is_monotone():
    means = []
    for delta in (0.0, 0.25, 0.5, 1.0):
        values = []
        for seed in range(100):
            generator = torch.Generator().manual_seed(seed)
            x = torch.randn(32, 4, generator=generator)
            y = torch.randn(32, 4, generator=generator) + delta
            values.append(float(unbiased_multi_kernel_mmd(x, y, bandwidths=KERNELS)))
        means.append(sum(values) / len(values))
    assert all(right >= left for left, right in zip(means, means[1:]))


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(1, 1, bias=False)
        self.fusion_score = nn.Parameter(torch.tensor([1.0]))
        self.classifier = nn.Linear(1, 1, bias=False)


@pytest.mark.parametrize("sign,expected_apply", [(-1.0, False), (1.0, True)])
def test_gradient_hard_gate_exact_formula_and_classifier_boundary(sign, expected_apply):
    model = TinyModel()
    source = model.encoder.weight.sum() + model.fusion_score.sum() + model.classifier.weight.sum()
    alignment = sign * model.encoder.weight.sum() + sign * model.fusion_score.sum()
    result = agreement_backward(model, source, alignment, 0.02)
    assert result["alignment_applied"] is expected_apply
    expected = 1.0 + (0.02 * sign if expected_apply else 0.0)
    assert torch.equal(model.encoder.weight.grad, torch.tensor([[expected]]))
    assert torch.equal(model.fusion_score.grad, torch.tensor([expected]))
    assert torch.equal(model.classifier.weight.grad, torch.ones_like(model.classifier.weight))


def test_trainer_cli_has_no_dev_or_final_argument():
    source = (ROOT / "protocol_next" / "train_channel_robust_alignment_v2.py").read_text(encoding="utf-8")
    assert '"--target-dev' not in source
    assert '"--target-final' not in source
    assert 'target_y == -1' in source
    assert 'LOSS_CONTRACT["alignment_weight"]' in source


def _valid_anchor_payload():
    contract = {
        "kind": "source_anchor_fixed_five_rbf", "pool_seed": 30042, "pool_size": 8,
        "pool_local_indices_sha256": "1" * 64, "multipliers": [0.25, 0.5, 1.0, 2.0, 4.0],
        "kernel_weights": [1.0] * 5,
        "fused": {"median_squared_distance": 2.0, "bandwidths": [0.5, 1.0, 2.0, 4.0, 8.0]},
        "channel": {"median_squared_distance": 4.0, "bandwidths": [1.0, 2.0, 4.0, 8.0, 16.0]},
        "source_feature_pool_sha256": "2" * 64, "target_inputs_used": False,
        "fixed_before_branch_training": True,
    }
    value = {"model_state_dict": {"x": torch.tensor([1.0])},
             "optimizer_state_dict": {"state": {}}, "scheduler_state_dict": {"last_epoch": 3},
             "rng_state": {"explicit_generators": {"x": torch.tensor([1], dtype=torch.uint8)}},
             "kernel_contract": contract}
    import hashlib, json
    value["anchor_replay_identity"] = {
        "model_state_sha256": object_sha256(value["model_state_dict"]),
        "optimizer_state_sha256": object_sha256(value["optimizer_state_dict"]),
        "scheduler_state_sha256": object_sha256(value["scheduler_state_dict"]),
        "rng_state_sha256": object_sha256(value["rng_state"]),
        "kernel_contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
    }
    return value


def test_anchor_payload_rejects_kernel_or_replay_tampering():
    value = _valid_anchor_payload()
    validate_anchor_payload(value)
    value["kernel_contract"]["target_inputs_used"] = True
    with pytest.raises(ValueError):
        validate_anchor_payload(value)
    value = _valid_anchor_payload()
    value["anchor_replay_identity"]["rng_state_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validate_anchor_payload(value)


def test_dev_predictor_rejects_missing_pair_provenance_before_model_use(tmp_path):
    from types import SimpleNamespace
    from evaluate_channel_robust_alignment_v2_dev import LockedPredictor
    payload = {
        "task_id": "T", "dataset": "D", "public_manifest_sha256": "m",
        "public_plan_sha256": "p", "channels": {"source": [0, 1, 2], "target": [2]},
        "target_statistics": "zscore_train_only", "target_train_fault_labels": "required_minus_one",
        "target_dev_or_final_loader_constructed": False, "target_dev_evaluated": False,
        "target_final_evaluated": False, "synchronous_pairs_used": False,
        "variant": "common_anchor_control",
    }
    path = tmp_path / "bad.pth"
    torch.save(payload, path)
    import hashlib
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    task = SimpleNamespace(task_id="T", dataset="D", manifest_sha256="m", plan_sha256="p",
                           num_classes=2, window_size=2048)
    with pytest.raises(Exception, match="provenance|boundary"):
        LockedPredictor(path, digest, task, {"source": [0, 1, 2], "target": [2]})


def test_recursive_checkpoint_nonfinite_tensor_rejected():
    with pytest.raises(RuntimeError, match="non-finite tensor"):
        assert_checkpoint_finite({"optimizer": {"state": [torch.tensor([float("inf")])]}})
