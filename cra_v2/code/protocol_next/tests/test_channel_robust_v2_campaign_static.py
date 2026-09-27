from __future__ import annotations

import ast
import copy
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_channel_robust_v2_campaign as runner

ROOT = Path(__file__).resolve().parents[2]


def test_training_contract_is_literal_and_no_adaptive_bandwidth():
    trainer = (ROOT / "protocol_next" / "train_channel_robust_alignment_v2.py").read_text(encoding="utf-8")
    loss = (ROOT / "workbench" / "loss" / "balanced_unbiased_mmd.py").read_text(encoding="utf-8")
    ast.parse(trainer)
    ast.parse(loss)
    assert '"alignment_weight": 0.02' in trainer
    assert "PRETRAIN_EPOCHS = 60" in trainer
    assert "ADAPT_EPOCHS = 20" in trainer
    loss_tree = ast.parse(loss)
    target = next(node for node in loss_tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "unbiased_multi_kernel_mmd")
    called_attributes = {
        node.func.attr for node in ast.walk(target)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    called_names = {
        node.func.id for node in ast.walk(target)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not ({"clamp", "clamp_min", "abs", "relu"} & called_attributes)
    assert not ({"abs", "relu"} & called_names)
    assert "pdist" not in called_attributes


def test_campaign_declares_four_anchors_and_32_branches():
    runner = (ROOT / "protocol_next" / "run_channel_robust_v2_campaign.py").read_text(encoding="utf-8")
    tree = ast.parse(runner)
    assert "common_anchor_control" in runner
    assert "channel_robust_alignment_v2a" in runner
    assert "strong_v4_reference" in runner
    assert "target_final" in runner
    # Matrix size is also asserted by runner.prepare before writing a plan.
    assert "len(matrix) != 32" in runner
    assert "len(anchor_matrix) != 4" in runner


def test_anchor_selection_reconstruction_rejects_epoch_tamper():
    rows = [{"epoch": epoch, "source_val": {"accuracy": 0.5, "loss": 1.0 - epoch / 1000}}
            for epoch in range(60)]
    assert runner.selected_anchor_record(rows)["epoch"] == 59
    bad = copy.deepcopy(rows)
    bad.pop(17)
    try:
        runner.selected_anchor_record(bad)
    except RuntimeError as error:
        assert "0..59" in str(error)
    else:
        raise AssertionError("truncated anchor log must fail")


def test_v4_runtime_path_must_stay_under_frozen_project(tmp_path):
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir(); outside.mkdir()
    key = "windows" if __import__("os").name == "nt" else "wsl"
    plan = {"strong_v4_reference": {
        "project_root": {key: str(project)}, "campaign_root": {key: str(outside)},
        "campaign_relative": "experiments/v4",
    }}
    try:
        runner._load_strong_v4_dev(plan)
    except RuntimeError as error:
        assert "escapes" in str(error)
    else:
        raise AssertionError("out-of-project v4 path must fail closed")


def test_seal_code_requires_independent_fit_audit_and_v4_receipt():
    source = (ROOT / "protocol_next" / "run_channel_robust_v2_campaign.py").read_text(encoding="utf-8")
    assert "audit_fit_artifacts(output, plan, results)" in source
    assert '"predev_fit_audit_sha256"' in source
    assert "channel_robust_screen_v4_completed_audit_20260925.json" in source
    assert "channel_robust_completed_campaign_audit_v1" in source
    assert 'completed_audit_receipt"].get("status") != "PASS"' in source


def test_common_anchor_cross_table_linkage_rejects_self_consistent_substitution():
    anchors = [{"anchor": {"task": task}, "checkpoint_sha256": task.lower().replace("-", "") * 16}
               for task in ("WP-S0", "WP-D1", "PG-S1", "PG-D1")]
    arms = []
    for task, digest in [(row["anchor"]["task"], row["checkpoint_sha256"]) for row in anchors]:
        arms.append({"arm": {"task": task}, "anchor_checkpoint_sha256": digest,
                     "branch_start_identity": {"anchor_checkpoint_sha256": digest}})
    mapping = runner.validate_branch_anchor_linkage({"anchors": anchors, "arms": arms})
    assert mapping["WP-S0"] == anchors[0]["checkpoint_sha256"]
    arms[0]["anchor_checkpoint_sha256"] = "f" * 64
    arms[0]["branch_start_identity"]["anchor_checkpoint_sha256"] = "f" * 64
    try:
        runner.validate_branch_anchor_linkage({"anchors": anchors, "arms": arms})
    except RuntimeError as error:
        assert "unique actual common anchor" in str(error)
    else:
        raise AssertionError("self-consistent branch anchor substitution must fail")


def test_verify_dev_results_rejects_last_registry_row_tamper(tmp_path):
    output = tmp_path
    (output / "dev").mkdir()
    arms, fit_rows, registry = [], [], []
    for index in range(32):
        arm = {"id": f"arm{index}", "task": "T", "target": [0],
               "variant": "common_anchor_control"}
        checkpoint_sha = f"{index:064x}"
        raw = {"checkpoint_sha256": checkpoint_sha, "accuracy": 0.5, "macro_f1": 0.4,
               "per_class_recall": [0.4], "confusion_matrix": [[1]], "support_per_class": [1],
               "target_final_evaluated": False,
               "checkpoint_identity": {"variant": arm["variant"], "task_id": "T",
                                       "channels": {"target": [0]}}}
        path = output / "dev" / f"{arm['id']}.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        digest = runner.sha256(path)
        row = {"arm": arm, "result_sha256": digest,
               **{key: raw[key] for key in runner.DEV_METRIC_KEYS}}
        arms.append(arm); fit_rows.append({"checkpoint_sha256": checkpoint_sha}); registry.append(row)
    (output / "dev_state.json").write_text(json.dumps({"status": "completed", "completed_arms": 32}), encoding="utf-8")
    (output / "dev_results.json").write_text(json.dumps(registry), encoding="utf-8")
    plan, fits = {"arms": arms}, {"arms": fit_rows}
    assert len(runner.verify_dev_results(output, plan, fits, all_required=True)) == 32
    registry[-1]["accuracy"] = 0.99
    (output / "dev_results.json").write_text(json.dumps(registry), encoding="utf-8")
    try:
        runner.verify_dev_results(output, plan, fits, all_required=True)
    except RuntimeError as error:
        assert "registry row differs" in str(error)
    else:
        raise AssertionError("last registry row tamper must fail")


def test_fit_path_boundary_and_recursive_nonfinite_fail_closed(tmp_path):
    output = tmp_path / "campaign"
    runs = output / "runs"
    runs.mkdir(parents=True)
    expected = "expected_run"
    actual = runs / expected
    actual.mkdir()
    (actual / "best_model.pth").write_bytes(b"x")
    row = {"run": str(actual), "checkpoint": str(actual / "best_model.pth")}
    assert runner.resolved_run(output, row, expected) == actual.resolve()
    outside = tmp_path / "outside"
    outside.mkdir(); (outside / "best_model.pth").write_bytes(b"x")
    row = {"run": str(outside), "checkpoint": str(outside / "best_model.pth")}
    try:
        runner.resolved_run(output, row, expected)
    except RuntimeError as error:
        assert "escapes" in str(error)
    else:
        raise AssertionError("out-of-campaign run must fail")
    try:
        runner.assert_recursive_finite({"epoch": [{"loss": float("nan")}]}, "record")
    except RuntimeError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("nested non-finite value must fail")


def test_running_commit_window_missing_or_partial_refuses_rerun(tmp_path):
    output = tmp_path / "campaign"
    (output / "runs").mkdir(parents=True)
    anchor = runner.anchors()[0]
    plan = {"anchors": runner.anchors(), "arms": runner.arms()}
    state = {"status": "running", "active": anchor, "active_kind": "anchor"}
    results = {"anchors": [], "arms": []}
    try:
        runner.recover_completed_unregistered(output, plan, results, state)
    except RuntimeError as error:
        assert "no exact orphan" in str(error)
    else:
        raise AssertionError("missing running orphan must fail")
    run = output / "runs" / runner.expected_run_id(anchor, anchor=True)
    run.mkdir()
    (run / "run_state.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    try:
        runner.recover_completed_unregistered(output, plan, results, state)
    except RuntimeError as error:
        assert "not completed" in str(error)
    else:
        raise AssertionError("partial running orphan must fail")


def test_running_completed_orphan_is_audited_then_committed(tmp_path, monkeypatch):
    output = tmp_path / "campaign"
    (output / "runs").mkdir(parents=True)
    anchor = runner.anchors()[0]
    plan = {"anchors": runner.anchors(), "arms": runner.arms()}
    campaign_state = {"status": "running", "active": anchor, "active_kind": "anchor"}
    results = {"anchors": [], "arms": []}
    run = output / "runs" / runner.expected_run_id(anchor, anchor=True)
    run.mkdir()
    state = {
        "status": "completed", "run_id": runner.expected_run_id(anchor, anchor=True),
        "checkpoint_sha256": "a" * 64, "selected_epoch": 3,
        "task_id": anchor["task"], "dataset": anchor["dataset"],
        "channels": {"source": [0, 1, 2], "target": None}, "seed": 42,
        "mask_seed": runner.TRAINING["mask_seed"],
        "source_draw_seed": runner.TRAINING["source_draw_seed"],
        "target_draw_seed": runner.TRAINING["target_draw_seed"], "code_sha256": {},
        "source_val": {"accuracy": 1.0}, "optimizer_steps": 10,
        "anchor_replay_identity": {}, "kernel_contract": {}, "training_seconds": 1.0,
    }
    (run / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    called = []
    monkeypatch.setattr(runner, "audit_fit_artifacts", lambda *args: called.append(args) or {"status": "PASS"})
    assert runner.recover_completed_unregistered(output, plan, results, campaign_state) is True
    assert called and len(results["anchors"]) == 1
    saved = json.loads((output / "fit_state.json").read_text(encoding="utf-8"))
    assert saved["last_commit_recovered"] == anchor["id"]


def test_source_only_role_seed_binding_and_historical_name_are_frozen():
    trainer = (ROOT / "protocol_next" / "train_channel_robust_alignment_v2.py").read_text(encoding="utf-8")
    loader = (ROOT / "protocol_next" / "public_loader.py").read_text(encoding="utf-8")
    campaign = (ROOT / "protocol_next" / "run_channel_robust_v2_campaign.py").read_text(encoding="utf-8")
    assert 'role="source_only" if source_only else "train"' in trainer
    assert "checkpoint.get(\"mask_seed\") != args.mask_seed" in trainer
    assert "checkpoint.get(\"source_draw_seed\") != args.source_draw_seed" in trainer
    assert "checkpoint.get(\"target_draw_seed\") != args.target_draw_seed" in trainer
    assert "class SourceOnlyTask" in loader
    assert '"historical_threshold_baseline": True' in campaign
    assert '"formal_gate_baseline"' not in campaign


def test_launcher_stale_lock_recovery_is_conservative_and_audited():
    source = (ROOT / "protocol_next" / "start_channel_robust_v2_campaign.ps1").read_text(encoding="utf-8")
    assert "$stale.output -ne $Output" in source
    assert "Get-Process -Id $stalePid" in source
    assert "$age -lt 300" in source
    assert "stale_lock_recovered" in source
    assert "launcher_events.jsonl" in source


def test_frozen_launcher_resolves_its_own_campaign_and_never_defaults_to_v1():
    launcher = (ROOT / "protocol_next" / "start_channel_robust_v2_campaign.ps1").read_text(
        encoding="utf-8"
    )
    assert '[string]$Output = ""' in launcher
    assert 'Join-Path $PSScriptRoot "..\\.."' in launcher
    assert 'Join-Path $candidate "plan.json"' in launcher
    assert "Non-frozen launcher requires an explicit -Output" in launcher
    assert "20260925_channel_robust_v2a_screen_v1" not in launcher


def test_launcher_persists_native_stderr_before_returning_child_exit_code():
    launcher = (ROOT / "protocol_next" / "start_channel_robust_v2_campaign.ps1").read_text(
        encoding="utf-8"
    )
    assert '$previousErrorAction = $ErrorActionPreference' in launcher
    assert '$ErrorActionPreference = "Continue"' in launcher
    assert 'Tee-Object -FilePath $logPath -Append' in launcher
    assert '$code = [int]$LASTEXITCODE' in launcher
    assert '$ErrorActionPreference = $previousErrorAction' in launcher
    assert 'launcher_events.jsonl' in launcher
