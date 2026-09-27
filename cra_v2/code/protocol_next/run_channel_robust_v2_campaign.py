"""Frozen CRA-v2A-U-fixedK campaign: 4 anchors plus 32 paired branches.

``prepare`` is metadata-only and never imports torch.  Fit actions are one
durable optimizer run at a time.  Development scoring is impossible until all
artifacts are verified and a pre-development seal has been written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocol_next"
WORKBENCH = ROOT / "workbench"
for path in (PROTOCOL, WORKBENCH):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

TASKS = (
    ("waterpump", "WP-S0", "same_condition_cross_sensor"),
    ("waterpump", "WP-D1", "cross_sensor_and_condition"),
    ("gearbox", "PG-S1", "same_condition_cross_sensor"),
    ("gearbox", "PG-D1", "cross_sensor_and_condition"),
)
TARGETS = ((0, 1, 2), (0,), (1,), (2,))
VARIANTS = ("common_anchor_control", "channel_robust_alignment_v2a")
SEED = 42
TRAINING = {
    "device": "cuda", "strict_cuda": True, "cublas_workspace_config": ":4096:8",
    "gpu_memory_fraction": 0.30, "cpu_threads": 1, "batch_size": 32,
    "pretrain_epochs": 60, "branch_epochs": 20, "lr": 0.001,
    "weight_decay": 0.0005, "alignment_weight": 0.02, "max_grad_norm": 5.0,
    "max_seconds": 1800, "spectral_grid": "low3k_pool2", "encoder_width": 64,
    "mask_seed": 424284, "source_draw_seed": 525294, "target_draw_seed": 626304,
}
CODE_FILES = (
    "protocol_next/run_channel_robust_v2_campaign.py",
    "protocol_next/start_channel_robust_v2_campaign.ps1",
    "protocol_next/train_channel_robust_alignment_v2.py",
    "protocol_next/evaluate_channel_robust_alignment_v2_dev.py",
    "protocol_next/public_loader.py", "protocol_next/private_evaluator.py",
    "protocol_next/antipair_builder.py",
    "protocol_next/tests/test_channel_robust_alignment_v2.py",
    "protocol_next/tests/test_channel_robust_v2_campaign_static.py",
    "workbench/models/channel_robust_alignment_v2.py",
    "workbench/models/channel_robust_alignment.py",
    "workbench/models/spectral_grid_pilot.py", "workbench/models/spectral_shared.py",
    "workbench/loss/balanced_unbiased_mmd.py", "workbench/experiment_protocol.py",
    "audit/channel_robust_alignment_v2_predev_20260925.md",
    "audit/channel_robust_v2_design_decision_20260925.md",
)
V4 = ROOT / "experiments" / "20260925_channel_robust_screen_v4"
V4_AUDIT = ROOT / "audit" / "channel_robust_screen_v4_completed_audit_20260925.json"
V4_AUDIT_SHA256 = "1961ad2e3a082ddca6fc27285f88b8aad9bccb9163c38876eda2f1aaaca56fb1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def target_tag(target) -> str:
    return "all3" if len(target) == 3 else f"slot{target[0]}"


def cross_platform_path(path: Path) -> dict:
    resolved = path.resolve()
    value = str(resolved)
    result = {"windows": value}
    if len(value) >= 3 and value[1:3] == ":\\":
        result["wsl"] = f"/mnt/{value[0].lower()}/{value[3:].replace(chr(92), '/')}"
    return result


def runtime_path(record: dict) -> Path:
    key = "windows" if os.name == "nt" else "wsl"
    if not isinstance(record, dict) or key not in record:
        raise RuntimeError("cross-platform path record is missing")
    return Path(record[key])


def anchors() -> list[dict]:
    result = [
        {"dataset": dataset, "task": task, "family": family, "seed": SEED,
         "id": f"{task}_source012_common_anchor_seed{SEED}"}
        for dataset, task, family in TASKS
    ]
    if len(result) != 4 or len({item["id"] for item in result}) != 4:
        raise RuntimeError("anchor matrix must contain four unique items")
    return result


def arms() -> list[dict]:
    result = []
    for dataset, task, family in TASKS:
        for target in TARGETS:
            for variant in VARIANTS:
                result.append({
                    "dataset": dataset, "task": task, "family": family,
                    "target": list(target), "variant": variant, "seed": SEED,
                    "id": f"{task}_target_{target_tag(target)}_{variant}_seed{SEED}",
                })
    if len(result) != 32 or len({item["id"] for item in result}) != 32:
        raise RuntimeError("branch matrix must contain 32 unique items")
    return result


def data_root(dataset: str) -> Path:
    return ROOT / "data_versions" / f"antipair_v1_{dataset}_source_first"


def _data_identity(root: Path, task: str) -> dict:
    names = ("version_manifest.json", "build_plan.json", f"tasks/{task}/task_manifest.json")
    return {name: sha256(root / name) for name in names}


def snapshot_code(destination: Path) -> dict:
    identities = {}
    for name in CODE_FILES:
        source, target = ROOT / name, destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        identities[name] = sha256(target)
        if identities[name] != sha256(source):
            raise RuntimeError("code changed while taking frozen snapshot")
    return identities


def strong_v4_reference() -> dict:
    """Freeze v4 control identities without opening any v4 development result."""
    plan_path, fit_path, seal_path = V4 / "plan.json", V4 / "fit_results.json", V4 / "predev_seal.json"
    plan, fits, seal = read_json(plan_path), read_json(fit_path), read_json(seal_path)
    audit = read_json(V4_AUDIT)
    if sha256(plan_path) != (V4 / "plan.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("v4 plan hash differs")
    if seal.get("plan_sha256") != sha256(plan_path) or seal.get("fit_results_sha256") != sha256(fit_path):
        raise RuntimeError("v4 pre-development seal differs")
    expected_inputs = {
        "plan.json": sha256(plan_path), "fit_results.json": sha256(fit_path),
        "dev_results.json": sha256(V4 / "dev_results.json"), "summary.json": sha256(V4 / "summary.json"),
    }
    if (sha256(V4_AUDIT) != V4_AUDIT_SHA256
            or audit.get("schema") != "channel_robust_completed_campaign_audit_v1"
            or audit.get("status") != "PASS" or audit.get("campaign_plan_sha256") != sha256(plan_path)
            or any(audit.get("input_sha256", {}).get(name) != digest for name, digest in expected_inputs.items())):
        raise RuntimeError("v4 completed campaign audit receipt differs")
    refs = []
    for row in fits:
        arm = row["arm"]
        if arm["variant"] != "matched_control" or arm["seed"] != SEED:
            continue
        checkpoint = V4 / "runs" / Path(row["run"]).name / "best_model.pth"
        if sha256(checkpoint) != row["checkpoint_sha256"]:
            raise RuntimeError("v4 strong-control checkpoint changed")
        refs.append({"arm": arm, "checkpoint_sha256": row["checkpoint_sha256"],
                     "checkpoint_relative": str(checkpoint.relative_to(ROOT)).replace("\\", "/")})
    if len(refs) != 16:
        raise RuntimeError("v4 must provide 16 strong matched controls")
    return {
        "campaign_root": cross_platform_path(V4), "project_root": cross_platform_path(ROOT),
        "campaign_relative": str(V4.relative_to(ROOT)).replace("\\", "/"),
        "plan_sha256": sha256(plan_path), "fit_results_sha256": sha256(fit_path),
        "predev_seal_sha256": sha256(seal_path), "controls": refs,
        # Hash-only freeze: prepare does not parse or inspect development scores.
        "dev_results_sha256": sha256(V4 / "dev_results.json"),
        "summary_sha256": sha256(V4 / "summary.json"),
        "report_sha256": sha256(V4 / "REPORT.md"),
        "completed_audit_receipt": {"path": cross_platform_path(V4_AUDIT),
                                    "sha256": V4_AUDIT_SHA256,
                                    "schema": audit["schema"], "status": audit["status"]},
        "dev_results_read_during_prepare": False, "historical_threshold_baseline": True,
    }


def prepare(output: Path) -> None:
    if output.exists():
        raise FileExistsError("fresh output directory required")
    anchor_matrix, matrix = anchors(), arms()
    if len(matrix) != 32 or len(anchor_matrix) != 4:
        raise RuntimeError("frozen matrix count differs")
    data, task_metadata = {}, {}
    for dataset, task, family in TASKS:
        root = data_root(dataset)
        manifest = read_json(root / "tasks" / task / "task_manifest.json")
        if manifest.get("task_id") != task or manifest.get("dataset") != dataset:
            raise RuntimeError("task manifest identity differs")
        data[task] = {"root": cross_platform_path(root), "sha256": _data_identity(root, task)}
        task_metadata[task] = {"dataset": dataset, "family": family,
                               "source": manifest.get("source"), "target": manifest.get("target"),
                               "classes": manifest.get("classes")}
    output.mkdir(parents=True)
    code = snapshot_code(output / "code")
    plan = {
        "kind": "channel_robust_v2a_u_fixedk_campaign_v1", "created_unix": time.time(),
        "output_name": output.name, "anchors": anchor_matrix, "arms": matrix,
        "counts": {"source_anchors": 4, "second_stage_control": 16,
                   "second_stage_v2a": 16, "second_stage_total": 32,
                   "optimizer_runs_total": 36},
        "data": data, "task_metadata": task_metadata, "training": TRAINING,
        "code_sha256": code, "strong_v4_reference": strong_v4_reference(),
        "information_boundary": {
            "anchor": "source-train labels plus source-val labels only; no target waveform",
            "branches": "source labels plus unlabeled target-train waveform; labels sentinel -1",
            "selection": "anchor source-val Acc/CE/earliest; branch fixed epoch 19",
            "target_dev": "blocked until all 4 anchors and 32 branches are SHA-sealed",
            "target_final_evaluated": False,
        },
        "screen_gate": {
            "causal_control": "common_anchor_control", "candidate": "channel_robust_alignment_v2a",
            "decides_expansion": True, "historical_control": "frozen v4 matched_control",
            "historical_comparison_decides_expansion": False,
            "minimum_jointly_improved_conditions": 10, "minimum_mean_accuracy_delta": 0.01,
            "minimum_mean_macro_f1_delta": 0.02, "minimum_double_cross_macro_f1_delta": 0.02,
            "minimum_positive_slots_per_3to1_task": 2,
            "weakest_3to1_macro_f1_must_improve": True, "no_new_zero_recall": True,
        },
        "target_dev_history_boundary": "reused development partitions; not untouched confirmation evidence",
    }
    write_json(output / "plan.json", plan)
    (output / "plan.sha256").write_text(sha256(output / "plan.json") + "\n", encoding="ascii")
    for directory in ("runs", "dev", "logs"):
        (output / directory).mkdir()
    write_json(output / "fit_results.json", {"anchors": [], "arms": []})
    write_json(output / "fit_state.json", {"status": "prepared", "completed_anchors": 0,
               "total_anchors": 4, "completed_arms": 0, "total_arms": 32, "active": None})
    write_json(output / "dev_results.json", [])
    write_json(output / "dev_state.json", {"status": "blocked_until_seal", "completed_arms": 0,
               "total_arms": 32, "active_arm": None})
    write_json(output / "pipeline_state.json", {"status": "prepared", "target_dev_evaluated": False,
               "target_final_evaluated": False})


def frozen(output: Path) -> dict:
    expected_runner = output / "code" / "protocol_next" / "run_channel_robust_v2_campaign.py"
    if Path(__file__).resolve() != expected_runner.resolve():
        raise RuntimeError("fit/dev/report must use the frozen campaign snapshot")
    if sha256(output / "plan.json") != (output / "plan.sha256").read_text(encoding="ascii").strip():
        raise RuntimeError("campaign plan changed after freeze")
    plan = read_json(output / "plan.json")
    if plan.get("output_name") != output.name or plan.get("anchors") != anchors() or plan.get("arms") != arms():
        raise RuntimeError("campaign matrix/output differs")
    if plan.get("training") != TRAINING:
        raise RuntimeError("training contract differs")
    for name, digest in plan["code_sha256"].items():
        if sha256(output / "code" / name) != digest:
            raise RuntimeError("frozen code changed: " + name)
    for task, record in plan["data"].items():
        for name, digest in record["sha256"].items():
            if sha256(runtime_path(record["root"]) / name) != digest:
                raise RuntimeError(f"data metadata changed: {task}/{name}")
    return plan


def resource_snapshot() -> dict:
    import torch
    values = {}
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        raise RuntimeError("fit is supported only inside the declared Linux/WSL runtime")
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            values[line.split(":", 1)[0]] = int(line.split()[1])
    result = {"time": time.time(), "wsl_mem_available_mib": values.get("MemAvailable", 0) // 1024,
              "load_1m_per_cpu": os.getloadavg()[0] / max(1, os.cpu_count() or 1),
              "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(gpu_name=torch.cuda.get_device_name(), gpu_free_mib=free // 2**20,
                      gpu_total_mib=total // 2**20)
        utilization = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True,
        ).stdout.strip().splitlines()
        processes = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
             "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True,
        ).stdout.strip().splitlines()
        result["nvidia_smi_gpu_rows"] = utilization
        result["nvidia_smi_compute_process_rows"] = processes
    audit_path = os.environ.get("CRA_V2_LAUNCH_AUDIT")
    if not audit_path:
        raise RuntimeError("fit requires a fresh Windows launch audit from start_channel_robust_v2_campaign.ps1")
    audit = read_json(Path(audit_path))
    if time.time() - float(audit.get("created_unix", 0)) > 120 or audit.get("approved") is not True:
        raise RuntimeError("Windows launch audit is stale or not approved")
    result["windows_launch_audit"] = audit
    result["windows_launch_audit_sha256"] = sha256(Path(audit_path))
    return result


def _common_argv(output: Path, plan: dict, task: str, seed: int) -> list[str]:
    return [
        "--public-root", str(runtime_path(plan["data"][task]["root"])), "--task-id", task,
        "--output-root", str(output / "runs"), "--spectral-grid", TRAINING["spectral_grid"],
        "--encoder-width", str(TRAINING["encoder_width"]), "--seed", str(seed),
        "--mask-seed", str(TRAINING["mask_seed"]),
        "--source-draw-seed", str(TRAINING["source_draw_seed"]),
        "--target-draw-seed", str(TRAINING["target_draw_seed"]),
        "--device", TRAINING["device"], "--gpu-memory-fraction", str(TRAINING["gpu_memory_fraction"]),
        "--cpu-threads", str(TRAINING["cpu_threads"]), "--batch-size", str(TRAINING["batch_size"]),
        "--lr", str(TRAINING["lr"]), "--weight-decay", str(TRAINING["weight_decay"]),
        "--max-grad-norm", str(TRAINING["max_grad_norm"]),
        "--max-seconds", str(TRAINING["max_seconds"]), "--strict-cuda", "--execute",
    ]


def anchor_argv(output: Path, plan: dict, item: dict) -> list[str]:
    return ["--phase", "anchor", *_common_argv(output, plan, item["task"], item["seed"])]


def arm_argv(output: Path, plan: dict, arm: dict, anchor: dict) -> list[str]:
    phase = "control" if arm["variant"] == "common_anchor_control" else "adapt"
    return [
        "--phase", phase, *_common_argv(output, plan, arm["task"], arm["seed"]),
        "--target-slots", ",".join(map(str, arm["target"])),
        "--anchor-checkpoint", anchor["checkpoint"],
        "--anchor-sha256", anchor["checkpoint_sha256"],
    ]


def expected_run_id(item: dict, *, anchor: bool) -> str:
    if anchor:
        return f"{item['task']}_source012_common_anchor_seed{item['seed']}_{TRAINING['device']}"
    return (f"{item['task']}_source012_target_{target_tag(item['target'])}_"
            f"{item['variant']}_seed{item['seed']}_{TRAINING['device']}")


def resolved_run(output: Path, row: dict, expected: str) -> Path:
    runs_root = (output / "runs").resolve(strict=True)
    run = Path(row.get("run", "")).resolve(strict=True)
    try:
        relative = run.relative_to(runs_root)
    except ValueError as error:
        raise RuntimeError("fit run escapes current campaign/runs") from error
    if relative != Path(expected) or run.name != expected or run.parent != runs_root:
        raise RuntimeError("fit run does not match its exact frozen run_id")
    checkpoint = Path(row.get("checkpoint", "")).resolve(strict=True)
    if checkpoint != run / "best_model.pth":
        raise RuntimeError("fit checkpoint path differs from exact run checkpoint")
    return run


def verify_fit_results(output: Path, plan: dict, *, all_required=False) -> dict:
    state, results = read_json(output / "fit_state.json"), read_json(output / "fit_results.json")
    anchor_rows, arm_rows = results.get("anchors"), results.get("arms")
    if not isinstance(anchor_rows, list) or not isinstance(arm_rows, list):
        raise RuntimeError("fit result shape differs")
    if len(anchor_rows) != state["completed_anchors"] or len(arm_rows) != state["completed_arms"]:
        raise RuntimeError("fit state/result count differs")
    if all_required and (len(anchor_rows) != 4 or len(arm_rows) != 32 or state["status"] != "completed"):
        raise RuntimeError("all 4 anchors and 32 branches are required")
    for row, item in zip(anchor_rows, plan["anchors"]):
        run = resolved_run(output, row, expected_run_id(item, anchor=True))
        if row["anchor"] != item or sha256(run / "best_model.pth") != row["checkpoint_sha256"]:
            raise RuntimeError("anchor order or hash differs")
    for row, arm in zip(arm_rows, plan["arms"]):
        run = resolved_run(output, row, expected_run_id(arm, anchor=False))
        if row["arm"] != arm or sha256(run / "best_model.pth") != row["checkpoint_sha256"]:
            raise RuntimeError("branch order or hash differs")
        if read_json(run / "run_state.json").get("status") != "completed":
            raise RuntimeError("branch run is not complete")
    return results


def _verify_completed_pair(rows: list[dict], pair_end: int) -> None:
    control, candidate = rows[pair_end - 1], rows[pair_end]
    if control["arm"]["variant"] != VARIANTS[0] or candidate["arm"]["variant"] != VARIANTS[1]:
        raise RuntimeError("paired branch order differs")
    if (control["arm"]["task"], control["arm"]["target"], control["arm"]["seed"]) != (
        candidate["arm"]["task"], candidate["arm"]["target"], candidate["arm"]["seed"]
    ):
        raise RuntimeError("control/candidate pair identity differs")
    for key in ("anchor_checkpoint_sha256", "branch_start_identity",
                "branch_source_indices_sha256", "branch_source_masks_sha256"):
        if control[key] != candidate[key]:
            raise RuntimeError("paired causal replay differs: " + key)


def validate_branch_anchor_linkage(results: dict) -> dict[str, str]:
    mapping = {}
    for row in results.get("anchors", []):
        task, digest = row.get("anchor", {}).get("task"), row.get("checkpoint_sha256")
        if task in mapping or not isinstance(task, str) or not isinstance(digest, str):
            raise RuntimeError("each task must have exactly one actual common anchor")
        mapping[task] = digest
    if results.get("arms") and set(mapping) != {task for _dataset, task, _family in TASKS}:
        raise RuntimeError("branches require the complete four-task common-anchor mapping")
    for row in results.get("arms", []):
        task = row.get("arm", {}).get("task")
        expected = mapping.get(task)
        if (row.get("anchor_checkpoint_sha256") != expected
                or row.get("branch_start_identity", {}).get("anchor_checkpoint_sha256") != expected):
            raise RuntimeError("branch does not link to the task's unique actual common anchor")
    return mapping


def read_jsonl(path: Path) -> list[dict]:
    records = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(line)
            assert_recursive_finite(record, f"{path}:{number}")
            records.append(record)
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"invalid JSONL at {path}:{number}") from error
    return records


def assert_recursive_finite(value, location: str) -> None:
    if isinstance(value, float) and not __import__("math").isfinite(value):
        raise RuntimeError(f"non-finite numeric value at {location}")
    if isinstance(value, dict):
        for key, child in value.items():
            assert_recursive_finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_recursive_finite(child, f"{location}[{index}]")


def assert_checkpoint_finite(value, location="checkpoint") -> None:
    import torch
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite tensor at {location}")
    elif isinstance(value, float):
        if not __import__("math").isfinite(value):
            raise RuntimeError(f"non-finite numeric value at {location}")
    elif isinstance(value, dict):
        for key, child in value.items():
            assert_checkpoint_finite(child, f"{location}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_checkpoint_finite(child, f"{location}[{index}]")


def selected_anchor_record(records: list[dict]) -> dict:
    if len(records) != 60 or [row.get("epoch") for row in records] != list(range(60)):
        raise RuntimeError("anchor must contain exactly epochs 0..59")
    for row in records:
        source_val = row.get("source_val")
        if not isinstance(source_val, dict) or not isinstance(source_val.get("accuracy"), (int, float)) \
                or not isinstance(source_val.get("loss"), (int, float)):
            raise RuntimeError("anchor source-val record is invalid")
    return max(records, key=lambda row: (row["source_val"]["accuracy"],
                                         -row["source_val"]["loss"], -row["epoch"]))


def audit_fit_artifacts(output: Path, plan: dict, results: dict) -> dict:
    """Reconstruct training facts before any target-development access."""
    import torch
    import train_channel_robust_alignment_v2 as trainer
    expected_code = trainer.code_identity()
    anchor_by_task = validate_branch_anchor_linkage(results)
    receipt = {"schema": "channel_robust_v2_predev_fit_audit_v1", "status": "PASS",
               "created_unix": time.time(), "anchors": [], "arms": [],
               "unique_anchor_sha256_by_task": anchor_by_task,
               "target_dev_read": False, "target_final_evaluated": False}
    for row in results["anchors"]:
        item = row["anchor"]
        run = resolved_run(output, row, expected_run_id(item, anchor=True))
        checkpoint_path = run / "best_model.pth"
        records = read_jsonl(run / "epoch_metrics.jsonl")
        selected = selected_anchor_record(records)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert_checkpoint_finite(checkpoint)
        trainer.validate_anchor_payload(checkpoint)
        state = read_json(run / "run_state.json")
        assert_recursive_finite(state, f"{run}/run_state.json")
        step_totals = [record.get("optimizer_steps_total") for record in records]
        if any(type(value) is not int or value <= 0 for value in step_totals) \
                or any(right <= left for left, right in zip(step_totals, step_totals[1:])):
            raise RuntimeError("anchor optimizer step totals are invalid")
        if (checkpoint.get("stage") != "source_anchor" or checkpoint.get("variant") != "source_anchor_v2a"
                or any(value.get("run_id") != expected_run_id(item, anchor=True)
                       for value in (row, state, checkpoint))
                or state.get("checkpoint_sha256") != row.get("checkpoint_sha256")
                or checkpoint.get("selected_epoch") != selected["epoch"]
                or checkpoint.get("source_val") != selected["source_val"]
                or row.get("source_val") != selected["source_val"]
                or state.get("source_val") != selected["source_val"]
                or state.get("selected_epoch") != selected["epoch"]
                or state.get("completed_pretrain_epochs") != 60
                or state.get("optimizer_steps") != step_totals[-1]
                or row.get("optimizer_steps") != step_totals[-1]
                or checkpoint.get("source_pretrain_optimizer_steps_at_selection")
                   != selected.get("optimizer_steps_total")
                or any(value.get("task_id") != item["task"] for value in (row, state, checkpoint))
                or any(value.get("dataset") != item["dataset"] for value in (row, state, checkpoint))
                or any(value.get("channels") != {"source": [0, 1, 2], "target": None}
                       for value in (row, state, checkpoint))
                or any(any(value.get(key) != expected for key, expected in {
                    "seed": item["seed"], "mask_seed": TRAINING["mask_seed"],
                    "source_draw_seed": TRAINING["source_draw_seed"],
                    "target_draw_seed": TRAINING["target_draw_seed"],
                }.items()) for value in (row, state, checkpoint))
                or any(value.get("code_sha256") != expected_code for value in (row, state, checkpoint))
                or checkpoint.get("loader_role") != "source_only"
                or checkpoint.get("target_train_dataset_constructed") is not False
                or checkpoint.get("target_waveform_samples_iterated") is not False
                or checkpoint.get("target_train_fault_labels") != "not_loaded"
                or checkpoint.get("target_dev_evaluated") is not False
                or checkpoint.get("target_final_evaluated") is not False):
            raise RuntimeError("anchor reconstructed selection or payload differs")
        receipt["anchors"].append({"id": row["anchor"]["id"], "epochs": 60,
            "selected_epoch": selected["epoch"], "epoch_metrics_sha256": sha256(run / "epoch_metrics.jsonl"),
            "checkpoint_sha256": sha256(checkpoint_path)})
    for row in results["arms"]:
        arm = row["arm"]
        run = resolved_run(output, row, expected_run_id(arm, anchor=False))
        checkpoint_path = run / "best_model.pth"
        records = read_jsonl(run / "epoch_metrics.jsonl")
        if len(records) != 20 or [item.get("branch_epoch") for item in records] != list(range(20)):
            raise RuntimeError("branch must contain exactly epochs 0..19")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        assert_checkpoint_finite(checkpoint)
        state = read_json(run / "run_state.json")
        assert_recursive_finite(state, f"{run}/run_state.json")
        expected_alignment = arm["variant"] == "channel_robust_alignment_v2a"
        optimizer_totals = [item.get("optimizer_steps_total") for item in records]
        alignment_totals = [item.get("alignment_steps_total") for item in records]
        if (any(type(value) is not int or value <= 0 for value in optimizer_totals)
                or any(right <= left for left, right in zip(optimizer_totals, optimizer_totals[1:]))
                or any(type(value) is not int or value < 0 for value in alignment_totals)
                or any(right < left for left, right in zip(alignment_totals, alignment_totals[1:]))):
            raise RuntimeError("branch optimizer/alignment step totals are invalid")
        checkpoint_step_values = [checkpoint.get(key) for key in (
            "adapt_optimizer_steps", "alignment_steps", "applied_alignment_steps", "suppressed_alignment_steps"
        )]
        if any(type(value) is not int or value < 0 for value in checkpoint_step_values):
            raise RuntimeError("branch checkpoint step totals are invalid")
        if (checkpoint.get("selected_epoch") != 19 or checkpoint.get("adaptation_endpoint") is not True
                or any(value.get("run_id") != expected_run_id(arm, anchor=False)
                       for value in (row, state, checkpoint))
                or state.get("checkpoint_sha256") != row.get("checkpoint_sha256")
                or checkpoint.get("adaptation_epochs") != 20 or checkpoint.get("variant") != arm["variant"]
                or checkpoint.get("source_val") != records[-1].get("source_val")
                or state.get("source_val") != records[-1].get("source_val")
                or row.get("source_val") != records[-1].get("source_val")
                or checkpoint.get("task_id") != arm["task"] or checkpoint.get("channels", {}).get("target") != arm["target"]
                or checkpoint.get("anchor_checkpoint_sha256") != anchor_by_task[arm["task"]]
                or row.get("anchor_checkpoint_sha256") != anchor_by_task[arm["task"]]
                or checkpoint.get("branch_start_identity", {}).get("anchor_checkpoint_sha256")
                   != anchor_by_task[arm["task"]]
                or checkpoint.get("branch_start_identity") != row["branch_start_identity"]
                or checkpoint.get("branch_source_indices_sha256") != row["branch_source_indices_sha256"]
                or checkpoint.get("branch_source_masks_sha256") != row["branch_source_masks_sha256"]
                or checkpoint.get("target_train_fault_labels") != "required_minus_one"
                or checkpoint.get("target_dev_evaluated") is not False
                or checkpoint.get("target_final_evaluated") is not False
                or state.get("completed_adaptation_epochs") != 20
                or state.get("selected_epoch") != 19
                or any(value.get("task_id") != arm["task"] for value in (row, state, checkpoint))
                or any(value.get("dataset") != arm["dataset"] for value in (row, state, checkpoint))
                or any(value.get("channels") != {"source": [0, 1, 2], "target": arm["target"]}
                       for value in (row, state, checkpoint))
                or any(any(value.get(key) != expected for key, expected in {
                    "seed": arm["seed"], "mask_seed": TRAINING["mask_seed"],
                    "source_draw_seed": TRAINING["source_draw_seed"],
                    "target_draw_seed": TRAINING["target_draw_seed"],
                }.items()) for value in (row, state, checkpoint))
                or any(value.get("code_sha256") != expected_code for value in (row, state, checkpoint))
                or any(item.get("alignment_weight") != (0.02 if expected_alignment else 0.0) for item in records)
                or bool(checkpoint.get("alignment_steps", 0) > 0) != expected_alignment
                or checkpoint.get("adapt_optimizer_steps") != optimizer_totals[-1]
                or state.get("optimizer_steps") != optimizer_totals[-1]
                or row.get("optimizer_steps") != optimizer_totals[-1]
                or checkpoint.get("alignment_steps") != alignment_totals[-1]
                or state.get("alignment_steps") != alignment_totals[-1]
                or row.get("alignment_steps") != alignment_totals[-1]
                or checkpoint.get("applied_alignment_steps") + checkpoint.get("suppressed_alignment_steps")
                   != checkpoint.get("alignment_steps")):
            raise RuntimeError("branch fixed endpoint or payload differs")
        receipt["arms"].append({"id": arm["id"], "epochs": 20, "endpoint": 19,
            "epoch_metrics_sha256": sha256(run / "epoch_metrics.jsonl"),
            "checkpoint_sha256": sha256(checkpoint_path)})
    for pair_end in range(1, len(results["arms"]), 2):
        _verify_completed_pair(results["arms"], pair_end)
    receipt["fit_results_sha256"] = sha256(output / "fit_results.json")
    receipt["plan_sha256"] = sha256(output / "plan.json")
    return receipt


def anchor_result_row(item: dict, run: Path, value: dict) -> dict:
    return {"anchor": item, "run": str(run), "run_id": value["run_id"],
            "checkpoint": str(run / "best_model.pth"),
            "checkpoint_sha256": value["checkpoint_sha256"], "selected_epoch": value["selected_epoch"],
            "task_id": value["task_id"], "dataset": value["dataset"], "channels": value["channels"],
            "seed": value["seed"], "mask_seed": value["mask_seed"],
            "source_draw_seed": value["source_draw_seed"], "target_draw_seed": value["target_draw_seed"],
            "code_sha256": value["code_sha256"],
            "source_val": value["source_val"], "optimizer_steps": value["optimizer_steps"],
            "anchor_replay_identity": value["anchor_replay_identity"],
            "kernel_contract": value["kernel_contract"], "training_seconds": value["training_seconds"]}


def arm_result_row(arm: dict, run: Path, value: dict) -> dict:
    return {"arm": arm, "run": str(run), "run_id": value["run_id"],
            "checkpoint": str(run / "best_model.pth"),
            "task_id": value["task_id"], "dataset": value["dataset"], "channels": value["channels"],
            "seed": value["seed"], "mask_seed": value["mask_seed"],
            "source_draw_seed": value["source_draw_seed"], "target_draw_seed": value["target_draw_seed"],
            "code_sha256": value["code_sha256"],
            **{key: value[key] for key in (
                "checkpoint_sha256", "selected_epoch", "source_val", "optimizer_steps",
                "alignment_steps", "applied_alignment_steps", "suppressed_alignment_steps",
                "anchor_checkpoint_sha256", "branch_start_identity",
                "branch_source_indices_sha256", "branch_source_masks_sha256", "training_seconds",
            )}}


def recover_completed_unregistered(
    output: Path, plan: dict, results: dict, campaign_state: dict
) -> bool:
    """Commit a completed orphan run only after the same full predev audit passes."""
    if len(results["anchors"]) < 4:
        item, kind = plan["anchors"][len(results["anchors"])], "anchor"
        run = output / "runs" / expected_run_id(item, anchor=True)
        make_row = lambda state: anchor_result_row(item, run, state)
    elif len(results["arms"]) < 32:
        item, kind = plan["arms"][len(results["arms"])], "arm"
        run = output / "runs" / expected_run_id(item, anchor=False)
        make_row = lambda state: arm_result_row(item, run, state)
    else:
        return False
    if campaign_state.get("status") == "running":
        if campaign_state.get("active") != item or campaign_state.get("active_kind") != kind:
            raise RuntimeError("running fit_state does not name the exact next orphan; refusing recovery")
    if not run.exists():
        if campaign_state.get("status") == "running":
            raise RuntimeError("running fit_state has no exact orphan directory; refusing silent rerun")
        return False
    state_path = run / "run_state.json"
    if not state_path.exists():
        raise RuntimeError("unregistered run directory exists without a durable run_state; refusing rerun")
    state = read_json(state_path)
    if state.get("status") != "completed":
        raise RuntimeError("unregistered run is not completed; refusing delete or silent rerun")
    row = make_row(state)
    proposed = {"anchors": list(results["anchors"]), "arms": list(results["arms"])}
    proposed["anchors" if kind == "anchor" else "arms"].append(row)
    audit_fit_artifacts(output, plan, proposed)
    results.clear(); results.update(proposed)
    write_json(output / "fit_results.json", results)
    completed = len(results["anchors"]) == 4 and len(results["arms"]) == 32
    write_json(output / "fit_state.json", {"status": "completed" if completed else "prepared",
               "completed_anchors": len(results["anchors"]), "total_anchors": 4,
               "completed_arms": len(results["arms"]), "total_arms": 32, "active": None,
               "last_commit_recovered": item["id"]})
    print(json.dumps({"phase": "fit", "recovered_commit": item["id"]}))
    return True


def fit_one(output: Path) -> int:
    plan = frozen(output)
    results = verify_fit_results(output, plan)
    state = read_json(output / "fit_state.json")
    if state["status"] == "completed":
        return 0
    if state["status"] not in ("prepared", "paused_resource", "running"):
        raise RuntimeError("fit phase is not at a resumable boundary")
    if recover_completed_unregistered(output, plan, results, state):
        return 0
    if state["status"] == "running":
        raise RuntimeError("running orphan did not pass completed-artifact recovery; refusing rerun")
    resources = resource_snapshot()
    if (not resources["cuda_available"] or resources["wsl_mem_available_mib"] < 2048
            or resources.get("gpu_free_mib", 0) < 1024 or resources["load_1m_per_cpu"] > 0.8):
        write_json(output / "fit_state.json", {**state, "status": "paused_resource", "resource": resources})
        return 3
    import train_channel_robust_alignment_v2 as trainer
    from public_loader import load_public_task
    if len(results["anchors"]) < len(plan["anchors"]):
        item = plan["anchors"][len(results["anchors"])]
        write_json(output / "fit_state.json", {**state, "status": "running", "active": item,
                   "active_kind": "anchor", "resource": resources})
        args = trainer.parse_args(anchor_argv(output, plan, item))
        value = trainer.run_phase(args, load_public_task)
        run = output / "runs" / trainer.run_id(args)
        row = anchor_result_row(item, run, value)
        results["anchors"].append(row)
    else:
        arm = plan["arms"][len(results["arms"])]
        anchor = next(row for row in results["anchors"] if row["anchor"]["task"] == arm["task"])
        write_json(output / "fit_state.json", {**state, "status": "running", "active": arm,
                   "active_kind": "branch", "resource": resources})
        args = trainer.parse_args(arm_argv(output, plan, arm, anchor))
        value = trainer.run_phase(args, load_public_task)
        run = output / "runs" / trainer.run_id(args)
        row = arm_result_row(arm, run, value)
        results["arms"].append(row)
        if len(results["arms"]) % 2 == 0:
            _verify_completed_pair(results["arms"], len(results["arms"]) - 1)
    write_json(output / "fit_results.json", results)
    completed = len(results["anchors"]) == 4 and len(results["arms"]) == 32
    write_json(output / "fit_state.json", {"status": "completed" if completed else "prepared",
               "completed_anchors": len(results["anchors"]), "total_anchors": 4,
               "completed_arms": len(results["arms"]), "total_arms": 32, "active": None})
    print(json.dumps({"phase": "fit", "anchors": len(results["anchors"]), "arms": len(results["arms"])}))
    return 0


def seal(output: Path) -> None:
    plan = frozen(output)
    results = verify_fit_results(output, plan, all_required=True)
    if (output / "predev_seal.json").exists():
        raise FileExistsError("pre-development seal already exists")
    fit_audit = audit_fit_artifacts(output, plan, results)
    write_json(output / "predev_fit_audit.json", fit_audit)
    value = {
        "created_unix": time.time(), "plan_sha256": sha256(output / "plan.json"),
        "anchor_checkpoints": [{"id": row["anchor"]["id"], "sha256": row["checkpoint_sha256"]}
                               for row in results["anchors"]],
        "branch_checkpoints": [{"id": row["arm"]["id"], "sha256": row["checkpoint_sha256"]}
                               for row in results["arms"]],
        "fit_results_sha256": sha256(output / "fit_results.json"),
        "predev_fit_audit_sha256": sha256(output / "predev_fit_audit.json"),
        "strong_v4_reference_sha256": json_sha(plan["strong_v4_reference"]),
        "target_dev_predictions_or_labels_read": False, "target_final_evaluated": False,
    }
    write_json(output / "predev_seal.json", value)
    write_json(output / "dev_state.json", {"status": "prepared", "completed_arms": 0,
               "total_arms": 32, "active_arm": None})
    write_json(output / "pipeline_state.json", {"status": "sealed_before_dev",
               "target_dev_evaluated": False, "target_final_evaluated": False})


def verify_seal(output: Path, results: dict, plan: dict) -> dict:
    value = read_json(output / "predev_seal.json")
    if (value.get("plan_sha256") != sha256(output / "plan.json")
            or value.get("fit_results_sha256") != sha256(output / "fit_results.json")
            or value.get("predev_fit_audit_sha256") != sha256(output / "predev_fit_audit.json")
            or read_json(output / "predev_fit_audit.json").get("status") != "PASS"
            or value.get("strong_v4_reference_sha256") != json_sha(plan["strong_v4_reference"])
            or value.get("target_dev_predictions_or_labels_read") is not False
            or len(value.get("anchor_checkpoints", [])) != 4
            or len(value.get("branch_checkpoints", [])) != 32):
        raise RuntimeError("pre-development seal differs")
    return value


DEV_METRIC_KEYS = (
    "checkpoint_sha256", "accuracy", "macro_f1", "per_class_recall",
    "confusion_matrix", "support_per_class",
)


def verify_dev_results(output: Path, plan: dict, fits: dict, *, all_required=False) -> list[dict]:
    """Rebuild the score registry from immutable per-arm result files."""
    state, registry = read_json(output / "dev_state.json"), read_json(output / "dev_results.json")
    if not isinstance(registry, list) or len(registry) != state.get("completed_arms"):
        raise RuntimeError("development state/result count differs")
    if len(registry) > len(plan["arms"]):
        raise RuntimeError("development registry exceeds frozen matrix")
    if all_required and (state.get("status") != "completed" or len(registry) != 32):
        raise RuntimeError("all 32 development results are required")
    rebuilt = []
    for index, (row, arm) in enumerate(zip(registry, plan["arms"])):
        result_path = output / "dev" / f"{arm['id']}.json"
        digest = sha256(result_path)
        raw = read_json(result_path)
        fit = fits["arms"][index]
        if (raw.get("checkpoint_sha256") != fit["checkpoint_sha256"]
                or raw.get("checkpoint_identity", {}).get("variant") != arm["variant"]
                or raw.get("checkpoint_identity", {}).get("task_id") != arm["task"]
                or raw.get("checkpoint_identity", {}).get("channels", {}).get("target") != arm["target"]
                or raw.get("target_final_evaluated") is not False):
            raise RuntimeError("development result checkpoint/arm identity differs")
        rebuilt_row = {"arm": arm, "result_sha256": digest,
                       **{key: raw[key] for key in DEV_METRIC_KEYS}}
        if row != rebuilt_row:
            raise RuntimeError("development registry row differs from its hashed result file")
        rebuilt.append(rebuilt_row)
    return rebuilt


def dev_one(output: Path) -> int:
    plan = frozen(output)
    fits = verify_fit_results(output, plan, all_required=True)
    verify_seal(output, fits, plan)
    state = read_json(output / "dev_state.json")
    prior = verify_dev_results(output, plan, fits)
    if state["status"] == "completed":
        return 0
    if state["status"] != "prepared" or len(prior) != state["completed_arms"] or len(prior) >= 32:
        raise RuntimeError("development phase is not at a resumable boundary")
    import evaluate_channel_robust_alignment_v2_dev as evaluator
    arm, fit = plan["arms"][len(prior)], fits["arms"][len(prior)]
    write_json(output / "dev_state.json", {**state, "status": "running", "active_arm": arm})
    public = runtime_path(plan["data"][arm["task"]]["root"])
    result_path = output / "dev" / f"{arm['id']}.json"
    args = evaluator.parse_args([
        "--public-root", str(public), "--private-root", str(public.parent / f"{public.name}_private_audit"),
        "--task-id", arm["task"], "--target-slots", ",".join(map(str, arm["target"])),
        "--checkpoint", fit["checkpoint"], "--checkpoint-sha256", fit["checkpoint_sha256"],
        "--output-json", str(result_path), "--cpu-threads", "1", "--batch-size", "64", "--execute",
    ])
    raw = evaluator.evaluate(args)
    prior.append({"arm": arm, "result_sha256": sha256(result_path), **{key: raw[key] for key in (
        "checkpoint_sha256", "accuracy", "macro_f1", "per_class_recall", "confusion_matrix",
        "support_per_class")}})
    write_json(output / "dev_results.json", prior)
    write_json(output / "dev_state.json", {"status": "completed" if len(prior) == 32 else "prepared",
               "completed_arms": len(prior), "total_arms": 32, "active_arm": None})
    print(json.dumps({"phase": "dev", "completed": len(prior), "total": 32}))
    return 0


def _pairs(rows: list[dict], control: str) -> list[dict]:
    by_key = {}
    for row in rows:
        arm = row["arm"]
        key = (arm["task"], tuple(arm["target"]), arm["seed"])
        by_key.setdefault(key, {})[arm["variant"]] = row
    result = []
    for key, values in sorted(by_key.items()):
        candidate, baseline = values["channel_robust_alignment_v2a"], values[control]
        result.append({
            "task": key[0], "target": list(key[1]), "seed": key[2],
            "family": candidate["arm"]["family"],
            "candidate_accuracy": candidate["accuracy"], "control_accuracy": baseline["accuracy"],
            "accuracy_delta": candidate["accuracy"] - baseline["accuracy"],
            "candidate_macro_f1": candidate["macro_f1"], "control_macro_f1": baseline["macro_f1"],
            "macro_f1_delta": candidate["macro_f1"] - baseline["macro_f1"],
            "jointly_improved": candidate["accuracy"] > baseline["accuracy"]
                                and candidate["macro_f1"] > baseline["macro_f1"],
            "new_zero_recall": any(new == 0 and old > 0 for new, old in zip(
                candidate["per_class_recall"], baseline["per_class_recall"])),
        })
    if len(result) != 16:
        raise RuntimeError("expected 16 paired comparisons")
    return result


def _formal_gate(pairs: list[dict]) -> dict:
    import statistics
    joint = sum(row["jointly_improved"] for row in pairs)
    mean_acc = statistics.fmean(row["accuracy_delta"] for row in pairs)
    mean_f1 = statistics.fmean(row["macro_f1_delta"] for row in pairs)
    double = statistics.fmean(row["macro_f1_delta"] for row in pairs if row["family"] == "cross_sensor_and_condition")
    slot_positive = {
        task: sum(row["jointly_improved"] for row in pairs if row["task"] == task and len(row["target"]) == 1)
        for task in ("WP-S0", "WP-D1", "PG-S1", "PG-D1")
    }
    single = [row for row in pairs if len(row["target"]) == 1]
    weakest = min(single, key=lambda row: row["control_macro_f1"])
    checks = {
        "joint_10_of_16": joint >= 10, "mean_accuracy_plus_1pp": mean_acc >= 0.01,
        "mean_macro_f1_plus_2pp": mean_f1 >= 0.02, "double_cross_macro_f1_plus_2pp": double >= 0.02,
        "two_positive_slots_each_task": all(value >= 2 for value in slot_positive.values()),
        "weakest_3to1_improves": weakest["macro_f1_delta"] > 0,
        "no_new_zero_recall": not any(row["new_zero_recall"] for row in pairs),
    }
    return {"passed": all(checks.values()), "checks": checks, "jointly_improved_conditions": joint,
            "mean_accuracy_delta": mean_acc, "mean_macro_f1_delta": mean_f1,
            "double_cross_macro_f1_delta": double, "positive_slots_by_task": slot_positive,
            "weakest_3to1": weakest}


def _load_strong_v4_dev(plan: dict) -> list[dict]:
    """Allowed only from report(), after the new campaign seal and scoring."""
    reference = plan["strong_v4_reference"]
    project_root = runtime_path(reference["project_root"]).resolve(strict=True)
    v4 = runtime_path(reference["campaign_root"]).resolve(strict=True)
    try:
        relative = v4.relative_to(project_root)
    except ValueError as error:
        raise RuntimeError("strong v4 root escapes the frozen project root") from error
    if relative.as_posix() != reference["campaign_relative"]:
        raise RuntimeError("strong v4 root relative identity differs")
    if (sha256(v4 / "plan.json") != reference["plan_sha256"]
            or sha256(v4 / "fit_results.json") != reference["fit_results_sha256"]
            or sha256(v4 / "predev_seal.json") != reference["predev_seal_sha256"]
            or sha256(v4 / "dev_results.json") != reference["dev_results_sha256"]
            or sha256(v4 / "summary.json") != reference["summary_sha256"]
            or sha256(v4 / "REPORT.md") != reference["report_sha256"]
            or sha256(runtime_path(reference["completed_audit_receipt"]["path"]))
               != reference["completed_audit_receipt"]["sha256"]
            or reference["completed_audit_receipt"].get("schema")
               != "channel_robust_completed_campaign_audit_v1"
            or reference["completed_audit_receipt"].get("status") != "PASS"):
        raise RuntimeError("strong v4 reference changed before report")
    rows = read_json(v4 / "dev_results.json")
    controls = [row for row in rows if row["arm"]["variant"] == "matched_control" and row["arm"]["seed"] == SEED]
    expected = {item["arm"]["id"]: item["checkpoint_sha256"] for item in reference["controls"]}
    if len(controls) != 16 or any(expected.get(row["arm"]["id"]) != row["checkpoint_sha256"] for row in controls):
        raise RuntimeError("v4 development controls differ from frozen checkpoint references")
    return controls


def report(output: Path) -> None:
    plan = frozen(output)
    fits = verify_fit_results(output, plan, all_required=True)
    verify_seal(output, fits, plan)
    rows = verify_dev_results(output, plan, fits, all_required=True)
    mechanism_pairs = _pairs(rows, "common_anchor_control")
    causal_gate = _formal_gate(mechanism_pairs)
    strong = _load_strong_v4_dev(plan)
    candidates = [row for row in rows if row["arm"]["variant"] == "channel_robust_alignment_v2a"]
    formal_rows = candidates + strong
    formal_pairs = _pairs(formal_rows, "matched_control")
    summary = {
        "method": "v2A-U-fixedK", "development_arms": 32,
        "causal_comparison": {"control": "common_anchor_control", "pairs": mechanism_pairs,
                              "expansion_gate": causal_gate},
        "formal_comparison": {"control": "frozen_v4_matched_control", "pairs": formal_pairs,
                              "historical_threshold_diagnostic": _formal_gate(formal_pairs),
                              "decides_expansion": False},
        "target_dev_history_boundary": plan["target_dev_history_boundary"],
        "target_final_evaluated": False,
    }
    write_json(output / "summary.json", summary)
    gate = causal_gate
    lines = ["# CRA-v2A-U-fixedK development report", "",
             "All four anchors and 32 second-stage branches were SHA-sealed before development scoring.",
             "The paired common-anchor control provides the causal expansion gate. The frozen v4 strong control is a separate historical comparison.",
             "Target final was not evaluated.", "", "## Paired causal expansion gate", "",
             f"Passed: **{gate['passed']}**.",
             f"Joint improvements: {gate['jointly_improved_conditions']}/16.",
             f"Mean Accuracy delta: {100*gate['mean_accuracy_delta']:+.2f} pp.",
             f"Mean Macro-F1 delta: {100*gate['mean_macro_f1_delta']:+.2f} pp.",
             f"Double-cross Macro-F1 delta: {100*gate['double_cross_macro_f1_delta']:+.2f} pp."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output / "pipeline_state.json", {"status": "completed", "fit_anchors": 4,
               "fit_arms": 32, "dev_arms": 32, "screen_gate_passed": gate["passed"],
               "target_dev_evaluated": True, "target_final_evaluated": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("prepare", "fit-one", "seal", "dev-one", "report", "status"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if args.mode == "prepare":
        prepare(output); return 0
    if args.mode == "status":
        print(json.dumps({name: read_json(output / f"{name}_state.json") for name in ("fit", "dev")}, ensure_ascii=False)); return 0
    if args.mode == "fit-one":
        return fit_one(output)
    if args.mode == "seal":
        seal(output)
    elif args.mode == "dev-one":
        return dev_one(output)
    else:
        report(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
