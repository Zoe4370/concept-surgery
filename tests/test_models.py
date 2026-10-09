import subprocess
import sys
import importlib.util
import json
import types
from pathlib import Path

import pytest

from concept_surgery.models import MODEL_PRESETS, model_slug, resolve_model_id


def test_model_presets_cover_multiple_families_and_scales():
    assert "gpt2" in MODEL_PRESETS
    assert MODEL_PRESETS["qwen-coder-0.5b"].model_id == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert MODEL_PRESETS["qwen-coder-1.5b"].model_id == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    assert MODEL_PRESETS["qwen-coder-3b"].model_id == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert MODEL_PRESETS["qwen-coder-7b"].model_id == "Qwen/Qwen2.5-Coder-7B-Instruct"
    assert MODEL_PRESETS["starcoder2-3b"].model_id == "bigcode/starcoder2-3b"
    assert MODEL_PRESETS["phi-2"].model_id == "microsoft/phi-2"
    assert MODEL_PRESETS["starcoder2-3b"].role.endswith("not instruction-tuned)")


def test_model_alias_resolution_and_safe_paths():
    assert resolve_model_id("qwen-coder-0.5b") == "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    assert resolve_model_id("org/my-model") == "org/my-model"
    assert model_slug("org/my-model") == "org-my-model"


def test_model_slugs_are_nonempty_and_deterministic():
    assert model_slug("gpt2") == model_slug("gpt2")
    try:
        model_slug("///")
    except ValueError:
        pass
    else:
        raise AssertionError("a path containing no usable characters should be rejected")


def test_model_suite_can_list_presets_without_torch():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_model_suite.py"
    result = subprocess.run(
        [sys.executable, str(script), "--list-models"],
        check=True,
        capture_output=True,
        text=True,
    )
    for alias in MODEL_PRESETS:
        assert alias in result.stdout


@pytest.fixture
def suite_runner():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_model_suite.py"
    spec = importlib.util.spec_from_file_location("run_model_suite_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_row_extracts_single_layer_and_legacy_result_fields(suite_runner):
    row = suite_runner._summary_row("tiny", "org/tiny", {
        "chosen": {"layer": 3, "method": "mean_diff"},
        "baseline": {"probe_acc": 0.8, "neutral_nll": 2.0},
        "ablated": {"probe_acc": 0.4, "neutral_nll": 2.1},
        "locality_kl": 0.03,
    })
    assert row == {
        "alias": "tiny", "model_id": "org/tiny", "status": "completed",
        "chosen_layer": 3, "chosen_method": "mean_diff",
        "baseline_probe_accuracy": 0.8, "edited_probe_accuracy": 0.4,
        "baseline_neutral_nll": 2.0, "edited_neutral_nll": 2.1,
        "locality_kl": 0.03,
    }


def test_runner_records_success_and_writes_manifest(monkeypatch, tmp_path, suite_runner):
    calls = []

    def fake_run_pipeline(config):
        calls.append(config)
        return {
            "chosen": {"layer": 2, "method": "leace"},
            "baseline": {"probe_acc": 0.9, "neutral_nll": 1.5},
            "ablated_single": {"probe_acc": 0.2, "neutral_nll": 1.6},
            "locality_kl": 0.01,
        }

    monkeypatch.setitem(sys.modules, "concept_surgery.pipeline", types.SimpleNamespace(run_pipeline=fake_run_pipeline))
    monkeypatch.setattr(sys, "argv", ["run_model_suite.py", "--models", "gpt2", "--out", str(tmp_path)])
    assert suite_runner.main() == 0
    assert len(calls) == 1
    assert calls[0].model_name == "gpt2"
    manifest = json.loads((tmp_path / "suite_results.json").read_text())
    assert manifest["models"][0]["status"] == "completed"
    assert manifest["models"][0]["edited_probe_accuracy"] == 0.2


def test_runner_records_failure_and_continues_by_default(monkeypatch, tmp_path, suite_runner):
    calls = []

    def fake_run_pipeline(config):
        calls.append(config.model_name)
        if config.model_name == "gpt2":
            raise RuntimeError("synthetic test failure")
        return {"chosen": {}, "baseline": {}, "ablated_single": {}}

    monkeypatch.setitem(sys.modules, "concept_surgery.pipeline", types.SimpleNamespace(run_pipeline=fake_run_pipeline))
    monkeypatch.setattr(sys, "argv", ["run_model_suite.py", "--models", "gpt2", "phi-2", "--out", str(tmp_path)])
    assert suite_runner.main() == 1
    assert calls == ["gpt2", "microsoft/phi-2"]
    manifest = json.loads((tmp_path / "suite_results.json").read_text())
    assert [row["status"] for row in manifest["models"]] == ["failed", "completed"]
    assert "synthetic test failure" in manifest["models"][0]["error"]


def test_runner_resume_uses_cached_results_without_loading_model(monkeypatch, tmp_path, suite_runner):
    result_dir = tmp_path / "gpt2"
    result_dir.mkdir()
    (result_dir / "results.json").write_text(json.dumps({
        "chosen": {"layer": 4, "method": "pca"},
        "baseline": {"probe_acc": 0.7, "neutral_nll": 1.2},
        "ablated_single": {"probe_acc": 0.3, "neutral_nll": 1.3},
        "locality_kl": 0.02,
    }))

    def unexpected_load(_config):
        pytest.fail("resume should not rerun an already completed model")

    monkeypatch.setitem(sys.modules, "concept_surgery.pipeline", types.SimpleNamespace(run_pipeline=unexpected_load))
    monkeypatch.setattr(sys, "argv", ["run_model_suite.py", "--models", "gpt2", "--resume", "--out", str(tmp_path)])
    assert suite_runner.main() == 0
    manifest = json.loads((tmp_path / "suite_results.json").read_text())
    assert manifest["models"][0]["chosen_layer"] == 4
