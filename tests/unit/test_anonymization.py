from unittest.mock import patch

from unittest.mock import patch
import pytest

pytest.importorskip("datasets")
pytest.importorskip("psutil")

from src.tasks.anonymization.orchestrator import AnonymizationOrchestrator


class Box(dict):
    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, key, value):
        self[key] = value


def boxify(d):
    if isinstance(d, dict):
        return Box({k: boxify(v) for k, v in d.items()})
    if isinstance(d, list):
        return [boxify(v) for v in d]
    return d


def build_config():
    return boxify({
        "anonymization": {
            "source": {"path": "/tmp/input"},
            "output": {"path": "/tmp/output"},
            "models_source": {"path": "/tmp/models"},
            "models": [{"mid": "1", "mtype": "type"}],
            "regexes": {"path": "/tmp/regexes"},
            "truecaser": {"path": "/tmp/true"},
            "format": "json",
            "method": "replace",
            "labels": ["name"],
            "store_original": True,
            "aggregate_output": False,
            "skip_existing": False,
            "docker_image": "anonymization:latest",
        }
    })


def test_validate_anonymization_config_missing_models():
    cfg = build_config()
    del cfg.anonymization.models
    orchestrator = AnonymizationOrchestrator(cfg)
    try:
        orchestrator._validate_anonymization_config()
    except ValueError as e:
        assert "models must be provided" in str(e)
    else:
        raise AssertionError("ValueError not raised")


def test_execute_runs_docker():
    cfg = build_config()
    orchestrator = AnonymizationOrchestrator(cfg)
    with patch("src.tasks.anonymization.orchestrator.subprocess.run") as mock_run:
        orchestrator._anonymize_data()
        mock_run.assert_called_once()
