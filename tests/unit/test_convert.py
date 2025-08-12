from unittest.mock import MagicMock, patch
import pytest

import pytest
from unittest.mock import MagicMock, patch

pytest.importorskip("transformers")

from src.tasks.convert.orchestrator import ConvertOrchestrator


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
        "convert": {
            "base_model": "model",
            "checkpoint_path": "/tmp/ckpt",
            "initial_format": "fsdp",
            "final_format": "huggingface",
            "output_dir": "/tmp/out",
        }
    })


def test_validate_config_missing_base_model():
    cfg = build_config()
    del cfg.convert.base_model
    orchestrator = ConvertOrchestrator(cfg)
    with pytest.raises(ValueError):
        orchestrator._validate_config()


def test_execute_calls_format_handler():
    cfg = build_config()
    orchestrator = ConvertOrchestrator(cfg)
    mock_handler_class = MagicMock()
    mock_handler_instance = mock_handler_class.return_value
    with patch.dict(
        "src.tasks.convert.orchestrator.FORMAT_HANDLERS",
        {"fsdp_huggingface": mock_handler_class},
    ):
        orchestrator.execute()
        mock_handler_class.assert_called_once_with("model", "/tmp/ckpt")
        mock_handler_instance.execute.assert_called_once_with(output_dir="/tmp/out")

