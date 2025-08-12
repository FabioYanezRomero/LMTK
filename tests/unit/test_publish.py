from unittest.mock import MagicMock, patch
import pytest

pytest.importorskip("transformers")
pytest.importorskip("torch")

from src.tasks.publish.orchestrator import PublishOrchestrator


from unittest.mock import MagicMock, patch
import pytest

from src.tasks.publish.orchestrator import PublishOrchestrator


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
        "publish": {
            "host": "hf",
            "base_model": "model",
            "repo_id": "user/repo",
            "checkpoint_path": "/tmp/ckpt",
            "format": "fsdp",
        }
    })


def test_validate_config_missing_host():
    cfg = build_config()
    del cfg.publish.host
    with patch("src.tasks.publish.orchestrator.AutoTokenizer.from_pretrained"):
        orchestrator = PublishOrchestrator(cfg)
    with pytest.raises(ValueError):
        orchestrator.validate_config()


def test_execute_runs_handlers():
    cfg = build_config()
    with patch("src.tasks.publish.orchestrator.AutoTokenizer.from_pretrained"):
        orchestrator = PublishOrchestrator(cfg)
    mock_format_class = MagicMock()
    mock_format_instance = mock_format_class.return_value
    mock_upload_class = MagicMock()
    mock_upload_instance = mock_upload_class.return_value
    with patch.dict(
        "src.tasks.publish.orchestrator.FORMAT_HANDLERS",
        {"fsdp": mock_format_class},
    ):
        with patch("src.tasks.publish.orchestrator.UploadHuggingface", mock_upload_class):
            orchestrator.execute()
    mock_format_class.assert_called_once_with("hf", "model", "/tmp/ckpt")
    mock_format_instance.execute.assert_called_once()
    mock_upload_class.assert_called_once_with(
        base_model="model",
        model=mock_format_instance.execute.return_value,
        tokenizer=orchestrator.tokenizer,
        repo_id="user/repo",
    )
    mock_upload_instance.execute.assert_called_once()
