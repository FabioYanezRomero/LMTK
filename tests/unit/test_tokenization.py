from unittest.mock import MagicMock, patch
import pytest

pytest.importorskip("datasets")
pytest.importorskip("transformers")

from src.tasks.tokenization.orchestrator import TokenizationOrchestrator


class Box(dict):
    """Simple stand-in for python-box's Box."""

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


CONFIGS = {
    "clm_training": boxify({
        "tokenizer": {
            "tokenizer_name": "gpt2",
            "context_length": 16,
            "task": "clm_training",
        },
        "dataset": {"source": "local"},
        "output": {"path": "/tmp/out"},
    }),
    "mlm_training": boxify({
        "tokenizer": {
            "tokenizer_name": "bert-base-uncased",
            "context_length": 16,
            "task": "mlm_training",
        },
        "dataset": {"source": "local"},
        "output": {"path": "/tmp/out"},
    }),
    "instruction": boxify({
        "tokenizer": {
            "tokenizer_name": "gpt2",
            "context_length": 16,
            "task": "instruction",
        },
        "dataset": {"source": "local"},
        "output": {"path": "/tmp/out"},
    }),
}


@pytest.mark.parametrize(
    "task, tokenizer_path",
    [
        ("clm_training", "src.tasks.tokenization.orchestrator.CausalLMTokenizer"),
        ("mlm_training", "src.tasks.tokenization.orchestrator.MaskedLMTokenizer"),
        ("instruction", "src.tasks.tokenization.orchestrator.InstructionTokenizer"),
    ],
)
def test_tokenize_dataset_uses_expected_tokenizer(task, tokenizer_path):
    config = CONFIGS[task]
    orchestrator = TokenizationOrchestrator(config)
    dataset = MagicMock()
    with patch(tokenizer_path) as MockTok:
        instance = MockTok.return_value
        orchestrator.tokenize_dataset(dataset)
        MockTok.assert_called_once()
        instance.tokenize.assert_called_once_with(dataset)

