from unittest.mock import MagicMock, patch
import pytest

pytest.importorskip("transformers")
pytest.importorskip("datasets")
pytest.importorskip("torch")

from src.tasks.training.fabric.trainer.base import FabricTrainerBase


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


CONFIGS = {
    "clm_training": boxify({
        "task": "clm_training",
        "batch_size": 1,
        "num_workers": 0,
        "model_name": "gpt2",
        "output_dir": "/tmp",
    }),
    "mlm_training": boxify({
        "task": "mlm_training",
        "batch_size": 1,
        "num_workers": 0,
        "model_name": "bert-base-uncased",
        "output_dir": "/tmp",
    }),
    "instruction": boxify({
        "task": "instruction",
        "batch_size": 1,
        "num_workers": 0,
        "model_name": "gpt2",
        "output_dir": "/tmp",
    }),
}


class DummyTrainer(FabricTrainerBase):
    def _setup_strategy(self):
        return None


@pytest.mark.parametrize("task, config", CONFIGS.items())
def test_instantiate_model_selects_correct_class(task, config):
    dataset = MagicMock()
    dummy_return = {"datasets": {}, "dataloaders": {}}
    with patch.object(FabricTrainerBase, "_load_fabric_datasets_dataloaders", return_value=dummy_return):
        trainer = DummyTrainer(devices=1, config=config, dataset=dataset)
    mock_model = MagicMock()
    with patch.dict("src.tasks.training.fabric.trainer.base.MODEL_CLASS_MAP", {task: mock_model}):
        trainer._instantiate_model()
        mock_model.assert_called_once()

