from __future__ import annotations

from opto.features.priority_search.search_template import save_train_config


class _DummyTrainer:
    @save_train_config
    def train(self, *, guide, train_dataset, batch_size=0, validate_dataset=None, test_dataset=None):
        return {
            "guide": guide,
            "train_dataset": train_dataset,
            "batch_size": batch_size,
            "validate_dataset": validate_dataset,
            "test_dataset": test_dataset,
        }


def test_save_train_config_accepts_positional_guide_and_dataset():
    trainer = _DummyTrainer()
    train_dataset = {"inputs": [], "infos": []}

    result = trainer.train("guide", train_dataset, batch_size=1)

    assert result["guide"] == "guide"
    assert result["train_dataset"] is train_dataset
    assert result["batch_size"] == 1
    assert trainer._train_last_kwargs == {
        "guide": "guide",
        "batch_size": 1,
    }
