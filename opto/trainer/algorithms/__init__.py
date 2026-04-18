"""Lazy public facade for trainer algorithms."""

__all__ = [
    "Trainer",
    "Minibatch",
    "MinibatchAlgorithm",
    "BasicSearchAlgorithm",
    "MinibatchCurriculumAccumulationCommonFeedbackAlgorithm",
    "BasicSearchCurriculumAccumulationCommonFeedbackAlgorithm",
    "BeamsearchAlgorithm",
    "BeamsearchHistoryAlgorithm",
    "UCBSearchAlgorithm",
]


def __getattr__(name):
    if name == "Trainer":
        from opto.trainer.algorithms.algorithm import Trainer

        return Trainer
    if name in {
        "Minibatch",
        "MinibatchAlgorithm",
        "BasicSearchAlgorithm",
        "MinibatchCurriculumAccumulationCommonFeedbackAlgorithm",
        "BasicSearchCurriculumAccumulationCommonFeedbackAlgorithm",
    }:
        from opto.trainer.algorithms import basic_algorithms as module

        return getattr(module, name)
    if name in {"BeamsearchAlgorithm", "BeamsearchHistoryAlgorithm"}:
        from opto.trainer.algorithms import beamsearch_algorithm as module

        return getattr(module, name)
    if name == "UCBSearchAlgorithm":
        from opto.trainer.algorithms.UCBsearch import UCBSearchAlgorithm

        return UCBSearchAlgorithm
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
