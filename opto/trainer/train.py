from typing import Union
import importlib

from opto import trace
from opto.trainer.algorithms import Trainer
from opto.trainer.guide import Guide
from opto.trainer.loggers import BaseLogger
from opto.optimizers.optimizer import Optimzier


def dataset_check(dataset):
    assert isinstance(dataset, dict), "Dataset must be a dictionary"
    assert 'inputs' in dataset and 'infos' in dataset, "Dataset must contain 'inputs' and 'infos' keys"
    assert len(dataset['inputs'])==len(dataset['infos']), "Inputs and infos must have the same length"


def train(
    model: trace.Module,
    guide: Guide,
    train_dataset: dict,
    # class of optimizer
    trainer: Union[Trainer, str] = 'BasicSearchAlgorithm',
    optimizer: Union[Optimizer, str] = "OptoPrimeV2",
    guide: Union[Guide, str] = 'LLMGuide',
    logger: Union[BaseLogger, str] = 'ConsoleLogger',
    # extra configs
    optimizer_kwargs: Union[dict, None] = None,
    guide_kwargs: Union[dict, None] = None,
    logger_kwargs: Union[dict, None] = None,
    # The rest is treated as trainer config
    **trainer_kwargs,
) -> None:
    """ A high-level helper function to train the model using trainer.

    A trainer algorithm applies an optimizer to train a model under a guide on a train_dataset.

    """
    optimizer_kwargs = optimizer_kwargs or {}  # this can be used to pass extra optimizer configs, like llm object explictly
    guide_kwargs = guide_kwargs or {}
    logger_kwargs = logger_kwargs or {}

    #  TODO check eligible optimizer, trainer
    dataset_check(train_dataset)

    # TODO remove duplicate codes

    # Check agent parameters is non-empty
    parameters = agent.parameters()
    assert len(parameters) >0, "Agent must have parameters."


    # Load optimizer from opto.optimizers
    if type(optimizer) is str:
        # check if optimizer is a valid class
        optimizers_module = importlib.import_module("opto.optimizers")
        optimizer_class = getattr(optimizers_module, optimizer)
    # else if optimizer is an instance
    elif issubclass(optimizer, Optimizer):
        optimizer_class = optimizer
    else:
        raise ValueError(f"Invalid optimizer type: {type(optimizer)}")
    optimizer = optimizer_class(
            model.parameters(),
            **optimizer_kwargs
    )

    # Load guide from opto.trainer.guide
    if type(guide) is str:
        # check if guide is a valid class
        guides_module = importlib.import_module("opto.trainer.guide")
        guide_class = getattr(guides_module, guide)
    # else if guide is an instance
    elif issubclass(guide, Guide):
        guide_class = guide
    else:
        raise ValueError(f"Invalid guide type: {type(guide)}")
    guide = guide_class(
        **guide_kwargs
    )

    # Load logger from opto.trainer.loggers
    if type(logger) is str:
        # check if logger is a valid class
        loggers_module = importlib.import_module("opto.trainer.loggers")
        logger_class = getattr(loggers_module, logger)
    # else if logger is an instance
    elif issubclass(logger, BaseLogger):
        logger_class = logger
    else:
        raise ValueError(f"Invalid logger type: {type(logger)}")
    logger = logger_class(**logger_kwargs)


    # Load trainer from opto.trainer.algorithms
    if type(trainer) is str:
        # check if trainer is a valid class
        trainers_module = importlib.import_module("opto.trainer.algorithms")
        trainer_class = getattr(trainers_module, trainer)
    # else if trainer is an instance
    elif issubclass(trainer, Trainer):
        trainer_class = trainer
    else:
        raise ValueError(f"Invalid trainer type: {type(trainer)}")
    trainer = trainer_class(
        agent,
        optimizer,
        logger
    )

    return trainer.train(
        guide=guide,
        train_dataset=train_dataset,
        **trainer_kwargs)
