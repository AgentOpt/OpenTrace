from typing import Union
import importlib

from opto import trace
from opto.train.algorithms import Trainer
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
    # TODO update the acceptable type of optimizer, trainer, guide, logger to be union of base class and str
    optimizer: Union[Optimizer, str] = "OptoPrimeV2",
    trainer: Union[Trainer, str] = 'BasicSearchAlgorithm',
    guide: Union[Guide, str] = 'LLMGuide',
    logger: Union[BaseLogger, str] = 'ConsoleLogger',
    # extra configs
    optimizer_kwargs: Union[dict, None] = None,
    trainer_kwargs: Union[dict, None] = None  # for train function
    # TODO other kwargs
) -> None:

    """ A high-level helper function to train the model using trainer. """
    optimizer_kwargs = optimizer_kwargs or {}  # this can be used to pass extra optimizer configs, like llm object explictly
    trainer_kwargs = trainer_kwargs or {}

    #  TODO check eligible optimizer, trainer
    dataset_check(train_dataset)

    # TODO remove duplicate codes

    # Load optimizer from opto.optimizers
    parameters = agent.parameters()
    assert len(parameters) >0, "Agent must have parameters."
    if type(optimizer) is str:
        # check if optimizer is a valid class
        optimizers_module = importlib.import_module("opto.optimizers")
        optimizer_class = getattr(optimizers_module, optimizer)
        optimizer = optimizer_class(
            model.parameters(),
            **optimizer_kwargs
    )
    # else if optimizer is an instance
    elif issubclass(optimizer, Optimizer):
        optimizer = optimizer(
            model.parameters(),
            **optimizer_kwargs
        )
    else:
        raise ValueError(f"Invalid optimizer type: {type(optimizer)}")

    # Load guide from opto.trainer.guide
    if type(guide) is str:
        # check if guide is a valid class
        guides_module = importlib.import_module("opto.trainer.guide")
        guide_class = getattr(guides_module, guide)
        guide = guide_class(
            **guide_kwargs
        )
    # else if guide is an instance
    elif issubclass(guide, Guide):
        guide = guide(
            **guide_kwargs
        )
    else:
        raise ValueError(f"Invalid guide type: {type(guide)}")

    # Load logger from opto.trainer.loggers
    if type(logger) is str:
        # check if logger is a valid class
        loggers_module = importlib.import_module("opto.trainer.loggers")
        logger_class = getattr(loggers_module, logger)
        logger = logger_class(**logger_kwargs)
    # else if logger is an instance
    elif issubclass(logger, BaseLogger):
        logger = logger(
            **logger_kwargs
        )
    else:
        raise ValueError(f"Invalid logger type: {type(logger)}")


    # Load trainer from opto.trainer.algorithms
    if type(trainer) is str:
        # check if trainer is a valid class
        trainers_module = importlib.import_module("opto.trainer.algorithms")
        trainer_class = getattr(trainers_module, trainer)
        trainer = trainer_class(
           agent,
           optimizer,
           logger
        )
    # else if trainer is an instance
    elif issubclass(trainer, Trainer):
        trainer = trainer(
            agent,
            optimizer,
            logger
        )
    else:
        raise ValueError(f"Invalid trainer type: {type(trainer)}")


    # TODO start training
    trainer.train(**trainer_kwargs)