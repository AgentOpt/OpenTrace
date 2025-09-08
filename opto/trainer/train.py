from typing import Union
import importlib

from opto import trace
from opto.trainer.algorithms import Trainer
from opto.trainer.guide import Guide
from opto.trainer.loggers import BaseLogger
from opto.optimizers.optimizer import Optimizer
from opto.trace.nodes import ParameterNode


def dataset_check(dataset):
    assert isinstance(dataset, dict), "Dataset must be a dictionary"
    assert 'inputs' in dataset and 'infos' in dataset, "Dataset must contain 'inputs' and 'infos' keys"
    assert len(dataset['inputs'])==len(dataset['infos']), "Inputs and infos must have the same length"


def train(
    *,
    model: Union[trace.Module, ParameterNode],
    train_dataset: dict,
    # class of optimizer
    algorithm: Union[Trainer, str] = 'MinibatchAlgorithm',
    optimizer: Union[Optimizer, str] = None,
    guide: Union[Guide, str] = 'LLMJudge',
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

    if optimizer is None:
        optimizer = "OPROv2" if isinstance(model, ParameterNode) else "OptoPrimeV2"

    # Convert ParameterNode to Module
    if isinstance(model, ParameterNode):
        assert model.trainable, "The parameter must be trainable."
        @trace.model
        class SingleNodeModel:
            def __init__(self, param):
                self.param = param  # ParameterNode
            def forward(self, x):
                return self.param
        model = SingleNodeModel(model)

    # Check model parameters is non-empty
    parameters = model.parameters()
    assert len(parameters) >0, "Model must have non-empty parameters."

    if isinstance(optimizer_kwargs, list):  # support multiple optimizers
        assert all(isinstance(d, dict) for d in optimizer_kwargs), "optimizer_kwargs must be a list of dictionaries."
        optimizer = [load_optimizer(optimizer, model, **d) for d in optimizer_kwargs ]
        assert all(isinstance(o, Optimizer) for o in optimizer)
    else:
        optimizer = load_optimizer(optimizer, model, **optimizer_kwargs)
        assert isinstance(optimizer, Optimizer)

    guide = load_guide(guide, **guide_kwargs)
    logger = load_logger(logger, **logger_kwargs)
    trainer_class = load_trainer_class(algorithm)

    assert isinstance(guide, Guide)
    assert isinstance(logger, BaseLogger)
    assert issubclass(trainer_class, Trainer)

    algo = trainer_class(
        model,
        optimizer,
        logger=logger
    )

    return algo.train(
        guide=guide,
        train_dataset=train_dataset,
        **trainer_kwargs)


def load_optimizer(optimizer: Union[Optimizer, str], model: trace.Module, **kwargs) -> Optimizer:
    if isinstance(optimizer, Optimizer):
        return optimizer
    elif isinstance(optimizer, str):
        optimizers_module = importlib.import_module("opto.optimizers")
        optimizer_class = getattr(optimizers_module, optimizer)
        return optimizer_class(model.parameters(), **kwargs)
    elif issubclass(optimizer, Optimizer):
        return optimizer(model.parameters(), **kwargs)
    else:
        raise ValueError(f"Invalid optimizer type: {type(optimizer)}")


def load_guide(guide: Union[Guide, str], **kwargs) -> Guide:
    if isinstance(guide, Guide):
        return guide
    elif isinstance(guide, str):
        guides_module = importlib.import_module("opto.trainer.guide")
        guide_class = getattr(guides_module, guide)
        return guide_class(**kwargs)
    elif issubclass(guide, Guide):
        return guide(**kwargs)
    else:
        raise ValueError(f"Invalid guide type: {type(guide)}")

def load_logger(logger: Union[BaseLogger, str], **kwargs) -> BaseLogger:
    if isinstance(logger, BaseLogger):
        return logger
    elif isinstance(logger, str):
        loggers_module = importlib.import_module("opto.trainer.loggers")
        logger_class = getattr(loggers_module, logger)
        return logger_class(**kwargs)
    elif issubclass(logger, BaseLogger):
        return logger(**kwargs)
    else:
        raise ValueError(f"Invalid logger type: {type(logger)}")

def load_trainer_class(trainer: Union[Trainer, str]) -> Trainer:
    if isinstance(trainer, str):
        if trainer.lower() == 'PrioritySearch'.lower():
            print('Warning: You are using PrioritySearch trainer, which is an experimental feature. Please report any issues you encounter.')
            trainers_module = importlib.import_module("opto.features.priority_search")
            trainer_class = getattr(trainers_module, trainer)
        else:
            trainers_module = importlib.import_module("opto.trainer.algorithms")
            trainer_class = getattr(trainers_module, trainer)
    elif issubclass(trainer, Trainer):
        trainer_class = trainer
    else:
        raise ValueError(f"Invalid trainer type: {type(trainer)}")

    return trainer_class