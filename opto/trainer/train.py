from typing import Union, Any
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


# TODO finish implementing resume function
# def resume(
#     save_path: str,
#     *,
#     algorithm: Union[Trainer, str] = 'MinibatchAlgorithm',
#     model: trace.Module,
#     train_dataset: dict,
#     validate_dataset = None,
#         test_dataset = None,
#         **kwargs):
#     """ Resume training from a checkpoint.

#     Args:
#         model: the model to be trained
#         train_dataset: the training dataset
#         resume_training: path to the checkpoint
#         validate_dataset: the validation dataset
#         test_dataset: the test dataset
#         **kwargs: additional keyword arguments for the training method. If not provided, the same parameters as the last training call are used.
#     """
#     dataset_check(train_dataset)
#     trainer_class = load_trainer_class(algorithm)
#     assert issubclass(trainer_class, Trainer)
#     assert isinstance(save_path, str), "resume_training must be a path string."
#     assert hasattr(trainer_class, 'resume'), f"{trainer_class} does not support resume."
#     assert hasattr(trainer_class, 'load'), f"{trainer_class} does not support load."
#     algo = trainer_class.load(save_path)  # load the saved state
#     return algo.resume(model=model,
#                         train_dataset=train_dataset,
#                         validate_dataset=validate_dataset,
#                         test_dataset=test_dataset,
#                         **kwargs)


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
) -> Any:
    """High-level training function for Trace models using optimization algorithms.

    Provides a unified interface for training Trace models by combining an optimizer,
    training algorithm, evaluation guide, and logging. Automatically configures
    components based on the model type and provided parameters.

    Parameters
    ----------
    model : Union[trace.Module, ParameterNode]
        The model to train. Can be a Trace Module with multiple parameters
        or a single ParameterNode for direct optimization.
    train_dataset : dict
        Training dataset with required keys:
        - 'inputs': List of input samples
        - 'infos': List of corresponding target/reference information
        Both lists must have the same length.
    algorithm : Union[Trainer, str], default='MinibatchAlgorithm'
        Training algorithm to use. Can be a Trainer instance or string name.
        Common algorithms: 'MinibatchAlgorithm', 'BeamSearchAlgorithm'.
    optimizer : Union[Optimizer, str], optional
        Optimizer for parameter updates. If None, automatically selected:
        - 'OPROv2' for ParameterNode models
        - 'OptoPrimeV2' for Module models
        Can be optimizer instance or string name.
    guide : Union[Guide, str], default='LLMJudge'
        Evaluation guide that provides feedback on model outputs.
        Common guides: 'LLMJudge', 'ExactMatchGuide'.
    logger : Union[BaseLogger, str], default='ConsoleLogger'
        Logger for tracking training progress and metrics.
    optimizer_kwargs : dict, optional
        Additional keyword arguments passed to optimizer constructor.
        Useful for specifying LLM instances, learning rates, etc.
    guide_kwargs : dict, optional
        Additional keyword arguments passed to guide constructor.
    logger_kwargs : dict, optional
        Additional keyword arguments passed to logger constructor.
    **trainer_kwargs
        Additional configuration passed to the training algorithm,
        such as batch size, number of epochs, early stopping criteria.

    Raises
    ------
    AssertionError
        If dataset format is invalid (missing keys, mismatched lengths).

    Notes
    -----
    The training process follows these steps:
    1. **Dataset Validation**: Ensures dataset has correct format and structure
    2. **Component Setup**: Instantiates optimizer, guide, and logger from strings/configs
    3. **Model Preparation**: Converts ParameterNode to Module if needed
    4. **Algorithm Execution**: Runs the specified training algorithm

    Training algorithms coordinate the optimization process:
    - Generate batches from the dataset
    - Apply the model to inputs
    - Use the guide to evaluate outputs and provide feedback
    - Update model parameters through the optimizer
    - Log progress and metrics

    Examples
    --------
    >>> # Train a simple text model
    >>> model = MyTextModel()
    >>> dataset = {
    ...     'inputs': ['What is AI?', 'Explain ML'],
    ...     'infos': ['Artificial Intelligence...', 'Machine Learning...']
    ... }
    >>> train(model=model, train_dataset=dataset, algorithm='MinibatchAlgorithm')

    >>> # Train with custom configuration
    >>> train(
    ...     model=model,
    ...     train_dataset=dataset,
    ...     optimizer='OptoPrimeV2',
    ...     guide='LLMJudge',
    ...     optimizer_kwargs={'max_tokens': 1000},
    ...     batch_size=8,
    ...     num_epochs=10
    ... )

    See Also
    --------
    Trainer : Base class for training algorithms
    Optimizer : Parameter optimization interface
    Guide : Evaluation and feedback interface
    """

    optimizer_kwargs = optimizer_kwargs or {}  # this can be used to pass extra optimizer configs, like llm object explictly
    guide_kwargs = guide_kwargs or {}
    logger_kwargs = logger_kwargs or {}

    #  TODO check eligible optimizer, trainer
    dataset_check(train_dataset)


    trainer_class = load_trainer_class(algorithm)
    assert issubclass(trainer_class, Trainer)

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
    assert isinstance(guide, Guide)

    logger = load_logger(logger, **logger_kwargs)
    assert isinstance(logger, BaseLogger)

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