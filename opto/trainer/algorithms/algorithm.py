from typing import Optional
from opto.trace.modules import Module
from opto.trainer.loggers import DefaultLogger
from opto.trainer.loader import DataLoader
from opto.trainer.guide import Guide
from opto.optimizers.optimizer import Optimizer
import os
import pickle

class AbstractAlgorithm:
    """Abstract base class for all training algorithms.

    This class provides a common interface for all algorithms that train agents.
    Subclasses should implement the train method with their specific training logic.

    Parameters
    ----------
    agent : Any
        The agent to be trained by the algorithm.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    agent : Any
        The agent instance being trained.
    """

    def __init__(self, agent, *args, **kwargs):
        """Initialize the abstract algorithm.

        Parameters
        ----------
        agent : Any
            The agent to be trained.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.
        """
        self.agent = agent

    def train(self, *args, **kwargs):
        """Train the agent using the algorithm's specific strategy.

        Parameters
        ----------
        *args
            Training arguments specific to the algorithm.
        **kwargs
            Training keyword arguments specific to the algorithm.

        Notes
        -----
        This method should be overridden by subclasses to implement
        their specific training logic.
        """
        pass


class Trainer(AbstractAlgorithm):
    """Base trainer class that defines the API for training agents from datasets.

    This class provides infrastructure for training trace.Module agents using datasets
    of (input, info) pairs and teacher/guide functions for feedback generation.

    Parameters
    ----------
    agent : trace.Module
        The trace module to be trained (e.g. constructed with @trace.model).
    num_threads : int, optional
        Maximum number of threads for parallel execution, by default None.
    logger : Logger, optional
        Logger instance for tracking training metrics, by default None.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    agent : trace.Module
        The agent being trained.
    num_threads : int or None
        Maximum number of threads for parallel operations.
    logger : Logger
        Logger instance for metric tracking.

    Notes
    -----
    The training paradigm involves:
    - agent: trace.Module (constructed via @trace.model decorator)
    - guide/teacher: function (question, student_answer, info) -> score, feedback
    - train_dataset: dataset of (x, info) pairs for training
    """

    def __init__(self,
                 agent,  # trace.model
                 num_threads: Optional[int] = None,   # maximum number of threads to use for parallel execution
                 logger=None,  # logger for tracking metrics
                 *args,
                 **kwargs):
        """Initialize the Trainer with an agent and configuration.

        Parameters
        ----------
        agent : trace.Module
            The trace module agent to be trained.
        num_threads : int, optional
            Maximum number of threads for parallel execution, by default None.
        logger : Logger, optional
            Logger for tracking training metrics, by default None (uses DefaultLogger).
        *args
            Additional positional arguments passed to parent class.
        **kwargs
            Additional keyword arguments passed to parent class.

        Raises
        ------
        AssertionError
            If agent is not a trace.Module instance.
        """
        assert isinstance(agent, Module), "Agent must be a trace Module. Getting {}".format(type(agent))
        super().__init__(agent, *args, **kwargs)
        self.num_threads = num_threads
        # Use DefaultLogger as default if logger is None
        self.logger = logger if logger is not None else DefaultLogger()

    def _use_asyncio(self, threads=None):
        """Determine whether to use asyncio based on the number of threads.

        Parameters
        ----------
        threads : int, optional
            Number of threads to use. If None, uses self.num_threads.

        Returns
        -------
        bool
            True if parallel execution should be used, False otherwise.

        Notes
        -----
        Parallel execution is enabled when the effective thread count is
        greater than 1. This helps optimize performance for batch operations.
        """
        effective_threads = threads or self.num_threads
        return effective_threads is not None and effective_threads > 1

    def save_agent(self, save_path, iteration=None):
        """Save the agent to the specified path with optional iteration numbering.

        Parameters
        ----------
        save_path : str
            Base path to save the agent to.
        iteration : int, optional
            Current iteration number for checkpoint naming, by default None.

        Returns
        -------
        str
            The actual path where the agent was saved.

        Notes
        -----
        If iteration is provided, it's appended to the filename. Final iterations
        (matching self.n_iters) get a "_final" suffix for easy identification.
        The directory structure is created automatically if it doesn't exist.
        """
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        # Add iteration number to filename if provided
        if iteration is not None:
            base, ext = os.path.splitext(save_path)
            # Add "_final" for the final checkpoint
            if hasattr(self, 'n_iters') and iteration == self.n_iters:
                save_path = f"{base}_iter{iteration}_final{ext}"
            else:
                save_path = f"{base}_iter{iteration}{ext}"

        # Save the agent
        self.agent.save(save_path)

        # Log if we have a logger and iteration is provided
        if hasattr(self, 'logger') and iteration is not None:
            self.logger.log('Saved agent', save_path, iteration, color='blue')

        return save_path

    def train(self,
              guide,
              train_dataset,  # dataset of (x, info) pairs
              num_threads: int = None,  # maximum number of threads to use (overrides self.num_threads)
              **kwargs
              ):
        """Train the agent using the provided guide and dataset.

        Parameters
        ----------
        guide : Guide
            Guide function to provide feedback during training.
        train_dataset : dict
            Training dataset containing 'inputs' and 'infos' keys.
        num_threads : int, optional
            Maximum number of threads to use, overrides self.num_threads.
        **kwargs
            Additional training arguments specific to the algorithm.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Notes
        -----
        Subclasses must implement this method with their specific training logic.
        The method should use the guide to evaluate agent outputs and update
        the agent parameters accordingly.
        """
        raise NotImplementedError

    def save(self, save_path: str):
        """Save the trainer and its components to a file.

        Parameters
        ----------
        path : str
            Base path to save the trainer state to.

        Notes
        -----
        This method serializes the trainer's state, saving different component
        types (Module, Guide, DataLoader, Optimizer) to separate files with
        appropriate extensions. The main trainer state is saved as a pickle file.
        """
        raise NotImplementedError

    @classmethod
    def load(cls,
             load_path: str):
        """Load the trainer and its components from a file.

        Parameters
        ----------
        path : str
            Path to the saved trainer state file.

        Notes
        -----
        This method deserializes the trainer's state and loads component files.
        It validates that the loaded attributes match the expected types and
        warns if attributes from the saved state are not found in the current
        trainer instance.
        """
        raise NotImplementedError

    def resume(self, *,
               model: Module,
               train_dataset: dict ,
               **kwargs):
        raise NotImplementedError




    # TODO remove these old save and load methods
    # def save(self, path: str):

    #     with open(path, 'wb') as f:
    #         d = {}
    #         for key, value in self.__dict__.items():
    #             if isinstance(value, Module):
    #                 _path = path+ f"_{key}.module"
    #                 value.save(_path)
    #                 d[key] = _path
    #             elif isinstance(value, Guide):
    #                 _path = path + f"_{key}.guide"
    #                 value.save(_path)
    #                 d[key] = _path
    #             elif isinstance(value, DataLoader):
    #                 _path = path + f"_{key}.dataloader"
    #                 value.save(_path)
    #                 d[key] = _path
    #             elif isinstance(value, Optimizer):
    #                 _path = path + f"_{key}.optimizer"
    #                 value.save(_path)
    #                 d[key] = _path
    #             else:
    #                 d[key] = value
    #         pickle.dump(d, f)

    # def load(self, path: str):
    #     """ Load the guide from a file. """
    #     with open(path, 'rb') as f:
    #         data = pickle.load(f)
    #         for key, value in data.items():
    #             if key not in self.__dict__:
    #                 warning_msg = f"Key '{key}' not found in the algorithm's attributes. Skipping loading for this key."
    #                 print(warning_msg)  # or use logging.warning(warning_msg)
    #                 continue

    #             # key is in the algorithm's attributes
    #             if isinstance(value, str):
    #                 if value.endswith('.module'):
    #                     attr = self.__dict__[key]
    #                     assert isinstance(attr, Module), f"Expected {key} to be a Module, got {type(attr)}"
    #                 elif value.endswith('.guide'):
    #                     attr = self.__dict__[key]
    #                     assert isinstance(attr, Guide), f"Expected {key} to be an Guide, got {type(attr)}"
    #                 elif value.endswith('.dataloader'):
    #                     attr = self.__dict__[key]
    #                     assert isinstance(attr, DataLoader), f"Expected {key} to be a DataLoader, got {type(attr)}"
    #                 elif value.endswith('.optimizer'):
    #                     attr = self.__dict__[key]
    #                     assert isinstance(attr, Optimizer), f"Expected {key} to be an Optimizer, got {type(attr)}"
    #                 attr.load(value)
    #             else:
    #                 self.__dict__[key] = value