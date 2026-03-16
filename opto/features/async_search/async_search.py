import asyncio
from tqdm.asyncio import tqdm
import numpy as np
from typing import Union, List, Dict, Any, Optional, Tuple
import uuid

from opto import trace
from opto.trace.errors import ExecutionError
from opto.optimizers.optimizer import Optimizer
from opto.trainer.algorithms.algorithm import Trainer
from opto.features.async_search.controller import Controller
from opto.features.async_search.async_sampler import AsyncSampler as Sampler
from opto.features.priority_search.sampler import BatchRollout
from opto.features.priority_search.search_template import Samples
from opto.trainer.loader import DataLoader
from opto.trainer.evaluators import evaluate
from opto.trainer.utils import safe_mean


WORKER_TASK = "worker_task"
EVAL_TASK = "eval_task"

def check_optimizer_parameters(optimizer: Optimizer, agent: trace.Module):
    """ Check if the optimizer's parameters are the same as the agent's parameters. """
    assert isinstance(optimizer, Optimizer), "optimizer must be an instance of Optimizer."
    agent_params = set(agent.parameters())
    optimizer_params = set(optimizer.parameters)
    assert agent_params == optimizer_params, "Optimizer parameters do not match agent parameters."


def as_async(func):
    """Decorator to make a synchronous function asynchronous."""
    async def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
    return wrapper

@as_async
def evaluate_agent(agent, guide, x, info, min_score=None):
    try:
        output = agent(x).data
        score = guide.metric(x, output, info)
    except ExecutionError as e:
        score = min_score
    return score


async def evaluate(agent, guide, inputs, infos, min_score=None, num_samples=1, description=None):
    """ Evaluate the agent on the inputs and return the scores

    Args:
        agent: The agent to evaluate
        guide: The guide to use for evaluation
        inputs: List of inputs to evaluate on
        infos: List of additional information for each input
        min_score: Minimum score to return when an exception occurs
        num_samples: Number of samples to use to evaluate each input
        num_threads: Maximum number of threads to use for parallel evaluation
        description: Description to display in the progress bar
    """
    assert len(inputs) == len(infos), "Inputs and infos must have the same length"
    N = len(inputs)
    # Use provided description or generate a default one
    eval_description = description or f"Evaluating {N} examples"
    # repeat each index num_samples times
    indices = [i for i in range(N) for _ in range(num_samples)]
    # Run the evaluation in parallel
    scores = await asyncio.gather(*(evaluate_agent(agent, guide, inputs[i], infos[i], min_score) for i in indices))
    scores = np.array(scores)
    if num_samples > 1:
        # scores will be of length N * num_samples
        # Reshape scores into an array of shape (N, num_samples)
        scores = scores.reshape(N, num_samples)
    return scores

class AsyncSearch(Trainer, Controller):

    def __init__(self,
                 agent: trace.Module,
                 optimizer: Union[Optimizer, List[Optimizer]],
                 num_threads: int = None,
                 logger: Optional[Any] = None,
                 *args, **kwargs):
        super().__init__(agent, num_threads=num_threads, logger=logger, *args, **kwargs)

        if not isinstance(optimizer, list):
            optimizer = [optimizer]
        assert len(optimizer) > 0, "Optimizers list is empty."
        for opt in optimizer:
            check_optimizer_parameters(opt, agent)
        self._optimizers = optimizer

        self.n_iters = 0
        self.n_epochs = 0
        self.n_samples = 0
        self._optimizer_index = -1
        self.train_sampler = None
        self._train_scores = []  # to store the scores of the agent during training
        self._train_num_samples = []  # to store the number of samples used to compute each score

    @property
    def optimizer(self):
        self._optimizer_index = (self._optimizer_index + 1) % len(self._optimizers)
        return self._optimizers[self._optimizer_index]

    # This is the synchronous interface
    def train(self, *args, num_threads = 1, **kwargs):
        self.num_threads = num_threads or self.num_threads
        return asyncio.run(self.run(num_threads, *args, **kwargs))

    async def async_train(self, *args, num_threads = 1, **kwargs):
        self.num_threads = num_threads or self.num_threads
        return await self.run(num_threads, *args, **kwargs)

    # TODO clean up
    async def init(self, *,
                   guide, # guide to provide feedback
                   train_dataset,  # dataset of (x, info) pairs to train the agent
                   # training loop
                   batch_size = 1,  # batch size for updating the agent
                   num_batches = 1,  # number of batches to use from the dataset in each iteration
                   score_range = None,  # minimum score to update the agent
                   num_epochs = 1,  # number of training epochs (int or None)
                   num_steps = None,  # number of training steps (int or None)
                   verbose = False,  # whether to print the output of the agent
                   # evaluation
                   test_dataset = None, # dataset of (x, info) pairs to evaluate the agent; if None, use train_dataset
                   test_guide = None, # guide to provide scores for the test set; if None, use guide
                   test_frequency: Union[int, None] = 1,  # frequency of evaluation NOTE set test_frequency < 0 to skip first evaluation
                   num_test_samples: int = 1,  # number of samples to use to evaluate each input
                   # logging
                   log_frequency = None,  # frequency of logging
                   save_frequency: Union[int, None] = None,  # frequency of saving the agent
                   save_path: str = "async_search_checkpoints/",  # path to save the agent
                   **kwargs
                   ):

        """Initializes the state for the asynchronous search."""
        self.guide = guide
        self.train_dataset = train_dataset
        # training loop
        self.batch_size = batch_size
        self.num_batches = num_batches
        self._score_range = score_range or (-np.inf, np.inf)
        assert len(self._score_range) == 2, "score_range must be a tuple (min_score, max_score)."
        assert self._score_range[1] >= self._score_range[0], "score_range must be a tuple (min_score, max_score)."
        self.num_epochs = num_epochs
        self.num_steps = num_steps if num_steps is not None else 0
        self.verbose = verbose
        # evaluation
        self.test_dataset = test_dataset or train_dataset
        self.test_guide = test_guide or guide
        self.test_frequency = test_frequency
        self.num_test_samples = num_test_samples
        # logging
        self.log_frequency = log_frequency or test_frequency
        self.save_frequency = save_frequency
        self.save_path = save_path

        if self.train_sampler is None:
            self.train_sampler = Sampler(
                DataLoader(train_dataset, batch_size=batch_size),
                guide,
                score_range=self._score_range
        )
        else:
            self.train_sampler.loader.dataset = train_dataset  # update the train dataset in the sampler

        # for managing tasks
        self._eval_tasks = set()
        self._worker_tasks = dict()

    def should_stop(self) -> bool:
        """Determine whether the training should stop based on the number of epochs or steps."""
        if self.n_epochs < self.num_epochs or self.n_iters < self.num_steps:
            return False
        return True

    async def create_task(self):
        # Check if it's time for evaluation
        if (self.test_frequency is not None) and (self.n_iters % self.test_frequency == 0):
            if not (self.n_iters == 0 and self.test_frequency < 0):
                # skip the first evaluation if test_frequency < 0
                task_id = EVAL_TASK + ':' + str(self.n_iters)
                if task_id not in self._eval_tasks:
                    # this is to prevent duplicate eval tasks
                    self._eval_tasks.add(task_id)
                    return self.eval_task(task_id=task_id)

        # Otherwise, run worker task
        print(f"Epoch: {self.n_epochs}. Iteration: {self.n_iters}")
        task_id = WORKER_TASK + ':' + str(self.n_iters) + ':' + uuid.uuid4().hex
        proposal, task_state = await self.get_task_state()
        self._worker_tasks[task_id] = (proposal, task_state)
        return self.worker_task(task_id=task_id,
                                proposal=proposal,
                                **task_state)

    async def eval_task(self, task_id):
        info_test = await self.evaluate_agent(self.test_dataset, self.test_guide)
        return task_id, info_test

    async def worker_task(self, task_id, proposal, **kwargs):
        samples, info_sample = await self.sample(proposal, verbose=self.verbose)
        result, info_update = await self.per_worker_update(samples, **kwargs)
        return task_id, (result, info_sample, info_update)

    async def process_result(self, result):
        task_id, result = result
        if task_id.startswith(EVAL_TASK):
            # evaluation task
            self._eval_tasks.remove(task_id)
            info_test = result
            self.log(info_test, prefix="Test/")
            return

        # Worker task is done
        result, info_sample, info_per_worker_update = result
        info_main_update = await self.main_update(result)

        # Log the reuslt of worker task
        if (self.save_frequency is not None and self.save_frequency > 0) and self.n_iters % self.save_frequency == 0:
            self.save(self.save_path)

        # Log information
        assert 'mean_score' in info_sample, "info_sample must contain 'mean_score'."
        assert 'self.n_epochs' in info_sample, "info_sample must contain 'self.n_epochs'."

        self._train_scores.append(info_sample['mean_score'])  # so that mean can be computed
        self._train_num_samples.append(info_sample['num_samples'])
        self.n_samples += len(samples)  # update the number of samples processed

        if self.n_iters % self.log_frequency == 0:
            avg_train_score = np.sum(np.array(self._train_scores) * np.array(self._train_num_samples)) / np.sum(self._train_num_samples)
            self.logger.log('Algo/Average train score', avg_train_score, self.n_iters, color='blue')
            self.log(info_main_update, prefix="Main Update/")
            self.log(info_per_worker_update, prefix="Per Worker Update/")
            self.log(info_sample, prefix="Sample/")
            self.logger.log('Algo/Number of training samples', self.n_samples, self.n_iters, color='blue')
            # Log parameters
            for p in self.agent.parameters():
                self.logger.log(f"Parameter/{p.name}", p.data, self.n_iters, color='red')

        # Update counters
        self.n_epochs = info_sample['self.n_epochs']
        self.n_iters += 1 / self.num_threads  # each worker contributes to 1/num_threads of an iteration

    async def post_process(self):
        """Final processing after the controller stops."""
        pass

    @property
    def max_score(self):
        """ Maximum score that can be achieved by the agent. """
        return self._score_range[1]

    @property
    def min_score(self):
        """ Minimum score that can be achieved by the agent. """
        return self._score_range[0]

    # Can be overridden by subclasses to implement specific sampling strategies
    async def sample(self, agents, verbose=False, **kwargs):
        """ Sample a batch of data based on the proposed parameters. All proposals are evaluated on the same batch of inputs.

        Args:
            agents (list): A list of trace.Modules (proposed parameters) to evaluate.
                **kwargs: Additional keyword arguments that may be used by the implementation.
        """
        samples = Samples(*await self.train_sampler.sample(agents, description_prefix='Sampling training minibatch: '))  # create a Samples object to store the samples and the minibatch
        # Log information about the sampling
        scores = [ g.get_scores() for g in samples.samples]  # list of list of scores for each BatchRollout
        scores = [item for sublist in scores for item in sublist if item is not None]  # flatten the list of scores
        log_info = {
            'mean_score': safe_mean(scores, 0),  # return 0, if num_samples == 0 so that the weighted mean can be computed
            'num_samples': len(scores),
            'self.n_epochs': self.train_sampler.n_epochs,
        }
        # check if the scores are within the score range
        if hasattr(self, '_score_range') and not (self.min_score <= log_info['mean_score'] <= self.max_score):
            print(f"Warning: Mean score {log_info['mean_score']} is out of the range {self._score_range}.")
        return samples, log_info


    async def evaluate_agent(self, test_dataset, guide):
        """ Evaluate the agent on the given test dataset. """
        # Use provided num_threads or fall back to self.num_threads
        test_scores = await evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                     min_score=self.min_score,
                                     num_samples=self.num_test_samples,
                                     description=f"Evaluating agent")
        test_score = safe_mean(test_scores)
        # check if the test_score is within the score range
        if hasattr(self, '_score_range') and not (self.min_score <= test_score <= self.max_score):
            print(f"Warning: Test score {test_score} is out of the range {self._score_range}.")
        return {'test_score': test_score}


    def log(self, info_log: Dict[str, Any], prefix=""):
        """Log information from the algorithm."""
        for key, value in info_log.items():
            if value is not None and self.logger:
                try:
                    self.logger.log(f"{prefix}{key}", value, self.n_iters)
                except Exception as e:
                    print(f"Logging failed for key {key}: {e}")

    # # TODO
    def save(self, save_path):
        # save agent, data iterator, optimizer states, and other necessary information to resume training
        # save self._worker_tasks
        pass

    # # TODO
    # @classmethod
    # def load(cls, load_path: str):

    # # TODO
    # def resume(self, *,
    #            model: trace.Module,
    #            train_dataset: dict ,
    #            validate_dataset = None,
    #            test_dataset = None,
    #            **kwargs):


    # Unimplemented methods that should be implemented by subclasses
    async def main_update(self, result) -> Tuple[Dict[trace.Parameter, Any], Dict[str, Any]]:
        """ Update the agent based on the provided result from all workers.

        Return:
            update_dict (dict of Parameter: Any): A dictionary containing the updated parameters of the agent.
            info_log (dict of str: Any): A dictionary containing logging information about the update process.
        """

        raise NotImplementedError("The update method should be implemented by subclasses.")

    async def get_task_state(self) -> Tuple[trace.Module, dict[str, Any]]:
        """ Get the proposal and state needed for the worker task.

        Returns:
            proposal (trace.Module): A proposed parameters (trace.Module) for the worker task.
            task_state (dict): A dictionary containing any additional state needed for the worker task.
        """
        # return proposal and state needed for the worker task
        raise NotImplementedError("The update method should be implemented by subclasses.")

    @classmethod
    async def per_worker_update(samples: Samples, **kwargs) -> Tuple[Any, dict[str, Any]]:
        """ Update the agent based on the provided samples.
        Args:
            samples (list): A list of samples collected by the proposal returned by get_task_state using training dataset.

        Returns:
            results (Any): The results of the update process, which will be passed to main_update.
            info_log (dict of str: Any): A dictionary containing logging information about the update process.
        """
        raise NotImplementedError("The update method should be implemented by subclasses.")
