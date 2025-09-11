import numpy as np
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.optimizers.optimizer import Optimizer
from opto.trainer.loggers import BaseLogger
from opto.trainer.algorithms.basic_algorithms import Trainer
from opto.trainer.loader import DataLoader
from opto.features.priority_search.sampler import Sampler, BatchRollout
from opto.trainer.evaluators import evaluate  # TODO update evaluate implementation
from dataclasses import dataclass
import pickle
# TODO save and load SearchTemplate
# TODO async version???
# TODO create SYNC and ASYNC versions of the base class; add an attribute to the class to indicate

@dataclass
class Samples:
    """ A container for samples collected during the search algorithm. It contains a list of BatchRollout objects
    and a dataset with inputs and infos which created the list of BatchRollout. """

    samples: List[BatchRollout]
    dataset: Dict[str, List[Any]]  # contains 'inputs' and 'infos' keys  # TODO do we need this?

    def __init__(self, samples: List[BatchRollout], dataset: Dict[str, List[Any]]):
        assert isinstance(samples, list), "samples must be a list of BatchRollout objects."
        assert all(isinstance(s, BatchRollout) for s in samples), "All samples must be BatchRollout objects."
        assert isinstance(dataset, dict), "dataset must be a dict."
        assert 'inputs' in dataset and 'infos' in dataset, "dataset must contain 'inputs' and 'infos' keys."

        self.samples = samples

        # TODO drop this
        self._dataset = dataset  # NOTE this cannot be extracted from the samples in general?

    def add_samples(self, samples):
        """ Add samples to the Samples object. """
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        samples = samples.samples  # extract the samples from the Samples object
        assert isinstance(samples, list), "samples must be a list of BatchRollout objects."
        assert all(isinstance(s, BatchRollout) for s in samples), "All samples must be BatchRollout objects."

        # TODO assert xs and infos are in self.minibatch
        # add a function to extract unique inputs and infos from the samples

        self.samples.extend(samples)

    # def get_batch(self):
    #     return self.dataset

    def __iter__(self):
        """ Iterate over the samples. """
        return iter(self.samples)

    def __len__(self):
        return sum(len(s) for s in self.samples)

    @property
    def n_batchrollouts(self) -> int:
        """ Number of sub-batches in the samples. """
        return len(self.samples)


def check_optimizer_parameters(optimizer: Optimizer, agent: trace.Module):
    """ Check if the optimizer's parameters are the same as the agent's parameters. """
    assert isinstance(optimizer, Optimizer), "optimizer must be an instance of Optimizer."
    agent_params = set(agent.parameters())
    optimizer_params = set(optimizer.parameters)
    assert agent_params == optimizer_params, "Optimizer parameters do not match agent parameters."


def save_train_config(function):
    """ Decorator to save the inputs of a class method. """
    def wrapper(self, **kwargs):
        _kwargs = kwargs.copy()
        del _kwargs['train_dataset']  # remove train_dataset from the saved kwargs
        if _kwargs.get('validate_dataset') is not None:
            del _kwargs['validate_dataset'] # remove validate_dataset from the saved kwargs
        if _kwargs.get('test_dataset') is not None:
            del _kwargs['test_dataset']  # remove test_dataset from the saved kwargs
        setattr(self, f'_train_last_kwargs', _kwargs)
        return function(self, **kwargs)
    return wrapper


class SearchTemplate(Trainer):
    # This only uses __init__ and evaluate of Minibatch class.
    """ This implements a generic template for search algorithm. """

    def __init__(self,
                agent: trace.Module,
                optimizer : Union[Optimizer, List[Optimizer]],
                num_threads: int = None,   # maximum number of threads to use for parallel execution
                logger: Union[BaseLogger, None] =None,
                *args,
                **kwargs,
                ):
        super().__init__(agent, num_threads=num_threads, logger=logger, *args, **kwargs)

        # TODO assert agent parameters are the same as optimizer.parameters
        if isinstance(optimizer, list):
            assert len(optimizer) > 0, "Optimizers list is empty."
            for opt in optimizer:
                check_optimizer_parameters(opt, agent)
            self._optimizers = optimizer
        else:
            check_optimizer_parameters(optimizer, agent)
            self._optimizers = [optimizer]

        self.n_iters = 0  # number of iterations
        self._optimizer_index = -1  # index of the current optimizer to use

    @property
    def optimizer(self):
        self._optimizer_index += 1
        return self._optimizers[self._optimizer_index % len(self._optimizers)]  # return the current optimizer

    @save_train_config
    def train(self,
              *,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              # validation
              validate_dataset = None, # same format as train_dataset; if None use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              num_batches = 1,  # number of batches to use from the dataset in each iteration
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              _init_epoch = 0,  # initial epoch number (for resuming training)
              _init_n_samples = 0,  # initial number of samples (for resuming training)
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent; if None, use train_dataset
              test_guide = None, # guide to provide scores for the test set; if None, use guide
              eval_frequency: Union[int, None] = 1,  # frequency of evaluation NOTE set test_frequency < 0 to skip first evaluation
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "search_checkpoints/",  # path to save the agent
              **kwargs
              ):
        assert 'subbatch_size' not in kwargs, "subbatch_size should not be provided in kwargs."

        ## Setup
        test_frequency = eval_frequency  # use eval_frequency as test_frequency  # NOTE legacy notation
        log_frequency = log_frequency or test_frequency  # frequency of logging (default to test_frequency)
        self.num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_dataset = test_dataset or train_dataset  # default to train_dataset if test_dataset is not provided
        test_guide = test_guide or guide
        self.num_eval_samples = num_eval_samples  # number of samples to use to evaluate each input
        if score_range is None:
            score_range = (-np.inf, np.inf)
        assert len(score_range) == 2, "score_range must be a tuple (min_score, max_score)."
        assert score_range[1] >= score_range[0], "score_range must be a tuple (min_score, max_score) with min_score <= max_score."
        self._score_range = score_range  # range of the score for the guide

        subbatch_size, batch_size = batch_size, batch_size*num_batches

        self.train_sampler = Sampler(
            DataLoader(train_dataset, batch_size=batch_size),
            guide,
            num_threads=self.num_threads,
            subbatch_size=subbatch_size,
            score_range=self._score_range
        )
        self._validate_dataset = validate_dataset  # if None, the current batch will be used for validation
        if validate_dataset is not None:
            self.validate_sampler = Sampler(
                DataLoader(validate_dataset, batch_size=batch_size),
                validate_guide or guide,
                num_threads=self.num_threads,
                subbatch_size=None,  # no sub-batch size for validation
                score_range=self._score_range
            )
        else:
            self.validate_sampler = self.train_sampler  # use the train_sampler for validation if no validation dataset is provided

        # Save the agent before learning if save_frequency > 0
        if (save_frequency is not None) and save_frequency > 0:
            self.save(save_path)

        samples = None
        n_epochs = _init_epoch  # number of epochs (full passes over the dataset) performed by the algorithm (This is incremented in sample)
        n_samples = _init_n_samples  # number of training samples processed by the algorithm (This is incremented in sample)
        train_scores = []  # to store the scores of the agent during training

        while n_epochs < num_epochs :

            print(f"Epoch: {n_epochs}. Iteration: {self.n_iters}")

            # 1. Propose new parameters given the current state of the algorithm
            # proposals: list of trace.Modules
            update_dict, proposals, info_update = self.update(samples, verbose=verbose, **kwargs)
            self.optimizer.update(update_dict)  # update self.agent with the proposed parameters

            # 2. Get feedback on the proposed parameters on the current batch
            # samples: Samples object containing the samples and the minibatch
            samples, info_sample = self.sample(proposals, verbose=verbose, **kwargs)

            # Evaluate the agent after update
            if (test_frequency is not None) and (self.n_iters % test_frequency == 0):
                if self.n_iters == 0 and test_frequency < 0:
                    print("Skipping first evaluation.")
                else:
                    info_test = self.test(test_dataset, test_guide)  # test self.agent
                    self.log(info_test, prefix="Test/")

            # Save the algorithm state
            if (save_frequency is not None and save_frequency > 0) and self.n_iters % save_frequency == 0:
                self.save(save_path)

            # Log information
            assert 'mean_score' in info_sample, "info_sample must contain 'mean_score'."
            assert 'n_epochs' in info_sample, "info_sample must contain 'n_epochs'."

            train_scores.append(info_sample['mean_score'])  # so that mean can be computed
            if self.n_iters % log_frequency == 0:
                self.logger.log('Algo/Average train score', np.mean(train_scores), self.n_iters, color='blue')
                self.log(info_update, prefix="Update/")
                self.log(info_sample, prefix="Sample/")
                n_samples += len(samples)  # update the number of samples processed
                self.logger.log('Algo/Number of samples', n_samples, self.n_iters, color='blue')
                # Log parameters
                for p in self.agent.parameters():
                    self.logger.log(f"Parameter/{p.name}", p.data, self.n_iters, color='red')

            # Update counters
            n_epochs = info_sample['n_epochs']  # update the number of epochs completed
            self.n_iters += 1
        return

    @property
    def max_score(self):
        """ Maximum score that can be achieved by the agent. """
        return self._score_range[1]

    @property
    def min_score(self):
        """ Minimum score that can be achieved by the agent. """
        return self._score_range[0]

    # Can be overridden by subclasses to implement specific sampling strategies
    def sample(self, agents, verbose=False, **kwargs):
        """ Sample a batch of data based on the proposed parameters. All proposals are evaluated on the same batch of inputs.

        Args:
            agents (list): A list of trace.Modules (proposed parameters) to evaluate.
                **kwargs: Additional keyword arguments that may be used by the implementation.
        """
        samples = Samples(*self.train_sampler.sample(agents, description_prefix='Sampling training minibatch: '))  # create a Samples object to store the samples and the minibatch
        # Log information about the sampling
        scores = [ g.get_scores() for g in samples.samples]  # list of list of scores for each BatchRollout
        scores = [item for sublist in scores for item in sublist]  # flatten the list of scores
        log_info = {
            'mean_score': np.mean(scores),
            'n_epochs': self.train_sampler.n_epochs,
        }
        # check if the scores are within the score range
        if not (self.min_score <= log_info['mean_score'] <= self.max_score):
            print(f"Warning: Mean score {log_info['mean_score']} is out of the range {self._score_range}.")

        return samples, log_info

    def log(self, info_log, prefix=""):
        """ Log the information from the algorithm. """
        for key, value in info_log.items():
            try:
                if value is not None:
                    self.logger.log(f"{prefix}{key}", value, self.n_iters)
            except Exception as e:
                print(e)

    def test(self, test_dataset, guide):
        min_score = self.min_score
        # Test the agent's performance
        test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                          min_score=min_score, num_threads=self.num_threads,
                          description=f"Evaluating agent")  # and log
        # check if the test_score is within the score range
        if not (self.min_score <= test_score <= self.max_score):
            print(f"Warning: Test score {test_score} is out of the range {self._score_range}.")
        return {'test_score': test_score}

    def evaluate(self, agent, guide, xs, infos, min_score=None, num_samples=1, num_threads=None, description=None):
        """ Evaluate the agent on the given dataset. """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_scores = evaluate(agent, guide, xs, infos, min_score=min_score, num_threads=num_threads,
                               num_samples=num_samples, description=description)
        if all([s is not None for s in test_scores]):
            return np.mean(test_scores)

    def save(self, save_path):
        with open(save_path+'/algo.pkl', 'wb') as f:
            pickle.dump(state, f)

    def load(self, load_path):
        with open(load_path+'/algo.pkl', 'rb') as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        print(f"Loaded algorithm state from {load_path}/algo.pkl")
        return


    def resume(self,
               load_path,
               train_dataset,
               validate_dataset = None,
               test_dataset = None,
               **kwargs):
        """ Resume training from a saved state.

        Args:
            load_path (str): Path to the saved state.
            train_dataset: Dataset to resume training.
            validate_dataset: Dataset for validation. If None, use the current batch.
            test_dataset: Dataset for testing. If None, use train_dataset.
            **kwargs: Additional keyword arguments for the training method. If not provided, the same parameters as the last training call are used.
        """
        self.load(load_path)  # load the saved state
        # Resume training with the same parameters as before
        last_train_kwargs = getattr(self, '_train_last_kwargs', {}).copy()
        last_train_kwargs['train_dataset'] = train_dataset
        last_train_kwargs['validate_dataset'] = validate_dataset
        last_train_kwargs['test_dataset'] = test_dataset
        last_train_kwargs.update(kwargs)  # update with any new parameters provided
        print(f"Resuming training with parameters: {last_train_kwargs}")
        self.train(**last_train_kwargs)


    # Unimplemented methods that should be implemented by subclasses
    def update(self, samples=None, verbose=False, **kwargs):
        """ Update the agent based on the provided samples.
        Args:
            samples (list): A list of samples from the previous iteration. If None, the agent's parameters are returned without updating.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            update_dict (dict of Parameter: Any): A dictionary containing the updated parameters of the agent.
            proposals (list of trace.Module): A list of proposed parameters (trace.Module) after the update.
            info_log (dict of str: Any): A dictionary containing logging information about the update process.

        This method updates the agent's parameters based on samples of the training dataset and validation dataset (provided by self.get_validate_dataset).
        In addition, it return new agents (proposals) that can be used for collecting data for the next iteration.
        """
        raise NotImplementedError("The update method should be implemented by subclasses.")
        # return update_dict, proposals, info_log
