import numpy as np
import copy
import heapq
import time
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trace.nodes import ParameterNode
from opto.optimizers.optimizer import Optimizer
from opto.trainer.utils import async_run
from opto.trainer.algorithms.basic_algorithms import batchify
from opto.features.priority_search.search_template import SearchTemplate, Samples, BatchRollout
from opto.features.priority_search.utils import set_module_parameters, remap_update_dict, create_module_from_update_dict, is_module_copy
from opto.features.priority_search.module_regressor import ModuleCandidateRegressor
from opto.utils.auto_retry import retry_with_exponential_backoff

class ModuleCandidate:
    """ A container used by PrioritySearch to store a candidate module as (its base module and update dictionary) and its statistics. """

    def __init__(self,
                 base_module: trace.Module,
                 update_dict: Optional[Dict[ParameterNode, Any]] = None,
                 optimizer: Optimizer = None,
                 ):
        """ A candidate module with its base module and update dictionary.
        Args:
            base_module (trace.Module): The base module to use as a template for the candidate.
            update_dict (dict): A dictionary of ParameterNode: value pairs to update the base module; the key can be a deep copy of the base module's parameters.
            stats (dict): A dictionary of statistics about the candidate.
        """
        assert isinstance(base_module, trace.Module), "base_module must be a trace.Module."
        if update_dict is not None:
            assert isinstance(optimizer, Optimizer), "optimizer must be an instance of Optimizer when update_dict is provided."

        self.base_module = base_module
        self.update_dict = update_dict if update_dict is not None else {}
        self.optimizer = optimizer  # the optimizer used to generate the update_dict; can be None, which indicates the base_module is used.
        self.update_dict = remap_update_dict(self.base_module, self.update_dict)
        self.rollouts = []  # list of dicts containing the rollout information (not BatchRollout, but a list of dicts)
        self.created_time = time.time()

    def get_module(self):
        """ Apply the update_dict to the base_module and return the updated module.
        A new module is always created so the base_module is not modified.
        The new module has a new attribute _module_candidate which is this candidate."""
        module = create_module_from_update_dict(self.base_module, self.update_dict) if self.update_dict else copy.deepcopy(self.base_module)  #
        setattr(module, '__TRACE_RESERVED_module_candidate_id', id(self))
        return module  # return the updated module

    def apply_update(self, base_module=None):
        """ Apply update to the base_module in place. """
        set_module_parameters(base_module or self.base_module, self.update_dict)

    def __getstate__(self):
        """ Get the state of the candidate for serialization. """
        state = copy.deepcopy(self.__dict__)  # this will detach the nodes from the computation graph
        return state

    def __setstate__(self, state):
        """ Set the state of the candidate from serialization. """
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        """ Create a deep copy, except for the base_module which is not copied, it is the original module. """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'base_module':
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, v)  # base_module is not copied, it is the original module
        return result

    def __eq__(self, other):
        """ Check if two candidates are equal based on their base_module and update_dict. """
        assert isinstance(other, ModuleCandidate), "other must be an instance of ModuleCandidate."
        return (self.update_dict == other.update_dict) and is_module_copy(self.base_module, other.base_module) and (id(self.optimizer) == id(other.optimizer))

    def __lt__(self, other):
        """ Compare two candidates based on their update_dict. """
        assert isinstance(other, ModuleCandidate), "other must be an instance of ModuleCandidate."
        return self.created_time > other.created_time
        # self < other if, self is created later than other
        # Since we will use minheap, and this would give priority to later created candidates in the heap memory.

    def __hash__(self):
        """ Hash the candidate based on its update_dict. """
        return hash((frozenset(self.update_dict.items()), id(self.optimizer), id(self.base_module)))

    def add_rollouts(self, rollouts: List[Dict[str, Any]]):
        """ Add rollouts to the candidate. """
        assert isinstance(rollouts, list), "rollouts must be a list of dicts."
        assert all(isinstance(r, dict) for r in rollouts), "All rollouts must be dicts."
        # Each rollout is a dict with keys: 'module', 'x', 'info', 'target', 'score', 'feedback'
        assert all('module' in r and 'x' in r and 'info' in r and 'target' in r and 'score' in r and 'feedback' in r for r in rollouts), \
            "Each rollout must contain 'module', 'x', 'info', 'target', 'score', and 'feedback' keys."

        self.rollouts.extend(rollouts)

    def mean_score(self):
        """ Compute the score of the candidate based on the rollouts. """
        if not self.rollouts:
            return None
        scores = [r['score'] for r in self.rollouts]
        return np.mean(scores) if scores else None

    def compute_score_confidence(self, min_score, max_score, scaling_constant=1.0, total_trials=1):
        """Compute the UCB, mean, LCB score for the candidate. After queried, the number of confidence queries is incremented.

        UCB = mean_score + scaling_constant * sqrt(ln(total_trials) / n_scores) * (max_score - min_score)
        UCB = clip(UCB, min_score, max_score)

        LCB = mean_score - scaling_constant * sqrt(ln(total_trials) / n_scores) * (max_score - min_score)
        LCB = clip(LCB, min_score, max_score)

        Args:
            min_score (float): The minimum score for clipping.
            max_score (float): The maximum score for clipping.
            scaling_constant (float): The scaling constant for the exploration term.
            total_trials (int): The total number of trials conducted. Must be at least 1.
        Returns:
            lcb_score (float): The lower confidence bound score.
            mean_score (float): The mean score.
            ucb_score (float): The upper confidence bound score.
        """
        # Get scores from rollouts
        scores = [r['score'] for r in self.rollouts]

        if not scores:
            return min_score, None, max_score

        # Calculate mean score for this candidate
        mean_score = np.mean(scores)
        n_scores = len(scores)
        assert n_scores == self.num_rollouts, "Number of scores should match number of rollouts."

        # Calculate how many times the confidence interval has been used to form a union bound
        assert total_trials >= 1, "total_trials must be at least 1."
        total_trials = total_trials + 1 # this is an upper bound, since log(1) = 0

        # Compute the exploration term based on Hoeffding's inequality
        exploration_term = scaling_constant * np.sqrt(np.log(total_trials) / n_scores) * (max_score - min_score)

        # Calculate UCB score
        ucb_score = mean_score + exploration_term
        ucb_score = np.clip(ucb_score, min_score, max_score)

        # Calculate LCB score
        lcb_score = mean_score - exploration_term
        lcb_score = np.clip(lcb_score, min_score, max_score)

        return lcb_score, mean_score, ucb_score


    @property
    def num_rollouts(self):
        """ Return the number of rollouts collected for this candidate. """
        return len(self.rollouts)


class HeapMemory:
    # This is a basic implementation of a heap memory that uses a priority queue to store candidates.
    # Later on this will be replaced by a memory DB.

    # NOTE that the heap memory is a max-heap, so we store negative scores to use the default min-heap behavior of heapq.
    def __init__(self, size=None):
        """ Initialize an empty heap memory. """
        self.memory = []
        self.size = size  # Optional size limit for the heap memory

    def push(self, score, data):
        """ Push an item to the heap memory. """
        heapq.heappush(self.memory, (-score, data))
        if self.size is not None and len(self.memory) > self.size:
            # NOTE a heuristic for now
            self.memory = self.memory[:self.size]  # Keep only the top `size` items

    def pop(self):
        """ Pop the top item from the heap memory. """
        if not self.memory:
            raise IndexError("pop from an empty heap memory")
        return heapq.heappop(self.memory)

    def __len__(self):
        """ Return the number of items in the heap memory. """
        return len(self.memory)

    def __bool__(self):
        """ Return True if the heap memory is not empty, False otherwise. """
        return len(self.memory) > 0

    def __iter__(self):
        """ Iterate over the items in the heap memory. """
        return iter(self.memory)

    def best(self, criterion=None):
        """ Return the best item in the heap memory without removing it.

        If criterion is None, return the item with the highest priority (lowest negative score).
        If criterion is a callable function, return the item that maximizes the criterion.
        """
        if not self.memory:
            raise IndexError("best from an empty heap memory")
        if criterion is None:
            return self.memory[0]  # return the item with the highest priority (lowest negative score)
        else:
            assert callable(criterion), "criterion must be a callable function."
            def _criterion(x):
                neg_score, candidate = x
                p = criterion(candidate)
                return p if p is not None else 0
            return max(self.memory, key=lambda x: _criterion(x))

    def reorder_according_to_predicted_scores(self):
        """ Reorder the heap memory according to the predicted scores. """
        # Now all ModuleCandidate objects in the heap memory have predicted scores. Should modify the old score to the negative predicted scores, then use heapq.heapify to reorder the heap memory.
        self.memory = [(-candidate.predicted_score, candidate) for _, candidate in self.memory]
        heapq.heapify(self.memory)

# TODO check saving and loading
class PrioritySearch(SearchTemplate):
    """ A search algorithm that uses a priority queue to explore the parameter space and propose new candidates.

        It provides a scalable template for implementing search algorithms based on asynchronous generation, validation, and testing.
        In each iteration,
            1. It proposes a best agent and a set of `num_candidates` exploration agents that have the highest scores in the priority queue.
            2. The best agent is tested for performance if test_frequency is met.
            3. `num_batches` minibatches of `batch_size` samples are drawn from the training dataset, and the exploration agents are run on the samples. This creates a set of agent rollouts, where each rollout contains the agent module, input, info, target, score, and feedback. For each agent, rollouts of each minibatch are grouped together as a connected subgraph (represented as the BatchRollout object). In total, this step creates `num_candidates * num_batches` subgraphs.
            4. Optimizer is run on each subgraph to propose new parameters for the agents. `num_proposals` proposals are generated for each subgraph. This results in `num_subgraphs * num_proposals` total proposals.
            5. The proposed parameters are validated by running the agents on the validation dataset, which can be the current batch or a separate validation dataset when provided. When validate_exploration_candidates is set to True, the exploration candidates are also validated.
            6. The validation results are used to update the priority queue, which stores the candidates and their scores. The candidates are stored as ModuleCandidate objects, which contain the base module, update dictionary, and rollouts (i.e. raw statistics of the candidate).

        This algorithm template can be subclassed to implement specific search algorithms by overriding the `exploit`, `explore`, and `compute_exploration_priority` methods.
        The `exploit` method is used to select the best candidate from the priority queue, the `explore` method is used to generate new candidates from the priority queue, and
        the `compute_exploration_priority` method is used to compute the score for ranking in the priority queue.

        By default, `compute_exploration_priority` computes the mean score of the rollouts. `exploit` simply returns the candidate with highest priority from the priority queue, and `explore` generates the top `num_candidates` candidates from the priority queue.


        `compute_exploration_priority`, `compute_exploitation_priority` can be overridden to implement different strategies for computing the priority and selecting the best candidate.
    """

    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              # validation
              validate_dataset = None, # same format as train_dataset; if None, use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              num_batches = 1,  # number of batches to use from the dataset in each iteration
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation (set it to be negative to skip the first evaluation)
              num_test_samples: int = 1,  # number of times to evaluate each input; when greater than 1, the scores are averaged.
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer
              validate_exploration_candidates: bool = True,  # whether to validate the proposed parameters for exploration
              use_best_candidate_to_explore: bool = True,  # whether to use the best candidate as part of the exploration candidates
              memory_size: Optional[int] = None,  # size of the heap memory to store the candidates; if None, no limit is set
              score_function: str = 'mean',  # function to compute the score for the candidates; 'mean' or 'ucb'
              ucb_exploration_constant: float = 1.0,  # exploration constant for UCB score function
              # Additional keyword arguments
              **kwargs
              ):
        """ Train the agent using the Priority Search algorithm.

        Args:
            guide (callable): A function that provides feedback for the agent.
            train_dataset (list): A list of (x, info) pairs to train the agent.
            validate_dataset (list, optional): A list of (x, info) pairs to validate the proposed candidates. If None, the current batch is used. Defaults to None.
            validate_guide (callable, optional): A function that provides feedback for the validation set. If None, the training guide is used. Defaults to None.
            batch_size (int, optional): The batch size for updating the agent. Defaults to 1.
            num_batches (int, optional): The number of batches to use from the dataset in each iteration. Defaults to 1.
            score_range (tuple, optional): A tuple of (min_score, max_score) to clip the scores. If None, no clipping is applied. Defaults to None.
            num_epochs (int, optional): The number of training epochs. Defaults to 1.
            num_threads (int, optional): The maximum number of threads to use. If None, it uses the number of CPU cores. Defaults to None.
            verbose (bool, optional): Whether to print the output of the agent. Defaults to False.
            test_dataset (list, optional): A list of (x, info) pairs to evaluate the agent. If None, no evaluation is performed. Defaults to None.
            test_frequency (int or None, optional): The frequency of evaluation. If None, no evaluation is performed. If negative, skips the first evaluation. Defaults to 1.
            num_test_samples (int, optional): The number of times to evaluate each input; when greater than 1, the scores are averaged. Defaults to 1.
            log_frequency (int or None, optional): The frequency of logging. If None, no logging is performed. Defaults to None.
            save_frequency (int or None, optional): The frequency of saving the agent. If None, no saving is performed. Defaults to None.
            save_path (str, optional): The path to save the agent. Defaults to "checkpoints/agent.pkl".
            num_candidates (int, optional): The number of candidates to propose for exploration. Defaults to 10.
            num_proposals (int, optional): The number of proposals to generate per optimizer. Defaults to 1.
            validate_exploration_candidates (bool, optional): Whether to validate the proposed parameters for exploration. Defaults to True.
            use_best_candidate_to_explore (bool, optional): Whether to use the best candidate as part of the exploration candidates. Defaults to True.
            memory_size (int, optional): The size of the heap memory to store the candidates. If None, no limit is set. Defaults to None.
            score_function (str, optional): The function to compute the score for the candidates; 'mean' or 'ucb'. Defaults to 'mean'.
            ucb_exploration_constant (float, optional): The exploration constant for UCB score function. Defaults to 1.0.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        """


        # Create agents and optimizers for search
        if num_candidates < len(self._optimizers):
            print(f"Warning: num_candidates {num_candidates} is less than the number of optimizers {len(self._optimizers)}. Setting num_candidates to {len(self._optimizers)}.")
            num_candidates = len(self._optimizers)
        self.num_candidates = num_candidates  # number of candidates for exploration
        self.num_proposals = num_proposals  # number of candidates to propose by each optimizer call

        self.validate_exploration_candidates = validate_exploration_candidates  # whether to validate the proposed parameters
        self.use_best_candidate_to_explore = use_best_candidate_to_explore
        self.score_function = score_function  # function to compute the score for the candidates
        if score_range is None:
            score_range = (0, 1)
        if score_function == 'ucb':  # this requires a bounded score range. By default, it is set to (0, 1)
            assert score_range[1]-score_range[0] < float('inf'), \
                "For UCB score function, score_range must be finite. Use 'mean' score function if you want to use unbounded scores."

        self.ucb_exploration_constant = ucb_exploration_constant
        self._exploration_candidates = None  # This stores the latest candidates used for exploration
        self._best_candidate = None  # This stores the latest best candidate used for exploitation

        self.memory = HeapMemory(size=memory_size)  # Initialize the heap memory with a size limit
        self.regressor = ModuleCandidateRegressor(memory=self.memory) # Initialize the

        super().train(guide=guide,
                      train_dataset=train_dataset,
                      validate_dataset=validate_dataset,
                      validate_guide=validate_guide,
                      batch_size=batch_size,
                      num_batches=num_batches,
                      score_range=score_range,
                      num_epochs=num_epochs,
                      num_threads=num_threads,
                      verbose=verbose,
                      test_dataset=test_dataset,
                      test_frequency=test_frequency,
                      num_test_samples=num_test_samples,
                      log_frequency=log_frequency,
                      save_frequency=save_frequency,
                      save_path=save_path,
                      **kwargs)

    def update(self,
               samples: Union[Samples, None] = None,
               verbose: bool = False,
               **kwargs): #-> Tuple[Dict[ParameterNode, Any], List[trace.Module], Dict[str, Any]]:
        """ Update the agent using the collected samples.
        """

        # samples is None in the first iteration
        if samples is not None:
            # 1. Propose new parameters based on running LLM optimizers on the collected samples
            
            candidates = retry_with_exponential_backoff(
                lambda: self.propose(samples, verbose=verbose, **kwargs),
                max_retries=10,
                base_delay=1.0,
                operation_name="propose_new_parameters"
            )  # List of ModuleCandidates
            # # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information

        else:  # The first iteration.
            max_mem_size = self.memory.size if self.memory.size is not None else float('inf')
            initial_update_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}
            while len(self.memory) < min(max_mem_size, self.num_candidates):
                self.memory.push(self.max_score, ModuleCandidate(self.agent, initial_update_dict, optimizer=self.optimizer))  # Push the base agent as the first candidate (This gives the initialization of the priority queue)
        self.update_memory_with_regressor()
        # 4. Explore and exploit the priority queue
        self._best_candidate, info_exploit = self.exploit(verbose=verbose, **kwargs)  # get the best candidate (ModuleCandidate) from the priority queue
        self._exploration_candidates, info_explore = self.explore(verbose=verbose, **kwargs)  # List of ModuleCandidates
        # TODO Log information about the update
        info_log = {
            'n_iters': self.n_iters,  # number of iterations
        }

        info_log.update(info_exploit)  # add the info from the exploit step
        info_log.update(info_explore)  # add the info from the explore step
        return self._best_candidate.update_dict, [c.get_module() for c in self._exploration_candidates], info_log


    ## Illustration of `propose``
    # Suppose we have 2 exploration candidates.
    # exploration_candidates = [candidate(param1, optimizer_1), candidate(param2, optimizer_2)]
    # and two batches are collected by sampler.
    #
    # In samples returned by sampler, we have data
    #   module(param1_copy1), batch_1
    #   module(param1_copy2), batch_2
    #   module(param2_copy1), batch_1
    #   module(param2_copy2), batch_2
    #
    # We first match the samples with the exploration candidates as
    #   candidate_batchrollouts_list =
    #       [ (candidate(param1, optimizer_1), batch_1), (candidate(param1, optimizer_1), batch_2),
    #         (candidate(param2, optimizer_2), batch_1), (candidate(param2, optimizer_2), batch_2) ]
    #
    # In backward, we create deepcopies of the optimizers for each batch, and run backward asynchronously.
    #    optimizer_1_copy_1(param1) <- feedback from batch_1
    #    optimizer_1_copy_2(param1) <- feedback from batch_2
    #    optimizer_2_copy_1(param2) <- feedback from batch_1
    #    optimizer_2_copy_2(param2) <- feedback from batch_2
    #
    # In step, we further create deepcopies of the optimizers for each proposal, and run step asynchronously.
    # for n_proposals = 2, we have
    #    optimizer_1_copy_1_copy_1(param1) -> proposal_1
    #    optimizer_1_copy_1_copy_2(param1) -> proposal_2
    #    ...
    #    optimizer_2_copy_2_copy_1(param2) -> proposal_7
    #    optimizer_2_copy_2_copy_2(param2) -> proposal_8
    # which form the new candidate list returned by `propose`.
    #
    def propose(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """ Analyzing samples and propose new parameters using self.optimizer. An independent optimizer is used for the minibatch generated by one agent and generates n_proposals proposals.

        Args:
            samples (Samples): Samples collected by the exploration candidates. If None, the agent's parameters are returned without updating.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.

        Returns:
            candidates (list of ModuleCandidate): A list of proposed candidates for the next iteration.
        """
        print("--- Proposing new parameters...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        samples = samples.samples  # list of BatchRollout objects
        n_proposals = self.num_proposals  # number of proposals to generate per optimizer

        # Associate each BatchRollout with self._exploration_candidates
        matched_candidates_and_samples = self.match_candidates_and_samples(self._exploration_candidates, samples)
        # NOTE len(matched_candidates_and_samples) <= len(self._exploration_candidates) since some exploration candidates might be duplicated.
        candidate_batchrollouts_list = [ (k,b) for k, v in matched_candidates_and_samples.items() for b in v]
        assert len(samples) == len(candidate_batchrollouts_list), "All samples must be associated with exploration candidates."
        n_batches = len(samples)  # number of batch rollouts in the samples

        # need to copy optimizer for the n_batches
        def _backward(n):
            candidate, rollouts = candidate_batchrollouts_list[n]
            optimizer = candidate.optimizer or self.optimizer
            # Create a copy of the optimizer to avoid modifying the original one and to allow parallel execution
            optimizer = copy.deepcopy(optimizer)
            optimizer.parameters = rollouts.module.parameters()  # set the optimizer's parameters to the proposal's parameters
            targets = [r.target for r in rollouts]
            feedbacks = [r.feedback for r in rollouts]
            # batchify the targets and feedbacks
            target = batchify(*targets)
            feedback = batchify(*feedbacks).data  # str
            # standard optimizer step
            optimizer.zero_feedback()  # reset the optimizer's feedback
            optimizer.backward(target, feedback)  # compute the gradients based on the targets and feedbacks
            return optimizer

        args_list = [(n,) for n in range(n_batches)]
        optimizers = async_run([_backward]*n_batches,  # run the optimizer step for each agent in parallel
                                 args_list=args_list,
                                 max_workers=self.num_threads,  # use the number of threads specified in the class
                                 description=None)
        assert len(optimizers) == n_batches, "Number of optimizers must match number of batch rollouts."
        # need to copy optimizer for the n_proposals
        # NOTE when optimizer is deepcopied, its parameters are not copied.
        optimizers = [copy.deepcopy(o) for o in optimizers ] * n_proposals  # repeat args_list n_proposals times
        assert len(optimizers) == n_batches * n_proposals, "Number of optimizers must match number of batch rollouts times number of proposals."

        # For each optimizer, containing the backward feedback, we call it n_proposals times to get the proposed parameters.
        def _step(n):
            optimizer = optimizers[n]
            update_dict = optimizer.step(verbose=verbose, num_threads=self.num_threads, bypassing=True, **kwargs)
            if not update_dict:  # if the optimizer did not propose any updates
                return None # return None to indicate no updates were proposed
            # update_dict may only contain some of the parameters of the agent, we need to make sure it contains all the parameters
            # since the current agent might have different parameters than the one used by the optimizer
            for param in optimizer.parameters: # for all parameters
                if param not in update_dict: # update_dict misses some parameters
                    update_dict[param] = param.data # add the parameter to the update_dict
            # the update_dict is linked to the copied parameters of the agent, we set it back to the agent's parameters
            update_dict = remap_update_dict(self.agent, update_dict)  # remap the update dict to the agent's parameters
            return update_dict  # return the proposed parameters

        args_list = [(n,) for n in range(n_batches*n_proposals)]
        update_dicts = async_run([_step]*n_batches*n_proposals,  # run the optimizer step for each agent in parallel
                                  args_list=args_list,
                                  max_workers=self.num_threads,  # use the number of threads specified in the class
                                  description=f"Calling optimizers: Generating {n_proposals} proposals for each of {n_batches} batches",)

        # update_dicts is a list of dicts of length n_batches * n_proposals
        # Create ModuleCandidate objects for each proposed update_dict that is non-trivial
        candidates = [ModuleCandidate(self.agent, update_dict, optimizer)
                        for update_dict, optimizer in zip(update_dicts, optimizers) if update_dict is not None]  # filter out None updates
        return candidates

    def validate(self,
                 candidates: List[ModuleCandidate],
                 samples: Samples,
                 verbose: bool = False,
                 **kwargs):
        """ Validate the proposed candidate parameters
        Args:
            candidates (list of ModuleCandidate): A list of ModuleCandidate objects representing the proposed parameters.
            samples (list of dict, optional): A list of samples collected in the current iteration. Defaults to None.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            results (dict): A dictionary where the keys are ids of ModuleCandidate objects and the values are ModuleCandidate and lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
        """
        print("--- Validating candidates...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        exploration_candidates = self._exploration_candidates  # exploration candidates from the previous iteration
        assert self._exploration_candidates is not None, "exploration_candidates must be set before calling validate."

        # The current batch of samples can be used to validate the exploration candidates
        validate_samples = copy.copy(samples)
        # Xuanfei: I commented all these below, only use training samples.
        # # Validate newly proposed candidates
        # use_prev_batch = self.use_prev_batch  # when True, self.validate_sampler == self.train_sampler, and the current batch is used for validation
        # candidate_agents = [c.get_module() for c in candidates]  # get the modules from the candidates
        # validate_samples.add_samples(Samples(*self.validate_sampler.sample(candidate_agents,
        #                                                         use_prev_batch=use_prev_batch,
        #                                                         description_prefix='Validating newly proposed candidates: ')))  # list of BatchRollout objects

        # if self.validate_exploration_candidates:
        #     if not use_prev_batch:   # validate the exploration candidates that collected the samples as well
        #         # validate the agents in the validate_dataset
        #         exploration_agents = [c.get_module() for c in exploration_candidates]  # get the modules from the exploration candidates
        #         exploration_samples = Samples(*self.validate_sampler.sample(exploration_agents,
        #                                                     description_prefix='Validating exploration candidates: '))  # sample the exploration agents
        #         validate_samples.add_samples(exploration_samples)  # append the exploration samples to the validate_samples


        matched_candidates_and_samples = self.match_candidates_and_samples(exploration_candidates + candidates, validate_samples.samples)
        results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c, rollouts in matched_candidates_and_samples.items():  # rollouts is a list of BatchRollouts
            results[c] = [ r for rr in rollouts for r in rr.to_list()]  # we only need the list of dicts

        return results

    def match_candidates_and_samples(
            self,
            candidates: List[ModuleCandidate],
            samples: List[BatchRollout]):
        """
        Match the given candidates with the provided samples.

        Args:
            candidates (list of ModuleCandidate): A list of ModuleCandidate objects representing the proposed parameters.
            samples (list of BatchRollout): A Samples object containing a list of BatchRollout objects, where each BatchRollout contains rollouts collected by an agent on different inputs.
        Returns:
            results (dict): A dictionary where the keys are ModuleCandidate objects and the values are lists of BatchRollouts collected by the corresponding ModuleCandidate.

        """
        # In general, there may be multiple BatchRollouts collected by the same ModuleCandidate.
        # We group the rollouts by the agent (ModuleCandidate) and return a dictionary
        # where the keys are the ModuleCandidate objects and the values are Samples

        # Group the samples by the ModuleCandidate id
        _results = { c: [] for c in candidates}  # dict of ModuleCandidate: list of BatchRollouts
        ids = {id(c): c for c in candidates}  # dict of ModuleCandidate id: ModuleCandidate

        for rollouts in samples:
            assert isinstance(rollouts, BatchRollout), "Each element in samples must be a BatchRollout object."
            # rollouts is a BatchRollout object
            module = rollouts.module  # trace.Module
            key = getattr(module, '__TRACE_RESERVED_module_candidate_id')  # use the candidate as the key
            if key not in ids:
                raise ValueError(f"ModuleCandidate with id {key} not found in results. Samples are not collected by known candidates.")
            # Append the rollouts to the list of rollouts for the key
            _results[ids[key]].append(rollouts)
        # assert all candidates have at least one rollout
        # Xuanfei: some candidates may not have rollouts
        # for c in candidates:
        #     assert len(_results[c]) > 0, f"ModuleCandidate with id {id(c)} has no rollouts. Samples are not collected by known candidates."

        return _results

    def update_memory(self, validate_results, verbose: bool = False, **kwargs):
        """ Update the priority queue with the validation results.
        Args:
            validate_results (dict): A dictionary where the keys are ModuleCandidate objects and the values are lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        """
        print("--- Updating memory with validation results...") if verbose else None
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the candidate

            # priority = self.compute_exploration_priority(candidate)  # compute the priority for the candidate
            placeholder_priority = self.max_score
            self.memory.push(placeholder_priority, candidate)

    def update_memory_with_regressor(self, verbose: bool = False, **kwargs):
        """ Update the priority queue with the regressor results.
        """
        print("--- Updating memory with regressor results...") if verbose else None
        # Update predicted scores for all candidates in the memory
        self.regressor.predict_scores()
        # Reorder the memory according to the predicted scores
        self.memory.reorder_according_to_predicted_scores()
        # For debugging, print the memory stats
        self.print_memory_stats()

    def print_memory_stats(self):
        # For debugging, print all candidates: number, mean_score(), num_rollouts, predicted_score
        for i, (neg_predicted_score, candidate) in enumerate(self.memory):
            print(f"Candidate {i}, Mean Score: {candidate.mean_score()}, Num Rollouts: {candidate.num_rollouts}, Predicted Score: {-neg_predicted_score}")

    def explore(self, verbose: bool = False, **kwargs):
        """ Explore the parameter space and propose new candidates.
        Args:
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            list: A list of proposed candidates.
            dict: A dictionary containing logging information about the exploration.
        """
        print(f"--- Generating {min(len(self.memory), self.num_candidates)} exploration candidates...")  if verbose else None
        # pop top self.num_candidates candidates from the priority queue
        # self._best_candidate is the exploited candidate from the previous iteration
        top_candidates = [self._best_candidate] if self.use_best_candidate_to_explore else []
        priorities = []  # to store the priorities of the candidates for logging
        while len(top_candidates) < self.num_candidates and len(self.memory) > 0:
            neg_priority, candidate = self.memory.pop()  # pop the top candidate from the priority queue
            priority = - neg_priority  # remember that we stored negative scores in the priority queue
            if self.use_best_candidate_to_explore:
                if candidate is self._best_candidate:  # skip if it is already in the top candidates
                    continue
            priorities.append(priority)  # store the priority of the candidate
            top_candidates.append(candidate)  # add the candidate to the top candidates
        # NOTE some top_candidates can be duplicates
        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [s for s in mean_scores if s is not None]  # filter out None scores
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': np.mean(priorities),  # list of priorities of the exploration candidates
            'exploration_candidates_mean_score': np.mean(mean_scores) if mean_scores else None,  # list of mean scores of the exploration candidates
            'exploration_candidates_average_num_rollouts': np.mean([c.num_rollouts for c in top_candidates]),
        }

        return top_candidates, info_dict

    def exploit(self, verbose: bool = False, **kwargs) -> Tuple[ModuleCandidate, Dict[str, Any]]:
        """ Exploit the best candidate from the priority queue. This method should not change the priority queue.
        Args:
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            ModuleCandidate: The best candidate from the priority queue.
        """
        print("--- Exploiting the best candidate...") if verbose else None
        if not self.memory:
            raise ValueError("The priority queue is empty. Cannot exploit.")
        neg_priority, best_candidate = self.memory.best(self.compute_exploitation_priority)  # (priority, candidate)
        priority = - neg_priority # remember that we stored negative scores in the priority queue
        return best_candidate, {
            'best_candidate_priority': priority,  # remember that we stored negative scores in the priority queue
            'best_candidate_mean_score': best_candidate.mean_score(),  # mean score of the candidate's rollouts
            'best_candidate_num_rollouts': best_candidate.num_rollouts,  # number of rollouts of the candidate
        }

    # TODO refactor below to reuse scoring
    def compute_exploitation_priority(self, candidate) -> float:
        # NOTE This function can be overridden by subclasses to compute a different score
        """ Compute the score for the candidate based on the rollouts during the validation phase.
        It can be overridden by subclasses to implement a different scoring strategy.

        Args:
            candidate (ModuleCandidate): The candidate for which to compute the score.
        Returns:
            float: The computed score for the candidate. Higher scores indicate higher priority.
        """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # By default, we compute the mean score of the rollouts
        return candidate.predicted_score

    def compute_exploration_priority(self, candidate) -> float:
        # NOTE This function can be overridden by subclasses to compute a different score
        """ Compute the score for the candidate based on the rollouts during the validation phase.
        It can be overridden by subclasses to implement a different scoring strategy.

        Args:
            candidate (ModuleCandidate): The candidate for which to compute the score.
        Returns:
            float: The computed score for the candidate. Higher scores indicate higher priority.
        """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # By default, we compute the mean score of the rollouts

        if self.score_function == 'mean':
            # Compute the mean score of the candidate's rollouts
            return candidate.mean_score()
        elif self.score_function == 'time':
            return -candidate.created_time  # latest candidates have higher priority
        elif self.score_function == 'ucb':
            # Compute the Upper Confidence Bound (UCB) score
            lcb_score, mean_score, ucb_score = candidate.compute_score_confidence(
                min_score=self.min_score,
                max_score=self.max_score,
                scaling_constant=self.ucb_exploration_constant,
                total_trials=self.n_iters + 1  # total number of trials conducted so far
            )
            return ucb_score  # return the UCB score
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")
