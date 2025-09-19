import numpy as np
import copy
import heapq
import time
from typing import Union, List, Tuple, Dict, Any, Optional
from opto import trace
from opto.trace.nodes import ParameterNode
from opto.trainer.utils import async_run
from opto.trainer.algorithms.basic_algorithms import batchify
from opto.features.priority_search.search_template import SearchTemplate, Samples
from opto.features.priority_search.utils import set_module_parameters, remap_update_dict, create_module_from_update_dict


class ModuleCandidate:
    """Container for storing candidate modules with their parameters and performance statistics.
    
    This class represents a candidate agent configuration consisting of a base module
    and an update dictionary that modifies its parameters. It tracks performance
    statistics through rollouts and provides confidence interval calculations.

    Parameters
    ----------
    base_module : trace.Module
        The base module to use as a template for the candidate.
    update_dict : dict[ParameterNode, Any], optional
        Dictionary of parameter updates to apply to the base module, by default None.

    Attributes
    ----------
    base_module : trace.Module
        The original module template.
    update_dict : dict[ParameterNode, Any]
        Parameter updates mapped to the base module's parameters.
    rollouts : list[dict]
        Performance statistics from agent evaluations.
    created_time : float
        Timestamp when the candidate was created.

    Notes
    -----
    The update dictionary is automatically remapped to ensure compatibility with
    the base module's parameter structure. Rollouts store detailed execution
    information including modules, inputs, targets, scores, and feedback.
    """

    def __init__(self,
                 base_module: Optional[trace.Module],
                 update_dict: Optional[Dict[ParameterNode, Any]] = None,
                 ):
        """Initialize a module candidate with base module and parameter updates.

        Parameters
        ----------
        base_module : trace.Module
            The base module to use as a template for the candidate.
        update_dict : dict[ParameterNode, Any], optional
            Dictionary of parameter updates to apply, by default None.

        Raises
        ------
        AssertionError
            If base_module is not a trace.Module instance.

        Notes
        -----
        The update dictionary is automatically remapped to ensure parameter
        compatibility with the base module. Internal tracking variables for
        confidence calculations are initialized.
        """
        assert isinstance(base_module, trace.Module), "base_module must be a trace.Module."
        self.base_module = base_module
        self.update_dict = update_dict if update_dict is not None else {}
        self.update_dict = remap_update_dict(self.base_module, self.update_dict)
        self.rollouts = []  # list of dicts containing the rollout information (not RolloutsGraph, but a list of dicts)
        self.created_time = time.time()
        self._n_updates = 0  # number of times this candidate has been updated
        self._n_confidence_queries = 1  # number of times the confidence score has been queried
        self._confidence_interval = None

    def get_module(self):
        """Create and return an updated module with applied parameter changes.

        Returns
        -------
        trace.Module
            New module instance with parameters updated according to update_dict.

        Notes
        -----
        A new module is always created to avoid modifying the base module.
        The returned module includes a special attribute marking its candidate ID
        for tracking purposes in the priority search algorithm.
        """
        module = create_module_from_update_dict(self.base_module, self.update_dict) if self.update_dict else copy.deepcopy(self.base_module)  #
        setattr(module, '__TRACE_RESERVED_module_candidate_id', id(self))
        return module  # return the updated module

    def apply_update(self, base_module=None):
        """Apply parameter updates to a module in place.

        Parameters
        ----------
        base_module : trace.Module, optional
            Module to update, uses self.base_module if None, by default None.

        Notes
        -----
        This method modifies the target module's parameters directly using
        the stored update dictionary.
        """
        set_module_parameters(base_module or self.base_module, self.update_dict)

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
        return self.update_dict == other.update_dict

    # TODO better way?
    def __lt__(self, other):
        """ Compare two candidates based on their update_dict. """
        assert isinstance(other, ModuleCandidate), "other must be an instance of ModuleCandidate."
        return self.created_time > other.created_time
        # This would give priority to later created candidates in the heap memory
        # since the heapq is a min-heap .

    def __hash__(self):
        """ Hash the candidate based on its update_dict. """
        return hash(frozenset(self.update_dict.items()))

    def add_rollouts(self, rollouts: List[Dict[str, Any]]):
        """Add performance rollouts to the candidate for statistics tracking.

        Parameters
        ----------
        rollouts : list[dict[str, Any]]
            List of rollout dictionaries containing execution results.

        Raises
        ------
        AssertionError
            If rollouts is not a list or contains non-dict elements.
            If rollout dicts missing required keys: 'module', 'x', 'info', 
            'target', 'score', 'feedback'.

        Notes
        -----
        Each rollout dictionary must contain complete execution information.
        Adding rollouts resets the confidence interval cache and increments
        the update counter.
        """
        assert isinstance(rollouts, list), "rollouts must be a list of dicts."
        assert all(isinstance(r, dict) for r in rollouts), "All rollouts must be dicts."
        # Each rollout is a dict with keys: 'module', 'x', 'info', 'target', 'score', 'feedback'
        assert all('module' in r and 'x' in r and 'info' in r and 'target' in r and 'score' in r and 'feedback' in r for r in rollouts), \
            "Each rollout must contain 'module', 'x', 'info', 'target', 'score', and 'feedback' keys."

        self.rollouts.extend(rollouts)
        self._confidence_interval = None  # reset the confidence interval
        self._n_updates += 1  # increment the number of updates

    def mean_score(self):
        """Calculate the mean performance score from rollout statistics.

        Returns
        -------
        float or None
            Average score across all rollouts, or None if no rollouts exist.

        Notes
        -----
        This is the primary performance metric used for ranking candidates
        in the priority queue.
        """
        if not self.rollouts:
            return None
        scores = [r['score'] for r in self.rollouts]
        return np.mean(scores) if scores else None

    def compute_score_confidence(self, min_score, max_score, scaling_constant=1.0):
        """Compute Upper and Lower Confidence Bounds for multi-armed bandit selection.

        Calculates confidence intervals using Hoeffding's inequality to balance
        exploration and exploitation in candidate selection.

        Parameters
        ----------
        min_score : float
            Minimum possible score value for clipping.
        max_score : float
            Maximum possible score value for clipping.
        scaling_constant : float, optional
            Exploration constant controlling confidence width, by default 1.0.

        Returns
        -------
        tuple[float, float, float]
            Lower confidence bound, mean score, upper confidence bound.

        Notes
        -----
        Uses the formula:
        - UCB = mean + scaling * sqrt(ln(total_queries) / trials) * (max - min)
        - LCB = mean - scaling * sqrt(ln(total_queries) / trials) * (max - min)
        
        Both bounds are clipped to [min_score, max_score]. The confidence query
        counter is incremented after each call for proper union bound calculation.
        """
        # Get scores from rollouts
        scores = [r['score'] for r in self.rollouts]

        # If no rollouts, return a high exploration score to encourage trying this candidate
        if not scores:
            return min_score, None, max_score

        # Calculate mean score for this candidate
        mean_score = np.mean(scores)
        candidate_trials = len(scores)

        # Calculate how many times the confidence interval has been used to form a union bound
        total_trials = min(self._n_confidence_queries) + 1 # this is an upper bound, since log(1) = 0

        # Compute the exploration term based on Hoeffding's inequality
        exploration_term = scaling_constant * np.sqrt(np.log(total_trials) / candidate_trials) * (max_score - min_score)

        # Calculate UCB score
        ucb_score = mean_score + exploration_term
        ucb_score = np.clip(ucb_score, min_score, max_score)

        # Calculate LCB score
        lcb_score = mean_score - exploration_term
        lcb_score = np.clip(lcb_score, min_score, max_score)

        self._n_confidence_queries += 1  # increment the number of confidence queries

        self._confidence_interval = dict(lcb_score=lcb_score, ucb_score=ucb_score, mean_score=mean_score)
        return lcb_score, mean_score, ucb_score

    @property
    def confidence_interval(self):
        # This is a cached property that returns the confidence interval of the candidate.
        # This is for accessing the confidence interval without increasing the number of confidence queries. E.g. this is useful when using both LCB and UCB of the same candidate.
        if self._confidence_interval is None:
            raise ValueError("Confidence interval has not been computed yet. Call compute_score_confidence() first.")
        return self._confidence_interval

    @property
    def num_rollouts(self):
        """ Return the number of rollouts collected for this candidate. """
        return len(self.rollouts)

    @property
    def n_updates(self):
        """ Return the number of times this candidate has been updated. """
        return self._n_updates

class HeapMemory:
    """Priority queue implementation for storing and retrieving module candidates.
    
    This class provides a max-heap interface using Python's min-heap heapq module
    by storing negative scores. It maintains the best-performing candidates with
    optional size limits for memory efficiency.

    Parameters
    ----------
    size : int, optional
        Maximum number of items to store in the heap, by default None (unlimited).

    Attributes
    ----------
    memory : list
        Internal heap storage containing (negative_score, candidate) tuples.
    size : int or None
        Maximum heap size limit.

    Notes
    -----
    Since heapq implements a min-heap, scores are stored as negative values to
    achieve max-heap behavior. This ensures the highest-scoring candidates are
    prioritized for selection.
    """
    def __init__(self, size=None):
        """Initialize an empty heap memory with optional size limit.

        Parameters
        ----------
        size : int, optional
            Maximum number of items to store, by default None (unlimited).
        """
        self.memory = []
        self.size = size  # Optional size limit for the heap memory

    def push(self, score, data):
        """Add an item to the heap memory with the given score.

        Parameters
        ----------
        score : float
            Priority score for the item (higher scores have higher priority).
        data : Any
            The item to store in the heap.

        Notes
        -----
        The score is negated before storage to achieve max-heap behavior.
        If the heap exceeds the size limit, it's truncated to maintain the limit.
        """
        heapq.heappush(self.memory, (-score, data))
        if self.size is not None and len(self.memory) > self.size:
            # NOTE a heuristic for now
            self.memory = self.memory[:self.size]  # Keep only the top `size` items

    def pop(self):
        """Remove and return the highest priority item from the heap.

        Returns
        -------
        tuple[float, Any]
            The (negative_score, data) tuple of the highest priority item.

        Raises
        ------
        IndexError
            If the heap is empty.
        """
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

    def best(self):
        """Return the highest priority item without removing it from the heap.

        Returns
        -------
        tuple[float, Any]
            The (negative_score, data) tuple of the highest priority item.

        Raises
        ------
        IndexError
            If the heap is empty.
        """
        if not self.memory:
            raise IndexError("best from an empty heap memory")
        return self.memory[0]


class PrioritySearch(SearchTemplate):
    """Priority-based search algorithm for exploring parameter space and optimizing agents.
    
    This algorithm uses a priority queue to systematically explore the parameter space
    through a cycle of proposal generation, validation, and candidate ranking. It balances
    exploration of new parameter configurations with exploitation of high-performing ones.

    The algorithm operates in iterative cycles:
    
    1. **Exploitation**: Select the best-performing candidate from the priority queue
    2. **Exploration**: Choose top candidates for parameter space exploration
    3. **Proposal Generation**: Use optimizers on collected samples to propose new parameters
    4. **Validation**: Evaluate proposed parameters on validation data
    5. **Memory Update**: Update priority queue with validation results
    
    Each iteration processes minibatches of training data, creating rollout graphs that
    capture agent execution statistics. These rollouts inform the optimization process
    and candidate evaluation.

    Attributes
    ----------
    memory : HeapMemory
        Priority queue storing module candidates with their performance scores.
    num_candidates : int
        Number of exploration candidates to select per iteration.
    num_proposals : int
        Number of parameter proposals per optimizer call.
    score_function : str
        Scoring method for candidate ranking ('mean' or 'ucb').
    ucb_exploration_constant : float
        Exploration parameter for Upper Confidence Bound scoring.

    Notes
    -----
    The algorithm can be customized by overriding key methods:
    - `exploit`: Strategy for selecting the best candidate
    - `explore`: Strategy for selecting exploration candidates  
    - `compute_priority`: Scoring function for candidate ranking
    
    Default implementations use mean rollout scores for ranking, simple best-candidate
    exploitation, and top-k candidate exploration.
    """

    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              # validation
              validate_dataset = None, # same format as train_dataset; if None use the current batch.
              validate_guide = None,  #  to provide scores for the validation set
              # training loop
              batch_size = 1,  # batch size for updating the agent
              sub_batch_size = None,  # sub-batch size that each optimizer attends to
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer
              validate_proposals: bool = True,  # whether to validate the proposed parameters for exploration
              use_best_candidate_to_explore: bool = True,  # whether to use the best candidate as part of the exploration candidates
              memory_size: Optional[int] = None,  # size of the heap memory to store the candidates; if None, no limit is set
              score_function: str = 'mean',  # function to compute the score for the candidates; 'mean' or 'ucb'
              ucb_exploration_constant: float = 1.0,  # exploration constant for UCB score function
              # Additional keyword arguments
              **kwargs
              ):
        """Train the agent using priority-based parameter space search.

        This method orchestrates the complete training process using a priority queue
        to guide parameter exploration and optimization.

        Parameters
        ----------
        guide : Guide
            Guide function to provide feedback during training.
        train_dataset : dict
            Training dataset containing 'inputs' and 'infos' keys.
        validate_dataset : dict, optional
            Validation dataset, uses current batch if None, by default None.
        validate_guide : Guide, optional
            Guide for validation scoring, uses train guide if None, by default None.
        batch_size : int, optional
            Batch size for agent updates, by default 1.
        sub_batch_size : int, optional
            Sub-batch size for optimizer attention, by default None.
        score_range : tuple[float, float], optional
            Score range for UCB calculations, by default None.
        num_epochs : int, optional
            Number of training epochs, by default 1.
        num_threads : int, optional
            Maximum threads for parallel processing, by default None.
        verbose : bool, optional
            Enable verbose output, by default False.
        test_dataset : dict, optional
            Test dataset for evaluation, by default None.
        test_frequency : int, optional
            Frequency of test evaluation, by default 1.
        num_eval_samples : int, optional
            Samples per input for evaluation, by default 1.
        log_frequency : int, optional
            Logging frequency, by default None.
        save_frequency : int, optional
            Model saving frequency, by default None.
        save_path : str, optional
            Path for saving checkpoints, by default "checkpoints/agent.pkl".
        num_candidates : int, optional
            Number of exploration candidates per iteration, by default 10.
        num_proposals : int, optional
            Number of proposals per optimizer call, by default 1.
        validate_proposals : bool, optional
            Whether to validate exploration candidates, by default True.
        use_best_candidate_to_explore : bool, optional
            Include best candidate in exploration set, by default True.
        memory_size : int, optional
            Maximum memory size for candidate storage, by default None.
        score_function : str, optional
            Scoring function ('mean' or 'ucb'), by default 'mean'.
        ucb_exploration_constant : float, optional
            UCB exploration parameter, by default 1.0.
        **kwargs
            Additional arguments passed to parent class.

        Notes
        -----
        The UCB score function requires a finite score_range for proper calculation.
        If score_range is None and UCB is selected, it defaults to (0, 1).
        """

        # Create agents and optimizers for search
        self.num_candidates = num_candidates  # number of candidates to propose by each optimizer call
        self.num_proposals = num_proposals
        self.validate_proposals = validate_proposals  # whether to validate the proposed parameters
        self.use_best_candidate_to_explore = use_best_candidate_to_explore
        self.score_function = score_function  # function to compute the score for the candidates
        if score_function == 'ucb':  # this requires a bounded score range. By default, it is set to (0, 1)
            if score_range is None:
                score_range = (0, 1)
            assert score_range[1]-score_range[0] < float('inf'), \
                "For UCB score function, score_range must be finite. Use 'mean' score function if you want to use unbounded scores."

        self.ucb_exploration_constant = 1.
        self._exploration_candidates = None

        self.memory = HeapMemory(size=memory_size)  # Initialize the heap memory with a size limit


        super().train(guide, train_dataset,
                      validate_dataset=validate_dataset,
                      validate_guide=validate_guide,
                      batch_size=batch_size,
                      sub_batch_size=sub_batch_size,
                      score_range=score_range,
                      num_epochs=num_epochs,
                      num_threads=num_threads,
                      verbose=verbose,
                      test_dataset=test_dataset,
                      test_frequency=test_frequency,
                      num_eval_samples=num_eval_samples,
                      log_frequency=log_frequency,
                      save_frequency=save_frequency,
                      save_path=save_path,
                      **kwargs)


    def update(self, samples=None, verbose=False, **kwargs):

        if samples is not None:
            # 1. Propose new parameters based on running LLM optimizers on the collected samples
            candidates = self.propose(samples, verbose=verbose, **kwargs)  # List of ModuleCandidates
            # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information
        else:
            if len(self.memory) == 0:
                self.memory.push(self.max_score, ModuleCandidate(self.agent))  # Push the base agent as the first candidate (This gives the initialization of the priority queue)
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

    def propose(self, samples, verbose=False, **kwargs):
        """ Analyzing samples and propose new parameters using self.optimizer. An independent optimizer is used for the minibatch generated by one agent and generates n_proposals proposals.

        Args:
            samples (list): A list of samples from the previous iteration. If None, the agent's parameters are returned without updating.
            n_proposals (int): Number of proposals to generate per optimizer. Defaults to 1.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
            **kwargs: Additional keyword arguments that may be used by the implementation.

        Returns:
            candidates (list of ModuleCandidate): A list of proposed candidates for the next iteration.
        """
        print("--- Proposing new parameters...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        samples = samples.samples  # list of RolloutsGraph objects
        n_proposals = self.num_proposals  # number of proposals to generate per optimizer

        def _backward(n):
            optimizer = copy.deepcopy(self.optimizer)  # create a copy of the optimizer to avoid modifying the original one
            rollouts = samples[n]  # RolloutsGraph
            # Make sure all rollouts are based on the same module, so they can be viewed as a minibatch.
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

        n_subgraphs = len(samples)  # number of subgraphs (agents) in the samples
        args_list = [(n,) for n in range(n_subgraphs)]
        optimizers = async_run([_backward]*n_subgraphs*n_proposals,  # run the optimizer step for each agent in parallel
                                  args_list=args_list,
                                  max_workers=self.num_threads,  # use the number of threads specified in the class
                                  description=None)

        # For each optimizer, containing the backward feedback, we call it n_proposals times to get the proposed parameters.
        def _step(optimizer):
            update_dict = optimizer.step(verbose=verbose, num_threads=self.num_threads, bypassing=True, **kwargs)
            if not update_dict:  # if the optimizer did not propose any updates
                return None # return None to indicate no updates were proposed
            # update_dict may only contain some of the parameters of the agent, we need to make sure it contains all the parameters
            for param in optimizer.parameters: # for all parameters
                if param not in update_dict: # update_dict misses some parameters
                    update_dict[param] = param.data # add the parameter to the update_dict
            # the update_dict is linked to the copied parameters of the agent, we set it back to the agent's parameters
            update_dict = remap_update_dict(self.agent, update_dict)  # remap the update dict to the agent's parameters
            return update_dict  # return the proposed parameters

        args_list = [(o,) for o in optimizers ] * n_proposals  # repeat args_list n_proposals times
        assert len(args_list) == n_subgraphs * n_proposals, "args_list must have length n_subgraphs * n_proposals"
        update_dicts = async_run([_step]*n_subgraphs*n_proposals,  # run the optimizer step for each agent in parallel
                                  args_list=args_list,
                                  max_workers=self.num_threads,  # use the number of threads specified in the class
                                  description=f"Calling optimizers: Generating {n_proposals} proposals for each of {n_subgraphs} sub batches",)

        # update_dicts is a list of dicts of length n_agents * n_proposals
        # Create ModuleCandidate objects for each proposed update_dict
        candidates = [ModuleCandidate(self.agent, update_dict) for update_dict in update_dicts if update_dict is not None]  # filter out None updates
        return candidates

    def validate(self, candidates, samples, verbose=False, **kwargs):
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

        # Get the validation dataset from the samples. If no validation dataset is provided, use the current batch.
        if self._validate_dataset is None:
            # If no validation dataset is provided, use the current batch
            validate_dataset = samples.get_batch()  # get the batch of inputs and infos from the samples
            self.validate_sampler.dataset = validate_dataset  # set the validation dataset in the sampler
            self.validate_sampler.batch_size = len(validate_dataset['inputs'])  # set the batch size to the number of inputs in the validation dataset

        candidate_agents = [c.get_module() for c in candidates]  # get the modules from the candidates
        validate_samples = Samples(*self.validate_sampler.sample(candidate_agents, description_prefix='Validating newly proposed candidates: '))  # list of RolloutsGraph objects


        exploration_candidates = self._exploration_candidates  # exploration candidates from the previous iteration
        assert exploration_candidates is not None, "exploration_candidates must be set before calling validate."
        if self.validate_proposals:
            if self._validate_dataset is None:
                # NOTE this might contain some duplicates due to sub_batch_size < batch_size
                validate_samples.add_samples(samples)  # if no validation dataset is provided, append the samples to the validate_samples
            else:  # validate the agents in the validate_dataset
                # exploration_agents = [rollouts.module for rollouts in samples.samples]  # NOTE this might contain some duplicates due to sub_batch_size < batch_size
                exploration_agents = [c.get_module() for c in exploration_candidates]  # get the modules from the exploration candidates
                exploration_samples = Samples(*self.validate_sampler.sample(exploration_agents, description_prefix='Validating exploration candidates: '))  # sample the exploration agents
                validate_samples.add_samples(exploration_samples)  # append the exploration samples to the validate_samples


        # TODO some ModuleCandidate are the same in parameters though they have different ids

        # In validate_samples, there may be multiple rollouts collected by the same agent (or their copies).
        # We need to group the rollouts by the agent (ModuleCandidate) and return a dictionary where the keys are the ModuleCandidate objects and the values are lists of rollouts (list of dicts).
        # Group the samples by the ModuleCandidate id
        _results = {}  # dict of ModuleCandidate: list of rollouts (list of dicts)
        for c in exploration_candidates + candidates:
            _results[id(c)] = []

        for rollouts in validate_samples.samples:
            module = rollouts.module  # trace.Module
            key = getattr(module, '__TRACE_RESERVED_module_candidate_id')  # use the candidate as the key
            if key not in _results:
                raise ValueError(f"ModuleCandidate with id {key} not found in results. Samples are not collected by known candidates.")
            # Append the rollouts to the list of rollouts for the key
            _results[key].extend(rollouts.to_list())

        # Merge rollouts of ModuleCandidates sharing the same parameters
        results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c in exploration_candidates + candidates:
            rollouts_list = _results[id(c)]
            matched = False
            for k in results.keys():
                if k == c:
                    matched = True
                    if id(k) != id(c):  # merging rollouts of candidates with the same parameters
                        rollouts_list += c.rollouts
                    results[k].extend(rollouts_list)  # add the rollouts to the candidate
                    break
            if not matched:  # key not found in results
                results[c] = rollouts_list  # add the rollouts to the candidate
        # NOTE what if propose creates multiple exploration_candidates that have the same parameters and the same rollouts stats?
        # For example, it copies candidates. This would create a bug.
        return results

    def update_memory(self, validate_results, verbose: bool = False, **kwargs):
        """ Update the priority queue with the validation results.
        Args:
            validate_results (dict): A dictionary where the keys are ModuleCandidate objects and the values are lists of rollouts (list of dicts) containing the module, x, info, target, score, feedback.
            **kwargs: Additional keyword arguments that may be used by the implementation.
        """
        print("--- Updating memory with validation results...") if verbose else None
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the candidate
            priority = self.compute_priority(candidate)  # compute the priority for the candidate
            self.memory.push(priority, candidate)

    ####
    def explore(self, verbose: bool = False, **kwargs):
        """ Explore the parameter space and propose new candidates.
        Args:
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            list: A list of proposed candidates.
            dict: A dictionary containing logging information about the exploration.
        """
        print(f"--- Generating {min(len(self.memory), self.num_candidates)} exploration candidates...") if verbose else None
        # pop top self.num_candidates candidates from the priority queue
        top_candidates = [self._best_candidate] if self.use_best_candidate_to_explore else []
        priorities = []  # to store the priorities of the candidates
        while len(top_candidates) < self.num_candidates and self.memory:
            priority, candidate = self.memory.pop()  # pop the top candidate from the priority queue
            priority = - priority  # remember that we stored negative scores in the priority queue
            priorities.append(priority)  # store the priority of the candidate
            if self.use_best_candidate_to_explore:
                if candidate == self._best_candidate:  # skip if it is already in the top candidates
                    continue
            top_candidates.append(candidate)  # add the candidate to the top candidates

        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [ s for s in mean_scores if s is not None]  # filter out None scores
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': np.mean(priorities),  # list of priorities of the exploration candidates
            'exploration_candidates_mean_score': np.mean(mean_scores),  # list of mean scores of the exploration candidates
        }

        return top_candidates, info_dict


    def exploit(self, verbose: bool = False, **kwargs):
        # NOTE This function can be overridden by subclasses to compute a different score
        """ Exploit the best candidate from the priority queue. This method should not change the priority queue.
        Args:
            **kwargs: Additional keyword arguments that may be used by the implementation.
        Returns:
            ModuleCandidate: The best candidate from the priority queue.
        """
        print("--- Exploiting the best candidate...") if verbose else None
        # Right now, we just return the best candidate from the priority queue
        # This function can be overridden by subclasses to implement a different exploitation strategy
        if not self.memory:
            raise ValueError("The priority queue is empty. Cannot exploit.")
        priority, best_candidate = self.memory.best()  # (priority, candidate)
        priority = - priority # remember that we stored negative scores in the priority queue
        return best_candidate, {
            'best_candidate_priority': priority,  # remember that we stored negative scores in the priority queue
            'best_candidate_mean_score': best_candidate.mean_score(),  # mean score of the candidate's rollouts
        }

    def compute_priority(self, candidate):
        # NOTE This function can be overridden by subclasses to compute a different score
        """ Compute the score for the candidate based on the rollouts during the validation phase.
        It can be overridden by subclasses to implement a different scoring strategy.

        Args:
            candidate (ModuleCandidate): The candidate for which to compute the score.
        Returns:
            float: The computed score for the candidate.
        """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # By default, we compute the mean score of the rollouts

        if self.score_function == 'mean':
            # Compute the mean score of the candidate's rollouts
            return candidate.mean_score()
        elif self.score_function == 'ucb':
            # Compute the Upper Confidence Bound (UCB) score
            lcb_score, mean_score, ucb_score = candidate.compute_score_confidence(
                min_score=self.min_score,
                max_score=self.max_score,
                scaling_constant=self.ucb_exploration_constant
            )
            return ucb_score  # return the UCB score
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")
