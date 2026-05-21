
import copy
import heapq
from typing import Union, Optional

from opto.optimizers.utils import print_color
from opto.trainer.algorithms.priority_search import PrioritySearch, ModuleCandidate
from opto.trainer.utils import safe_mean

# Below we define several algorithms that use the PrioritySearch class.


class SequentialUpdate(PrioritySearch):
    """ A basic algorithm that explores the parameter space and proposes new candidates one by one.

        This is realized by setting

            num_candidates = 1
            num_proposals = 1
            memory_size = 1

        This is the same as MinibatchAlgorithm when
            1. no validation set is provided
            2. num_batches = 1

        validate_proposals here acts the same as `ensure_improvement` flag in MinibatchAlgorithm
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
              num_batches = 1,  # number of batches to use from the dataset in each iteration
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation
              num_test_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer
              default_score: float = float('inf'),  # default score assigned to priority queue candidates
              validate_proposals: bool = True,  # whether to validate the proposed parameters
              memory_size: Optional[int] = None,  # size of the heap memory to store the candidates; if None, no limit is set
              # Additional keyword arguments
              **kwargs
              ):

        num_candidates = 1  # SequentialSearch only proposes one candidate at a time
        num_proposals = 1  # SequentialSearch only generates one proposal at a time
        memory_size = 1  # SequentialSearch only stores one candidate at a time in the heap memory
        # validate_proposals is the same as `ensure_improvement` flag in MinibatchAlgorithm

        return super().train(guide, train_dataset,
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
                      num_candidates=num_candidates,
                      num_proposals=num_proposals,
                      default_score=default_score,
                      validate_proposals=validate_proposals,
                      memory_size=memory_size, **kwargs)


class SequentialSearch(PrioritySearch):
    """ A sequential search that generates one candidate in each iteration by validating multiple proposals.

        This is realized by setting
            num_proposals = 1
            memory_size = 1

        This is the same as BasicSearchAlgorithm when
            1. a validation set is provided
            2. validate_proposals is True.
            3. num_batches is 1.
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
              num_batches = 1,  # number of batches to use from the dataset in each iteration
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation
              num_test_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer
              default_score: float = float('inf'),  # default score assigned to priority queue candidates
              validate_proposals: bool = True,  # whether to validate the proposed parameters
              memory_size: Optional[int] = None,  # size of the heap memory to store the candidates; if None, no limit is set
              # Additional keyword arguments
              **kwargs
              ):

        num_candidates = 1  # SequentialSearch only generates one candidate at a time
        memory_size = 1  # MultiSequentialUpdate only stores one candidate at a time in the heap memory
        # validate_proposals is the same as `ensure_improvement` flag in MinibatchAlgorithm

        return super().train(guide, train_dataset,
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
                      num_candidates=num_candidates,
                      num_proposals=num_proposals,
                      default_score=default_score,
                      validate_proposals=validate_proposals,
                      memory_size=memory_size, **kwargs)

class BeamSearch(PrioritySearch):
    """ A beam search algorithm that explores the parameter space and proposes new candidates based on the best candidates in the priority queue.

        This is realized by setting
            num_proposals = beam_size
            memory_size = beam_size

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
              num_batches = 1,  # number of batches to use from the dataset in each iteration
              score_range = None,  # minimum score to update the agent
              num_epochs = 1,  # number of training epochs
              num_threads = None,  # maximum number of threads to use
              verbose = False,  # whether to print the output of the agent
              # evaluation
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              test_frequency: Union[int, None] = 1, # frequency of evaluation
              num_test_samples: int = 1,  # number of samples to use to evaluate each input
              # logging
              log_frequency = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              # Priority Search specific parameters
              num_candidates: int = 10,  # number of candidates to propose for exploration
              num_proposals: int = 1,  # number of proposals to generate per optimizer; this is beam_size in beam search.
              default_score: float = float('inf'),  # default score assigned to priority queue candidates
              validate_proposals: bool = True,  # whether to validate the proposed parameters
              memory_size: Optional[int] = None,  # size of the heap memory to store the candidates; if None, no limit is set
              **kwargs):

        # num_candidates acts as the beam size in beam search.
        memory_size = num_candidates

        return super().train(guide, train_dataset,
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
                       num_candidates=num_candidates,  # beam size
                       num_proposals=num_proposals,  # number of proposals to generate per optimizer
                       default_score=default_score,
                       validate_proposals=validate_proposals,
                       memory_size=memory_size, **kwargs)

class ParetobasedPS(PrioritySearch):
    """GEPA-style Pareto-based exploration on top of the PrioritySearch pipeline.

    Instead of popping the top candidates by a scalar priority (the default
    PrioritySearch behavior), this algorithm selects exploration candidates
    via the Pareto frontier of per-task scores:

        1. For every training input x, find the candidate(s) with the highest
           empirical mean score on x (the "best set" for x).
        2. Collect all candidates that appear in at least one such best set.
        3. Remove strictly dominated candidates: candidate ``a`` strictly
           dominates ``b`` iff the set of tasks on which ``a`` is best is a
           proper superset of the set of tasks on which ``b`` is best.
        4. Return the remaining (Pareto-optimal) candidates, truncated to
           ``num_candidates`` (sorted by overall mean score as a tie-breaker).

    Notes
    -----
    * To compute per-task scores we need the original ``x`` of each rollout,
      so ``compress_candidate_memory`` is overridden to keep ``x`` and
      ``score`` (instead of only ``score`` / ``score_dict`` as in the base
      class).
    * Scalar priorities are still pushed into the priority queue by
      ``update_memory`` (via ``compute_exploration_priority``), so exploit
      still works. Only ``explore`` is replaced with the Pareto selection.
    """

    def compress_candidate_memory(self, candidate: ModuleCandidate) -> ModuleCandidate:
        """Keep ``x`` and ``score`` per rollout. Needed because Pareto selection groups rollouts by task ``x``; the parent class would drop it."""
        def _process_rollout(rollout):
            for k in rollout:
                if k not in ['x', 'score']:
                    rollout[k] = None
        candidate = copy.copy(candidate)
        candidate.rollouts = copy.deepcopy(candidate.rollouts)
        for rollout in candidate.rollouts:
            _process_rollout(rollout)
        return candidate

    def compute_score_for_task_x(self, candidate: ModuleCandidate, x) -> float:
        """Empirical mean score of ``candidate`` on task ``x`` (0 if unseen)."""
        scores = [r['score'] for r in candidate.rollouts if r.get('x') == x]
        return safe_mean(scores, missing_value=0)

    def get_best_candidates_for_x(self, x, candidates):
        """Return the subset of ``candidates`` with the max score on task ``x``."""
        if not candidates:
            return []
        scores = [self.compute_score_for_task_x(c, x) for c in candidates]
        highest = max(scores)
        return [c for c, s in zip(candidates, scores) if s == highest]

    def explore(self, verbose: bool = False, **kwargs):
        """Select exploration candidates from the Pareto frontier.

        Returns
        -------
        top_candidates : list[ModuleCandidate]
            The Pareto-frontier candidates, truncated to ``self.num_candidates``.
        priorities : list[float]
            Priorities associated with the selected candidates (as stored in
            the heap memory, for logging).
        info_dict : dict
            Logging info analogous to ``PrioritySearch.explore``.
        """
        print_color("Using Pareto-based exploration to explore the parameter space...", "green")

        # Gather all candidates currently in memory.
        all_candidates = [c for _, c in self.memory.memory]
        if not all_candidates:
            return [], [], {
                'num_exploration_candidates': 0,
                'exploration_candidates_mean_priority': None,
                'exploration_candidates_mean_score': None,
                'exploration_candidates_average_num_rollouts': None,
            }

        # Training inputs (deduplicated in a stable way).
        raw_xs = list(self.train_sampler.dataset['inputs'])
        seen, xs = set(), []
        for x in raw_xs:
            try:
                key = x
                if key in seen:
                    continue
                seen.add(key)
            except TypeError:
                # unhashable x: keep all occurrences; semantics unchanged
                pass
            xs.append(x)

        # Best candidates per task.
        best_for_x = {i: self.get_best_candidates_for_x(x, all_candidates)
                      for i, x in enumerate(xs)}

        # Candidates that are best on at least one task.
        frontier_pool = list({id(c): c for cs in best_for_x.values() for c in cs}.values())

        # Strict Pareto dominance on task-index sets.
        def tasks_where_best(c):
            return frozenset(i for i, cs in best_for_x.items() if c in cs)

        tasks_of = {id(c): tasks_where_best(c) for c in frontier_pool}

        non_dominated = []
        for b in frontier_pool:
            tb = tasks_of[id(b)]
            dominated = False
            for a in frontier_pool:
                if a is b:
                    continue
                ta = tasks_of[id(a)]
                # Strict superset: ta ⊋ tb  (a is best everywhere b is best, and strictly more)
                if tb.issubset(ta) and ta != tb:
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(b)

        self.logger.log('Update/num_pareto_candidates',
                        len(non_dominated), self.n_iters, color='green')
        print_color(
            f"Pareto frontier size: {len(non_dominated)} / {len(frontier_pool)} "
            f"(taking up to {self.num_candidates} for exploration).", "green")

        # Truncate to num_candidates (break ties by mean score, descending).
        non_dominated.sort(
            key=lambda c: c.mean_score() if c.mean_score() is not None else 0.0,
            reverse=True,
        )
        top_candidates = non_dominated[:self.num_candidates]

        # Remove selected candidates from the heap memory and re-heapify.
        selected_ids = {id(c) for c in top_candidates}
        priorities = []
        items_to_remove = []
        for neg_priority, candidate in self.memory.memory:
            if id(candidate) in selected_ids:
                priorities.append(-neg_priority)
                items_to_remove.append((neg_priority, candidate))
        for item in items_to_remove:
            self.memory.memory.remove(item)
        heapq.heapify(self.memory.memory)

        mean_scores = [c.mean_score() for c in top_candidates]
        mean_scores = [s for s in mean_scores if s is not None]
        info_dict = {
            'num_exploration_candidates': len(top_candidates),
            'exploration_candidates_mean_priority': safe_mean(priorities),
            'exploration_candidates_mean_score': safe_mean(mean_scores),
            'exploration_candidates_average_num_rollouts':
                safe_mean([c.num_rollouts for c in top_candidates]),
        }
        return top_candidates, priorities, info_dict

