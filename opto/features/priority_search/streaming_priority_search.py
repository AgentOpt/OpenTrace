
import numpy as np
from typing import List
from opto.features.priority_search.search_template import Samples, BatchRollout, save_train_config
from opto.features.priority_search.priority_search import PrioritySearch, ModuleCandidate



class StreamingPrioritySearch(PrioritySearch):
    """ A variant of PrioritySearch that uses only the most recent samples for proposing new candidates.
        It overrides the `propose` method to match candidates with samples differently.
    """

    @save_train_config
    def train(self,
              *args,
              exploration_ratio = 0.5,
              exploration_temperature = 1.0,
              **kwargs):
        assert 0 < exploration_ratio <= 1, "exploration_ratio must be in (0, 1]."
        self._exploration_ratio = exploration_ratio
        self._exploration_temperature = exploration_temperature
        return super().train(*args, **kwargs)

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
        # NOTE since we overwrite validate, this function is only called in propose

        # Associate each BatchRollout candidates
        matched_candidates_and_samples = super().match_candidates_and_samples(candidates, samples)

        # Update the candidates with all the rollouts collected so far so we can compute their scores
        results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c, rollouts in matched_candidates_and_samples.items():  # rollouts is a list of BatchRollouts
            results[c] = [ r for rr in rollouts for r in rr.to_list()]  # we only need the list of dicts
        for candidate, rollouts in results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the candidate

        # set exploration candidates to those with merged stats (they are the ones that will be added to memory)
        self._exploration_candidates = list(results.keys())

        # Now these candidates have all the rollouts collected so far
        # compute the score for each candidate using compute_exploration_priority
        candidate_batchrollouts_list = [ (k,b) for k, v in matched_candidates_and_samples.items() for b in v]
        scores = [self.compute_exploration_priority(c) for c, _ in candidate_batchrollouts_list]

        # We use the top K to improve over, where K is determined by exploration_ratio.

        # ensure it is possible to select K>=1 such that K * num_proposals <= num_candidates * exploration_ratio
        if self.num_proposals > self.num_candidates * self._exploration_ratio:
            print(f"Warning: num_proposals {self.num_proposals} is greater than num_candidates {self.num_candidates * self._exploration_ratio}. Setting num_proposals to num_candidates * exploration_ratio.")
            self.num_proposals = int(self.num_candidates * self._exploration_ratio)

        currently_available = len(self._exploration_candidates) + len(self.memory)
        K = max(int(self.num_candidates * self._exploration_ratio / self.num_proposals), 1)  # K>=1
        # make sure we have enough candidates to explore
        if K * self.num_proposals + currently_available < self.num_candidates:
            # Increase K to ensure we have enough candidates
            K += int((self.num_candidates - (K * self.num_proposals + currently_available)) / self.num_proposals)
        # make sure K * self.num_proposals <= self.num_candidates
        K = min(K, int(self.num_candidates / self.num_proposals))

        # Randomly sample K candidates from the pool
        if len(candidate_batchrollouts_list) <= K:
            return matched_candidates_and_samples
        weight = np.array(scores)/self._exploration_temperature
        weight = np.exp(weight - np.max(weight))
        indices = np.random.choice(len(candidate_batchrollouts_list), size=K, replace=False, p=weight/np.sum(weight))
        candidate_batchrollouts_list = [candidate_batchrollouts_list[i] for i in indices]
        # candidate_batchrollouts_list = sorted(candidate_batchrollouts_list, key=lambda x: scores[candidate_batchrollouts_list.index(x)], reverse=True)[:K]
        assert len(candidate_batchrollouts_list) == K, f"Number of selected candidates {len(candidate_batchrollouts_list)} must be equal to K {K}."
        # convert it back to the format of matched_candidates_and_samples
        matched_candidates_and_samples = {c: [] for c, _ in candidate_batchrollouts_list}
        for c, b in candidate_batchrollouts_list:
            matched_candidates_and_samples[c].append(b)
        return matched_candidates_and_samples


    def validate(self,
                 candidates: List[ModuleCandidate],
                 samples: Samples,
                 verbose: bool = False,
                 **kwargs):
        print("--- Skip validating candidates...") if verbose else None
        exploration_candidates = self._exploration_candidates  # exploration candidates from the previous iteration
        assert self._exploration_candidates is not None, "exploration_candidates must be set before calling validate."
        results = {c: []  for c in (exploration_candidates + candidates)}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        print(f'Adding {len(exploration_candidates)} exploration candidates and {len(candidates)} proposed candidates to validate results.')
        assert len(candidates) <= self.num_candidates, f"Number of proposed candidates {len(candidates)} must be no larger than num_candidates {self.num_candidates}."
        if len(candidates) == self.num_candidates:
            print("Warning: Number of proposed candidates is equal to num_candidates. Running in pure exploration mode.")
        # remove this assertion since some candidates might be duplicates
        # assert len(results) == len(exploration_candidates) + len(candidates), f"Number of candidates in results must match the number of exploration candidates and proposed candidates. Getting {len(results)} vs {len(exploration_candidates) + len(candidates)}."
        return results

    def compute_exploration_priority(self, candidate) -> float:
        if candidate.num_rollouts == 0:
            return self.max_score  # candidates with no rollouts have the highest priority
        return super().compute_exploration_priority(candidate)