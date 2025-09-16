import numpy as np
import copy
from typing import Union, List, Tuple, Dict, Any, Optional
from opto.features.priority_search.search_template import Samples, SearchTemplate
from opto.features.priority_search.module_regressor import ModuleCandidateRegressor
from opto.features.priority_search.priority_search import PrioritySearch, ModuleCandidate, HeapMemory
import heapq

class PrioritySearch_with_Regressor(PrioritySearch):
    """ 
    A subclass of PrioritySearch that uses a regressor to predict the scores of the candidates.
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
              score_range = None,  # range of (min_score, max_score) to clip the scores; if None, no clipping is applied
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
              memory_size: Optional[int] = None,  # size of the long-term heap memory to store the candidates; if None, no limit is set
              short_term_memory_size: Optional[int] = None,  # size of the short-term memory to store the most recent candidates; if None, no limit is set
              short_term_memory_duration: Optional[int] = 0,  # number of iterations to keep the candidates in the short-term memory before merging them into the long-term memory. 0 means only long-term memory is used.
              score_function: str = 'mean',  # function to compute the score for the candidates; 'mean' or 'ucb'
              ucb_exploration_constant: float = 1.0,  # exploration constant for UCB score function
              # Regressor specific parameters
              regressor_embedding_model: str = "gemini/text-embedding-004",  # embedding model for the regressor
              regressor_learning_rate: float = 0.2,  # learning rate for the regressor
              regressor_regularization_strength: float = 1e-4,  # L2 regularization strength for the regressor
              regressor_max_iterations: int = 20000,  # maximum iterations for regressor training
              regressor_tolerance: float = 5e-3,  # convergence tolerance for the regressor
              # Additional keyword arguments
              **kwargs
              ):
        """ Train the agent using the Priority Search algorithm with regressor.

        This extends the parent PrioritySearch by adding a regressor that predicts
        candidate scores based on the long-term memory.

        Args:
            All parameters from the parent PrioritySearch.train() method, plus:
            regressor_embedding_model (str, optional): Embedding model for the regressor. Defaults to "gemini/text-embedding-004".
            regressor_learning_rate (float, optional): Learning rate for the regressor. Defaults to 0.2.
            regressor_regularization_strength (float, optional): L2 regularization strength for the regressor. Defaults to 1e-4.
            regressor_max_iterations (int, optional): Maximum iterations for regressor training. Defaults to 20000.
            regressor_tolerance (float, optional): Convergence tolerance for the regressor. Defaults to 5e-3.
        """

        # Initialize the search parameters and memory
        self._initialize_search_parameters(
            num_candidates=num_candidates,
            num_proposals=num_proposals,
            validate_exploration_candidates=validate_exploration_candidates,
            use_best_candidate_to_explore=use_best_candidate_to_explore,
            score_function=score_function,
            score_range=score_range,
            ucb_exploration_constant=ucb_exploration_constant,
            memory_size=memory_size,
            short_term_memory_size=short_term_memory_size,
            short_term_memory_duration=short_term_memory_duration
        )
        
        # Initialize the regressor with the long-term memory and custom parameters - this is the only difference from parent class
        self.regressor = ModuleCandidateRegressor(
            memory=self.long_term_memory,
            embedding_model=regressor_embedding_model,
            num_threads=num_threads,
            learning_rate=regressor_learning_rate,
            regularization_strength=regressor_regularization_strength,
            max_iterations=regressor_max_iterations,
            tolerance=regressor_tolerance
        )

        SearchTemplate.train(self, guide=guide,
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
            candidates = self.propose(samples, verbose=verbose, **kwargs)  # List of ModuleCandidates
            # 2. Validate the proposed parameters
            validate_results = self.validate(candidates, samples, verbose=verbose, **kwargs)  # this updates the priority queue
            # 3. Update the priority queue with the validation results
            self.update_memory(validate_results, verbose=verbose, **kwargs)  # samples are provided here in case candidates do not capture full information
        else:  # The first iteration.
            max_mem_size = self.memory.size if self.memory.size is not None else float('inf')
            while len(self.memory) < min(max_mem_size, self.num_candidates):
                self.memory.push(self.max_score, ModuleCandidate(self.agent, optimizer=self.optimizer))  # Push the base agent as the first candidate (This gives the initialization of the priority queue)

        
        self.update_memory_with_regressor(verbose=verbose, **kwargs)

        # 4. Explore and exploit the priority queue
        self._best_candidate, self._best_candidate_priority, info_exploit = self.exploit(verbose=verbose, **kwargs)  # get the best candidate (ModuleCandidate) from the priority queue
        self._exploration_candidates, self._exploration_candidates_priority, info_explore = self.explore(verbose=verbose, **kwargs)  # List of ModuleCandidates
        # TODO Log information about the update
        info_log = {
            'n_iters': self.n_iters,  # number of iterations
            'short_term_memory_size': len(self.short_term_memory),  # size of the short-term memory
            'long_term_memory_size': len(self.long_term_memory),  # size of the long-term memory
            'using_short_term_memory': self.memory is self.short_term_memory,  # whether the current memory is the short-term memory
            'using_long_term_memory': self.memory is self.long_term_memory,  # whether the current memory is the long-term memory
        }

        info_log.update(info_exploit)  # add the info from the exploit step
        info_log.update(info_explore)  # add the info from the explore step
        return self._best_candidate.update_dict, [c.get_module() for c in self._exploration_candidates], info_log

    def validate(self,
                 candidates: List[ModuleCandidate],
                 samples: Samples,
                 verbose: bool = False,
                 **kwargs):
        """ Override the validate method. In this version we only use training data to update arm statistics. No validation is performed.
        """
        print("--- Validating candidates...") if verbose else None
        assert isinstance(samples, Samples), "samples must be an instance of Samples."
        exploration_candidates = self._exploration_candidates  # exploration candidates from the previous iteration
        assert self._exploration_candidates is not None, "exploration_candidates must be set before calling validate."

        # The current batch of samples can be used to validate the exploration candidates
        validate_samples = copy.copy(samples)
        matched_candidates_and_samples = self.match_candidates_and_samples(exploration_candidates, validate_samples.samples)
        results = {}  # dict of ModuleCandidate id: (ModuleCandidate, list of rollouts)
        for c, rollouts in matched_candidates_and_samples.items():  # rollouts is a list of BatchRollouts
            results[c] = [ r for rr in rollouts for r in rr.to_list()]  # we only need the list of dicts

        return results

    def update_memory(self, validate_results, verbose: bool = False, **kwargs):
        """ Override the update_memory method. In this subclass, we update the priority of all candidates together. Cannot use the parent class's update_memory method, because now some candidates may not have predicted scores.
        """
        print("--- Updating memory with validation results...") if verbose else None
        for candidate, rollouts in validate_results.items():
            candidate.add_rollouts(rollouts)  # add the rollouts to the 
            placeholder_priority = self.max_score
            self.memory.push(placeholder_priority, candidate)

    def update_memory_with_regressor(self, verbose: bool = False, **kwargs):
        """ Update the priority queue with the regressor results.
        """
        print("--- Updating memory with regressor results...") if verbose else None
        if self.memory is self.long_term_memory:    # Only update the regressor if we are using the long-term memory
            self.regressor.update()
        self.regressor.predict_scores(self.memory) # The only difference from the parent class
        # Reorder the memory according to the predicted scores
        # Extract candidates from memory tuples and reorder by predicted scores
        candidates_with_scores = [(-candidate.predicted_score, candidate) for _, candidate in self.memory]
        self.memory.memory = candidates_with_scores  # Update the internal list of HeapMemory
        heapq.heapify(self.memory.memory)  # Heapify the internal list

    def print_memory_stats(self):
        # For debugging, print all candidates: number, mean_score(), num_rollouts, predicted_score. It is better to see an increasing trend in the predicted scores.
        for i, (neg_predicted_score, candidate) in enumerate(self.memory):
            print(f"Candidate {i}, Mean Score: {candidate.mean_score()}, Num Rollouts: {candidate.num_rollouts}, Predicted Score: {-neg_predicted_score}")

    # TODO refactor below to reuse scoring
    def compute_exploitation_priority(self, candidate) -> float:
        """ Compute the priority for the candidate based on the predicted score. """
        if not isinstance(candidate, ModuleCandidate):
            raise TypeError("candidate must be an instance of ModuleCandidate.")
        # By default, we compute the mean score of the rollouts
        return candidate.predicted_score
