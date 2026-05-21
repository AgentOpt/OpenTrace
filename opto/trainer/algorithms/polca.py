from opto.trainer.algorithms.priority_search import PrioritySearch, ModuleCandidate
from opto.trainer.summarizer import Summarizer
from typing import Union, List, Tuple, Dict, Any, Optional, Callable
from opto.optimizers.utils import print_color
from opto.trainer.utils import async_run
from opto.utils.llm import embed
import numpy as np
from opto.trainer.search_template import Samples


class _CandidateEmbedder:
    """Adds an `embedding` attribute to ModuleCandidate objects via an embedding API."""

    def __init__(self, embedding_model: str = "gemini/gemini-embedding-001"):
        self.embedding_model = embedding_model

    def _get_parameter_text(self, candidate):
        assert hasattr(candidate, 'update_dict'), "ModuleCandidate must have an update_dict"
        params_with_names = {k.py_name: v for k, v in candidate.update_dict.items()}
        return str(params_with_names)

    def _get_embedding(self, candidate, max_retries: int = 10, base_delay: float = 1.0):
        parameter_text = self._get_parameter_text(candidate)
        return embed(self.embedding_model, parameter_text,
                     max_retries=max_retries, base_delay=base_delay)

    def add_embeddings_to_candidates(self, candidates: List[ModuleCandidate], max_workers: int = 50):
        """Attach embeddings to each candidate that doesn't already have one."""
        candidates_needing_embeddings = [c for c in candidates if not hasattr(c, "embedding")]
        if not candidates_needing_embeddings:
            return
        embedding_functions = [lambda c=candidate: self._get_embedding(c)
                               for candidate in candidates_needing_embeddings]
        new_embeddings = async_run(
            embedding_functions,
            max_workers=max_workers,
            description=f"Generating embeddings for {len(candidates_needing_embeddings)} candidates",
        )
        for candidate, embedding in zip(candidates_needing_embeddings, new_embeddings):
            candidate.embedding = embedding


def calculate_distance_to_memory(memory, new_candidate):
    """For a new candidate, calculate the distance to the current memory. That's the least L2 distance to any candidate in the memory.
    
    To use this funciton in PrioritySearch, set memory to be self.memory.memory.
    """
    assert hasattr(new_candidate, 'embedding') and all(hasattr(candidate, 'embedding') for _, candidate in memory), "All candidates should have the embedding attribute."
    min_distance = float('inf')
    for _, candidate in memory:
        distance = np.linalg.norm(np.array(new_candidate.embedding) - np.array(candidate.embedding))
        if distance < min_distance:
            min_distance = distance
    return min_distance

class POLCA(PrioritySearch):
    """
    A subclass of PrioritySearch, which keeps an epsilon-net as the memory. Reject new candidates that are in the epsilon-net of the memory.

    This class uses a summarizer to summarize the memory and the exploration candidates. It then sets the context for the optimizer to use the summary to guide the exploration.

    Args:
        epsilon: The epsilon value for the epsilon-net. 0 means no filtering, the same as vanilla PrioritySearch.
        use_summarizer: Whether to use a summarizer to summarize the memory and the exploration candidates.
        embedding_model: The embedding model used to compute candidate embeddings for the epsilon-net.
        summarizer_model_name: The model name for the summarizer.
        *args: Additional arguments for the parent class.
        **kwargs: Additional keyword arguments for the parent class.
    """
    def __init__(self,
                 epsilon: float = 0.1,
                 use_summarizer: bool = False,
                 context: str = "Concrete recommendations for generating better agent parameters based on successful patterns observed in the trajectories: ",
                 embedding_model: str = "gemini/gemini-embedding-001",
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.use_summarizer = use_summarizer
        self.regressor = _CandidateEmbedder(embedding_model=embedding_model)
        self.summarizer = Summarizer()
        self.context = context
        
    def filter_candidates(self, new_candidates: List[ModuleCandidate]) -> List[ModuleCandidate]:
        """ Filter candidates by their embeddings.
        """
        if self.epsilon == 0: # no filtering
            print_color(f"No filtering of candidates.", "green")
            return new_candidates
        exploration_memory = [(0, candidate) for candidate in self._exploration_candidates]
        current_memory = self.memory.memory + exploration_memory
        # Add embeddings to all the candidates. The regressor will check if the candidates have embeddings, and if not, it will add them in parallel.
        current_candidates = [candidate for _, candidate in current_memory]
        self.regressor.add_embeddings_to_candidates(current_candidates+new_candidates)

        # filter new candidates based on the distance to the current memory.
        num_new_candidates = len(new_candidates)

        added_candidates = []
        success_distances = []
        
        while len(new_candidates) > 0:
            # calculate the distance to the memory for each new candidate
            distances = [calculate_distance_to_memory(current_memory, new_candidate) for new_candidate in new_candidates]
            
            # filter candidates: keep only those with distance > epsilon
            filtered_candidates = []
            filtered_distances = []
            for i, (candidate, distance) in enumerate(zip(new_candidates, distances)):
                if distance > self.epsilon:
                    filtered_candidates.append(candidate)
                    filtered_distances.append(distance)
            
            # if no candidates remain, exit the loop
            if len(filtered_candidates) == 0:
                break
            
            # add the candidate with the largest distance to the memory
            max_distance_idx = np.argmax(filtered_distances)
            new_node = filtered_candidates[max_distance_idx]
            current_memory.append((0, new_node))
            added_candidates.append(new_node)
            success_distances.append(float(filtered_distances[max_distance_idx]))
            
            # remove the added candidate from new_candidates list
            new_candidates = [c for c in filtered_candidates if c is not new_node]

        print_color(f"Proposed {num_new_candidates} new candidates, {len(added_candidates)} of them are added to the memory.", "green")
        # print the distances between the added candidates and the memory before adding them.
        print_color(f"Distances between the added candidates and the memory before adding them: {success_distances}", "green")
        return added_candidates
    
    def compress_candidate_memory(self, candidate: ModuleCandidate) -> ModuleCandidate:
        """ For the summarizer usage, we keep the entire rollout. """
        if self.use_summarizer:
            return candidate
        else:
            return super().compress_candidate_memory(candidate)
    
    def propose(self,
                samples : Samples,
                verbose : bool = False,
                **kwargs):
        """ 
        Override the propose method to include a summary into the context of the optimizer.
        """
        
        # Use the summarizer to summarize the memory and the exploration candidates.
        if self.use_summarizer:
            # Summarize the memory and the exploration candidates.
            exploration_memory = [(0, candidate) for candidate in self._exploration_candidates]
            print_color(f"Summarizing the history...", "green")
            try: 
                summary = self.summarizer.summarize(self.memory.memory+exploration_memory)
                print_color(f"Summary: {summary}", "green")
                self.context = f"Concrete recommendations for generating better agent parameters based on successful patterns observed in the trajectories: {summary}"
            except Exception as e:
                print_color(f"Error: {e}", "red")
                print_color(f"Using fallback context: {self.context}", "red")
            # Set the context for the optimizer.
            for candidate in self._exploration_candidates:
                candidate.optimizer.set_context(self.context)
        return super().propose(samples, verbose, **kwargs)
