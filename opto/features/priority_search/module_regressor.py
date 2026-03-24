import numpy as np
from opto.trainer.utils import batch_run, async_run
from opto.optimizers.utils import print_color
from typing import Union, List, Tuple, Dict, Any, Optional
from opto.utils.auto_retry import retry_with_exponential_backoff
import litellm
import time
from opto.features.priority_search.priority_search import ModuleCandidate


def embed_text(model, text):
    """Call the embedding API for a given model and text string.

    This is a standalone function so users can easily replace it with a custom
    embedding provider (e.g. local model, different API) without subclassing.
    Must return a litellm-compatible response with response.data[0].embedding.
    """
    return litellm.embedding(model=model, input=text)


class RegressorTemplate:
    """Base class template for regression-based predictors for ModuleCandidate objects.
    
    Provides common functionality for embedding generation and candidate processing.
    Subclasses should implement update() and predict_scores() methods.

    Regressors can be built on this template by implementing the update() and predict_scores() methods. 
    This class itself is enough for getting embeddings for candidates.
    """
    
    def __init__(self, embedding_model="gemini/gemini-embedding-001", num_threads=None, regularization_strength=1, linear_dim=None, rich_text=True,verbose: bool = False, max_candidates_to_predict=500,original_embedding_dim=768):
        '''
        Args:
            embedding_model: The embedding model to use.
            num_threads: The number of threads to use for the embedding generation.
            regularization_strength: The regularization strength for the logistic regression.
            linear_dim: The dimension of the linear space.
            rich_text: Whether to use rich text for the parameter text.
            verbose: Whether to print the verbose output.
            max_candidates_to_predict: The maximum number of candidates to predict.
            original_embedding_dim: The original dimension of the embedding.
        '''
    def _get_parameter_text(self, candidate):
        """Get the parameter text for a ModuleCandidate."""
        if not hasattr(candidate, 'update_dict'):
            print(candidate)
        assert hasattr(candidate, 'update_dict'), "ModuleCandidate must have an update_dict"
        # Convert parameter nodes to readable names for deterministic embedding
        params_with_names = {k.py_name: v for k, v in candidate.update_dict.items()}
        return str(params_with_names)
    

    def _get_embedding(self, candidate,max_retries=10,base_delay=1.0):
        """Get the embedding for a ModuleCandidate."""
        parameter_text = self._get_parameter_text(candidate)
        
        try:
            response = retry_with_exponential_backoff(
                lambda: embed_text(self.embedding_model, parameter_text),
                max_retries=max_retries,
                base_delay=base_delay,
                operation_name="Embedding API call"
            )
            embedding = response.data[0].embedding
            if self.random_projector is not None:
                embedding_array = np.array(embedding).reshape(1, -1)
                projected = self.random_projector.transform(embedding_array)
                embedding = projected.flatten().tolist()
            return embedding
        except Exception as e:
            print_color(f"ERROR: Embedding API call failed after retries: {e}", "red")
            return None

    def add_embeddings_to_candidates(self, candidates: List[ModuleCandidate]):
        """Add embeddings to a list of candidates. This function could be used outside."""
        self._update_memory_embeddings_for_batch(candidates)

    def _update_memory_embeddings_for_batch(self, batch,max_workers=50,max_retries=10,base_delay=1.0):
        """Update the embeddings for a batch of candidates."""
        # Separate candidates that need embeddings from those that already have them
        candidates_needing_embeddings = []
        for candidate in batch:
            if not hasattr(candidate, "embedding"):
                candidates_needing_embeddings.append(candidate)
        
        # Generate embeddings in parallel for candidates that need them
        if candidates_needing_embeddings:
            def get_embedding_for_candidate(candidate):
                return self._get_embedding(candidate)
            
            # Create function list for async_run
            embedding_functions = [lambda c=candidate: get_embedding_for_candidate(c) 
                                 for candidate in candidates_needing_embeddings]
            
            # Run embedding generation in parallel
            new_embeddings = async_run(
                embedding_functions,
                max_workers=max_workers,
                description=f"Generating embeddings for {len(candidates_needing_embeddings)} candidates"
            )
            
            # Assign embeddings back to candidates
            for candidate, embedding in zip(candidates_needing_embeddings, new_embeddings):
                candidate.embedding = embedding

    def update(self, memory: List[Tuple[float, ModuleCandidate]]):
        """Update the regression model parameters. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the update method")
    
    def predict_scores(self, memory: List[Tuple[float, ModuleCandidate]]):
        """Predict scores for candidates. Should be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement the predict_scores method")

class ModuleCandidateRegressor:
    """
    Predict scores using embedding logistic regression for ModuleCandidate objects. 
    Should have two key methods: predict_scores and predict_scores_for_batch. 
    predict_scores has no parameters, it could return predicted scores for all candidates in the memory. 
    predict_scores_for_batch has one parameter, a batch of candidates, it could return predicted scores for the batch of candidates."""
    
    def __init__(self, memory=None, embedding_model="gemini/text-embedding-004", num_threads=None, learning_rate=0.2, regularization_strength=1e-4, max_iterations=20000, tolerance=5e-3, max_candidates_to_predict=500, original_embedding_dim=768,patience=20,lr_decay_factor=0.8):
        self.max_candidates_to_predict = max_candidates_to_predict
        self.memory = memory
        self.embedding_model = embedding_model
        self.num_threads = num_threads
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.regularization_strength = regularization_strength  # L2 regularization strength (lambda)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = patience  # Early stopping patience
        self.lr_decay_factor = lr_decay_factor   # Learning rate decay factor
        self.linear_dim = original_embedding_dim
        # Initialize weights with larger values for more aggressive learning
        self.weights = np.random.normal(0, 0.1, self.linear_dim)
        self.bias = 0.0
        
    def _sigmoid(self, z):
        """Sigmoid activation function for logistic regression."""
        return 1.0 / (1.0 + np.exp(-z))

    def _get_parameter_text(self, candidate):
        """Get the parameter text for a ModuleCandidate."""
        if not hasattr(candidate, 'update_dict'):
            print(candidate)
        assert hasattr(candidate, 'update_dict'), "ModuleCandidate must have an update_dict"
        # Convert parameter nodes to readable names for deterministic embedding
        params_with_names = {k.py_name: v for k, v in candidate.update_dict.items()}
        return str(params_with_names)

    def _get_embedding(self, candidate,max_retries=10,base_delay=1.0):
        """Get the embedding for a ModuleCandidate."""
        parameter_text = self._get_parameter_text(candidate)
        
        try:
            response = retry_with_exponential_backoff(
                lambda: embed_text(self.embedding_model, parameter_text),
                max_retries=max_retries,
                base_delay=base_delay,
                operation_name="Embedding API call"
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print_color(f"ERROR: Embedding API call failed after retries: {e}", "red")
            print_color("Using random embedding as fallback", "yellow")
            fallback_embedding = np.random.normal(0, 0.01, self.linear_dim)
            return fallback_embedding / np.linalg.norm(fallback_embedding)
    
    def _update_memory_embeddings_for_batch(self, batch,max_workers=1000,max_retries=10,base_delay=1.0):
        """Update the embeddings for a batch of candidates."""
        # Separate candidates that need embeddings from those that already have them
        candidates_needing_embeddings = []
        for candidate in batch:
            if not hasattr(candidate, "embedding"):
                candidates_needing_embeddings.append(candidate)
        
        # Generate embeddings in parallel for candidates that need them
        if candidates_needing_embeddings:
            def get_embedding_for_candidate(candidate):
                return self._get_embedding(candidate)
            
            # Create function list for async_run
            embedding_functions = [lambda c=candidate: get_embedding_for_candidate(c) 
                                 for candidate in candidates_needing_embeddings]
            
            # Run embedding generation in parallel
            new_embeddings = async_run(
                embedding_functions,
                max_workers=max_workers,
                description=f"Generating embeddings for {len(candidates_needing_embeddings)} candidates"
            )
            
            # Assign embeddings back to candidates
            for candidate, embedding in zip(candidates_needing_embeddings, new_embeddings):
                candidate.embedding = embedding

    def update(self):
        """Update the regression model parameters using the current memory with logistic regression."""
        start_time = time.time()
        if self.verbose:
            print_color("Updating regression model using the current memory with logistic regression...", "blue")
        # Extract candidates from memory (memory contains (neg_score, candidate) tuples)
        batch = [candidate for _, candidate in self.memory]
        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Get training data from memory (only candidates with rollout data)
        training_candidates = [candidate for neg_score, candidate in self.memory if candidate.num_rollouts > 0 and candidate.mean_score() is not None]
        
        if len(training_candidates) == 0:
            if self.verbose:
                print_color("Warning: No training data available for regression model.", "yellow")
            end_time = time.time()
            elapsed_time = end_time - start_time
            if self.verbose:
                print_color(f"Regressor update completed in {elapsed_time:.4f} seconds (no training data)", "cyan")
            return
            
        # Extract raw binary training data from each candidate
        X_list = []
        y_list = []
        
        for candidate in training_candidates:
            embedding = candidate.embedding
            eval_count = candidate.num_rollouts
            mean_score = candidate.mean_score()
            
            if mean_score is None:
                continue
                
            # Calculate score_sum from mean_score and eval_count
            # Assuming scores are binary (0 or 1), score_sum = mean_score * eval_count
            score_sum = mean_score * eval_count
            
            # score_sum directly represents the number of successes
            num_successes = int(round(score_sum))
            num_failures = eval_count - num_successes
            
            # Ensure non-negative values
            num_successes = max(0, num_successes)
            num_failures = max(0, num_failures)
            
            # Create binary training samples: 1 for success, 0 for failure
            for _ in range(num_successes):
                X_list.append(embedding)
                y_list.append(1.0)
            
            for _ in range(num_failures):
                X_list.append(embedding)
                y_list.append(0.0)
        
        if len(X_list) == 0:
            print_color("Warning: No binary training samples generated.", "yellow")
            end_time = time.time()
            elapsed_time = end_time - start_time
            if self.verbose:
                print_color(f"Regressor update completed in {elapsed_time:.4f} seconds (no binary samples)", "cyan")
            return
            
        # Convert to numpy arrays
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Ensure X has the right dimensions
        if X.shape[1] != self.linear_dim:
            self.linear_dim = X.shape[1]
            # Initialize weights with larger values for more aggressive learning
            self.weights = np.random.normal(0, 0.1, self.linear_dim)
        
        # Convergence-based regularized logistic regression training using all raw binary data
        m = len(X_list)        
        # Training loop until convergence with adaptive learning rate and early stopping
        prev_cost = float('inf')
        best_cost = float('inf')
        converged = False
        iteration = 0
        patience_counter = 0
        
        # Reset learning rate
        self.learning_rate = self.initial_learning_rate
        
        for iteration in range(self.max_iterations):
            # Forward pass
            z = X.dot(self.weights) + self.bias
            predictions = self._sigmoid(z)
            
            # Compute cost with L2 regularization
            epsilon = 1e-15  # Small value to prevent log(0)
            predictions_clipped = np.clip(predictions, epsilon, 1 - epsilon)
            log_likelihood = -np.mean(y * np.log(predictions_clipped) + (1 - y) * np.log(1 - predictions_clipped))
            l2_penalty = self.regularization_strength * np.sum(self.weights ** 2)
            total_cost = log_likelihood + l2_penalty
            
            # Check for improvement and early stopping
            cost_change = abs(prev_cost - total_cost)
            if total_cost < best_cost:
                best_cost = total_cost
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Backward pass (compute gradients with L2 regularization)
            dw = (1/m) * X.T.dot(predictions - y) + 2 * self.regularization_strength * self.weights
            db = (1/m) * np.sum(predictions - y)
            gradient_norm = np.linalg.norm(dw)
            
            # Check convergence criteria (stricter)
            if cost_change < self.tolerance and gradient_norm < self.tolerance:
                converged = True
                print_color(f"Converged at iteration {iteration + 1}: cost change {cost_change:.10f}, gradient norm {gradient_norm:.10f}", "green")
                break
            
            # Early stopping if no improvement
            if patience_counter >= self.patience:
                print_color(f"Early stopping at iteration {iteration + 1}: no improvement for {self.patience} iterations", "yellow")
                break
            
            # Adaptive learning rate: decay if no improvement for several iterations
            if patience_counter > 0 and patience_counter % 10 == 0:
                self.learning_rate *= self.lr_decay_factor
                print_color(f"Reducing learning rate to {self.learning_rate:.6f}", "yellow")
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            prev_cost = total_cost
        
        # Final status
        if converged:
            print_color(f"Logistic regression converged after {iteration + 1} iterations. Final cost: {total_cost:.6f} (Log-likelihood: {log_likelihood:.6f}, L2 penalty: {l2_penalty:.6f}), bias: {self.bias:.6f}", "green")
        else:
            print_color(f"Logistic regression reached max iterations ({self.max_iterations}). Final cost: {total_cost:.6f} (Log-likelihood: {log_likelihood:.6f}, L2 penalty: {l2_penalty:.6f}), bias: {self.bias:.6f}", "yellow")
        
        # Print timing information
        end_time = time.time()
        elapsed_time = end_time - start_time
        print_color(f"Regressor update completed in {elapsed_time:.4f} seconds", "cyan")
    
    def predict_scores(self,memory = None):
        """Predict scores for all candidates in the memory."""
        # Extract all candidates from memory (memory is a list of (neg_score, candidate) tuples)
        if memory is None:
            memory = self.memory
        batch = [candidate for _, candidate in memory]

        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Collect all embeddings in order
        embeddings = []
        for candidate in batch:
            embeddings.append(candidate.embedding)
        

        # Batch prediction using vectorized operations
        X_batch = np.array(embeddings)
        z = X_batch.dot(self.weights) + self.bias
        predicted_scores = self._sigmoid(z)
        
        # Update each candidate with predicted score as attribute
        for candidate, predicted_score in zip(batch, predicted_scores):
            candidate.predicted_score = predicted_score
            
        return predicted_scores
       