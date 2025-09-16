import numpy as np
import copy
from typing import Union
from opto.trainer.loader import DataLoader
from opto.trainer.utils import batch_run, async_run
from opto.optimizers.utils import print_color
# from opto.trainer.evaluators import evaluate
from typing import Union, List, Tuple, Dict, Any, Optional
from collections import deque
from opto.utils.llm import LLM # For the selector LLM
# from opto.trace.nodes import ParameterNode
import json
# import warnings
# from black import format_str, FileMode
import random
# import mathX
from opto.utils.auto_retry import retry_with_exponential_backoff
import litellm
import time

class ModuleCandidateRegressor:
    """
    Predict scores using embedding logistic regression for ModuleCandidate objects. 
    Should have two key methods: predict_scores and predict_scores_for_batch. 
    predict_scores has no parameters, it could return predicted scores for all candidates in the memory. 
    predict_scores_for_batch has one parameter, a batch of candidates, it could return predicted scores for the batch of candidates."""
    
    def __init__(self, memory=None, embedding_model="gemini/text-embedding-004", num_threads=None, learning_rate=0.2, regularization_strength=1e-4, max_iterations=20000, tolerance=5e-3):
        # In the regressor, no need for calling LLM to make the prediction. So we could predict the entire memory at once.
        self.max_candidates_to_predict = 500
        self.memory = memory
        self.embedding_model = embedding_model
        self.num_threads = num_threads
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.regularization_strength = regularization_strength  # L2 regularization strength (lambda)
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = 20  # Early stopping patience
        self.lr_decay_factor = 0.8   # Learning rate decay factor
        # default linear dimension is 768
        self.linear_dim = 768
        # Initialize weights with larger values for more aggressive learning
        self.weights = np.random.normal(0, 0.1, self.linear_dim)
        self.bias = 0.0
        
    def _sigmoid(self, z):
        """Sigmoid activation function for logistic regression."""
        return 1.0 / (1.0 + np.exp(-z))

    def _get_parameter_text(self, candidate):
        """Get the parameter text for a ModuleCandidate."""
        if not candidate.update_dict:
            # If update_dict is empty, use a default text or base module info
            return "base_module_parameters"
        
        # Get the first value from update_dict (similar to additional_instructions)
        # TODO: support for multiple parameters
        parameter_text = list(candidate.update_dict.values())[0]
        return str(parameter_text)

    def _get_embedding(self, candidate):
        """Get the embedding for a ModuleCandidate."""
        parameter_text = self._get_parameter_text(candidate)
        
        def single_embedding_call():
            return litellm.embedding(
                model=self.embedding_model,
                input=parameter_text
            )
        
        try:
            response = retry_with_exponential_backoff(
                single_embedding_call,
                max_retries=10,
                base_delay=1.0,
                operation_name="Embedding API call"
            )
            embedding = response.data[0].embedding
            return embedding
        except Exception as e:
            print_color(f"ERROR: Embedding API call failed after retries: {e}", "red")
            # Return a random embedding as fallback to prevent complete failure
            print_color("Using random embedding as fallback", "yellow")
            fallback_embedding = np.random.normal(0, 0.01, self.linear_dim)
            return fallback_embedding / np.linalg.norm(fallback_embedding)
    
    def _update_memory_embeddings_for_batch(self, batch):
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
                max_workers=1000,
                description=f"Generating embeddings for {len(candidates_needing_embeddings)} candidates"
            )
            
            # Assign embeddings back to candidates
            for candidate, embedding in zip(candidates_needing_embeddings, new_embeddings):
                candidate.embedding = embedding

    def update(self):
        """Update the regression model parameters using the current memory with logistic regression."""
        start_time = time.time()
        print_color("Updating regression model using the current memory with logistic regression...", "blue")
        # Extract candidates from memory (memory contains (neg_score, candidate) tuples)
        batch = [candidate for _, candidate in self.memory]
        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Get training data from memory (only candidates with rollout data)
        training_candidates = [candidate for neg_score, candidate in self.memory if candidate.num_rollouts > 0 and candidate.mean_score() is not None]
        
        if len(training_candidates) == 0:
            print_color("Warning: No training data available for regression model.", "yellow")
            end_time = time.time()
            elapsed_time = end_time - start_time
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
        # print_color(f"Training regularized logistic regression with {m} binary samples from {len(training_candidates)} candidates until convergence.", "blue")
        # print_color(f"Using L2 regularization strength: {self.regularization_strength}, learning rate: {self.learning_rate}", "blue")
        # print_color(f"Max iterations: {self.max_iterations}, tolerance: {self.tolerance}", "blue")
        
        # Debug: Print initial weight statistics
        initial_weight_norm = np.linalg.norm(self.weights)
        # print_color(f"Initial weight norm: {initial_weight_norm:.6f}", "yellow")
        
        # Debug: Print embedding statistics
        embedding_mean = np.mean(X)
        embedding_std = np.std(X)
        embedding_norm_mean = np.mean([np.linalg.norm(row) for row in X])
        # print_color(f"Embedding stats - mean: {embedding_mean:.6f}, std: {embedding_std:.6f}, avg norm: {embedding_norm_mean:.6f}", "yellow")
        
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
            
            # Print progress periodically
            # if iteration == 0 or (iteration + 1) % max(1, min(50, self.max_iterations // 20)) == 0:
            #     z_mean, z_std = np.mean(z), np.std(z)
            #     weight_norm = np.linalg.norm(self.weights)
                # print_color(f"Iteration {iteration + 1}: Cost: {total_cost:.6f} (change: {cost_change:.8f}), LR: {self.learning_rate:.6f}, Weight norm: {weight_norm:.6f}, Gradient norm: {gradient_norm:.8f}", "cyan")
                # print_color(f"  Logits - mean: {z_mean:.6f}, std: {z_std:.6f}, range: [{np.min(z):.6f}, {np.max(z):.6f}]", "cyan")
                # print_color(f"  Predictions - range: [{np.min(predictions):.6f}, {np.max(predictions):.6f}], mean: {np.mean(predictions):.6f}", "cyan")
                # print_color(f"  Patience: {patience_counter}/{self.patience}", "cyan")
            
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
       