import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from opto.trainer.utils import  async_run
from opto.optimizers.utils import print_color
from typing import Union, List, Tuple, Dict, Any, Optional
from opto.utils.auto_retry import retry_with_exponential_backoff
import litellm
import time
from opto.features.priority_search.priority_search import ModuleCandidate


class GaussianRandomProjection:
    """
    Gaussian Random Projection for dimensionality reduction of linear features.
    
    Projects high-dimensional embeddings to a lower-dimensional space using a random
    Gaussian matrix. This preserves approximate distances (Johnson-Lindenstrauss lemma)
    while reducing computational complexity.
    """
    
    def __init__(self, input_dim: int, output_dim: int, random_seed: Optional[int] = None):
        """
        Initialize Gaussian Random Projection.
        
        Args:
            input_dim: Original dimensionality of features
            output_dim: Target dimensionality after projection
            random_seed: Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.random_seed = random_seed
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Create random projection matrix: N(0, 1/sqrt(output_dim))
        # This scaling ensures approximate preservation of distances
        self.projection_matrix = np.random.normal(
            0, 1/np.sqrt(self.output_dim), 
            size=(self.input_dim, self.output_dim)
        )
        
        print_color(f"Initialized Gaussian Random Projection: {input_dim}D → {output_dim}D", "green")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using the fitted projection matrix.
        
        Args:
            X: Input features of shape (n_samples, input_dim)
            
        Returns:
            Projected features of shape (n_samples, output_dim)
        """
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input dimension mismatch: expected {self.input_dim}, got {X.shape[1]}")
        
        # Apply random projection: X_projected = X @ projection_matrix

        # calculate the norm of the features, before and after projection, for debugging
        # before_norm = np.linalg.norm(X, axis=1)
        # after_norm = np.linalg.norm(X @ self.projection_matrix, axis=1)
        # print_color(f"Before projection norm: {before_norm}, after projection norm: {after_norm}", "green")

        # I have tested the norm of the features before projection is 1, after projection is almost 1.
        return X @ self.projection_matrix
    
    def get_projected_dim(self) -> int:
        """Return the output dimensionality after projection."""
        return self.output_dim


class RegressorTemplate:
    """Base class template for regression-based predictors for ModuleCandidate objects.
    
    Provides common functionality for embedding generation and candidate processing.
    Subclasses should implement update() and predict_scores() methods.
    """
    
    def __init__(self, embedding_model="gemini/text-embedding-004", num_threads=None, regularization_strength=1, linear_dim=None):
        # In the regressor, no need for calling LLM to make the prediction. So we could predict the entire memory at once.
        self.max_candidates_to_predict = 500
        self.embedding_model = embedding_model
        self.num_threads = num_threads
        self.regularization_strength = regularization_strength  # L2 regularization strength (lambda)
        
        # Default original embedding dimension (from text-embedding-004)
        self.original_embedding_dim = 768
        
        if linear_dim is not None:
            # Use random projection from 768D to linear_dim
            self.linear_dim = linear_dim
            print_color(f"Using random projection: {self.original_embedding_dim}D → {linear_dim}D", "blue")
            self.random_projector = GaussianRandomProjection(
                input_dim=self.original_embedding_dim,
                output_dim=linear_dim,
                random_seed=42
            )
        else:
            # Use default 768D without projection
            self.linear_dim = self.original_embedding_dim
            self.random_projector = None
            
        # Initialize weights with larger values for more aggressive learning
        self.weights = np.random.normal(0, 0.1, self.linear_dim)
        self.bias = 0.0

    def _get_parameter_text(self, candidate):
        """Get the parameter text for a ModuleCandidate."""
        if not candidate.update_dict:
            # If update_dict is empty, use a default text or base module info
            return "base_module_parameters"
        
        # Get the first value from update_dict (similar to additional_instructions)
        # Support for multiple parameters
        # Return a text representation of the dictionary
        parameter_text = str(candidate.update_dict)
        return parameter_text

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
            if self.random_projector is not None:
                # Convert to numpy array and reshape for transform (expects 2D: n_samples x n_features)
                embedding_array = np.array(embedding).reshape(1, -1)
                projected = self.random_projector.transform(embedding_array)
                # Convert back to list and flatten
                embedding = projected.flatten().tolist()
            return embedding
        except Exception as e:
            print_color(f"ERROR: Embedding API call failed after retries: {e}", "red")
            return None
    
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
                max_workers=50,
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

class LogisticRegressor(RegressorTemplate):
    """
    Predict scores using embedding logistic regression for ModuleCandidate objects. 
    Should have two key methods: predict_scores and predict_scores_for_batch. 
    predict_scores has no parameters, it could return predicted scores for all candidates in the memory. 
    predict_scores_for_batch has one parameter, a batch of candidates, it could return predicted scores for the batch of candidates."""
    
    def __init__(self, embedding_model="gemini/text-embedding-004", num_threads=None, learning_rate=0.001, regularization_strength=1, max_iterations=20000, tolerance=5e-3, linear_dim=None):
        super().__init__(embedding_model, num_threads, regularization_strength,linear_dim)
        # Logistic regression specific parameters
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.patience = 20  # Early stopping patience
        
        # Convert weights and bias to PyTorch tensors
        self.weights_tensor = torch.tensor(self.weights, dtype=torch.float32, requires_grad=True)
        self.bias_tensor = torch.tensor(self.bias, dtype=torch.float32, requires_grad=True)
        
        # Initialize Adam optimizer (will be reinitialized when dimensions change)
        self.optimizer = None
        
    def _sigmoid(self, z):
        """Sigmoid activation function for logistic regression."""
        return 1.0 / (1.0 + np.exp(-z))

    def update(self, memory: List[Tuple[float, ModuleCandidate]]):
        """This function update the regression model parameters using the input batch of candidates.
        Input:
            batch: a list of candidates.
        """
        start_time = time.time()
        batch = [candidate for _, candidate in memory]
        print_color("Updating regression model using the memory with logistic regression...", "blue")
        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Get training data from memory (only candidates with rollout data)
        training_candidates = [candidate for candidate in batch if candidate.num_rollouts > 0 and candidate.mean_score() is not None]
        
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
            scores = [r['score'] for r in candidate.rollouts]
            for score in scores:
                if score is None:
                    continue
                else:
                    X_list.append(embedding)
                    y_list.append(score)
                
        
        if len(X_list) == 0:
            print_color("Warning: No binary training samples generated.", "yellow")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print_color(f"Regressor update completed in {elapsed_time:.4f} seconds (no binary samples)", "cyan")
            return
        print_color(f"Updating regression model with {len(training_candidates)} candidates ({len(X_list)} binary samples)...", "blue")
        # Convert to PyTorch tensors
        X = torch.tensor(X_list, dtype=torch.float32)
        y = torch.tensor(y_list, dtype=torch.float32)
        
        # Assert that dimensions match - this should never fail if properly initialized
        assert X.shape[1] == self.linear_dim, f"Feature dimension mismatch: expected {self.linear_dim}, got {X.shape[1]}. This indicates an initialization error."
        
        # Initialize Adam optimizer with current parameters
        self.optimizer = optim.Adam([self.weights_tensor, self.bias_tensor], 
                                   lr=self.learning_rate, 
                                   weight_decay=self.regularization_strength)
        
        # Training loop with PyTorch and Adam optimizer
        prev_loss = float('inf')
        best_loss = float('inf')
        converged = False
        patience_counter = 0
        
        print_color(f"Training with PyTorch and Adam optimizer: lr={self.learning_rate}, weight_decay={self.regularization_strength}", "blue")
        
        for iteration in range(self.max_iterations):
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            z = X @ self.weights_tensor + self.bias_tensor
            predictions = torch.sigmoid(z)
            
            # Compute binary cross-entropy loss (PyTorch handles regularization via weight_decay)
            loss = nn.functional.binary_cross_entropy(predictions, y)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            current_loss = loss.item()
            
            # Check for improvement and early stopping
            loss_change = abs(prev_loss - current_loss)
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Check convergence criteria
            if loss_change < self.tolerance:
                converged = True
                print_color(f"Converged at iteration {iteration + 1}: loss change {loss_change:.10f}", "green")
                break
            
            # Early stopping if no improvement
            if patience_counter >= self.patience:
                print_color(f"Early stopping at iteration {iteration + 1}: no improvement for {self.patience} iterations", "yellow")
                break
            
            prev_loss = current_loss
        
        # Update numpy versions for compatibility with existing code
        self.weights = self.weights_tensor.detach().numpy()
        self.bias = self.bias_tensor.item()
        
        # Final status
        final_loss = loss.item()
        if converged:
            print_color(f"PyTorch logistic regression converged after {iteration + 1} iterations. Final loss: {final_loss:.6f}, bias: {self.bias:.6f}", "green")
        else:
            print_color(f"PyTorch logistic regression reached max iterations ({self.max_iterations}). Final loss: {final_loss:.6f}, bias: {self.bias:.6f}", "yellow")
        
        # Print timing information
        end_time = time.time()
        elapsed_time = end_time - start_time
        print_color(f"Regressor update completed in {elapsed_time:.4f} seconds", "cyan")
    
    def predict_scores(self,memory):
        """Predict scores for all candidates in the memory."""
        # Extract all candidates from memory (memory is a list of (neg_score, candidate) tuples)
        if len(memory) == 0:
            return
        batch = [candidate for _, candidate in memory]

        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Collect all embeddings in order
        embeddings = []
        for candidate in batch:
            if candidate.embedding:
                embeddings.append(candidate.embedding)
            else:
                candidate.embedding = self._get_embedding(candidate)
                embeddings.append(candidate.embedding)
        

        # Batch prediction using vectorized operations
        X_batch = np.array(embeddings)
        z = X_batch @ self.weights + self.bias
        predicted_scores = self._sigmoid(z)
        
        # Update each candidate with predicted score as attribute
        for candidate, predicted_score in zip(batch, predicted_scores):
            candidate.predicted_score = predicted_score
            
        return predicted_scores

class LinearRegressor(RegressorTemplate):
    """Use closed-form solution for regularized linear regression."""
    
    def __init__(self, embedding_model="gemini/text-embedding-004", num_threads=None, regularization_strength=1, transformation_exploration_factor=0.0, linear_dim=None):
        super().__init__(embedding_model, num_threads, regularization_strength, linear_dim)
        # The transformation exploration factor should lie in [0,1]
        assert 0 <= transformation_exploration_factor <= 1, "Transformation exploration factor must be between 0 and 1"
        self.transformation_exploration_factor = transformation_exploration_factor  # 0: [0,1] -> [0,1], 1: [0,1] -> [-1,0]
    
    def _transform_targets(self, y):
        """Transform targets from [0,1] to exploration range based on transformation_exploration_factor."""
        # General formula: [0,1] -> [0-factor, 1-factor] = [-factor, 1-factor]
        return y - self.transformation_exploration_factor
    
    def _inverse_transform_predictions(self, predictions):
        """Transform predictions back from exploration range to [0,1]."""
        # Inverse of the transformation: add back the transformation factor
        return predictions + self.transformation_exploration_factor
    
    def update(self, memory: List[Tuple[float, ModuleCandidate]]):
        """Update the regression model parameters using the input batch of candidates.
        Uses closed-form solution for L2 regularized linear regression: w = (X^T X + λI)^(-1) X^T y
        """
        start_time = time.time()
        batch = [candidate for _, candidate in memory]
        print_color("Updating linear regression model using closed-form solution...", "blue")
        
        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Get training data from memory (only candidates with rollout data)
        training_candidates = [candidate for candidate in batch if candidate.num_rollouts > 0 and candidate.mean_score() is not None]
        
        if len(training_candidates) == 0:
            print_color("Warning: No training data available for linear regression model.", "yellow")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print_color(f"Linear regressor update completed in {elapsed_time:.4f} seconds (no training data)", "cyan")
            return
            
        # Extract features (embeddings) and targets (mean scores)
        X_list = []
        y_list = []
        
        for candidate in training_candidates:
            embedding = candidate.embedding
            scores = [r['score'] for r in candidate.rollouts]
            # Filter out None scores
            for score in scores:
                if score is None:
                    continue
                else:
                    X_list.append(embedding)
                    y_list.append(score)
        
        if len(X_list) == 0:
            print_color("Warning: No valid training samples for linear regression.", "yellow")
            end_time = time.time()
            elapsed_time = end_time - start_time
            print_color(f"Linear regressor update completed in {elapsed_time:.4f} seconds (no valid samples)", "cyan")
            return
            
        print_color(f"Updating linear regression model with {len(training_candidates)} candidates ({len(X_list)} samples)...", "blue")
        
        # Convert to numpy arrays
        X = np.array(X_list)  # Shape: (n_samples, n_features)
        y = np.array(y_list)  # Shape: (n_samples,)
        
        # Apply transformation to target values for exploration
        y_transformed = self._transform_targets(y)
        
        # Update dimensions if needed
        if X.shape[1] != self.linear_dim:
            self.linear_dim = X.shape[1]
            
        # Closed-form solution for L2 regularized linear regression
        # w = (X^T X + λI)^(-1) X^T y
        # where λ is the regularization strength
        
        n_features = X.shape[1]
        lambda_reg = self.regularization_strength
        
        # Add bias term by augmenting X with a column of ones
        X_augmented = np.column_stack([X, np.ones(X.shape[0])])  # Shape: (n_samples, n_features + 1)
        
        # Create regularization matrix (regularize both weights and bias)
        reg_matrix = lambda_reg * np.eye(n_features + 1)
        
        try:
            # Solve the normal equation: (X^T X + λI) w = X^T y
            XTX_reg = X_augmented.T @ X_augmented + reg_matrix
            XTy = X_augmented.T @ y_transformed
            
            # Use numpy's linear solver for better numerical stability
            w_augmented = np.linalg.solve(XTX_reg, XTy)
            
            # Save the regularized Gram matrix
            self.cov = XTX_reg
            
            # Extract weights and bias
            self.weights = w_augmented[:-1]  # All but last element
            self.bias = w_augmented[-1]     # Last element
            
            # Calculate training error for reporting (using transformed targets)
            predictions_transformed = X @ self.weights + self.bias
            mse = np.mean((predictions_transformed - y_transformed) ** 2)
            l2_penalty = lambda_reg * np.sum(self.weights ** 2)
            total_cost = mse + l2_penalty
            
            print_color(f"Linear regression solved successfully. MSE: {mse:.6f}, L2 penalty: {l2_penalty:.6f}, Total cost: {total_cost:.6f}, Bias: {self.bias:.6f}", "green")
            
        except np.linalg.LinAlgError as e:
            print_color(f"Warning: Linear algebra error in closed-form solution: {e}", "yellow")
            print_color("Falling back to pseudo-inverse solution...", "yellow")
            
            # Fallback to pseudo-inverse if regular solve fails
            try:
                XTX_reg_stabilized = XTX_reg + 1e-6 * np.eye(XTX_reg.shape[0])
                w_augmented = np.linalg.solve(XTX_reg_stabilized, XTy)
                self.cov = XTX_reg_stabilized
                self.weights = w_augmented[:-1]
                self.bias = w_augmented[-1]
                print_color("Regularized solve solution computed successfully.", "green")
            except Exception as e2:
                print_color(f"Error: Both regular and pseudo-inverse solutions failed: {e2}", "red")
                print_color("Keeping previous weights and bias.", "yellow")
        
        # Print timing information
        end_time = time.time()
        elapsed_time = end_time - start_time
        print_color(f"Linear regressor update completed in {elapsed_time:.4f} seconds", "cyan")

    def predict_scores(self, memory: List[Tuple[float, ModuleCandidate]]):
        """Predict scores for all candidates in the memory."""
        # Extract all candidates from memory (memory is a list of (neg_score, candidate) tuples)
        if len(memory) == 0:
            return
        batch = [candidate for _, candidate in memory]

        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Collect all embeddings in order
        embeddings = []
        for candidate in batch:
            embeddings.append(candidate.embedding)
        

        # Batch prediction using vectorized operations
        X_batch = np.array(embeddings)
        predicted_scores_transformed = X_batch @ self.weights + self.bias
        
        # Transform predictions back to [0,1] range
        predicted_scores = self._inverse_transform_predictions(predicted_scores_transformed)

        # Clip predicted scores to be between 0 and 1
        # predicted_scores = np.clip(predicted_scores, 0, 1)
        
        # Update each candidate with predicted score as attribute
        for candidate, predicted_score in zip(batch, predicted_scores):
            candidate.predicted_score = float(predicted_score)
            
        return predicted_scores

class LinearUCBRegressor(LinearRegressor):
    """Linear UCB regressor that uses Upper Confidence Bound scores for exploration-exploitation balance."""
    
    def __init__(self, embedding_model="gemini/text-embedding-004", num_threads=None, regularization_strength=1, alpha=0.3, transformation_exploration_factor=0.0, linear_dim=None):
        super().__init__(embedding_model, num_threads, regularization_strength, transformation_exploration_factor, linear_dim)
        self.alpha = alpha  # UCB exploration parameter
        self.cov = None     # Will be set during update()
    
    def predict_scores(self, memory: List[Tuple[float, ModuleCandidate]]):
        """Predict UCB scores for all candidates in the memory.
        UCB score = predicted_reward + alpha * sqrt(x^T * Cov * x)
        where Cov is the covariance matrix from linear regression.
        """
        # Extract all candidates from memory (memory is a list of (neg_score, candidate) tuples)
        if len(memory) == 0:
            return
        batch = [candidate for _, candidate in memory]

        # Ensure all candidates have embeddings
        self._update_memory_embeddings_for_batch(batch)
        
        # Check if we have a trained model
        if not hasattr(self, 'weights') or self.weights is None or not hasattr(self, 'cov') or self.cov is None:
            print_color("Warning: No trained model available for UCB predictions. Using random scores.", "yellow")
            # Return random scores as fallback
            predicted_scores = np.random.uniform(0, 1, len(batch))
            for candidate, predicted_score in zip(batch, predicted_scores):
                candidate.predicted_score = float(predicted_score)
                # Also set mean_prediction for consistency with main prediction path
                candidate.mean_prediction = float(predicted_score)
            return predicted_scores
        
        # Collect all embeddings in order
        embeddings = []
        for candidate in batch:
            embeddings.append(candidate.embedding)

        # Batch prediction using vectorized operations
        X_batch = np.array(embeddings)  # Shape: (n_candidates, n_features)
        
        # Augment features with bias term (add column of ones)
        X_augmented = np.column_stack([X_batch, np.ones(X_batch.shape[0])])  # Shape: (n_candidates, n_features + 1)
        
        # Compute mean predictions in transformed space (exploitation term)
        w_augmented = np.concatenate([self.weights, [self.bias]])  # Combine weights and bias
        mean_predictions_transformed = X_augmented @ w_augmented
        
        # Compute confidence bounds (exploration term) - vectorized
        # confidence = sqrt(x^T * Cov^(-1) * x) for each candidate
        # Since self.cov now stores XTX_reg, we need to solve instead of multiply
        # Vectorized computation using solve for better numerical stability
        cov_X_T = np.linalg.solve(self.cov, X_augmented.T)
        confidence_bounds = np.sqrt(np.sum(X_augmented * cov_X_T.T, axis=1))
        
        # print the mean scores and confidence bounds for debugging and tuning hyperparameters
        print_color(f"Mean predictions: {mean_predictions_transformed}", "green")
        print_color(f"Bonus (without alpha): {confidence_bounds}", "green")
        
        # Compute UCB scores in transformed space: mean + alpha * confidence
        ucb_scores_transformed = mean_predictions_transformed + self.alpha * confidence_bounds
        
        # Transform predictions back to [0,1] range
        ucb_scores = self._inverse_transform_predictions(ucb_scores_transformed)
        mean_predictions = self._inverse_transform_predictions(mean_predictions_transformed)
        
        # Clip UCB scores to be between 0 and 1
        # predicted_scores = np.clip(ucb_scores, 0, 1)
        # mean_predictions = np.clip(mean_predictions, 0, 1)
        
        # Update each candidate with predicted UCB score as attribute
        for i, candidate in enumerate(batch):
            candidate.predicted_score = float(predicted_scores[i])
            # Also store the individual components for debugging/analysis
            candidate.mean_prediction = float(mean_predictions[i])
            # candidate.confidence_bound = float(confidence_bounds[i])
            # candidate.ucb_score = float(ucb_scores[i])
            
        return predicted_scores
DOMAIN_CONTEXT = """## Problem Context and Domain Knowledge
                    You are a score prediction model for tau-bench agent configurations. You are optimizing agents for tool-agent-user interaction in real-world domains (airline and retail environments).

                    **Core Optimization Task:**
                    - **Agent Type**: Tool-calling agents that help users complete complex multi-step tasks
                    - **Parameters**: Agents have configurable parameters like tools_info (tool descriptions) and additional_instructions (strategic guidance)
                    - **Performance Metric**: Success rate on completing user tasks correctly within the domain constraints
                    - **Environments**: Airline (flight bookings, cancellations, changes) and Retail (orders, returns, exchanges)

                    **Key Success Factors:**
                    - **Tool Usage**: Agents must use the right tools at the right time with correct parameters
                    - **User Interaction**: Effective communication and confirmation of actions with users
                    - **Domain Constraints**: Following business rules (authentication requirements, policy compliance)
                    - **Error Handling**: Graceful recovery from failures and providing alternative solutions
                    - **Workflow Efficiency**: Completing tasks with minimal back-and-forth while being thorough

                    **Common Failure Modes:**
                    - Using wrong tools or incorrect tool parameters
                    - Missing critical authentication or verification steps
                    - Poor user communication leading to misunderstandings
                    - Incomplete task completion or taking unintended actions
                    - Not following domain-specific business rules and constraints

                    **Optimization Strategy:**
                    Better parameter configurations lead to higher task success rates. The goal is to find parameter settings that maximize agent performance across diverse scenarios in the target domain.
                 """

from opto.utils.llm import LLM
import copy
import random
import xml.etree.ElementTree as ET
from xml.etree.ElementTree import ParseError
import re

class LLMRegressor:
    """
    A LLM regressor to predict scores for a batch of candidates.
    """
    def __init__(self, model_name="gemini/gemini-2.0-flash", temperature=0.0, max_candidates_per_prompt=50, max_candidates_to_predict=20, num_repetitions=5, num_threads=None):
        self.LLM = LLM(model=model_name)
        self.temperature = temperature
        self.max_candidates_per_prompt = max_candidates_per_prompt
        self.max_candidates_to_predict = max_candidates_to_predict
        self.num_repetitions = num_repetitions
        self.num_threads = num_threads
        self.training_data = []  # Store candidates with evaluation data
    
    def update(self, memory: List[Tuple[float, ModuleCandidate]]):
        """Update the training data with candidates that have evaluation results."""
        start_time = time.time()
        batch = [candidate for _, candidate in memory]
        print_color("Updating LLM regressor training data...", "blue")
        
        # Extract candidates with rollout data and calculate their statistics
        self.training_data = []
        for candidate in batch:
            if candidate.num_rollouts > 0 and candidate.mean_score() is not None:
                # Create training entry with statistics
                training_entry = {
                    'candidate': candidate,
                    'params': self._get_candidate_params(candidate),
                    'eval_count': candidate.num_rollouts,
                    'mean_score': candidate.mean_score(),
                    'score_sum': candidate.mean_score() * candidate.num_rollouts,
                    'score_variance': self._calculate_score_variance(candidate)
                }
                self.training_data.append(training_entry)
        if len(self.training_data) == 0:
            print_color("Warning: No training data available for LLM regressor.", "yellow")
        else:
            print_color(f"LLM regressor updated with {len(self.training_data)} training candidates", "green")
        
        # Print timing information
        end_time = time.time()
        elapsed_time = end_time - start_time
        print_color(f"LLM regressor update completed in {elapsed_time:.4f} seconds", "cyan")
    
    def _get_candidate_params(self, candidate):
        """Extract parameters from a ModuleCandidate for LLM processing."""
        assert hasattr(candidate, 'update_dict'), "Candidate must have an update_dict"
        assert candidate.update_dict, "Candidate update_dict must not be empty"
        
        return {k.py_name if hasattr(k, 'py_name') else str(k): str(v) for k, v in candidate.update_dict.items()}
        
    
    def predict_scores(self, memory: List[Tuple[float, ModuleCandidate]]):
        """Predict scores for all candidates in the memory."""
        if len(memory) == 0:
            print_color("Warning: No candidates in memory. Returning empty array.", "yellow")
            return np.array([])
        
        start_time = time.time()
        batch = [candidate for _, candidate in memory]
        print_color(f"LLM regressor predicting scores for {len(batch)} candidates...", "blue")
        
        # Check if we have training data
        if len(self.training_data) == 0:
            print_color("Warning: No training data available for LLM regressor. Using random scores.", "yellow")
            # Return random scores as fallback
            predicted_scores = np.random.uniform(0, 1, len(batch))
            for candidate, predicted_score in zip(batch, predicted_scores):
                candidate.predicted_score = float(predicted_score)
            return predicted_scores
        
        # Divide candidates into smaller batches
        candidate_batches = [batch[i:i+self.max_candidates_to_predict] 
                           for i in range(0, len(batch), self.max_candidates_to_predict)]
        
        all_predicted_scores = []
        
        # Process each batch
        if hasattr(self, 'num_threads') and self.num_threads and self.num_threads > 1:
            # Parallel processing of batches
            batch_functions = [lambda cb=candidate_batch: self._predict_scores_for_batch(cb) 
                             for candidate_batch in candidate_batches]
            batch_results = async_run(
                batch_functions,
                max_workers=200,
                description=f"Processing {len(candidate_batches)} candidate batches"
            )
            # Flatten results
            for batch_scores in batch_results:
                all_predicted_scores.extend(batch_scores)
        else:
            # Sequential processing
            for candidate_batch in candidate_batches:
                batch_scores = self._predict_scores_for_batch(candidate_batch)
                all_predicted_scores.extend(batch_scores)
        
        # Update candidates with predicted scores
        for candidate, predicted_score in zip(batch, all_predicted_scores):
            candidate.predicted_score = float(predicted_score)
        
        # Print timing information
        end_time = time.time()
        elapsed_time = end_time - start_time
        print_color(f"LLM regressor prediction completed in {elapsed_time:.4f} seconds", "cyan")
        
        return np.array(all_predicted_scores)

    def _calculate_score_variance(self, candidate):
        """Calculate the variance of the scores for a candidate."""
        scores = [rollout['score'] for rollout in candidate.rollouts if rollout['score'] is not None]
        return np.var(scores)

    def _predict_scores_for_batch(self, candidate_batch):
        """Predict scores for a single batch of candidates."""
        # Convert candidates to prediction format
        batch_to_predict = []
        for candidate in candidate_batch:
            # calculate the variance of the scores for the candidate
            
            entry = {
                'candidate': candidate,
                'params': self._get_candidate_params(candidate),
                # 'eval_count': getattr(candidate, 'num_rollouts', 0),
                # 'mean_score': candidate.mean_score() if hasattr(candidate, 'mean_score') and candidate.mean_score() is not None else 0.0,
                # 'score_variance': self._calculate_score_variance(candidate)
            }
            batch_to_predict.append(entry)
        
        # Perform multiple prediction rounds with different training subsets
        if hasattr(self, 'num_threads') and self.num_threads and self.num_threads > 1:
            # Parallel repetitions
            def single_round():
                subset = self._sample_training_subset()
                return self._call_regressor(subset, batch_to_predict)
            
            round_functions = [single_round for _ in range(self.num_repetitions)]
            predicted_scores_all_rounds = async_run(
                round_functions,
                max_workers=self.num_threads,
                description=f"Running {self.num_repetitions} prediction rounds"
            )
        else:
            # Sequential repetitions
            predicted_scores_all_rounds = []
            for round_idx in range(self.num_repetitions):
                subset = self._sample_training_subset()
                predicted_scores = self._call_regressor(subset, batch_to_predict)
                predicted_scores_all_rounds.append(predicted_scores)
        
        # Calculate average predicted scores across all rounds
        # There may be None values in individual candidate scores, only keep the non-None scores to calculate the average
        num_candidates = len(batch_to_predict)
        avg_predicted_scores = []
        
        # Convert to numpy array for easier handling, but handle None values
        scores_matrix = np.array(predicted_scores_all_rounds, dtype=object)
        
        for candidate_idx in range(num_candidates):
            # Extract all scores for this candidate across all rounds
            candidate_scores = scores_matrix[:, candidate_idx]
            
            # Filter out None values and calculate average
            valid_scores = [score for score in candidate_scores if score is not None]
            avg_score = np.mean(valid_scores) if valid_scores else 0.0 # Predict 0 at this step, so this candidate with prediction error will not be selected
            
            avg_predicted_scores.append(avg_score)
        
        return avg_predicted_scores
    
    def _sample_training_subset(self):
        """Sample a subset of training data for prompt construction."""
        if len(self.training_data) == 0:
            return []
        
        batch_size = min(self.max_candidates_per_prompt, len(self.training_data))
        subset = random.sample(self.training_data, batch_size)
        return subset
    
    def _call_regressor(self, training_subset, batch_to_predict):
        """Call the LLM regressor to make predictions."""
        # Default fallback: return zeros for all candidates to predict
        default_scores = np.zeros(len(batch_to_predict))
        
        # Randomly shuffle the training subset for randomized LLM presentation
        shuffled_subset = training_subset.copy()
        random.shuffle(shuffled_subset)
        
        # Prepare XML for training data
        if not shuffled_subset:
            training_candidates_xml = "<training_candidates>\n  <note>No available data</note>\n</training_candidates>"
        else:
            training_candidates_xml = "<training_candidates>\n"
            for idx, entry in enumerate(shuffled_subset):
                training_candidates_xml += f"  <candidate index='{idx}'>\n"
                training_candidates_xml += f"    <eval_count>{entry['eval_count']}</eval_count>\n"
                training_candidates_xml += f"    <mean_score>{entry['mean_score']}</mean_score>\n"
                training_candidates_xml += f"    <score_variance>{entry['score_variance']}</score_variance>\n"
                training_candidates_xml += "    <parameters>\n"
                for param_name, param_value in entry['params'].items():
                    param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
                    training_candidates_xml += f"      <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
                training_candidates_xml += "    </parameters>\n"
                training_candidates_xml += "  </candidate>\n"
            training_candidates_xml += "</training_candidates>"
        
        # Randomly shuffle prediction batch and track original order
        shuffled_prediction_with_original_idx = [(i, entry) for i, entry in enumerate(batch_to_predict)]
        random.shuffle(shuffled_prediction_with_original_idx)
        shuffled_prediction_batch = [entry for _, entry in shuffled_prediction_with_original_idx]
        shuffled_to_original_idx = {shuffled_idx: original_idx for shuffled_idx, (original_idx, _) in enumerate(shuffled_prediction_with_original_idx)}
        
        # Prepare XML for prediction candidates
        prediction_candidates_xml = "<prediction_candidates>\n"
        for idx, entry in enumerate(shuffled_prediction_batch):
            prediction_candidates_xml += f"  <candidate index='{idx}'>\n"
            prediction_candidates_xml += "    <parameters>\n"
            for param_name, param_value in entry['params'].items():
                param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
                prediction_candidates_xml += f"      <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
            prediction_candidates_xml += "    </parameters>\n"
            prediction_candidates_xml += "  </candidate>\n"
        prediction_candidates_xml += "</prediction_candidates>"
        
        # Create parameter schema XML
        if shuffled_subset:
            example_params = shuffled_subset[0]['params']
        elif batch_to_predict:
            example_params = batch_to_predict[0]['params']
        else:
            example_params = {}
        
        example_param_schema_xml = "<parameter_schema>\n"
        for param_name, param_value in example_params.items():
            param_value_escaped = str(param_value).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&apos;')
            example_param_schema_xml += f"  <parameter name='{param_name}'><![CDATA[{param_value_escaped}]]></parameter>\n"
        example_param_schema_xml += "</parameter_schema>"
        
        # Create the prompt
        example_format = '''<prediction_result>
        <Deep Data Examination>
            [Comprehensive analysis of ALL training data - examine all training candidates' parameters and scores, identify parameter-performance relationships, understand what makes parameters effective or ineffective, compare similar and different candidates to understand patterns, build overall understanding that will inform all predictions]
        </Deep Data Examination>
        <score_estimates>
            <candidate index="0">
            <reasoning>[Detailed reasoning for this specific candidate - analyze the parameters thoroughly, compare to similar training examples from your analysis above, explain your logic for the predicted score, discuss confidence level and any uncertainty]</reasoning>
            <predicted_score>0.XX</predicted_score>
            </candidate>
        </score_estimates>
        </prediction_result>'''
        
        prompt_messages = [
            {
                "role": "system",
                "content": f"""
        {DOMAIN_CONTEXT}

        ## Function Approximation Objective
        You are a **parameter-to-score function approximator**. Your goal is to learn the mapping from candidate parameters to their performance scores using training data, then apply this learned function to predict scores for new candidates.

        ## Core Capabilities
        1. **Pattern Learning**: Extract parameter-performance correlations from observed data
        2. **Function Mapping**: Build a parameter → score mapping function from patterns
        3. **Noise Reduction**: Use cross-candidate patterns to denoise observed scores
        4. **Score Prediction**: Apply learned function to predict scores for all candidates (observed and unobserved)

        ## Key Insights for Function Approximation
        - **Observed scores contain noise**: Raw scores may not reflect true performance due to evaluation variance
        - **Parameters reveal true performance**: Similar parameters should yield similar scores
        - **Cross-candidate learning**: Information from one candidate can improve predictions for others
        - **Pattern-based denoising**: Use parameter similarities to correct noisy observations

        ## Analysis Approach

        ### Step 1: Deep Data Examination
        **First, thoroughly analyze ALL available training data:**
        - Examine all training candidates' parameters and their observed scores
        - Look for relationships between parameter characteristics and performance
        - Identify what makes parameters effective or ineffective
        - Compare similar and different candidates to understand patterns
        - Build overall understanding of the parameter-performance relationship

        ### Step 2: Individual Candidate Reasoning
        **Then, for each prediction candidate, provide detailed reasoning:**
        - Analyze the candidate's specific parameters thoroughly
        - Compare to similar training examples from your analysis
        - Explain your reasoning for the predicted score
        - Be honest about uncertainty and confidence level

        ## Prediction Methodology
        1. **For training candidates**: Use parameter patterns to denoise raw scores
        - If raw score seems inconsistent with parameter quality, adjust based on similar candidates
        - Consider eval_count (higher count = more reliable, but still may need correction. The mean_score is the empirical success rate on eval_count tasks. So if eval_count is very small, the mean_score may not be reliable.)
        2. **For prediction candidates**: Use parameter-based function approximation
        - Find candidates with similar parameter profiles from training data
        - Apply learned parameter-performance mappings
        - Predict score based on parameter quality indicators

        ## Output Requirements
        Return ONLY an XML structure with these two elements:
        - <Deep Data Examination>: **Comprehensive analysis of ALL training data** - Examine all training candidates' parameters and scores, identify parameter-performance relationships, understand what makes parameters effective, and build foundational insights for predictions.
        - <score_estimates>: **For each prediction candidate, provide detailed reasoning** - Analyze the specific parameters, compare to training examples, explain prediction logic, and assess confidence level. Each candidate needs thorough reasoning.

        ## Example Output Format
        {example_format}

        **CRITICAL**: Ensure all XML tags are properly closed. Focus on learning from training data to predict scores for new candidates.
        """,
            },
            {
                "role": "user", 
                "content": f"""
        ## Training Data (Candidates with Observed Scores)
        {training_candidates_xml}

        ## Prediction Candidates (Need Score Predictions)
        {prediction_candidates_xml}

        ## Parameter Schema
        {example_param_schema_xml}

        ## Task
        **Your Mission**: Learn from training data to predict scores for all prediction candidates.

        **Simple Process**:
        1. **Deep Data Examination**: First, thoroughly analyze ALL training data - examine all candidates' parameters and scores, understand what makes parameters effective, identify patterns and relationships.
        2. **Individual Reasoning**: Then, for each prediction candidate, provide detailed reasoning - analyze the specific parameters, compare to training examples, explain your prediction logic.

        **Key Points**: 
        - Start with comprehensive analysis of all training data
        - Provide thorough reasoning for each individual prediction
        - Be honest about uncertainty and confidence levels

        Return ONLY the XML structure with your analysis and reasoned score predictions.
        """,
        },
        ]
        
        # Call LLM with retry logic
        def single_llm_call():
            return self.LLM(prompt_messages, temperature=self.temperature)
        
        try:
            llm_response = retry_with_exponential_backoff(
                single_llm_call,
                max_retries=10,
                base_delay=1.0,
                operation_name="LLM Regressor call"
            )
        except Exception as e:
            print_color(f"WARNING: LLM Regressor call failed: {e}, returning default scores.", "red")
            return default_scores
        
        llm_response_str = getattr(getattr(llm_response, 'choices', [{}])[0], 'message', None)
        llm_response_str = getattr(llm_response_str, 'content', None)
        
        if not llm_response_str:
            print_color("WARNING: LLM Regressor returned empty response. Using default scores.", "red")
            return default_scores

        cleaned_llm_response_str = llm_response_str.strip()
        
        # Parse XML response
        def parse_xml_response(xml_content):
            """Parse XML response to extract predicted scores"""
            score_estimates = {}
            
            # Try to extract score_estimates section
            estimates_match = re.search(r'<score_estimates>.*?</score_estimates>', xml_content, re.DOTALL)
            if estimates_match:
                estimates_section = estimates_match.group(0)
                
                # Extract individual candidate scores
                candidate_pattern = r'<candidate[^>]*index=["\'](\d+)["\'][^>]*>.*?<predicted_score>(.*?)</predicted_score>'
                for match in re.finditer(candidate_pattern, estimates_section, re.DOTALL):
                    index = match.group(1)
                    try:
                        predicted_score = float(match.group(2).strip())
                    except (ValueError, TypeError):
                        predicted_score = None
                    score_estimates[index] = predicted_score
            
            return score_estimates
        
        try:
            score_estimates = parse_xml_response(cleaned_llm_response_str)
        except Exception as e:
            print_color(f"WARNING: Failed to parse LLM regressor XML output: {e}. Using default scores.", "red")
            return default_scores

        # Extract predicted scores in original batch order
        predicted_scores = []
        for idx in range(len(shuffled_prediction_batch)):
            candidate_key = str(idx)
            original_idx = shuffled_to_original_idx[idx]

            assert candidate_key in score_estimates, f"Candidate key {candidate_key} not in score_estimates"
            # assert score_estimates[candidate_key] is not None, f"Predicted score for candidate {candidate_key} is None"
            # predicted_score could be None
            predicted_score = score_estimates[candidate_key]
            
            
            predicted_scores.append((original_idx, predicted_score))
        
        # Sort by original index to maintain order
        predicted_scores.sort(key=lambda x: x[0])
        predicted_scores = [score for _, score in predicted_scores]
        
        return np.array(predicted_scores)