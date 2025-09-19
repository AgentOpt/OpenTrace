import numpy as np
import copy
from typing import Union
from opto import trace
from opto.trainer.algorithms.algorithm import Trainer
from opto.trainer.loader import DataLoader
from opto.trainer.utils import batch_run, async_run
from opto.optimizers.utils import print_color
from opto.trainer.evaluators import evaluate


def standard_optimization_step(agent, x, guide, info, min_score=0):
    """Execute a standard forward pass and feedback computation step.

    This function runs the agent on input data, evaluates the output using the guide,
    and handles any execution errors that may occur during the process.

    Parameters
    ----------
    agent : trace.Module
        The agent module to execute the forward pass on.
    x : Any
        Input data for the agent.
    guide : callable
        Guide function with signature (question, student_answer, info) -> (score, feedback).
    info : Any
        Additional information passed to the guide function.
    min_score : float, optional
        Minimum score to assign when execution exceptions occur, by default 0.

    Returns
    -------
    tuple[trace.Node, float, str]
        A tuple containing:
        - target: Output node from the agent (or exception node if error occurred)
        - score: Numeric score from the guide evaluation
        - feedback: Text feedback from the guide evaluation

    Notes
    -----
    If a trace.ExecutionError occurs during agent execution, the function catches
    it and returns the exception node with the minimum score and full feedback.
    """
    try:
        target = agent(x)
        score, feedback = guide(x, target.data, info)
    except trace.ExecutionError as e:
        target = e.exception_node
        score, feedback = min_score, target.create_feedback('full')
    return target, score, feedback


class Minibatch(Trainer):
    """General minibatch optimization algorithm with comprehensive training infrastructure.
    
    This class provides a complete training framework that handles minibatch sampling,
    evaluation, logging, checkpointing, and improvement validation. It serves as a
    base class for various optimization algorithms that operate on minibatches.

    Parameters
    ----------
    agent : trace.Module
        The agent module to be trained.
    optimizer : Optimizer
        The optimizer instance for parameter updates.
    num_threads : int, optional
        Maximum number of threads for parallel execution, by default None.
    logger : Logger, optional
        Logger instance for tracking metrics, by default None.
    *args
        Additional positional arguments passed to parent class.
    **kwargs
        Additional keyword arguments passed to parent class.

    Attributes
    ----------
    agent : trace.Module
        The agent being trained.
    optimizer : Optimizer
        The optimizer used for parameter updates.
    n_iters : int
        Number of training iterations completed.
    num_eval_samples : int
        Number of samples used for evaluation.

    Notes
    -----
    This class implements the core training loop including:
    - Minibatch sampling and processing
    - Periodic evaluation and logging
    - Model checkpointing at specified intervals
    - Optional improvement validation and rollback
    """

    def __init__(self,
                 agent,
                 optimizer,
                 num_threads: int = None,   # maximum number of threads to use for parallel execution
                 logger=None,
                 *args,
                 **kwargs,
                 ):
        """Initialize the Minibatch algorithm.

        Parameters
        ----------
        agent : trace.Module
            The agent module to be trained.
        optimizer : Optimizer
            The optimizer instance for parameter updates.
        num_threads : int, optional
            Maximum number of threads for parallel execution, by default None.
        logger : Logger, optional
            Logger instance for tracking metrics, by default None.
        *args
            Additional positional arguments passed to parent class.
        **kwargs
            Additional keyword arguments passed to parent class.
        """
        super().__init__(agent, num_threads=num_threads, logger=logger, *args, **kwargs)
        self.optimizer = optimizer
        self.n_iters = 0  # number of iterations


    def train(self,
              guide,
              train_dataset,
              *,
              ensure_improvement: bool = True,  # whether to check the improvement of the agent
              improvement_threshold: float = 0.,  # threshold for improvement
              num_epochs: int = 1,  # number of training epochs
              batch_size: int = 1,  # batch size for updating the agent
              test_dataset = None,  # dataset of (x, info) pairs to evaluate the agent
              eval_frequency: int = 1,  # frequency of evaluation
              num_eval_samples: int = 1,  # number of samples to use to evaluate each input
              log_frequency: Union[int, None] = None,  # frequency of logging
              save_frequency: Union[int, None] = None,  # frequency of saving the agent
              save_path: str = "checkpoints/agent.pkl",  # path to save the agent
              min_score: Union[int, None] = None,  # minimum score to update the agent
              verbose: Union[bool, str] = False,  # whether to print the output of the agent
              num_threads: int = None,  # maximum number of threads to use (overrides self.num_threads)
              **kwargs
              ):
        """Train the agent using minibatch optimization with comprehensive monitoring.

        This method implements a complete training loop that processes the dataset in
        minibatches, applies parameter updates, and tracks progress through evaluation,
        logging, and checkpointing.

        Parameters
        ----------
        guide : Guide
            Guide function to provide feedback during training.
        train_dataset : dict
            Training dataset containing 'inputs' and 'infos' keys.
        ensure_improvement : bool, optional
            Whether to validate that updates improve performance, by default False.
        improvement_threshold : float, optional
            Minimum improvement threshold for accepting updates, by default 0.0.
        num_epochs : int, optional
            Number of training epochs to run, by default 1.
        batch_size : int, optional
            Size of minibatches for parameter updates, by default 1.
        test_dataset : dict, optional
            Test dataset for evaluation, defaults to train_dataset if None.
        eval_frequency : int, optional
            Frequency of evaluation (every N iterations), by default 1.
        num_eval_samples : int, optional
            Number of samples per input for evaluation, by default 1.
        log_frequency : int, optional
            Frequency of logging, defaults to eval_frequency if None.
        save_frequency : int, optional
            Frequency of saving checkpoints, by default None (no saving).
        save_path : str, optional
            Path template for saving agent checkpoints, by default "checkpoints/agent.pkl".
        min_score : int, optional
            Minimum score threshold for processing, by default None.
        verbose : bool or str, optional
            Verbosity level for training output, by default False.
        num_threads : int, optional
            Number of threads for parallel processing, by default None.
        **kwargs
            Additional arguments passed to subclass methods.

        Returns
        -------
        tuple[list[float], float or None]
            Training scores and final test score.

        Notes
        -----
        The training procedure follows these steps for each minibatch:
        1. Forward pass: Run agent on inputs and compute feedback using guide
        2. Parameter update: Apply optimizer to update agent parameters
        3. Improvement check: Optionally validate and potentially rollback updates
        4. Evaluation: Periodically evaluate agent performance on test data
        5. Logging: Track training metrics and parameter values
        6. Checkpointing: Save agent state at specified intervals
        """

        log_frequency = log_frequency or eval_frequency  # frequency of logging (default to eval_frequency)
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_dataset = test_dataset or train_dataset  # default to train_dataset if test_dataset is not provided
        self.num_eval_samples = num_eval_samples  # number of samples to use to evaluate each input

        # Evaluate the agent before learning
        if eval_frequency > 0:
            test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                          min_score=min_score, num_threads=num_threads,
                          num_samples=self.num_eval_samples,
                          description=f"Evaluating agent (iteration {self.n_iters})")  # and log
            self.logger.log('Average test score', test_score, self.n_iters, color='green')

        # Save the agent before learning if save_frequency > 0
        if save_frequency is not None and save_frequency > 0:
            self.save_agent(save_path, self.n_iters)

        # TODO random sampling with replacement
        loader = DataLoader(train_dataset, batch_size=batch_size)
        train_scores = []
        test_score = None

        for i in range(num_epochs):
            # Train agent
            for xs, infos in loader:
                # Backup the current value of the parameters
                backup_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}

                # Forward the agent on the inputs and compute the feedback using the guide
                forward = batch_run(max_workers=num_threads, description=f"Forward pass (batch size: {len(xs)})")(self.forward)
                outputs = forward(self.agent, xs, guide, infos)

                # Update the agent
                score = self.update(outputs, verbose=verbose, num_threads=num_threads, **kwargs)

                # Reject the update if the score on the current batch is not improved
                if ensure_improvement:
                    changes = any([backup_dict[p] != p.data for p in self.agent.parameters() ])
                    if changes: # Only check improvement if there're changes in the parameters for efficiency
                        if not self.has_improvement(xs, guide, infos, score, outputs, backup_dict,
                                               threshold=improvement_threshold, num_threads=num_threads):
                            self.optimizer.update(backup_dict) # Restore the backup

                self.n_iters += 1

                # Evaluate the agent after update
                if test_dataset is not None and self.n_iters % eval_frequency == 0:
                    test_score = self.evaluate(self.agent, guide, test_dataset['inputs'], test_dataset['infos'],
                                  min_score=min_score, num_threads=num_threads,
                                  num_samples=self.num_eval_samples,
                                  description=f"Evaluating agent (iteration {self.n_iters})")  # and log
                    self.logger.log('Average test score', test_score, self.n_iters, color='green')

                # Save the agent
                if save_frequency is not None and save_frequency > 0 and self.n_iters % save_frequency == 0:
                    self.save_agent(save_path, self.n_iters)

                # Logging
                if score is not None:  # so that mean can be computed
                    train_scores.append(score)
                if self.n_iters % log_frequency == 0:
                    print(f"Epoch: {i}. Iteration: {self.n_iters}")
                    self.logger.log("Instantaneous train score", score, self.n_iters)
                    self.logger.log("Average train score", np.mean(train_scores), self.n_iters)
                    for p in self.agent.parameters():
                        self.logger.log(f"Parameter: {p.name}", p.data, self.n_iters, color='red')

        return train_scores, test_score

    def evaluate(self, agent, guide, xs, infos, min_score=None, num_samples=1, num_threads=None, description=None):
        """Evaluate the agent on a dataset and return the average score.

        Parameters
        ----------
        agent : trace.Module
            The agent to evaluate.
        guide : Guide
            Guide function to provide evaluation scores.
        xs : list
            List of input data points.
        infos : list
            List of additional information for each input.
        min_score : float, optional
            Minimum score for evaluation, by default None.
        num_samples : int, optional
            Number of samples per input for evaluation, by default 1.
        num_threads : int, optional
            Number of threads for parallel evaluation, by default None.
        description : str, optional
            Description for progress tracking, by default None.

        Returns
        -------
        float or None
            Average evaluation score, or None if any scores are invalid.
        """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        test_scores = evaluate(agent, guide, xs, infos, min_score=min_score, num_threads=num_threads,
                               num_samples=num_samples, description=description)
        if all([s is not None for s in test_scores]):
            return np.mean(test_scores)

    def has_improvement(self, xs, guide, infos, current_score, current_outputs, backup_dict, threshold=0, num_threads=None, *args, **kwargs):
        """Check if the updated agent shows improvement over the previous version.

        This method evaluates the updated agent and compares its performance to the
        current score to determine whether the parameter update should be accepted.

        Parameters
        ----------
        xs : list
            Input data points for evaluation.
        guide : Guide
            Guide function to provide evaluation scores.
        infos : list
            Additional information for each input.
        current_score : float
            Score of the agent before the update.
        current_outputs : list
            Outputs from the agent-guide interaction.
        backup_dict : dict
            Backup of parameter values before the update.
        threshold : float, optional
            Minimum improvement threshold, by default 0.
        num_threads : int, optional
            Number of threads for evaluation, by default None.
        *args
            Additional positional arguments.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        bool
            True if the update shows improvement, False otherwise.

        Notes
        -----
        This method can be overridden by subclasses to implement custom
        improvement validation logic. The default implementation evaluates
        the updated agent and compares against the threshold.
        """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        new_score = self.evaluate(self.agent, guide, xs, infos, num_threads=num_threads,
                                 description=f"Checking improvement (iteration {self.n_iters})",
                                 num_samples=self.num_eval_samples,
                                 *args, **kwargs)  # evaluate the updated agent
        if new_score is None or new_score <= current_score - threshold:
            print_color(f"Update rejected: Current score {current_score}, New score {new_score}", 'red')
            return False
        else:
            print_color(f"Update accepted: Current score {current_score}, New score {new_score}", 'green')
            return True


    def forward(self, agent, x, guide, info):
        """Execute forward pass and compute feedback for a single input.

        This method must be implemented by subclasses to define how the agent
        processes individual inputs and generates outputs for parameter updates.

        Parameters
        ----------
        agent : trace.Module
            The agent module to execute.
        x : Any
            Input data for the agent.
        guide : callable
            Guide function with signature (question, student_answer, info) -> (score, feedback).
        info : Any
            Additional information for the guide.

        Returns
        -------
        Any
            Outputs that will be used by the update method to modify agent parameters.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def update(self, outputs, verbose=False, num_threads=None, **kwargs):
        """Update the agent parameters based on forward pass outputs.

        This method must be implemented by subclasses to define how parameter
        updates are computed and applied based on the forward pass results.

        Parameters
        ----------
        outputs : Any
            Outputs returned from the forward method.
        verbose : bool, optional
            Whether to print verbose update information, by default False.
        num_threads : int, optional
            Maximum number of threads to use, overrides self.num_threads, by default None.
        **kwargs
            Additional keyword arguments for the update process.

        Returns
        -------
        float or None
            Average score of the minibatch inputs, or None if no valid scores.

        Raises
        ------
        NotImplementedError
            This method must be implemented by subclasses.

        Notes
        -----
        The update method should process the outputs from forward passes,
        apply parameter updates using the optimizer, and return performance metrics.
        """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads
        raise NotImplementedError("Subclasses must implement this method")



@trace.bundle()
def batchify(*items):
    """Concatenate multiple items into a formatted batch string.

    Parameters
    ----------
    *items : Any
        Variable number of items to concatenate into a batch.

    Returns
    -------
    str
        Formatted string with each item labeled by ID.

    Notes
    -----
    This function is decorated with @trace.bundle() and creates a formatted
    string where each item is prefixed with 'ID [i]:' for identification.
    """
    output = ''
    for i, item in enumerate(items):
        output += f'ID {[i]}: {item}\n'
    return output


class MinibatchAlgorithm(Minibatch):
    """Standard minibatch algorithm that aggregates outputs for batch feedback.

    This algorithm processes each instance in the minibatch individually, then
    concatenates the outputs and feedback to provide a single batched update
    to the agent. This approach allows the agent to learn from multiple examples
    simultaneously in each optimization step.

    Attributes
    ----------
    agent : trace.Module
        The agent being trained (inherited from parent).
    optimizer : Optimizer
        The optimizer for parameter updates (inherited from parent).

    Notes
    -----
    The algorithm follows these steps:
    1. Execute standard optimization steps for each minibatch instance
    2. Aggregate targets and feedback using the batchify function
    3. Apply optimizer backward pass with batched feedback
    4. Update agent parameters using the optimizer step
    """

    def forward(self, agent, x, guide, info):
        """Execute standard optimization step for a single input.

        Parameters
        ----------
        agent : trace.Module
            The agent to execute.
        x : Any
            Input data.
        guide : callable
            Guide function for feedback generation.
        info : Any
            Additional information for the guide.

        Returns
        -------
        tuple[trace.Node, float, str]
            Tuple of (target, score, feedback) from standard optimization step.
        """
        return standard_optimization_step(agent, x, guide, info)  # (target, score, feedback)

    def update(self, outputs, verbose=False, num_threads=None, **kwargs):
        """ Subclasses can implement this method to update the agent.
            Args:
                outputs: returned value from self.step
                verbose: whether to print the output of the agent
                num_threads: maximum number of threads to use (overrides self.num_threads)
            Returns:
                score: average score of the minibatch of inputs

        """
        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads

        scores, targets, feedbacks = [], [], []
        # Concatenate the targets and feedbacks into a single string
        for target, score, feedback in outputs:
            scores.append(score)
            targets.append(target)
            feedbacks.append(feedback)
        target = batchify(*targets)
        feedback = batchify(*feedbacks).data  # str
        average_score = np.mean(scores) if all([s is not None for s in scores]) else None

        # Update the agent using the feedback
        self.optimizer.zero_feedback()
        self.optimizer.backward(target, feedback)
        self.optimizer_step(verbose=verbose, num_threads=num_threads, **kwargs)  # update the agent

        return average_score  # return the average score of the minibatch of inputs

    def optimizer_step(self, bypassing=False, verbose=False, num_threads=None, **kwargs):
        """ Subclasses can implement this method to update the agent. """
        # We separate this method from the update method to allow subclasses to implement their own optimization step.
        return self.optimizer.step(bypassing=bypassing, verbose=verbose, **kwargs)


class BasicSearchAlgorithm(MinibatchAlgorithm):
    """ A basic search algorithm that calls the optimizer multiple times to get candidates and selects the best one based on validation set. """

    def train(self,
              guide, # guide to provide feedback
              train_dataset,  # dataset of (x, info) pairs to train the agent
              *,
              validate_dataset = None, # dataset of (x, info) pairs to evaluate the agent for candidate selection
              validate_guide = None,  #  to provide scores for the validation set
              num_proposals = 4,  # number of proposals to get from the optimizer
              num_epochs = 1,  # number of training epochs
              batch_size = 1,  # batch size for updating the agent
              test_dataset = None, # dataset of (x, info) pairs to evaluate the agent
              eval_frequency = 1, # frequency of evaluation
              log_frequency = None,  # frequency of logging
              min_score = None,  # minimum score to update the agent
              verbose = False,  # whether to print the output of the agent
              num_threads = None,  # maximum number of threads to use
              **kwargs
              ):

        self.num_proposals = num_proposals
        self.validate_dataset = validate_dataset or train_dataset  # default to train_dataset
        self.validate_guide = validate_guide or guide
        self.min_score = min_score
        self.current_score = None

        return super().train(guide, train_dataset, num_epochs=num_epochs, batch_size=batch_size,
                      test_dataset=test_dataset, eval_frequency=eval_frequency, log_frequency=log_frequency,
                      min_score=min_score, verbose=verbose, num_threads=num_threads, **kwargs)

    # This code should be reusable for other algorithms
    def optimizer_step(self, bypassing=False, verbose=False, num_threads=None, **kwargs):
        """ Use the optimizer to propose multiple updates and select the best one based on validation score. """

        num_threads = num_threads or self.num_threads  # Use provided num_threads or fall back to self.num_threads

        def validate():
            """ Validate the agent on the validation dataset. """
            scores = evaluate(self.agent,
                              self.validate_guide,
                              self.validate_dataset['inputs'],
                              self.validate_dataset['infos'],
                              min_score=self.min_score,
                              num_threads=num_threads,
                              description="Validating proposals")
            return np.mean(scores) if all([s is not None for s in scores]) else -np.inf

        # TODO perhaps we can ask for multiple updates in one query or use different temperatures in different queries
        # Generate different proposals
        step_kwargs = dict(bypassing=True, verbose='output' if verbose else False)  # we don't print the inner full message
        step_kwargs.update(kwargs)  # update with additional kwargs if provided

        # Use aysnc_run to run the optimizer_step in parallel
        # NOTE optimizer_step is coupled via async_run
        update_dicts = async_run([super().optimizer_step]*self.num_proposals,
                                kwargs_list=[step_kwargs] * self.num_proposals,
                                max_workers=num_threads,
                                description=f"Generating {self.num_proposals} proposals")  # async step
        # Validate the proposals
        candidates = []
        backup_dict = {p: copy.deepcopy(p.data) for p in self.agent.parameters()}  # backup the current value
        for update_dict in update_dicts:
            if len(update_dict) == 0:
                continue
            self.optimizer.update(update_dict)  # set the agent with update_dict
            score = validate()  # check the score on the validation set
            candidates.append((score, update_dict))
            self.optimizer.update(backup_dict)  # restore the backup

        # Include the current parameter as a candidate
        if self.current_score is None:
            self.current_score = validate()
        candidates.append((self.current_score, backup_dict))

        # Find the candidate with the best score
        best_score, best_update = max(candidates, key=lambda x: x[0])
        self.current_score = best_score

        if verbose:
            print_color(f"Best score: {best_score} out of scores {[c[0] for c in candidates]}", 'green')
            print(f"Selected Update:\n{best_update}")

        # Make the best update
        self.optimizer.update(best_update)

        # Logging
        self.logger.log('Validation score', best_score, self.n_iters, color='green')