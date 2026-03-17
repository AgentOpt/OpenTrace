from typing import Any, List, Dict
import copy, pickle, os
from opto.trace.nodes import ParameterNode, Node
from opto.trace.propagators import GraphPropagator
from opto.trace.propagators.propagators import Propagator
from opto.trace.utils import sum_feedback


class AbstractOptimizer:
    """Abstract base class for all optimizers in the Trace framework.

    Defines the interface that all optimizers must implement for parameter
    optimization based on feedback from the computation graph.

    Parameters
    ----------
    parameters : list[ParameterNode]
        List of trainable parameters to optimize. Must be non-empty and contain
        only ParameterNode instances.
    *args
        Additional positional arguments for optimizer configuration.
    **kwargs
        Additional keyword arguments for optimizer configuration.

    Attributes
    ----------
    parameters : list[ParameterNode]
        The parameters being optimized.

    Methods
    -------
    step()
        Perform one optimization step.
    zero_feedback()
        Clear accumulated feedback from parameters.
    propagator
        Property returning the feedback propagator.

    Raises
    ------
    AssertionError
        If parameters is not a list, contains non-ParameterNode objects,
        or is empty.

    Notes
    -----
    This abstract class establishes the optimizer protocol:

    1. **Parameter Management**: Optimizers maintain a list of parameters
       they are responsible for updating.

    2. **Feedback Processing**: Optimizers process feedback accumulated
       in parameters during backward passes.

    3. **Update Steps**: The step() method applies optimization logic
       to update parameter values.

    4. **Feedback Clearing**: zero_feedback() resets accumulated feedback
       for the next iteration.

    Subclasses must implement all abstract methods to create functional
    optimizers.

    See Also
    --------
    Optimizer : Concrete base class with graph-based optimization
    ParameterNode : Trainable parameters that optimizers update
    Propagator : Handles feedback propagation through the graph
    """

    def __init__(self, parameters: List[ParameterNode], *args, **kwargs):
        self.parameter_check(parameters)
        # this is a guaranteed basic check, not possible to be overloaded by subclasses
        assert type(parameters) is list, "Parameters must be a list."
        assert all([isinstance(p, ParameterNode) for p in parameters]), "Parameters must be a list of ParameterNode instances."
        assert len(parameters) > 0, 'Parameters list is empty.'
        for p in parameters:
            assert p.trainable, "Parameter {} must be trainable.".format(p.name)
        self.parameters = parameters

    def step(self):
        """Update the parameters based on the feedback."""
        raise NotImplementedError

    def zero_feedback(self):
        """Reset the feedback."""
        raise NotImplementedError

    @property
    def propagator(self):
        """Return a Propagator object that can be used to propagate feedback in backward."""
        raise NotImplementedError

    def parameter_check(self, parameters: List[ParameterNode]):
        """Check if the parameters are valid.
        This can be overloaded by subclasses to add more checks.

        Args:
            parameters: List[ParameterNode]
                The parameters to check.
        """
        pass



class Optimizer(AbstractOptimizer):
    """Base class for graph-based optimizers in the Trace framework.

    Extends AbstractOptimizer with concrete implementations for graph-based
    optimization, including feedback propagation, parameter projection, and
    update mechanisms.

    Parameters
    ----------
    parameters : list[ParameterNode]
        List of trainable parameters to optimize.
    propagator : Propagator, optional
        Custom propagator for feedback processing. If None, uses default
        GraphPropagator.
    *args
        Additional positional arguments.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    parameters : list[ParameterNode]
        The parameters being optimized.
    propagator : Propagator
        The feedback propagator used during backward passes.
    trace_graph : Any
        Aggregated computation graph from all parameters.

    Methods
    -------
    step(bypassing=False, *args, **kwargs)
        Perform one optimization step with optional update bypassing.
    propose(*args, **kwargs)
        Generate proposed parameter updates based on feedback.
    project(update_dict)
        Apply constraints/projections to proposed updates.
    update(update_dict)
        Apply updates to trainable parameters.
    backward(node, *args, **kwargs)
        Propagate feedback through the graph. Feedback is passed in through *args and **kwargs.
    zero_feedback()
        Clear accumulated feedback from all parameters.
    save(path)
        Save optimizer state (placeholder).
    load(path)
        Load optimizer state (placeholder).
    _step(*args, **kwargs)
        Abstract method for computing parameter updates.
    default_propagator()
        Return the default propagator instance.

    Notes
    -----
    The Optimizer class implements a three-stage update process:

    1. **Propose**: Generate candidate updates based on feedback
       (implemented in _step by subclasses).

    2. **Project**: Apply constraints and projections to ensure
       updates remain in valid parameter space.

    3. **Update**: Apply the projected updates to parameters
       (can be bypassed for analysis).

    Key features:

    - **Feedback Aggregation**: Automatically collects and aggregates
      feedback from the computation graph.

    - **Projection Support**: Integrates with parameter projections
      for constrained optimization.

    - **Flexible Propagation**: Supports custom propagators for
      different feedback processing strategies.

    - **State Management**: Provides hooks for saving/loading
      optimizer state (implementation-specific).

    Subclasses must implement _step() to define the optimization
    algorithm.

    See Also
    --------
    AbstractOptimizer : Abstract base class
    GraphPropagator : Default feedback propagator
    ParameterNode : Parameters being optimized
    Projection : Constraints applied during optimization

    Usage
    --------
    result = traced_computation(x)
    optimizer.zero_feedback()
    optimizer.backward(result, 'user feedback')

    Examples
    --------
    >>> class MyOptimizer(Optimizer):
    ...     def _step(self):
    ...         updates = {}
    ...         for p in self.parameters:
    ...             feedback = sum_feedback(p.feedback)
    ...             updates[p] = p.data - 0.01 * feedback
    ...         return updates
    """

    def __init__(
        self,
        parameters: List[ParameterNode],
        *args,
        propagator: Propagator = None,
        **kwargs
    ):
        super().__init__(parameters)
        propagator = propagator if propagator is not None else self.default_propagator()
        assert isinstance(propagator, Propagator)
        self._propagator = propagator

    @property
    def propagator(self):
        return self._propagator

    @property
    def trace_graph(self):
        """Aggregate the graphs of all the parameters."""
        return sum_feedback(self.parameters)

    def step(self, bypassing=False, *args, **kwargs):
        """Perform one optimization step.

        Parameters
        ----------
        bypassing : bool, default=False
            If True, computes updates but doesn't apply them to parameters.
            Useful for analysis or debugging.
        *args
            Additional arguments passed to propose().
        **kwargs
            Additional keyword arguments passed to propose().

        Returns
        -------
        dict[ParameterNode, Any]
            Dictionary mapping parameters to their (projected) updates.

        Notes
        -----
        The step executes in three phases:
        1. Propose updates via _step()
        2. Apply projections to maintain constraints
        3. Update parameters (unless bypassing=True)
        """
        update_dict = self.propose(*args, **kwargs)
        self.project(update_dict)
        if not bypassing:
            self.update(update_dict)
        return update_dict

    def project(self, update_dict: Dict[ParameterNode, Any]):
        """Apply projections to constrain parameter updates.

        Parameters
        ----------
        update_dict : dict[ParameterNode, Any]
            Proposed updates for each parameter.

        Notes
        -----
        Modifies update_dict in-place by applying each parameter's
        projection operators sequentially. Only applies to trainable
        parameters with defined projections.
        """
        for p, d in update_dict.items():
            if p.trainable:
                for projection in p.projections:
                    d = projection.project(d)
            update_dict[p] = d

    def propose(self, *args, **kwargs):
        """Generate proposed parameter updates based on feedback.

        Parameters
        ----------
        *args
            Arguments passed to _step().
        **kwargs
            Keyword arguments passed to _step().

        Returns
        -------
        dict[ParameterNode, Any]
            Proposed new values for each parameter.

        Notes
        -----
        Delegates to _step() which must be implemented by subclasses.
        """
        return self._step(*args, **kwargs)

    def update(self, update_dict: Dict[ParameterNode, Any]):
        """Apply updates to trainable parameters.

        Parameters
        ----------
        update_dict : dict[ParameterNode, Any]
            New values for each parameter.

        Notes
        -----
        Only updates parameters marked as trainable. Updates are
        applied by directly modifying the parameter's _data attribute.
        """
        for p, d in update_dict.items():
            if p.trainable:
                p._data = d

    def zero_feedback(self):
        """Clear accumulated feedback from all parameters.

        Notes
        -----
        Should be called after each optimization step to prepare
        for the next iteration's feedback accumulation.
        """
        for p in self.parameters:
            p.zero_feedback()

    # Subclass should implement the methods below.
    def _step(self, *args, **kwargs) -> Dict[ParameterNode, Any]:
        """Compute parameter updates based on accumulated feedback.

        Parameters
        ----------
        *args
            Optimizer-specific arguments.
        **kwargs
            Optimizer-specific keyword arguments.

        Returns
        -------
        dict[ParameterNode, Any]
            Proposed new values for each parameter.

        Notes
        -----
        Must be implemented by subclasses to define the optimization
        algorithm. Has access to self.parameters and their feedback.
        """
        raise NotImplementedError

    def default_propagator(self):
        """Return the default feedback propagator.

        Returns
        -------
        GraphPropagator
            Default propagator for feedback processing.

        Notes
        -----
        Subclasses can override to provide custom default propagators.
        """
        return GraphPropagator()

    def backward(self, node: Node, *args, **kwargs):
        """Propagate feedback backward through the computation graph.

        Parameters
        ----------
        node : Node
            Starting node for backward propagation.
        *args
            Additional arguments passed to node.backward(*args, **kwargs).
            This corresponds to the positional arguments in node.backward
        **kwargs
            Additional keyword arguments passed to node.backward(*args, **kwargs).
            This corresponds to the keyword arguments in node.backward
            If 'propagator' is not provided, uses the optimizer's propagator.

        Returns
        -------
        Any
            Result from node.backward(), typically a visualization graph.

        Notes
        -----
        Uses the optimizer's propagator for feedback processing by default.

        Usage
        ------
        optimizer.backward(result, 'make this number bigger', propagator=custom_propagator)
        optimizer.backward(result, feedback='make this number bigger')
        """
        kwargs.setdefault('propagator', self.propagator)
        return node.backward(*args, **kwargs)

    def save(self, path: str):
        """Save the optimizer state to a file."""
        # check if the directory exists
        directory = os.path.dirname(path)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.__getstate__(), f)

    def load(self, path: str):
        """Load the optimizer state from a file."""
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.__setstate__(state)

    # NOTE: overload __getstate__ and __setstate__ in subclasses to customize pickling behavior
    def __getstate__(self):
        state = self.__dict__.copy()
        # don't pickle the parameters, as they are part of the model
        state['parameters'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        # deepcopy everything except self.parameters
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k != 'parameters':
                setattr(result, k, copy.deepcopy(v, memo))
            else:
                setattr(result, k, v)  # parameters is not copied, it is the original parameters
        return result