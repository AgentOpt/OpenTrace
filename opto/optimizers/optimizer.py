from typing import Any, List, Dict
import copy, pickle, os
from opto.trace.nodes import ParameterNode, Node
from opto.trace.propagators import GraphPropagator
from opto.trace.propagators.propagators import Propagator
from opto.trace.utils import sum_feedback


class AbstractOptimizer:
    """An optimizer is responsible for updating the parameters based on the feedback."""

    def __init__(self, parameters: List[ParameterNode], *args, **kwargs):
        assert type(parameters) is list
        assert all([isinstance(p, ParameterNode) for p in parameters])
        assert len(parameters) > 0, 'Parameters list is empty.'
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


class Optimizer(AbstractOptimizer):
    """Optimizer based on Trace graph."""

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
        update_dict = self.propose(*args, **kwargs)
        self.project(update_dict)
        if not bypassing:
            self.update(update_dict)
        return update_dict  # TODO add reasoning

    def project(self, update_dict: Dict[ParameterNode, Any]):
        """Project the update dictionary onto the feasible set."""
        for p, d in update_dict.items():
            if p.trainable:
                for projection in p.projections:
                    d = projection.project(d)
            update_dict[p] = d

    def propose(self, *args, **kwargs):
        """Propose the new data of the parameters based on the feedback."""
        return self._step(*args, **kwargs)

    def update(self, update_dict: Dict[ParameterNode, Any]):
        """Update the trainable parameters given a dictionary of new data."""
        for p, d in update_dict.items():
            if p.trainable:
                p._data = d

    def zero_feedback(self):
        for p in self.parameters:
            p.zero_feedback()

    # Subclass should implement the methods below.
    def _step(self, *args, **kwargs) -> Dict[ParameterNode, Any]:
        """Return the new data of parameter nodes based on the feedback."""
        raise NotImplementedError

    def default_propagator(self):
        """Return the default Propagator object of the optimizer."""
        return GraphPropagator()

    def backward(self, node: Node, *args, **kwargs):
        """Propagate the feedback backward."""
        return node.backward(*args, propagator=self.propagator, **kwargs)

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