from typing import Any, List, Dict, Tuple
from opto.trace.nodes import Node, MessageNode, get_op_name


class AbstractPropagator:
    """Abstract base class for feedback propagation in computation graphs.
    
    This class defines the interface for propagating feedback from child nodes
    to their parent nodes in traced computation graphs. Concrete implementations
    define specific propagation strategies for different types of operations.

    Notes
    -----
    Propagators are essential components of the trace system that enable
    backward passes for optimization. They determine how feedback flows
    through the computation graph during parameter updates.
    """
    
    def __call__(self, child: MessageNode):
        """Propagate feedback from child node to its parent nodes.
        
        This method validates the feedback structure and delegates to the
        concrete propagate implementation.

        Parameters
        ----------
        child : MessageNode
            Child node containing feedback to propagate.

        Returns
        -------
        dict[Node, Any]
            Dictionary mapping parent nodes to their propagated feedback.

        Raises
        ------
        AssertionError
            If child is not a MessageNode or feedback format is invalid.

        Notes
        -----
        All MessageNode feedback should have at most one feedback entry per key.
        The propagated feedback must include entries for all parent nodes.
        """
        assert isinstance(child, MessageNode)
        assert all(
            [len(f) <= 1 for f in child.feedback.values()]
        )  # All MessageNode feedback should be at most length 1
        propagated_feedback = self.propagate(child)
        # Check propagated feedback has the right format
        # It should be a dictionary with the parents as keys and the feedback as values
        assert isinstance(propagated_feedback, dict)
        assert all((p in propagated_feedback for p in child.parents))
        return propagated_feedback

    def propagate(self, child: MessageNode) -> Dict[Node, Any]:
        """Compute propagated feedback for parent nodes.

        This abstract method must be implemented by concrete propagator classes
        to define how feedback is computed and distributed to parent nodes.

        Parameters
        ----------
        child : MessageNode
            Child node containing feedback to propagate.

        Returns
        -------
        dict[Node, Any]
            Dictionary mapping each parent node to its computed feedback.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError


class AbstractFeedback:
    """Abstract base class for feedback objects in propagation systems.
    
    This class defines the interface for feedback objects that can be combined
    and accumulated during backward propagation. Concrete implementations must
    support addition operations for proper feedback aggregation.

    Notes
    -----
    Feedback objects are used to carry gradient information through the
    computation graph. The addition operation enables accumulation of
    feedback from multiple sources.
    """

    def __add__(self, other):
        """Add two feedback objects together.

        Parameters
        ----------
        other : AbstractFeedback
            Another feedback object to combine with this one.

        Returns
        -------
        AbstractFeedback
            Combined feedback object.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def __radd__(self, other):
        """Support right-hand addition and sum() function.

        Parameters
        ----------
        other : Any
            Other object to add. If 0, returns self for sum() compatibility.

        Returns
        -------
        AbstractFeedback
            Result of addition operation.
        """
        if other == 0:  # for support sum
            return self
        else:
            return self.__add__(other)


class Propagator(AbstractPropagator):
    """Configurable propagator with operator-specific override support.
    
    This propagator allows registration of custom propagation functions for
    specific operators while providing a default propagation strategy for
    unregistered operators.

    Attributes
    ----------
    override : dict[str, callable]
        Dictionary mapping operator names to custom propagation functions.

    Notes
    -----
    This design enables flexible customization of propagation behavior for
    different types of operations while maintaining a unified interface.
    """
    
    def __init__(self):
        """Initialize propagator with empty override registry."""
        self.override = dict()  # key: operator name: data: override propagate function

    def register(self, operator_name, propagate_function):
        """Register a custom propagation function for an operator.

        Parameters
        ----------
        operator_name : str
            Name of the operator to override.
        propagate_function : callable
            Custom propagation function with signature (child: MessageNode) -> dict.

        Notes
        -----
        Registered functions take precedence over the default propagation logic.
        The function should return a dictionary mapping parent nodes to feedback.
        """
        self.override[operator_name] = propagate_function

    def propagate(self, child: MessageNode) -> Dict[Node, Any]:
        """Propagate feedback using operator-specific or default logic.

        Parameters
        ----------
        child : MessageNode
            Child node containing feedback to propagate.

        Returns
        -------
        dict[Node, Any]
            Dictionary mapping parent nodes to propagated feedback.

        Notes
        -----
        First checks for registered operator-specific propagation functions.
        Falls back to the default _propagate method if no override is found.
        """
        operator_name = child.op_name
        if operator_name in self.override:
            return self.override[operator_name](child)
        else:
            return self._propagate(child)

    def init_feedback(self, node: Node, feedback: Any):
        """Initialize feedback object for propagation.

        This method converts raw feedback into the appropriate format for
        recursive propagation through the computation graph.

        Parameters
        ----------
        node : Node
            Node receiving the feedback.
        feedback : Any
            Raw feedback data to initialize.

        Returns
        -------
        Any
            Initialized feedback object ready for propagation.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _propagate(self, child: MessageNode) -> Dict[Node, Any]:
        """Default propagation logic for operators without custom overrides.

        This method implements the default strategy for propagating feedback
        from child nodes to their parents using the node's description, data,
        and feedback information.

        Parameters
        ----------
        child : MessageNode
            Child node containing feedback to propagate.

        Returns
        -------
        dict[Node, Any]
            Dictionary mapping parent nodes to their computed feedback.

        Raises
        ------
        NotImplementedError
            Must be implemented by subclasses to define default behavior.
        """
        raise NotImplementedError


# Note:
# if len(feedback) > 1, it means there are two or more child nodes from this node,
# we might need to perform a "merge" feedback action


class SumPropagator(Propagator):
    """Simple propagator that sums or concatenates feedback from multiple sources.
    
    This propagator implements a basic aggregation strategy where feedback from
    multiple child nodes is combined by summation (for numeric types) or
    concatenation (for string types) before being distributed to parent nodes.

    Notes
    -----
    This is a concrete implementation suitable for scenarios where feedback
    can be meaningfully combined through simple aggregation operations.
    """
    
    def init_feedback(self, feedback: Any):
        """Initialize feedback without transformation.

        Parameters
        ----------
        feedback : Any
            Raw feedback to initialize.

        Returns
        -------
        Any
            The feedback object unchanged.
        """
        return feedback

    def _propagate(self, child: MessageNode):
        """Propagate feedback by summing or concatenating multiple sources.

        Parameters
        ----------
        child : MessageNode
            Child node containing feedback to propagate.

        Returns
        -------
        dict[Node, Any]
            Dictionary mapping each parent node to the aggregated feedback.

        Notes
        -----
        User feedback takes precedence and is used directly if present.
        Otherwise, feedback from all sources is aggregated by type:
        - Strings are concatenated
        - Numeric types are summed
        All feedback values must be of the same type for proper aggregation.
        """
        if "user" in child.feedback:
            assert len(child.feedback) == 1, "user feedback should be the only feedback"
            assert len(child.feedback["user"]) == 1
            feedback = child.feedback["user"][0]
        else:
            # Simply sum the feedback
            feedback_list = [v[0] for k, v in child.feedback.items()]
            assert len(feedback_list) > 0
            assert all(
                [type(feedback_list[0]) is type(f) for f in feedback_list]
            ), "error in propagate"
            if isinstance(feedback_list[0], str):
                feedback = "".join(feedback_list)
            else:
                feedback = sum(feedback_list)
        return {parent: feedback for parent in child.parents}
