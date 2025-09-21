from opto.trace.nodes import node, Node, ExceptionNode
from typing import Any

from opto.trace.bundle import bundle
import opto.trace.operators as ops
from opto.trace.errors import ExecutionError
import numpy as np

# List[Nodes], Node[List]
def iterate(x: Any):
    """Create an iterator for node containers.

    Parameters
    ----------
    x : Any
        A node or value to iterate over. Can be list, tuple, set, dict,
        string, or numpy array.

    Returns
    -------
    SeqIterable or DictIterable
        An iterator object that yields nodes during iteration.

    Raises
    ------
    ExecutionError
        If the input is not iterable.

    Notes
    -----
    This function enables iteration over node containers in traced code:
    - Lists, tuples, strings, arrays → SeqIterable
    - Sets → Converted to list then SeqIterable
    - Dicts → SeqIterable over keys
    - Non-iterables → Raises ExecutionError with ExceptionNode

    The returned iterator creates child nodes during iteration,
    maintaining proper parent-child relationships in the graph.
    """
    if not isinstance(x, Node):
        x = node(x)
    if issubclass(x.type, list) or issubclass(x.type, tuple) or issubclass(x.type, str) or issubclass(x.type, np.ndarray):
        return SeqIterable(x)
    elif issubclass(x.type, set):
        converted_list = ops.to_list(x)
        return SeqIterable(converted_list)
    elif issubclass(x.type, dict):
        return SeqIterable(x.keys())
    else:
        raw_traceback = "TypeError: Cannot unpack non-iterable {} object".format(
            type(x._data)
        )
        ex = TypeError(raw_traceback)
        e = ExceptionNode(
            ex,
            inputs=[x],
            info={
                "traceback": raw_traceback,
            },
        )
        raise ExecutionError(e)


# List, Tuple, Set share an Iterable
class SeqIterable:
    """Iterator for sequence-like node containers.

    Provides iteration over nodes containing lists, tuples, sets,
    strings, or arrays. Creates child nodes for each element during
    iteration.

    Parameters
    ----------
    wrapped_list : Node
        A node containing a sequence-like object.

    Attributes
    ----------
    wrapped_list : Node
        The node being iterated over.
    _index : int
        Current iteration index.

    Methods
    -------
    __iter__()
        Reset iterator to beginning.
    __next__()
        Get next element as a node.

    Notes
    -----
    Each iteration:
    1. Accesses the element using node indexing (wrapped_list[index])
    2. Creates a MessageNode for the accessed element
    3. Maintains parent-child relationship in the graph
    4. Returns the element node

    This ensures all iterations are traced in the computation graph.
    """
    def __init__(self, wrapped_list):
        assert isinstance(wrapped_list, Node)
        self._index = 0
        self.wrapped_list = wrapped_list

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.wrapped_list._data):
            result = self.wrapped_list[self._index]
            self._index += 1
            assert isinstance(result, Node)
            assert self.wrapped_list in result.parents
            return result
        else:
            raise StopIteration


class DictIterable:
    """Iterator for dictionary nodes.

    Provides iteration over dictionary nodes, yielding (key, value)
    tuples where values are nodes.

    Parameters
    ----------
    wrapped_dict : Node
        A node containing a dictionary.

    Attributes
    ----------
    wrapped_dict : Node
        The dictionary node being iterated.
    keys : Node
        Node containing the dictionary keys.
    _index : int
        Current iteration index.

    Methods
    -------
    __iter__()
        Reset iterator to beginning.
    __next__()
        Get next (key, value) tuple.

    Notes
    -----
    Iteration process:
    1. Extracts keys using ops.keys()
    2. For each key, accesses value using wrapped_dict[key]
    3. Returns (key, value_node) tuples
    4. Maintains graph relationships

    Used by Node.items() to provide dictionary iteration in traced code.
    """
    def __init__(self, wrapped_dict):
        assert isinstance(wrapped_dict, Node)
        self._index = 0
        self.wrapped_dict = wrapped_dict
        self.keys = ops.keys(wrapped_dict)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.keys):

            key = self.keys[self._index]
            result = (key, self.wrapped_dict[key])
            self._index += 1

            assert self.wrapped_dict in result[1].parents

            return result
        else:
            raise StopIteration
