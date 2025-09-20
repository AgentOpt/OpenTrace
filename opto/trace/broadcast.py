import copy

from opto.trace.containers import NodeContainer
from opto.trace.nodes import Node


def recursive_conversion(true_func, false_func):
    """Recursively apply functions to nodes and non-nodes in nested structures.

    Creates a function that traverses nested data structures, applying
    different functions to Node objects versus other objects.

    Parameters
    ----------
    true_func : callable
        Function to apply to Node objects.
    false_func : callable
        Function to apply to non-Node objects.

    Returns
    -------
    callable
        A function that recursively processes nested structures.

    Notes
    -----
    Supported container types:
    - tuple, list, dict, set: Recursively processed
    - NodeContainer: Attributes recursively processed
    - Node: true_func applied
    - Other: false_func applied

    The returned function preserves the structure while transforming
    the contents. Commonly used for:
    - Extracting data from nested nodes
    - Converting between node and non-node representations
    - Applying transformations while maintaining structure

    Examples
    --------
    >>> # Extract data from nested nodes
    >>> extract = recursive_conversion(
    ...     true_func=lambda n: n.data,
    ...     false_func=lambda x: x
    ... )
    >>> result = extract(nested_structure)
    """

    def func(obj):
        if isinstance(obj, Node):  # base case
            return true_func(obj)
        elif isinstance(obj, tuple):
            return tuple(func(x) for x in obj)
        elif isinstance(obj, list):
            return [func(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: func(v) for k, v in obj.items()}
        elif isinstance(obj, set):
            return {func(x) for x in obj}
        elif isinstance(obj, NodeContainer):
            output = copy.copy(obj)
            for k, v in obj.__dict__.items():
                setattr(output, k, func(v))
            return output
        else:
            return false_func(obj)

    return func


# TODO to test it and clean up the code
def apply_op(op, output, *args, **kwargs):
    """Apply an operator to containers of nodes with broadcasting.

    Enables element-wise operations on mixed containers of nodes and
    regular values, similar to NumPy broadcasting but for Node objects.

    Parameters
    ----------
    op : callable
        The operator to apply element-wise.
    output : Any
        Container template determining output structure.
        Can be list, tuple, dict, or NodeContainer.
    *args : Any
        Positional arguments for the operator.
        Each can be a Node or container matching output type.
    **kwargs : Any
        Keyword arguments for the operator.
        Each can be a Node or container matching output type.

    Returns
    -------
    Any
        Result with same structure as output, containing results
        of applying op element-wise.

    Raises
    ------
    AssertionError
        If container types don't match or lengths differ.

    Notes
    -----
    Broadcasting rules:
    1. If all inputs are Nodes, applies op directly
    2. For containers, applies op element-wise:
       - Lists/tuples: By index
       - Dicts: By key
       - NodeContainers: By attribute
    3. Node inputs are broadcast to all elements
    4. Container inputs must match output structure

    The function modifies output in-place for most containers
    but returns a new tuple for tuple inputs.
    """

    inputs = list(args) + list(kwargs.values())
    containers = [x for x in inputs if not isinstance(x, Node)]
    if len(containers) == 0:  # all inputs are Nodes, we just apply op
        return op(*args, **kwargs)

    # # there is at least one container
    # output = copy.deepcopy(containers[0])  # this would be used as the template of the output

    def admissible_type(x, base):
        return type(x) is type(base) or isinstance(x, Node)

    assert all(
        admissible_type(x, output) for x in inputs
    )  # All inputs are either Nodes or the same type as output

    if isinstance(output, list) or isinstance(output, tuple):
        assert all(
            isinstance(x, Node) or len(output) == len(x) for x in inputs
        ), f"output {output} and inputs {inputs} are of different lengths."
        for k in range(len(output)):
            _args = [x if isinstance(x, Node) else x[k] for x in args]
            _kwargs = {
                kk: vv if isinstance(vv, Node) else vv[k] for kk, vv in kwargs.items()
            }
            output[k] = apply_op(op, output[k], *_args, **_kwargs)
        if isinstance(output, tuple):
            output = tuple(output)

    elif isinstance(output, dict):
        for k, v in output.items():
            _args = [x if isinstance(x, Node) else x[k] for x in args]
            _kwargs = {
                kk: vv if isinstance(vv, Node) else vv[k] for kk, vv in kwargs.items()
            }
            output[k] = apply_op(op, output[k], *_args, **_kwargs)

    elif isinstance(output, NodeContainer):  # this is a NodeContainer object instance
        for k, v in output.__dict__.items():
            _args = [x if isinstance(x, Node) else getattr(x, k) for x in args]
            _kwargs = {
                kk: vv if isinstance(v, Node) else getattr(vv, k)
                for kk, vv in kwargs.items()
            }
            new_v = apply_op(op, v, *_args, **_kwargs)
            setattr(output, k, new_v)
    else:
        pass
    return output
