import inspect
from collections import UserDict, UserList
from opto.trace.nodes import ParameterNode
import functools
import copy


class NodeContainer:
    """Base marker class for containers that hold nodes.

    This class serves as an identifier to distinguish containers of nodes
    from regular Python containers. It has no implementation, serving only
    as a type marker for isinstance checks.

    Notes
    -----
    NodeContainer is used as a base class to identify objects that contain
    Node objects and may need special handling during graph construction
    and parameter collection.

    See Also
    --------
    ParameterContainer : Extends NodeContainer with parameter management
    Seq : List-like container for nodes
    Map : Dict-like container for nodes
    """

    ...


def trainable_method(method):
    """Check if a method is trainable.

    Parameters
    ----------
    method : Any
        The method or attribute to check.

    Returns
    -------
    bool
        True if the method is a trainable FunModule, False otherwise.

    Notes
    -----
    Used internally to identify trainable bundled methods when collecting
    parameters from a container.
    """
    from opto.trace.bundle import FunModule

    if isinstance(method, FunModule):
        return method.trainable
    return False


class ParameterContainer(NodeContainer):
    """Base class for containers that manage parameter nodes.

    ParameterContainer provides automatic collection and management of
    ParameterNode objects and nested containers. It serves as the foundation
    for models and modules in the Trace framework.

    Methods
    -------
    parameters()
        Return a flattened list of all parameters.
    parameters_dict()
        Return a dictionary of all parameters and containers.
    copy()
        Create a deep copy with shared parameter references.

    Notes
    -----
    ParameterContainer implements sophisticated parameter collection:

    1. **Automatic Discovery**: Scans attributes to find ParameterNodes,
       trainable methods, and nested ParameterContainers.

    2. **Recursive Collection**: Traverses nested containers to collect
       all parameters in the hierarchy.

    3. **Method Support**: Recognizes and collects parameters from
       trainable bundled methods.

    4. **Efficient Copying**: The copy() method creates new container
       instances while sharing parameter references, useful for
       creating model variants.

    The parameter collection logic handles:
    - Direct ParameterNode attributes
    - Trainable FunModule methods
    - Nested ParameterContainers
    - Class methods wrapped with functools.partial

    See Also
    --------
    Module : Extends ParameterContainer with forward() method
    ParameterNode : The parameters being collected
    bundle : Decorator that can make methods trainable
    """

    def parameters(self):
        """Return a flattened list of all parameters in the container.

        Returns
        -------
        list[ParameterNode]
            All ParameterNode objects in this container and nested containers.

        Raises
        ------
        ValueError
            If the container contains an unknown parameter type.

        Notes
        -----
        Recursively traverses nested ParameterContainers to collect all
        parameters. The returned list is suitable for passing to optimizers.
        """
        parameters = []
        for k, v in self.parameters_dict().items():
            if isinstance(v, ParameterNode):
                parameters.append(v)
            elif isinstance(v, ParameterContainer):
                parameters.extend(v.parameters())
            else:
                raise ValueError("The model contains an unknown parameter type.")

        return parameters

    def parameters_dict(self):
        """Return a dictionary of all the parameters in the model, including
        both trainable and non-trainable parameters. The dict contains
        ParameterNodes or ParameterContainers.
        """
        from opto.trace.bundle import FunModule

        parameters = {}
        for name, attr in inspect.getmembers(self):
            if name.startswith('__TRACE_RESERVED_'):
                # These are reserved for internal use.
                continue

            if isinstance(attr, functools.partial):  # this is a class method
                method = attr.func.__self__
                if trainable_method(method):
                    parameters[name] = method.parameter
            elif isinstance(attr, FunModule):
                # when a bundle method is not trainable
                # it shows up as a FunModule attribute
                if trainable_method(attr):
                    parameters[name] = attr.parameter
            elif trainable_method(attr):  # method attribute
                parameters[name] = attr.parameter
            elif isinstance(attr, ParameterNode):
                parameters[name] = attr
            elif isinstance(attr, ParameterContainer):
                parameters[name] = attr

        assert all(
            isinstance(v, (ParameterNode, ParameterContainer))
            for v in parameters.values()
        )

        return parameters  # include both trainable and non-trainable parameters

    def copy(self):
        """Create a deep copy with shared parameter references.

        Returns
        -------
        ParameterContainer
            A new container with copied structure but shared parameters.

        Notes
        -----
        This method creates new container instances while maintaining
        references to the original ParameterNode objects. This is useful
        for creating model variants that share parameters but have
        independent structure.

        The copying process:
        1. Deep copies the entire container structure
        2. Replaces parameter references with originals
        3. Recursively applies to nested containers
        """

        # NOTE This current code is not optimized for speed; it does extra traversals and copying.

        new_container = copy.deepcopy(self)

        # Set the parameters to the original ones
        for name, attr in inspect.getmembers(self):
            if isinstance(attr, functools.partial):  # this is a class method
                method = attr.func.__self__
                if trainable_method(method):
                    new_attr = getattr(new_container, name)
                    setattr(new_attr.func.__self__, 'parameter', method.parameter)
            elif trainable_method(attr):  # method attribute
                new_attr = getattr(new_container, name)
                new_attr.parameter = attr.parameter
            elif isinstance(attr, ParameterNode):
                setattr(new_container, name, attr)
            elif isinstance(attr, ParameterContainer):
                setattr(new_container, name, attr.copy())  # recursion

        return new_container

class Seq(UserList, ParameterContainer):
    """List-like container for managing sequences of nodes and parameters.

    Seq provides a list interface while supporting automatic parameter
    collection from contained nodes and nested containers.

    Parameters
    ----------
    *args
        Either a single sequence-like object or multiple items to store.
        If a single argument with __len__ and __getitem__ is provided,
        it's used as the sequence. Otherwise, all arguments become items.

    Attributes
    ----------
    data : list
        The underlying list storage (inherited from UserList).

    Methods
    -------
    parameters_dict()
        Return dictionary of contained parameters.

    Notes
    -----
    Seq is automatically used when converting Python lists/tuples that
    contain nodes. It maintains list semantics while enabling:
    - Parameter collection from contained ParameterNodes
    - Recursive parameter discovery in nested containers
    - Standard list operations (indexing, iteration, etc.)

    See Also
    --------
    Map : Dictionary-like container for nodes
    ParameterContainer : Base class for parameter management
    """

    def __init__(self, *args):
        if (
            len(args) == 1
            and hasattr(args[0], "__len__")
            and hasattr(args[0], "__getitem__")
        ):
            seq = args[0]
        else:
            seq = args
        super().__init__(initlist=seq)

    def parameters_dict(self):
        """Return a dictionary of all the parameters in the model, including
        both trainable and non-trainable parameters. The dict contains
        ParameterNodes or ParameterContainers.
        """
        parameters = {}
        for attr in self.data:
            if isinstance(attr, ParameterNode):
                parameters[attr.name] = attr
            elif isinstance(attr, ParameterContainer):
                parameters[str(attr)] = attr  # TODO: what is the name of the container?

        assert all(
            isinstance(v, (ParameterNode, ParameterContainer))
            for v in parameters.values()
        )
        return parameters


class Map(UserDict, ParameterContainer):
    """Dictionary-like container for managing mappings of nodes and parameters.

    Map provides a dictionary interface while supporting automatic parameter
    collection from contained nodes and nested containers.

    Parameters
    ----------
    mapping : dict
        Initial dictionary of key-value pairs.

    Attributes
    ----------
    data : dict
        The underlying dictionary storage (inherited from UserDict).

    Methods
    -------
    parameters_dict()
        Return dictionary of contained parameters.

    Notes
    -----
    Map is automatically used when converting Python dictionaries that
    contain nodes. It maintains dictionary semantics while enabling:
    - Parameter collection from contained ParameterNodes
    - Recursive parameter discovery in nested containers
    - Standard dictionary operations (key access, iteration, etc.)

    The parameters_dict() method uses dictionary values as parameter
    identifiers when they are ParameterNodes or containers.

    See Also
    --------
    Seq : List-like container for nodes
    ParameterContainer : Base class for parameter management
    """

    def __init__(self, mapping):
        super().__init__(mapping)

    def parameters_dict(self):
        """Return a dictionary of all the parameters in the model, including
        both trainable and non-trainable parameters. The dict contains
        ParameterNodes or ParameterContainers.
        """
        parameters = {}
        for k, v in self.data.items():
            if isinstance(v, ParameterNode):
                parameters[k] = v
            elif isinstance(v, ParameterContainer):
                parameters[str(v)] = v  # TODO: what is the name of the container?

            if isinstance(k, ParameterNode):
                parameters[str(k)] = k
            elif isinstance(k, ParameterContainer):
                raise Exception("The key of a Map cannot be a container.")

        assert all(
            isinstance(v, (ParameterNode, ParameterContainer))
            for v in parameters.values()
        )
        return parameters


#
