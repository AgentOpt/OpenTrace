import os
import ast
import pickle
import copy
import sys
import inspect
import textwrap
from opto.trace.containers import ParameterContainer, trainable_method
from opto.trace.nodes import ParameterNode, Node
from opto.trace.projections import Projection, BlackCodeFormatter

import functools
from typing import List, Optional

def model(cls):
    """Decorator to transform a class into a Trace-compatible model with parameter collection.

    The model decorator wraps a class to enable automatic parameter collection,
    optimization support, and code export functionality. Decorated classes become
    Module subclasses with enhanced capabilities for the Trace framework.

    Parameters
    ----------
    cls : type
        The class to be decorated. Should define methods and attributes that
        may include trainable parameters.

    Returns
    -------
    type
        A wrapped version of the class that:
        - Inherits from both the original class and Module
        - Automatically collects parameters for optimization
        - Provides export functionality for code generation
        - Supports saving/loading of parameters

    Notes
    -----
    The model decorator provides several key features:

    1. **Parameter Collection**: Automatically identifies and collects all
       ParameterNode and bundled method parameters for optimization.

    2. **Code Export**: Can export the current state of the model (including
       learned parameters) as executable Python code.

    3. **Integration**: Seamlessly integrates with Trace optimizers and training
       loops through the Module interface.

    4. **State Management**: Inherits save/load functionality from Module for
       parameter persistence.

    Limitations:
    - Decorated classes cannot be pickled directly due to dynamic wrapping
    - Use the save/load methods for persistence instead

    See Also
    --------
    Module : Base class providing core functionality
    bundle : Decorator for making methods trainable
    ParameterNode : Trainable parameters within models

    Examples
    --------
    >>> @model
    >>> class MyModel:
    ...     def __init__(self):
    ...         self.weight = node(0.5, trainable=True)
    ...     
    ...     @bundle(trainable=True)
    ...     def forward(self, x):
    ...         return x * self.weight
    >>> 
    >>> m = MyModel()
    >>> m.parameters() returns all trainable parameters
    >>> m.export('model.py') saves current state as code
    """
    name = f"{cls.__name__}Model"
    bases = (cls, Model)
    # for export to work, we save the references to the original cls
    __TRACE_RESERVED_cls_name = cls.__name__
    temp_cls_members = inspect.getmembers(cls)
    __TRACE_RESERVED_cls_members = []
    __TRACE_RESERVED_cls_name_to_source = {}
    for name, member in temp_cls_members:
        if name.startswith('__TRACE_RESERVED_'):
            continue
        if not name.startswith('__'):
            __TRACE_RESERVED_cls_members.append((name, member))
        elif name.startswith('__'):
            try:
                if hasattr(member, '__qualname__') and cls.__name__ in member.__qualname__:
                    inspect.getsource(member)  # additionally we see if this works
                    __TRACE_RESERVED_cls_members.append((name, member))
            except (AttributeError, TypeError):
                continue

    for name, member in __TRACE_RESERVED_cls_members:
        if 'FunModule' in str(member):
            # for these class method members, we need to access their content dynamically
            continue
        __TRACE_RESERVED_cls_name_to_source[name] = inspect.getsource(member)

    new_class = type(name, bases, {})
    new_class.__module__ = cls.__module__

    # for export
    new_class.reserved_cls_name = __TRACE_RESERVED_cls_name
    new_class.reserved_cls_members = __TRACE_RESERVED_cls_members
    new_class.reserved_cls_name_to_source = __TRACE_RESERVED_cls_name_to_source

    mod = sys.modules[cls.__module__]
    setattr(mod, name, new_class)
    return new_class

class Module(ParameterContainer):
    """Base class for all Trace models and wrapped functions.

    Module extends ParameterContainer to provide a standard interface for
    components in the Trace framework. It defines the forward computation
    pattern and provides parameter management functionality.

    Methods
    -------
    forward(*args, **kwargs)
        Define the forward computation. Must be overridden by subclasses.
    __call__(*args, **kwargs)
        Makes the module callable, delegating to forward().
    save(file_name)
        Save model parameters to a pickle file.
    load(file_name)
        Load model parameters from a pickle file.
    _set(new_parameters)
        Update parameters from a dictionary or ParameterContainer.

    Attributes
    ----------
    Inherits all attributes from ParameterContainer, including:
    - Automatic parameter collection
    - Parameter dictionary access
    - Recursive parameter traversal

    Notes
    -----
    Module serves as the foundation for:

    1. **Model Classes**: Classes decorated with @model inherit from Module
       to gain parameter management capabilities.

    2. **Function Wrappers**: FunModule extends Module to wrap functions
       as traceable operators.

    3. **Custom Components**: Users can subclass Module directly to create
       custom traceable components.

    The forward() method follows PyTorch's design pattern, providing a
    familiar interface for defining computations.

    Parameter Management:
    - Parameters are automatically collected from attributes
    - Supports nested modules and recursive parameter collection
    - Save/load functionality preserves learned parameters

    See Also
    --------
    ParameterContainer : Base class for parameter management
    model : Decorator that creates Module subclasses
    FunModule : Module subclass for wrapped functions

    Examples
    --------
    >>> class LinearLayer(Module):
    ...     def __init__(self, input_dim, output_dim):
    ...         self.weight = node(np.random.randn(input_dim, output_dim), trainable=True)
    ...         self.bias = node(np.zeros(output_dim), trainable=True)
    ...     
    ...     def forward(self, x):
    ...         return x @ self.weight + self.bias
    >>> 
    >>> layer = LinearLayer(10, 5)
    >>> output = layer(input_data)  # Calls forward()
    >>> layer.save('layer_params.pkl')  # Save parameters
    """

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def save(self, file_name: str):
        """Save the parameters of the model to a pickle file.

        Parameters
        ----------
        file_name : str
            Path to the output pickle file. Directories are created if needed.

        Notes
        -----
        Saves a deep copy of parameters to prevent reference issues.
        The saved file can be loaded with the load() method.
        """
        # detect if the directory exists
        directory = os.path.dirname(file_name)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(copy.deepcopy(self.parameters_dict()), f)

    def load(self, file_name):
        """Load the parameters of the model from a pickle file.

        Parameters
        ----------
        file_name : str
            Path to the pickle file containing saved parameters.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        AssertionError
            If loaded parameters don't match model structure.
        """
        with open(file_name, "rb") as f:
            loaded_data = pickle.load(f)
        self._set(loaded_data)

    def _set(self, new_parameters):
        """Update model parameters from a dictionary or ParameterContainer.

        Parameters
        ----------
        new_parameters : dict or ParameterContainer
            New parameter values to set. Keys must match existing parameter names.

        Raises
        ------
        AssertionError
            If not all model parameters are present in new_parameters.

        Notes
        -----
        This method updates existing parameters in-place and adds any new
        parameters that don't exist in the current model.
        """
        assert isinstance(new_parameters, (dict, ParameterContainer))
        if isinstance(new_parameters, ParameterContainer):
            new_parameters_dict = new_parameters.parameters_dict()
        else:
            new_parameters_dict = new_parameters  # dictionary

        parameters_dict = self.parameters_dict()

        assert all(
            k in new_parameters_dict for k in parameters_dict.keys()
        ), """ Not all model parameters are in the new parameters dictionary. """

        for k, v in new_parameters_dict.items():
            if k in parameters_dict:  # if the parameter exists
                assert isinstance(v, (ParameterNode, ParameterContainer))
                parameters_dict[k]._set(v)
            else:  # if the parameter does not exist
                assert k not in self.__dict__
                setattr(self, k, v)


class Model(Module):
    """ Base class for all models. A model is a container of parameters with methods. """

    def _replace_self_assignments_with_node_data(self, source: str) -> str:
        """
        Replace any `self.<attr> = ...` in `source` with `self.<attr> = <attr.data>`
        when `getattr(self, '<attr>')` is a Node and has a `.data` attribute.

        If `.data` is a string of Python code, it will be inserted as code.
        Otherwise `.data` is inserted via repr().
        """

        class Rewriter(ast.NodeTransformer):
            def __init__(self, outer_self):
                self._self = outer_self

            def visit_Assign(self, node: ast.Assign) -> ast.AST:
                # Check if any target is `self.<name>`
                replace_names = []
                for t in node.targets:
                    if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                        replace_names.append(t.attr)

                # Nothing to do
                if not replace_names:
                    return self.generic_visit(node)

                # Decide replacement expression ONCE per assignment
                new_value = node.value
                for name in replace_names:
                    try:
                        attr = getattr(self._self, name)
                    except AttributeError:
                        continue
                    if isinstance(attr, Node) and hasattr(attr, "data"):
                        data = attr.data
                        # If it's a string, assume it's code; otherwise, literal via repr
                        if isinstance(data, str):
                            try:
                                new_value = ast.parse(data, mode="eval").body
                            except SyntaxError:
                                # fall back to literal repr if not valid code
                                new_value = ast.parse(repr(data), mode="eval").body
                        else:
                            new_value = ast.parse(repr(data), mode="eval").body
                        # Once we have a replacement value, we can stop; it applies to the whole RHS
                        break

                node.value = new_value
                return node

        tree = ast.parse(source)
        tree = Rewriter(self).visit(tree)
        ast.fix_missing_locations(tree)
        # Python 3.9+: ast.unparse available
        return ast.unparse(tree)

    def export(self, filename, projections: Optional[List[Projection]] = None):
        if projections is None:
            projections = [BlackCodeFormatter()]
        cls = self.__class__
        name = cls.reserved_cls_name
        trace_model_body = f"class {name}:\n"
        cls_members = cls.reserved_cls_members

        for i, (name, member) in enumerate(cls_members):
            if 'FunModule' in str(member):
                if member.parameter is not None:
                    source = member.parameter.data
                else:
                    source = member.info['source']
                source = textwrap.dedent(source)
                indented = textwrap.indent(source, "    ")
                trace_model_body += indented
            else:
                source = cls.reserved_cls_name_to_source[name] # inspect.getsource(member)
                source = textwrap.dedent(source)
                indented = textwrap.indent(source, "    ")
                trace_model_body += indented
            if i < len(cls_members) - 1:
                trace_model_body += "\n"

        trace_model_body = self._replace_self_assignments_with_node_data(trace_model_body)

        trace_model_body = functools.reduce(lambda body, proj: proj.project(body), projections, trace_model_body)
        with open(filename, "w") as f:
            f.write(trace_model_body)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, copy.deepcopy(v, memo))
        return result

    def __getstate__(self):
        parameters_dict = self.parameters_dict()
        non_parameters_dict = {}
        for k, v in self.__dict__.items():
            if k not in parameters_dict:
                if k.startswith('__TRACE_RESERVED_'):
                    continue
                non_parameters_dict[k] = v
        return dict(parameters_dict=parameters_dict,
                    non_parameters_dict=non_parameters_dict)

    def __setstate__(self, state):
        parameters_dict = state['parameters_dict']
        non_parameters_dict = state['non_parameters_dict']
        self._set(parameters_dict)
        self.__dict__.update(non_parameters_dict)

    def save(self, file_name: str):
        directory = os.path.dirname(file_name)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(copy.deepcopy(self.__getstate__()), f)

    def load(self, file_name):
        with open(file_name, "rb") as f:
            loaded_data = pickle.load(f)
            self.__setstate__(loaded_data)
