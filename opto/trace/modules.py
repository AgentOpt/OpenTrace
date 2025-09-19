import os
import pickle
import copy
import inspect
import textwrap
from opto.trace.containers import ParameterContainer, trainable_method
from opto.trace.nodes import ParameterNode
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
    >>> # m.parameters() returns all trainable parameters
    >>> # m.export('model.py') saves current state as code
    """

    class ModelWrapper(cls, Module):

        def export(self, filename, projections: Optional[List[Projection]] = None):
            """Dump the model's source code to a file, including all methods and attributes.
            Ignores dunder methods unless they were overridden by the user.
            """
            if projections is None:
                projections = [BlackCodeFormatter()]

            trace_model_body = f"class {cls.__name__}:\n"

            # Get all members of the class
            all_members = inspect.getmembers(self)
            cls_members = inspect.getmembers(cls)
            cls_member_names = [m[0] for m in cls_members]

            # Filter out dunder methods unless they were overridden
            filtered_members = []
            for name, member in all_members:
                # Skip internal trace reserved members
                if name.startswith('__TRACE_RESERVED_'):
                    continue

                if name not in cls_member_names:
                    continue

                # Include if it's not a dunder method or if it was overridden
                if not name.startswith('__'):
                    filtered_members.append((name, member))
                elif name.startswith('__'):
                    # For dunder methods, check if they were overridden
                    try:
                        print(cls.__name__, "<>", member.__qualname__)
                        # MixedClass <> test_export_mixed_trainable.<locals>.MixedClass.__init__
                        # if we wrap it inside a function, the qualname is different than when we dont
                        if hasattr(member, '__qualname__') and cls.__name__ in member.__qualname__:
                            filtered_members.append((name, member))
                    except (AttributeError, TypeError):
                        # Skip if we can't determine if it was overridden
                        continue

            # Process each member
            for i, (name, member) in enumerate(filtered_members):
                print(name, member)
                if 'FunModule' in str(member):
                    # Handle methods
                    if member.parameter is not None:
                        source = member.parameter.data
                    else:
                        source = member.info['source']
                    source = textwrap.dedent(source)
                    indented = textwrap.indent(source, "    ")
                    trace_model_body += indented
                else:  # this is a class method
                    source = inspect.getsource(member)
                    source = textwrap.dedent(source)
                    indented = textwrap.indent(source, "    ")
                    trace_model_body += indented

                if i < len(all_members) - 1:
                    trace_model_body += "\n"  # only one newline between members

            # Replace node initializations with their current values
            # WARNING: there might be corner cases that this static analysis does not cover
            import re
            node_pattern = r'self\.(\w+)\s*=\s*node\([^)]*\)'

            def replace_node(match):
                attr_name = match.group(1)
                if hasattr(self, attr_name):
                    attr = getattr(self, attr_name)
                    if hasattr(attr, 'data'):
                        return f"self.{attr_name} = {attr.data}"
                return match.group(0)  # Return original if replacement not possible

            trace_model_body = re.sub(node_pattern, replace_node, trace_model_body)

            trace_model_body = functools.reduce(lambda body, proj: proj.project(body), projections, trace_model_body)

            with open(filename, "w") as f:
                f.write(trace_model_body)

    return ModelWrapper


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