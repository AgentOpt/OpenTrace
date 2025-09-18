import os
import pickle
import copy
import sys
import inspect
import textwrap
from opto.trace.containers import ParameterContainer, trainable_method
from opto.trace.nodes import ParameterNode
from opto.trace.projections import Projection, BlackCodeFormatter

import functools
from typing import List, Optional


def model(cls):
    """
    Wrap a class with this decorator. This helps collect parameters for the optimizer. This decorated class cannot be pickled.
    """
    name = f"{cls.__name__}Model"
    bases = (cls, Model)
    new_class = type(name, bases, {})
    new_class.__module__ = cls.__module__
    mod = sys.modules[cls.__module__]
    setattr(mod, name, new_class)
    return new_class

# Old code
# def model(cls):
#     """
#     Wrap a class with this decorator. This helps collect parameters for the optimizer. This decorated class cannot be pickled.
#     """

#     class ModelWrapper(cls, Module):

#         def export(self, filename, projections: Optional[List[Projection]] = None):
#             """Dump the model's source code to a file, including all methods and attributes.
#             Ignores dunder methods unless they were overridden by the user.
#             """
#             if projections is None:
#                 projections = [BlackCodeFormatter()]

#             trace_model_body = f"class {cls.__name__}:\n"

#             # Get all members of the class
#             all_members = inspect.getmembers(self)
#             cls_members = inspect.getmembers(cls)
#             cls_member_names = [m[0] for m in cls_members]

#             # Filter out dunder methods unless they were overridden
#             filtered_members = []
#             for name, member in all_members:
#                 # Skip internal trace reserved members
#                 if name.startswith('__TRACE_RESERVED_'):
#                     continue

#                 if name not in cls_member_names:
#                     continue

#                 # Include if it's not a dunder method or if it was overridden
#                 if not name.startswith('__'):
#                     filtered_members.append((name, member))
#                 elif name.startswith('__'):
#                     # For dunder methods, check if they were overridden
#                     try:
#                         print(cls.__name__, "<>", member.__qualname__)
#                         # MixedClass <> test_export_mixed_trainable.<locals>.MixedClass.__init__
#                         # if we wrap it inside a function, the qualname is different than when we dont
#                         if hasattr(member, '__qualname__') and cls.__name__ in member.__qualname__:
#                             filtered_members.append((name, member))
#                     except (AttributeError, TypeError):
#                         # Skip if we can't determine if it was overridden
#                         continue

#             # Process each member
#             for i, (name, member) in enumerate(filtered_members):
#                 print(name, member)
#                 if 'FunModule' in str(member):
#                     # Handle methods
#                     if member.parameter is not None:
#                         source = member.parameter.data
#                     else:
#                         source = member.info['source']
#                     source = textwrap.dedent(source)
#                     indented = textwrap.indent(source, "    ")
#                     trace_model_body += indented
#                 else:  # this is a class method
#                     source = inspect.getsource(member)
#                     source = textwrap.dedent(source)
#                     indented = textwrap.indent(source, "    ")
#                     trace_model_body += indented

#                 if i < len(all_members) - 1:
#                     trace_model_body += "\n"  # only one newline between members

#             # Replace node initializations with their current values
#             # WARNING: there might be corner cases that this static analysis does not cover
#             import re
#             node_pattern = r'self\.(\w+)\s*=\s*node\([^)]*\)'

#             def replace_node(match):
#                 attr_name = match.group(1)
#                 if hasattr(self, attr_name):
#                     attr = getattr(self, attr_name)
#                     if hasattr(attr, 'data'):
#                         return f"self.{attr_name} = {attr.data}"
#                 return match.group(0)  # Return original if replacement not possible

#             trace_model_body = re.sub(node_pattern, replace_node, trace_model_body)

#             trace_model_body = functools.reduce(lambda body, proj: proj.project(body), projections, trace_model_body)

#             with open(filename, "w") as f:
#                 f.write(trace_model_body)


#         def __deepcopy__(self, memo):
#             # regular deepcopy behavior, because we will overwrite __setstate__ and __getstate__ for pickling
#             cls = self.__class__
#             result = cls.__new__(cls)
#             memo[id(self)] = result
#             for k, v in self.__dict__.items():
#                 setattr(result, k, copy.deepcopy(v, memo))
#             return result

#         def __getstate__(self):
#             parameters_dict = self.parameters_dict()
#             non_parameters_dict = {}
#             for k, v in self.__dict__.items():
#                 if k not in parameters_dict:
#                     if k.startswith('__TRACE_RESERVED_'):
#                         # These are reserved for internal use.
#                         continue
#                     non_parameters_dict[k] = v
#             return dict(parameters_dict=parameters_dict,
#                         non_parameters_dict=non_parameters_dict)

#         def __setstate__(self, state):
#             parameters_dict = state['parameters_dict']
#             non_parameters_dict = state['non_parameters_dict']
#             self._set(parameters_dict)
#             # self.__dict__.update(non_parameters_dict)

#         def save(self, file_name: str):
#             """Save the parameters of the model to a pickle file."""
#             directory = os.path.dirname(file_name)
#             if directory != "":
#                 os.makedirs(directory, exist_ok=True)
#             with open(file_name, "wb") as f:
#                 pickle.dump(copy.deepcopy(self.__getstate__()), f)

#         def load(self, file_name):
#             """Load the parameters of the model from a pickle file."""
#             with open(file_name, "rb") as f:
#                 loaded_data = pickle.load(f)
#                 self.__setstate__(loaded_data)

#     # return ModelWrapper
#     name = f"{cls.__name__}ModelWrapper"
#     ModelWrapper.__name__ = name
#     ModelWrapper.__qualname__ = name

#     # register the class in the module namespace for pickle
#     ModelWrapper.__module__ = cls.__module__
#     mod = sys.modules[cls.__module__]
#     setattr(mod, name, ModelWrapper)
#     return ModelWrapper

class Module(ParameterContainer):
    """Module is a ParameterContainer which has a forward method."""

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def save(self, file_name: str):
        """Save the parameters of the model to a pickle file."""
        # detect if the directory exists
        directory = os.path.dirname(file_name)
        if directory != "":
            os.makedirs(directory, exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(copy.deepcopy(self.parameters_dict()), f)

    def load(self, file_name):
        """Load the parameters of the model from a pickle file."""
        with open(file_name, "rb") as f:
            loaded_data = pickle.load(f)
        self._set(loaded_data)

    def _set(self, new_parameters):
        """Set the parameters of the model from a dictionary.
        new_parameters is a ParamterContainer or a parameter dict.
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

    def export(self, filename, projections: Optional[List[Projection]] = None):
        if projections is None:
            projections = [BlackCodeFormatter()]
        cls = self.__class__
        trace_model_body = f"class {cls.__name__}:\n"
        all_members = inspect.getmembers(self)
        cls_members = inspect.getmembers(cls)
        cls_member_names = [m[0] for m in cls_members]
        filtered_members = []
        for name, member in all_members:
            if name.startswith('__TRACE_RESERVED_'):
                continue
            if name not in cls_member_names:
                continue
            if not name.startswith('__'):
                filtered_members.append((name, member))
            elif name.startswith('__'):
                try:
                    if hasattr(member, '__qualname__') and cls.__name__ in member.__qualname__:
                        filtered_members.append((name, member))
                except (AttributeError, TypeError):
                    continue
        for i, (name, member) in enumerate(filtered_members):
            if 'FunModule' in str(member):
                if member.parameter is not None:
                    source = member.parameter.data
                else:
                    source = member.info['source']
                source = textwrap.dedent(source)
                indented = textwrap.indent(source, "    ")
                trace_model_body += indented
            else:
                source = inspect.getsource(member)
                source = textwrap.dedent(source)
                indented = textwrap.indent(source, "    ")
                trace_model_body += indented
            if i < len(all_members) - 1:
                trace_model_body += "\n"
        import re
        node_pattern = r'self\.(\w+)\s*=\s*node\([^)]*\)'
        def replace_node(match):
            attr_name = match.group(1)
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if hasattr(attr, 'data'):
                    return f"self.{attr_name} = {attr.data}"
            return match.group(0)
        trace_model_body = re.sub(node_pattern, replace_node, trace_model_body)
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
