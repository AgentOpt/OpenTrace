import copy
import ctypes
import functools
import inspect
import re
import sys
import traceback
import asyncio

from typing import List, Dict, Callable, Union, Any

from opto.trace.broadcast import recursive_conversion
from opto.trace.errors import ExecutionError, TraceMissingInputsError
from opto.trace.modules import Module
from opto.trace.nodes import GRAPH
from opto.trace.nodes import (
    MessageNode,
    USED_NODES,
    Node,
    ParameterNode,
    ExceptionNode,
    node,
    get_op_name,
)
from opto.trace.utils import contain

# This is a global flag to allow external dependencies to be used in the operator.
ALLOW_EXTERNAL_DEPENDENCIES = None
def disable_external_dependencies_check(allow_external_dependencies: bool):
    """Set the global flag for allowing external dependencies in operators.

    Parameters
    ----------
    allow_external_dependencies : bool
        If True, operators can use nodes that are not explicitly passed as inputs.
        If False, all node dependencies must be explicit inputs.

    Notes
    -----
    This global setting affects all subsequently created bundle decorators.
    Use with caution as it can make dependency tracking less explicit.
    """
    global ALLOW_EXTERNAL_DEPENDENCIES
    ALLOW_EXTERNAL_DEPENDENCIES = allow_external_dependencies

def bundle(
    description=None,
    traceable_code=False,
    _process_inputs=True,
    trainable=False,
    catch_execution_error=True,
    allow_external_dependencies=False,
    overwrite_python_recursion=False,
    projections=None,
):
    """Decorator to wrap functions for integration with the Trace computation graph.

    The bundle decorator transforms regular Python functions into traced operators that
    automatically create MessageNodes when called with Node inputs. This enables automatic
    differentiation and optimization of the wrapped code.

    Parameters
    ----------
    description : str, optional
        Description of the operator in format "[op_name] details". If not provided,
        generated from function name and docstring.
    traceable_code : bool, default=False
        If True, the function's internal operations are also traced, creating a nested
        graph structure. Enables fine-grained optimization but increases overhead.
    _process_inputs : bool, default=True
        If True, automatically extracts data from Node inputs before passing to function.
        If False, passes Node objects directly to the function.
    trainable : bool, default=False
        If True, treats the function's source code as a trainable parameter that can
        be modified during optimization. Enables code synthesis and modification.
    catch_execution_error : bool, default=True
        If True, exceptions are caught and converted to ExceptionNodes in the graph.
        If False, exceptions propagate normally.
    allow_external_dependencies : bool, default=False
        If True, permits the function to use nodes not passed as explicit inputs.
        These hidden dependencies are tracked separately.
    overwrite_python_recursion : bool, default=False
        If True, recursive calls within the function call the original function directly.
        If False, recursive calls go through the wrapped version.
    projections : list[Projection], optional
        Constraints to apply when updating trainable code parameters.

    Returns
    -------
    FunModule
        A wrapped version of the function that:
        - Creates MessageNodes when called with Node inputs
        - Preserves the original function signature
        - Can be used with other decorators

    Notes
    -----
    The bundle decorator enables several key features:

    1. **Automatic Tracing**: Functions automatically participate in the computation
       graph when called with Node arguments.

    2. **Error Handling**: Exceptions can be captured as part of the graph, enabling
       error-aware optimization.

    3. **Code as Data**: With trainable=True, function source code becomes data that
       can be optimized, enabling program synthesis.

    4. **Nested Tracing**: With traceable_code=True, operations inside the function
       are individually traced, enabling fine-grained optimization.

    5. **Compatibility**: The wrapped function maintains its original signature and
       can be used normally with non-Node inputs when tracing is disabled.

    See Also
    --------
    FunModule : The class that implements bundled functions
    MessageNode : Nodes created by bundled functions
    ParameterNode : Created when trainable=True

    Examples
    --------
    >>> # Simple bundled function
    >>> @bundle()
    >>> def add(a, b):
    ...     return a + b
    >>> 
    >>> # Bundled function with trainable code
    >>> @bundle(trainable=True, description="[optimizer] Optimization step")
    >>> def optimize_step(gradient, parameter):
    ...     return parameter - 0.01 * gradient
    >>> 
    >>> # Using bundled functions
    >>> x = node(5)
    >>> y = node(3)
    >>> z = add(x, y)  # Creates MessageNode with value 8
    """
    prev_f_locals = inspect.stack()[1].frame.f_locals

    def decorator(fun):
        fun_module = FunModule(
            fun=fun,
            description=description,
            traceable_code=traceable_code,
            _process_inputs=_process_inputs,
            trainable=trainable,
            catch_execution_error=catch_execution_error,
            allow_external_dependencies=allow_external_dependencies,
            overwrite_python_recursion=overwrite_python_recursion,
            _ldict=prev_f_locals,  # Get the locals of the calling function
            projections=projections,
        )
        return fun_module

    return decorator


class trace_nodes:
    """Context manager for tracking node usage within operators.

    Maintains a thread-local stack of sets that record which nodes are accessed
    during operator execution. This enables detection of implicit dependencies
    and proper graph construction.

    Notes
    -----
    This context manager is used internally by FunModule to track node dependencies.
    It uses contextvars for thread-safety and async compatibility.

    The tracking works by:
    1. Creating a new set for nodes on __enter__
    2. Nodes add themselves to this set when their data is accessed
    3. The set is removed from the stack on __exit__

    See Also
    --------
    USED_NODES : Global context variable maintaining the stack
    Node.data : Property that registers access in USED_NODES
    """

    def __init__(self):
        self.token = None

    def __enter__(self):
        # nodes = set()
        # USED_NODES.append(nodes)
        # return nodes

        nodes = set()
        current = USED_NODES.get()
        new_list = current + [nodes]
        self.token = USED_NODES.set(new_list)
        return nodes

    def __exit__(self, type, value, traceback):
        # USED_NODES.pop()
        if self.token:
            USED_NODES.reset(self.token)

class FunModule(Module):
    """Wrapper class that traces function execution in the computation graph.

    FunModule implements the bundle decorator's functionality, transforming regular
    Python functions into graph-aware operators that create MessageNodes. It handles
    input processing, error catching, dependency tracking, and code trainability.

    Parameters
    ----------
    fun : callable
        The function to be wrapped and traced.
    description : str, optional
        Operator description in format "[op_name] details". Auto-generated if not provided.
    traceable_code : bool, default=False
        Whether to trace operations inside the function.
    _process_inputs : bool, default=True
        Whether to extract data from Node inputs before passing to function.
    trainable : bool, default=False
        Whether the function's source code is a trainable parameter.
    catch_execution_error : bool, default=True
        Whether to catch and wrap exceptions as ExceptionNodes.
    allow_external_dependencies : bool, default=False
        Whether to permit hidden node dependencies.
    overwrite_python_recursion : bool, default=False
        Whether recursive calls bypass the wrapper.
    projections : list[Projection], optional
        Constraints for trainable code updates.
    _ldict : dict, optional
        Local namespace for code execution.

    Attributes
    ----------
    info : dict
        Metadata about the wrapped function including source, signature, and execution details.
    parameter : ParameterNode or None
        The trainable code parameter when trainable=True.
    trainable : bool
        Whether the code is trainable.

    Methods
    -------
    forward(*args, **kwargs)
        Execute the wrapped function with tracing.
    get_source(fun)
        Extract source code from a function.

    Notes
    -----
    FunModule implements sophisticated function wrapping:

    1. **Input Processing**: Automatically converts Node inputs to data values
       for the wrapped function, unless _process_inputs=False.

    2. **Dependency Tracking**: Uses trace_nodes context to detect all nodes
       accessed during execution, even indirect ones.

    3. **Error Handling**: Can catch exceptions and convert them to ExceptionNodes,
       preserving error information in the graph.

    4. **Code Trainability**: When trainable=True, the function's source code
       becomes a ParameterNode that optimizers can modify.

    5. **Traceable Execution**: When traceable_code=True, creates detailed graphs
       of internal operations for fine-grained optimization.

    See Also
    --------
    bundle : Decorator that creates FunModule instances
    Module : Base class for Trace modules
    MessageNode : Nodes created by FunModule execution
    ParameterNode : Code parameters when trainable=True
    """

    def __init__(
        self,
        fun: Callable,
        description: str = None,
        traceable_code: bool = False,
        _process_inputs: bool = True,
        trainable=False,
        catch_execution_error=True,
        allow_external_dependencies=False,
        overwrite_python_recursion=False,
        projections=None,
        _ldict=None,
    ):

        assert _ldict is None or isinstance(
            _ldict, dict
        ), "_ldict must be a dictionary. or None"
        self._ldict = {} if _ldict is None else _ldict.copy()

        assert callable(fun), "fun must be a callable."

        # Get the source code of the function, excluding the decorator line
        source, line_number = self.get_source(fun)

        # Construct the info dictionary
        docstring = inspect.getdoc(fun)
        self.info = dict(  # TODO explain the info dict
            # info about the decorated function
            fun=None,  # to be defined at run time
            fun_name=fun.__qualname__,
            _fun_name=fun.__name__,  # this saves the pure fun name (without the class name); this should not be modified.map=
            doc=inspect.cleandoc(docstring) if docstring is not None else "",
            signature=inspect.signature(fun),
            source=source,
            line_number=line_number,
            file=inspect.getfile(fun),
            error_comment=None,
            traceback=None,
            # for traceable_code == True
            output=None,  # output of the function
            inputs={"args": [], "kwargs": {}},  # inputs of the function
            # misc
            external_dependencies=None,
        )

        if description is None:
            # Generate the description from the function name and docstring.
            description = f"[{self.info['fun_name']}] {self.info['doc']}"
        assert len(get_op_name(description)) > 0

        self.traceable_code = traceable_code
        self._fun = fun
        self.description = description
        self._process_inputs = _process_inputs
        self.catch_execution_error = catch_execution_error
        self.allow_external_dependencies = allow_external_dependencies
        self.parameter = None
        self.overwrite_python_recursion = overwrite_python_recursion
        if trainable:
            # trainable code uses exec which has an effect of overwrite_python_recursion==True.
            self.overwrite_python_recursion = True
            # assert overwrite_python_recursion, "trainable requires overwrite_python_recursion to be True."

            signature_sr = re.search(r"\s*(def.*\"\"\")", source, re.DOTALL)
            if (
                signature_sr is None
            ):  # if there is no docstring just take the first line
                signature = re.search(r"\s*(def.*:)", source).group(1)
            else:
                signature = signature_sr.group(1)

            self.parameter = ParameterNode(
                self.info["source"],
                name="__code",
                description="The code should start with:\n" + signature,
                projections=projections,
            )

    @property
    def trainable(self):
        return self.parameter is not None

    @property
    def fun(self, *args, **kwargs):
        """Return a callable function. Return the decorated function if the parameter is None. Otherwise, return the function defined by the parameter. When exception happens during the defining the function with the parameter, raise a trace.ExecutionError."""

        # This function should be later called within trace_nodes context manager.
        if self.parameter is None:
            return self._fun
        else:
            code = (
                self.parameter._data
            )  # This is not traced, but we will add this as the parent later.
            # before we execute,  we should try to import all the global name spaces from the original function
            try:
                _ldict = {}
                gdict = self._fun.__globals__.copy()
                gdict.update(self._ldict)
                exec(code, gdict, _ldict)  # define the function
                fun_name =  self.info["_fun_name"]
                fun = _ldict[fun_name]
                fun.__globals__.update(_ldict)  # update the global dict (This would allow for recursive calls and defining new functions)
            except SyntaxError as err:
                error_class = err.__class__.__name__
                detail = err.args[0]
                line_number = err.lineno
                e = err
            except Exception as err:
                # TODO would this ever happen?
                error_class = err.__class__.__name__
                detail = err.args[0]
                cl, exc, tb = sys.exc_info()
                line_number = traceback.extract_tb(tb)[-1][1]
                e = err
            else:
                return fun

            base_message = f"({error_class}) {detail}."
            commented_code = (
                self.generate_comment(code, base_message, line_number, 1)
                + f"\n{base_message}"
            )
            raw_traceback = (
                "SyntaxError in trainable code definition.\n" + commented_code
                if "SyntaxError" == error_class
                else traceback.format_exc()
            )
            self.info["error_comment"] = commented_code
            self.info["traceback"] = raw_traceback  # This is saved for user debugging

            e_node = ExceptionNode(
                e,
                inputs={"code": self.parameter},
                description=f"[exception] The code parameter {self.parameter.py_name} has an error.",
                name="exception_" + self.parameter.py_name,
                info=self.info,
            )

            raise ExecutionError(e_node)

    @property
    def name(self):
        return get_op_name(self.description)

    def _wrap_inputs(self, fun, args, kwargs):
        """Wrap the inputs to a function as nodes when they're not.

        Args:
            fun (callable): the function to be wrapped.
            args (list): the positional arguments of the function.
            kwargs (dict): the keyword arguments of the function.

        Returns:
            inputs (dict): the inputs dict to construct the MessageNode (constructed by args and kwargs).
            args (list): the wrapped positional arguments.
            kwargs (dict): the wrapped keyword arguments.
            _args (list): the original positional arguments (including the default values).
            _kwargs (dict): the original keyword arguments (including the default values).
        """
        # Wrap the inputs as nodes

        # add default into kwargs
        ba = inspect.signature(fun).bind(*args, **kwargs)
        a0 = ba.arguments.copy()
        ba.apply_defaults()
        a1 = ba.arguments
        fullargspec = inspect.getfullargspec(fun)
        # include default into the kwargs
        for k, v in a1.items():
            if k not in a0:
                if k != fullargspec.varargs and k != fullargspec.varkw:
                    kwargs[k] = v
        # convert args and kwargs to nodes, except for FunModule
        _args, _kwargs = args, kwargs  # back up

        args = [
            (
                node(
                    a,
                    name=(
                        fullargspec.args[i]
                        if i < len(fullargspec.args) and not isinstance(a, Node)
                        else None
                    ),
                )
                if not isinstance(a, FunModule)
                else a
            )
            for i, a in enumerate(args)
        ]
        kwargs = {
            k: (
                node(v, name=k if not isinstance(v, Node) else None)
                if not isinstance(v, FunModule)
                else v
            )
            for k, v in kwargs.items()
        }

        # Construct the input dict of the MessageNode from function inputs
        inputs = {}
        # args, varargs, varkw, defaults, kwonlyargs, kwonlydefaults, ann
        _, varargs, varkw, _, _, _, _ = inspect.getfullargspec(fun)

        # bind the node version of args and kwargs
        ba = inspect.signature(fun).bind(*args, **kwargs)
        spec = ba.arguments

        def extract_param(n):
            return (
                n.parameter
                if isinstance(n, FunModule) and n.parameter is not None
                else n
            )

        # expand varargs and varkw
        for k, v in spec.items():
            if k == varargs:  # unpack varargs
                for i, n in enumerate(v):
                    inputs[f"args_{i}"] = extract_param(
                        n
                    )  # TODO different representation?
            elif k == varkw:  # unpack varkw
                for kk, n in v.items():
                    inputs[kk] = extract_param(n)
            else:
                inputs[k] = extract_param(v)
        assert all(
            [isinstance(n, Node) for n in inputs.values()]
        ), "All values in inputs must be nodes."

        return inputs, args, kwargs, _args, _kwargs

    def _get_tracer(self):
        """Get a tracer to overwrite the python recursion behavior of calling the decorated function."""

        # Define a tracer to deal with recursive function calls
        _bundled_func = None

        def tracer(frame, event, arg=None):
            """This tracer modifies the local/global dict of the frame, so that
            when a recursive call of the wrapped function is made, it calls the
            unwrapped function."""
            nonlocal _bundled_func

            if frame.f_code == self._fun.__code__:  # entering the wrapped function
                # Use the original function, rather than the bundled function
                if event == "call":  # Detect potential recursive calls
                    if frame.f_code.co_name in frame.f_locals:
                        # # the function is not defined globally at the top level
                        current_fun = frame.f_locals[frame.f_code.co_name]
                        if current_fun != self._fun:
                            update_local(frame, frame.f_code.co_name, self._fun)
                    elif frame.f_code.co_name in frame.f_globals:
                        current_fun = frame.f_globals[frame.f_code.co_name]
                        if current_fun != self._fun:
                            assert isinstance(current_fun, FunModule)
                            _bundled_func = current_fun  # save the original function
                            frame.f_globals[frame.f_code.co_name] = self._fun

                elif event == "return":
                    if frame.f_code.co_name in frame.f_globals:
                        frame.f_globals[frame.f_code.co_name] = _bundled_func
            return tracer

        return tracer

    def _construct_error_comment(self, e):
        """Construct the error comment on the source code and traceback."""
        self.info["traceback"] = (
            traceback.format_exc()
        )  # This is saved for user debugging
        # Construct message to optimizer
        error_class = e.__class__.__name__
        detail = e.args[0]
        cl, exc, tb = sys.exc_info()
        assert tb is not None  # we're in the except block, so tb should not be None
        n_fun_calls = len(traceback.extract_tb(tb))
        # Step through the traceback stack
        comments = []
        base_message = f"({error_class}) {detail}."
        for i, (f, ln) in enumerate(traceback.walk_tb(tb)):
            if i > 0:  # ignore the first one, since that is the try statement above
                error_message = (
                    base_message
                    if i == n_fun_calls - 1
                    else "Error raised in function call. See below."
                )

                if (
                    i == 1 and self.parameter is not None
                ):  # this is the trainable function defined by exec, which needs special treatment. inspect.getsource doesn't work here.
                    comment = self.generate_comment(
                        self.parameter._data, error_message, ln, 1
                    )
                    comment_backup = self.generate_comment(
                        self.parameter._data, base_message, ln, 1
                    )
                else:
                    try:
                        f_source, f_source_ln = self.get_source(f, bug_mode=True)
                    except OSError:  # OSError: could not get source code
                        # we reach the compiled C level, so the previous level is actually the bottom
                        comments[-1] = comment_backup  # replace the previous comment
                        break  # exit the loop
                    comment = self.generate_comment(
                        f_source, error_message, ln, f_source_ln
                    )
                    comment_backup = self.generate_comment(
                        f_source, base_message, ln, f_source_ln
                    )
                comments.append(comment)
        commented_code = "\n\n".join(comments)
        self.info["error_comment"] = commented_code + f"\n{base_message}"
        output = e
        return output

    def sync_call_fun(self, fun, *_args, **_kwargs):
        """Call the operator fun and return the output. Catch the exception if catch_execution_error is True."""
        oldtracer = sys.gettrace()
        if (
            self.overwrite_python_recursion and self.parameter is None
        ):  # Overwrite the python recursion behavior
            # Running a tracer would slow down the execution, so we only do this when necessary.
            sys.settrace(self._get_tracer())

        if self.catch_execution_error:
            try:
                output = fun(*_args, **_kwargs)
            except Exception as e:
                output = self._construct_error_comment(e)
        else:
            output = fun(*_args, **_kwargs)

        sys.settrace(oldtracer)
        return output

    async def async_call_fun(self, fun, *_args, **_kwargs):
        oldtracer = sys.gettrace()
        if (
            self.overwrite_python_recursion and self.parameter is None
        ):  # Overwrite the python recursion behavior
            # Running a tracer would slow down the execution, so we only do this when necessary.
            sys.settrace(self._get_tracer())

        if self.catch_execution_error:
            try:
                output = await fun(*_args, **_kwargs)
            except Exception as e:
                output = self._construct_error_comment(e)
        else:
            output = await fun(*_args, **_kwargs)

        sys.settrace(oldtracer)
        return output

    def preprocess_inputs(self, args, kwargs, _args, _kwargs):
        # NOTE This function must be put inside the used_nodes context manager.
        """Preprocess the inputs for the operator fun.

        Args:
            _args (list): the original positional arguments. This includes the default values.
            _kwargs (dict): the original keyword arguments. This includes the default values.
            args (list): the wrapped positional arguments.
            kwargs (dict): the wrapped keyword arguments.
        """
        # Construct the inputs to call self.fun
        if self._process_inputs:  # This is for handling hierarchical graph
            if self.traceable_code:
                _args, _kwargs = detach_inputs(args), detach_inputs(kwargs)
            else:  # NOTE Extract data from the nodes and pass them to the function; This line must be put inside the used_nodes context manager.
                _args, _kwargs = to_data(args), to_data(
                    kwargs
                )  # read node.data; this ensures the inputs are treated as used nodes
        # else the inputs are passed directly to the function
        # so we don't change _args and _kwargs
        return _args, _kwargs  # this will be passed as the input to the function

    def postprocess_output(self, output, fun, _args, _kwargs, used_nodes, inputs):
        """
            Wrap the output as a MessageNode. Log the inputs and output of the function call.

        Args:
            output (Any): the output of the operator fun.
            fun (callable): the operator fun.
            _args (list): the original positional arguments. This includes the default values.
            _kwargs (dict): the original keyword arguments. This includes the default values.
            used_nodes (List[Node]): the nodes used in the operator fun.
            inputs (Dict[str, Node]): the inputs of the operator fun.
        """

        # Log inputs and output of the function call
        self.info["output"] = output
        self.info["inputs"]["args"] = _args
        self.info["inputs"]["kwargs"] = _kwargs

        # Nodes used to create the output but not in the inputs are external dependencies.
        external_dependencies = [
            node for node in used_nodes if not contain(inputs.values(), node)
        ]
        self.info["external_dependencies"] = external_dependencies

        # Make sure all nodes in used_nodes are in the parents of the returned node.
        if ALLOW_EXTERNAL_DEPENDENCIES is not None:  # This is a global overwrite
            allow_external_dependencies = ALLOW_EXTERNAL_DEPENDENCIES
        else:
            allow_external_dependencies = self.allow_external_dependencies

        if len(external_dependencies) > 0 and not allow_external_dependencies:
            raise TraceMissingInputsError(
                f"Not all nodes used in the operator {fun} are specified as inputs of the returned node. Missing {[(node.name, node.data) for node in external_dependencies]} "
            )

        if not GRAPH.TRACE:
            inputs = (
                {}
            )  # We don't need to keep track of the inputs if we are not tracing.
        # Wrap the output as a MessageNode or an ExceptionNode
        nodes = self.wrap(output, inputs, external_dependencies)
        return nodes

    def forward(self, *args, **kwargs):
        fun = self.fun  # Define the function (only once)
        self.info["fun"] = fun
        if inspect.iscoroutinefunction(fun):
            return self.async_forward(
                fun, *args, **kwargs
            )  # Return a coroutine that returns a MessageNode
        else:
            return self.sync_forward(fun, *args, **kwargs)  # Return a MessageNode

    def sync_forward(self, fun, *args, **kwargs):
        """
        Call the operator fun and return a MessageNode. All nodes used in
        the operator fun are added to used_nodes during the execution. If
        the output is not a Node, we wrap it as a MessageNode, whose inputs
        are nodes in used_nodes. Sync version.
        """
        # Wrap the inputs as nodes
        inputs, args, kwargs, _args, _kwargs = self._wrap_inputs(fun, args, kwargs)
        # Execute fun
        with trace_nodes() as used_nodes:
            # After exit, used_nodes contains the nodes whose data attribute is read in the operator fun.
            _args, _kwargs = self.preprocess_inputs(args, kwargs, _args, _kwargs)
            output = self.sync_call_fun(fun, *_args, **_kwargs)
        # Wrap the output as a MessageNode or an ExceptionNode
        nodes = self.postprocess_output(output, fun, _args, _kwargs, used_nodes, inputs)
        return nodes

    async def async_forward(self, fun, *args, **kwargs):
        """
        Call the operator fun and return a MessageNode. All nodes used in
        the operator fun are added to used_nodes during the execution. If
        the output is not a Node, we wrap it as a MessageNode, whose inputs
        are nodes in used_nodes. Async version.
        """
        # Wrap the inputs as nodes
        inputs, args, kwargs, _args, _kwargs = self._wrap_inputs(fun, args, kwargs)
        # Execute fun
        with trace_nodes() as used_nodes:
            # After exit, used_nodes contains the nodes whose data attribute is read in the operator fun.
            _args, _kwargs = self.preprocess_inputs(args, kwargs, _args, _kwargs)
            output = await self.async_call_fun(
                fun, *_args, **_kwargs
            )  # use await to call the async function
        # Wrap the output as a MessageNode or an ExceptionNode
        nodes = self.postprocess_output(output, fun, _args, _kwargs, used_nodes, inputs)
        return nodes

    def wrap(
        self,
        output: Any,
        inputs: Union[List[Node], Dict[str, Node]],
        external_dependencies: List[Node],
    ):
        """Wrap the output as a MessageNode of inputs as the parents."""
        # Some nodes are used in the operator fun, we need to wrap the output as a MessageNode.
        if self.parameter is not None:
            # This is a trainiable op. Create a new op eval.
            inputs.update({"__code": self.parameter})
            description = "[eval] This operator eval(__code, *args, **kwargs) evaluates the code block, where __code is the code (str) and *args and **kwargs are the arguments of the function. The output is the result of the evaluation, i.e., __code(*args, **kwargs)."
            name = "eval"
            self.info["fun_name"] = "eval"
        else:
            description = self.description
            name = self.name
        info = self.info.copy()
        if isinstance(output, Exception):
            e_node = ExceptionNode(
                output,
                inputs=inputs,
                description=f'[exception] The operator {self.info["fun_name"]} raises an exception.',
                name="exception_" + name,
                info=info,
            )
            raise ExecutionError(e_node)
        else:
            return MessageNode(
                output, description=description, inputs=inputs, name=name, info=info
            )

    @staticmethod
    def is_valid_output(output):
        return isinstance(output, Node) or (
            isinstance(output, tuple) and all([isinstance(o, Node) for o in output])
        )


    # Define __set_name__ and __get__ for FunModule to act as a descriptor.
    def __get__(self, obj, db_type):
        if obj is None:  # class method
            return self
        # Support instance methods.
        method_name = f'__TRACE_RESERVED_bundle_{self.name}'  # NOTE we assume these are secret names not taken
        obj_node_name = f'__TRACE_RESERVED_self_node'
        if not hasattr(obj, obj_node_name):
            setattr(obj, obj_node_name, node(obj))
        if not hasattr(obj, method_name):
            funmodule = copy.deepcopy(self)  # instance specific version
            funmodule.forward = functools.partial(funmodule.forward, getattr(obj, obj_node_name))
            setattr(obj, method_name, funmodule)
        fun = getattr(obj, method_name)
        assert fun is not self  # self is defined in the class level
        assert isinstance(fun, FunModule), f"Expected {method_name} to be a FunModule, but got {type(fun)}"
        # fun = functools.partial(self.__call__, obj)
        return fun

    def detach(self):
        return copy.deepcopy(self)

    def generate_comment(
        self,
        code: str,
        comment: str,
        comment_line_number: int,
        base_line_number: int = 0,
    ):
        commented_code = []
        for i, l in enumerate(code.split("\n")):
            if i == comment_line_number - base_line_number:
                commented_code.append(f"{l} <--- {comment}")
            else:
                commented_code.append(f"{l}")
        commented_code = "\n".join(commented_code)
        return commented_code

    def get_source(self, obj: Any, bug_mode=False):
        """Get the source code of the function and its line number, excluding the @bundle decorator line.
        bug_mode=True means
        We are in the forward() function, but there is an error during execution.
        The error can be caused by a lambda function which does not have `def` in the source code.
        We turn off the error raising in the end of this function.

        Allowable two types of usages:

        Examples:

        >>>    @blah
        >>>    ...
        >>>    @bundle    # or  @ ....bundle()
        >>>    def fun(...): # ...
        >>>        ....
            or inline usage
        >>>    bundle()(fun)  # or ....bundle()(fun)
        """
        source = inspect.getsource(
            obj
        )  # the source includes @bundle, or @trace.bundle, etc. we will remove those parts.
        line_number = int(inspect.getsourcelines(obj)[1])  # line number of obj

        # Check if it's a decorator or an inline usage.
        decorator_usage = False
        lines = source.split("\n")
        for i, line in enumerate(lines):
            line = line.strip().split("#")[0]  # remove spacing and comment
            if line == "":
                continue
            if line[0] == "@":  # decorator line. check whether it's using bundle
                # use cases
                # @bundle(
                # @bundle\   i.e., change line
                # @......bundle(
                # @......bundle\
                if (
                    ("@bundle(" in line)
                    or ("@bundle\\" in line)
                    or (re.search(r"@.*\.bundle\(.*", line) is not None)
                    or (re.search(r"@.*\.bundle\\.*", line) is not None)
                ):
                    decorator_usage = True
                    break  # i is the where the bundle decorator is used

        if decorator_usage:
            line_offset = i  # to account for @bundle is not the top decorator

            # Extract the lines after @bundle(....)
            inner_source = "\n".join(lines[i:])  # i is where @bundle is used
            assert "def " in inner_source
            # str after the first bundle
            after_bundle = "bundle".join(
                inner_source.split("bundle")[1:]
            )  # NOTE there may be multiple usages of bundle in the comments

            # Find where the scope of brackets
            count = 0
            for i, t in enumerate(after_bundle):
                if t == "(":
                    count += 1
                elif t == ")":
                    count -= 1
                if count == 0:
                    break
            # Get the decorated source code
            after_bundle_call = after_bundle[i + 1:]  # after bundle(....)
            extracted_source = "\n".join(
                after_bundle_call.split("\n")[1:]
            )  # remove the first \n
            extracted_source = extracted_source.strip()
            # Get the line number of the decorated source code
            within_bundle_call = after_bundle[: i + 1]
            n_line_changes = (
                line_offset + 1 + within_bundle_call.count("\n")
            )  # the latter is the lines within the bundle call
            line_number += n_line_changes
        else:
            # The inline usecase of
            # fun = @bundle(...)fun(...)
            #   ...
            extracted_source = inspect.getsource(obj).strip()

        if not bug_mode:
            assert (
                "def" in extracted_source
            ), f"def is not in the source code: {extracted_source}"

        return extracted_source, line_number


def to_data(obj):
    """Extract data from nodes in nested structures.

    Parameters
    ----------
    obj : Any
        A node, container of nodes, or nested structure containing nodes.

    Returns
    -------
    Any
        The same structure with Node objects replaced by their data values.

    Notes
    -----
    Recursively traverses nested structures (lists, dicts, tuples, sets,
    NodeContainers) and extracts the data from all Node objects while
    preserving the structure. Non-Node objects are returned unchanged.

    Examples
    --------
    >>> x = node(5)
    >>> y = node(10)
    >>> to_data([x, y, 15])  # Returns [5, 10, 15]
    """
    return recursive_conversion(lambda x: x.data, lambda x: x)(obj)


def wrap_node(obj):
    """Wrap non-Node objects in nodes throughout nested structures.

    Parameters
    ----------
    obj : Any
        Any object or nested structure potentially containing non-Node values.

    Returns
    -------
    Any
        The same structure with non-Node objects wrapped in Node instances.

    Notes
    -----
    Recursively traverses nested structures and wraps any non-Node objects
    in Node instances. Existing Node objects are left unchanged. Useful for
    ensuring all values in a structure are nodes before processing.
    """
    return recursive_conversion(lambda x: x, lambda x: node(x))(obj)


def detach_inputs(obj):
    """Detach nodes from the computation graph in nested structures.

    Parameters
    ----------
    obj : Any
        A node, container of nodes, or nested structure containing nodes.

    Returns
    -------
    Any
        The same structure with all nodes detached from the graph.

    Notes
    -----
    Recursively traverses nested structures and calls detach() on all
    Node objects, removing them from gradient computation. Non-Node
    objects are returned unchanged. Useful for creating non-differentiable
    copies of values.
    """
    return recursive_conversion(lambda x: x.detach(), lambda x: x)(obj)


def update_local(frame, name, value):
    """Update a local variable in a Python frame.

    Parameters
    ----------
    frame : frame
        The Python frame object to modify.
    name : str
        Name of the local variable to update.
    value : Any
        New value for the variable.

    Notes
    -----
    This low-level function modifies Python frame locals directly using
    ctypes. It's used internally for trainable code execution where
    local variables need to be dynamically updated. The PyFrame_LocalsToFast
    call ensures the change is reflected in the actual frame.

    Warning: This is an advanced internal function that modifies Python
    internals and should be used with caution.
    """
    frame.f_locals[name] = value
    ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(frame), ctypes.c_int(0))


if __name__ == "__main__":
    x = node("hello")

    @bundle("[Custom] This is a test function.")
    def test(x):
        return x.data + " world"

    y = test(x)
    print(y)
    print("Parents", y.parents)
    print("Children", y.children)
    print("Level", y._level)
