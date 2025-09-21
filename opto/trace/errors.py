from opto.trace.nodes import ExceptionNode


class ExecutionError(Exception):
    """Exception raised when traced code execution fails.

    Wraps an ExceptionNode to preserve error information in the computation
    graph while still raising a Python exception.

    Parameters
    ----------
    exception_node : ExceptionNode
        The ExceptionNode containing error details from the failed operation.

    Attributes
    ----------
    exception_node : ExceptionNode
        The wrapped exception node with full error context.

    Notes
    -----
    ExecutionError enables error-aware optimization by:
    1. Preserving error information in the computation graph
    2. Providing full traceback for debugging
    3. Allowing optimizers to learn from execution failures

    The string representation shows the full traceback from the original
    error, making debugging easier.

    See Also
    --------
    ExceptionNode : Node type that captures exceptions in the graph
    bundle : Decorator that can catch and wrap ExecutionErrors
    """

    def __init__(self, exception_node: ExceptionNode):
        self.exception_node = exception_node
        super().__init__(self.exception_node.data)

    def __str__(self):
        return "\n\n" + self.exception_node.info["traceback"]  # show full traceback


class TraceMissingInputsError(Exception):
    """Exception raised when required inputs are missing during tracing.

    This error occurs when a traced operation cannot find all necessary
    input nodes in the computation graph.

    Parameters
    ----------
    message : str
        Description of which inputs are missing.

    Attributes
    ----------
    message : str
        The error message describing missing inputs.

    Notes
    -----
    This exception typically indicates:
    1. A node was used before being defined
    2. External dependencies are used without allow_external_dependencies=True
    3. Input processing failed to extract required nodes

    The error helps identify graph construction issues early in the
    execution process.
    """
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return self.message  # f"TraceMissingInputsError: {self.message}"
