from typing import Union
from opto.trace.bundle import bundle, ExecutionError
from opto.trace.modules import Module, model
from opto.trace.containers import NodeContainer
from opto.trace.broadcast import apply_op
import opto.trace.propagators as propagators
import opto.trace.operators as operators
import opto.trace.projections as projections

from opto.trace.nodes import Node, GRAPH
from opto.trace.nodes import node
from opto.utils.llm import AbstractModel


class stop_tracing:
    """A contextmanager to disable tracing."""

    def __enter__(self):
        GRAPH.TRACE = False

    def __exit__(self, type, value, traceback):
        GRAPH.TRACE = True


# TODO defined it somewhere else?
@model
class trace_llm:
    """ This is callable class of accessing LLM as a trace operator. """

    def __init__(self,
                 system_prompt: Union[str, None, Node] = None,
                 llm: AbstractModel = None,):
        self.system_prompt = node(system_prompt)
        if llm is None:
            from opto.utils.llm import LLM
            llm = LLM()
        assert isinstance(llm, AbstractModel), f"{llm} must be an instance of AbstractModel"
        self.llm = llm

    def forward(self, user_prompt):
        return operators.call_llm(self.llm, self.system_prompt, user_prompt)


__all__ = [
    "node",
    "stop_tracing",
    "GRAPH",
    "Node",
    "bundle",
    "ExecutionError",
    "Module",
    "NodeContainer",
    "model",
    "apply_op",
    "propagators",
    "utils"
]
