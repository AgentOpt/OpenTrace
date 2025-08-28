import pydantic
import opto.trace as trace
from typing import Union
from opto.utils.llm import AbstractModel, LLM

"""
TracedLLM:
1. special operations that supports specifying inputs (system_prompt, user_prompt) to LLM and parsing of outputs, wrap
   everything under one command.
2. Easy to use interface -- can be inherited by users.

Usage patterns:

1. Direct use: (only supports single input, single output) (signature: str -> str)
llm = TracedLLM("You are a helpful assistant.")
llm("Hello, what's the weather in France today?")

2. Inheritance:
class Scorer(TracedLLM):
   "This is a class that scores the response from LLM"
   doc: opto.flows.types.TracedInput
   score: opto.flows.types.TracedOutput

scorer = Scorer("You are a helpful assistant that scores the response from LLM")
scorer(doc="The response is ...")
"""

@trace.model
class TracedLLM:
    def __init__(self,
                 system_prompt: Union[str, None, trace.Node] = None,
                 llm: AbstractModel = None):
        """Initialize TracedLLM with a system prompt.

        Args:
            system_prompt: The system prompt to use for LLM calls
            llm: The LLM model to use for inference
        """
        self.system_prompt = trace.node(system_prompt)
        if llm is None:
            llm = LLM()
        assert isinstance(llm, AbstractModel), f"{llm} must be an instance of AbstractModel"
        self.llm = llm

    def forward(self, user_prompt: str) -> str:
        """Call the LLM with user prompt, using the configured system prompt."""
        return trace.operators.call_llm(self.llm, self.system_prompt, user_prompt)


if __name__ == '__main__':
    pass
