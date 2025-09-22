import pydantic
from pydantic import BaseModel, ValidationError, Field, create_model
import opto.trace as trace
from typing import Union, get_type_hints, Any, Dict, List
from opto.utils.llm import AbstractModel, LLM
from opto.flows.types import TracedInput, TracedOutput, DynamicModelMixin
from opto.optimizers.utils import extract_xml_like_data
import inspect
import json
import re


"""
TracedLLM:
1. special operations that supports specifying inputs (system_prompt, user_prompt) to LLM and parsing of outputs, wrap
   everything under one command.
2. Easy to use interface -- can be inherited by users.
3. Support multi-turn chatting (message history)

Usage patterns:

Direct use: (only supports single input, single output) (signature: str -> str)
llm = TracedLLM("You are a helpful assistant.")
response = llm("Hello, what's the weather in France today?")
"""

@trace.bundle(catch_execution_error=False)
def call_llm(llm, system_prompt: str, *user_prompts: List[str], **kwargs) -> str:
    """Call the LLM model.

    Args:
        llm: The language model to use for generating responses.
        system_prompt: the system prompt to the agent. By tuning this prompt, we can control the behavior of the agent. For example, it can be used to provide instructions to the agent (such as how to reason about the problem, how to use tools, how to answer the question), or provide in-context examples of how to solve the problem.
        user_prompt: the input to the agent. It can be a query, a task, a code, etc.
    Returns:
        The response from the agent.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    for user_prompt in user_prompts:
        messages.append({"role": "user", "content": user_prompt})
    # TODO auto-parsing results
    response = llm(messages=messages, **kwargs)
    return response.choices[0].message.content

@trace.model
class TracedLLM(DynamicModelMixin):
    def __init__(self,
                 system_prompt: Union[str, None, trace.Node] = None,
                 llm: AbstractModel = None, chat_history_on=False):
        """Initialize TracedLLM with a system prompt.

        Args:
            system_prompt: The system prompt to use for LLM calls. If None and the class has a docstring, the docstring will be used.
            llm: The LLM model to use for inference
            chat_history_on: if on, maintain chat history for multi-turn conversations
        """
        # Use class docstring as system prompt if none provided
        if system_prompt is None:
            class_docstring = self.__class__.__doc__
            if class_docstring and class_docstring.strip():
                system_prompt = class_docstring.strip()
        
        self.system_prompt = trace.node(system_prompt)
        if llm is None:
            llm = LLM()
        assert isinstance(llm, AbstractModel), f"{llm} must be an instance of AbstractModel"
        self.llm = llm
        
        # Initialize fields for structured input/output
        self._input_fields = []
        self._output_fields = []
        self._field_types = {}  # Store type annotations for each field

    def forward(self, *args, **kwargs) -> str:
        """Main function that handles both direct call and inheritance patterns.
        
        Args:
            *args: For direct pattern - single string argument
            **kwargs: For inheritance pattern - named input fields
            
        Returns:
            str: For direct pattern
            TracedResponse: For inheritance pattern with structured output fields
        """
        # Direct pattern: single string argument
        if len(args) == 1 and isinstance(args[0], str):
            return self._call_llm(args[0])
        else:
            raise ValueError("Direct usage requires a single string argument")
    
    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM with user prompt and system prompt."""
        return call_llm(self.llm, self.system_prompt, user_prompt)
