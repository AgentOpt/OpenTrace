import pydantic
from pydantic import BaseModel, ValidationError, Field, create_model
import opto.trace as trace
from typing import Union, get_type_hints, Any, Dict, List, Optional
from opto.utils.llm import AbstractModel, LLM
from opto.features.flows.types import TraceObject
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


class ChatHistory(TraceObject):
    def __init__(self, max_len=10, auto_summary=False):
        """Initialize chat history for multi-turn conversation.
        
        Args:
            max_len: Maximum number of messages to keep in history
            auto_summary: Whether to automatically summarize old messages
        """
        self.messages = []
        self.max_len = max_len
        self.auto_summary = auto_summary

    def add(self, content, role):
        """Add a message to history with role validation.
        
        Args:
            content: The content of the message
            role: The role of the message ("user" or "assistant")
        """
        if role not in ["user", "assistant"]:
            raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")

        # Check for alternating user/assistant pattern
        if len(self.messages) > 0:
            last_msg = self.messages[-1]
            if last_msg["role"] == role:
                print(f"Warning: Adding consecutive {role} messages. Consider alternating user/assistant messages.")

        self.messages.append({"role": role, "content": content})
        self._trim_history()

    def _trim_history(self):
        """Trim history to max_len while preserving first user message."""
        if len(self.messages) <= self.max_len:
            return

        # Find first user message index
        first_user_idx = None
        for i, msg in enumerate(self.messages):
            if msg["role"] == "user":
                first_user_idx = i
                break

        # Keep first user message
        protected_messages = []
        if first_user_idx is not None:
            first_user_msg = self.messages[first_user_idx]
            protected_messages.append(first_user_msg)

        # Calculate how many recent messages we can keep
        remaining_slots = self.max_len - len(protected_messages)
        if remaining_slots > 0:
            # Get recent messages
            recent_messages = self.messages[-remaining_slots:]
            # Avoid duplicating first user message
            if first_user_idx is not None:
                first_user_msg = self.messages[first_user_idx]
                recent_messages = [msg for msg in recent_messages if msg != first_user_msg]

            self.messages = protected_messages + recent_messages
        else:
            self.messages = protected_messages

    def get_messages(self, system_prompt: Optional[Union[str, trace.Node]] = None):
        """Get messages from history.

        Args:
            system_prompt: If this is passed in, then we construct a node/graph that
                           builds system_prompt -> chat_history graph
        
        Returns:
            List of messages
        """

        @trace.bundle()
        def converse_with_llm(system_prompt: Union[str, trace.Node]):
            """The conversation history with the LLM using the given system prompt.
            Args:
                system_prompt: The system prompt to use for the conversation.
            Returns:
                The conversation history from the LLM.
            """
            return self

        if system_prompt is None:
            return self.messages.copy()
        else:
            return converse_with_llm(system_prompt)

    def __str__(self):
        """String representation of the chat history. Mostly for the optimizer."""
        if len(self.messages) == 0:
            return "There is no chat history so far."

        lines = [">>ChatHistory<<"]

        for msg in self.messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")

        lines.append(">>End<<")
        return "\n".join(lines)


@trace.bundle(catch_execution_error=False)
def call_llm(llm, system_prompt: str, user_prompt: str, chat_history: Optional[ChatHistory] = None, **kwargs) -> str:
    """Call the LLM model.

    Args:
        llm: The language model to use for generating responses.
        system_prompt: the system prompt to the agent. By tuning this prompt, we can control the behavior of the agent. For example, it can be used to provide instructions to the agent (such as how to reason about the problem, how to use tools, how to answer the question), or provide in-context examples of how to solve the problem.
        user_prompt: the input to the agent. It can be a query, a task, a code, etc.
        chat_history: The conversation between the user and LLM so far. Can be empty.
    Returns:
        The response from the agent.
    """
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(chat_history.get_messages())
    messages.append({"role": "user", "content": user_prompt})

    # TODO auto-parsing results
    response = llm(messages=messages, **kwargs)
    return response.choices[0].message.content


DEFAULT_SYSTEM_PROMPT_DESCRIPTION = ("the system prompt to the agent. By tuning this prompt, we can control the "
                                     "behavior of the agent. For example, it can be used to provide instructions to "
                                     "the agent (such as how to reason about the problem, how to use tools, "
                                     "how to answer the question), or provide in-context examples of how to solve the "
                                     "problem.")


@trace.model
class TracedLLM:
    def __init__(self,
                 system_prompt: Union[str, None, trace.Node] = None,
                 llm: AbstractModel = None, chat_history_on=False,
                 trainable=False):
        """Initialize TracedLLM with a system prompt.

        Args:
            system_prompt: The system prompt to use for LLM calls. If None and the class has a docstring, the docstring will be used.
            llm: The LLM model to use for inference
            chat_history_on: if on, maintain chat history for multi-turn conversations
        """
        if system_prompt is None:
            system_prompt = "You are a helpful assistant."

        self.system_prompt = system_prompt if isinstance(system_prompt, trace.Node) else trace.node(system_prompt,
                                                                                                    name='system_prompt',
                                                                                                    description=DEFAULT_SYSTEM_PROMPT_DESCRIPTION,
                                                                                                    trainable=trainable)
        if llm is None:
            llm = LLM()
        assert isinstance(llm, AbstractModel), f"{llm} must be an instance of AbstractModel"
        self.llm = llm
        self.chat_history = ChatHistory()

    def forward(self, user_query: str, **kwargs) -> str:
        """Main function that handles both direct call and inheritance patterns.
        
        Args:
            *args: For direct pattern - single string argument
            **kwargs: For inheritance pattern - named input fields
            
        Returns:
            str: For direct pattern
            TracedResponse: For inheritance pattern with structured output fields
        """
        messages = []
        messages.append({"role": "system", "content": self.system_prompt.data})
        messages.extend(self.chat_history.get_messages())
        messages.append({"role": "user", "content": user_query})

        response = self.llm(messages=messages, **kwargs)

        @trace.bundle()
        def call_llm(chat_history: ChatHistory, user_query: str) -> str:
            """Call the LLM model.
            Args:
                user_query
            Returns:
                response from the LLM
            """
            return response.choices[0].message.content

        return call_llm(self.chat_history.get_messages(self.system_prompt), user_query)