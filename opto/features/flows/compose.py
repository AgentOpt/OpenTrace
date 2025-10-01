import opto.trace as trace
from typing import Union, get_type_hints, Any, Dict, List, Optional
from opto.utils.llm import AbstractModel, LLM
import random

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


class ChatHistory:
    def __init__(self, max_len=50, auto_summary=False):
        """Initialize chat history for multi-turn conversation.

        Args:
            max_len: Maximum number of messages to keep in history
            auto_summary: Whether to automatically summarize old messages
        """
        self.messages: List[Dict[str, Union[str, trace.Node]]] = []
        self.max_len = max_len
        self.auto_summary = auto_summary

    def __len__(self):
        return len(self.messages)

    def add(self, content: Union[trace.Node, str], role):
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

    def append(self, message: Dict[str, Union[str, trace.Node]]):
        """Append a message directly to history."""
        if "role" not in message or "content" not in message:
            raise ValueError("Message must have 'role' and 'content' fields.")
        self.add(message["content"], message["role"])

    def __iter__(self):
        return iter(self.messages)

    def get_messages(self) -> List[Dict[str, str]]:
        messages = []
        for message in self.messages:
            if type(message['content']) is trace.Node:
                messages.append({"role": message["role"], "content": message["content"].data})
            else:
                messages.append(message)
        return messages

    def get_messages_as_node(self, llm_name="") -> List[trace.Node]:
        node_list = []
        for message in self.messages:
            # issue: if user query is a node and has other computation attached, we can't rename it :(
            if type(message['content']) is trace.Node:
                node_list.append(message['content'])
            else:
                role = message["role"]
                content = message["content"]
                name = f"{llm_name}_{role}" if llm_name else f"{role}"
                if role == 'user':
                    name += "_query"
                elif role == 'assistant':
                    name += "_response"
                node_list.append(trace.node(content, name=name))

        return node_list

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
                 trainable=False, model_name=None):
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
        self.chat_history_on = chat_history_on
        self.model_name = model_name if model_name else f"TracedLLM{random.randint(1, 9999)}"

    def forward(self, user_query: str) -> str:
        """We build the TraceGraph in two ways.

        If there is no chat history, then the graph would look like:

        llm = UF_LLM(system_prompt)
        response = llm.chat(user_prompt)

        If there is chat history, the graph would look like:

        llm = UF_LLM(system_prompt)
        response = llm.chat(user_prompt)
        response_2 = llm.chat(user_prompt_2)

        Args:
            *args: For direct pattern - single string argument
            **kwargs: For inheritance pattern - named input fields

        Returns:
            str: For direct pattern
        """
        messages = [{"role": "system", "content": self.system_prompt.data}]
        messages.extend(self.chat_history.get_messages())
        messages.append({"role": "user", "content": user_query})

        response = self.llm(messages=messages)

        @trace.bundle()
        def call_llm(*args) -> str:
            """Call the LLM model.
            Args:
                All the conversation history so far, starting from system prompt, to alternating user/assistant messages, ending with the current user query.

            Returns:
                response from the LLM
            """
            return response.choices[0].message.content

        user_query_node = trace.node(user_query, name=f"{self.model_name}_user_query")
        arg_list = ([self.system_prompt] + self.chat_history.get_messages_as_node(self.model_name)
                    + [user_query_node])

        # save to chat history
        if self.chat_history_on:
            self.chat_history.add(user_query_node, role="user")
            response_node = trace.node(response.choices[0].message.content,
                                       name=f"{self.model_name}_assistant_response")

            self.chat_history.add(response_node, role="assistant")

        return call_llm(*arg_list)

    def chat(self, user_query: str) -> str:
        return self.forward(user_query)