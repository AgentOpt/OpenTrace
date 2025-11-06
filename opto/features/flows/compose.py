import opto.trace as trace
from typing import Union, get_type_hints, Any, Dict, List, Optional
from opto.utils.llm import AbstractModel, LLM
import contextvars

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

USED_TracedLLM = contextvars.ContextVar('USED_TracedLLM', default=list())


class ChatHistory:
    def __init__(self, max_len=50, auto_summary=False):
        """Initialize chat history for multi-turn conversation.

        Args:
            max_len: Maximum number of messages to keep in history
            auto_summary: Whether to automatically summarize old messages
        """
        self.messages: List[Dict[str, Any]] = []
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

    def append(self, message: Dict[str, Any]):
        """Append a message directly to history."""
        if "role" not in message or "content" not in message:
            raise ValueError("Message must have 'role' and 'content' fields.")
        self.add(message["content"], message["role"])

    def __iter__(self):
        return iter(self.messages)

    def get_messages(self) -> List[Dict[str, str]]:
        messages = []
        for message in self.messages:
            if isinstance(message['content'], trace.Node):
                messages.append({"role": message["role"], "content": message["content"].data})
            else:
                messages.append(message)
        return messages

    def get_messages_as_node(self, llm_name="") -> List[trace.Node]:
        node_list = []
        for message in self.messages:
            # If user query is a node and has other computation attached, we can't rename it
            if isinstance(message['content'], trace.Node):
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
    """
    This high-level model provides an easy-to-use interface for LLM calls with system prompts and optional chat history.

    Python usage patterns:

        llm = UF_LLM(system_prompt)
        response = llm.chat(user_prompt)
        response_2 = llm.chat(user_prompt_2)

    The underlying Trace Graph:
        TracedLLM_response0 = TracedLLM.forward.call_llm(args_0=system_prompt0, args_1=TracedLLM0_user_query0)
        TracedLLM_response1 = TracedLLM.forward.call_llm(args_0=system_prompt0, args_1=TracedLLM0_user_query0, args_2=TracedLLM_response0, args_3=TracedLLM0_user_query1)
        TracedLLM_response2 = TracedLLM.forward.call_llm(args_0=system_prompt0, args_1=TracedLLM0_user_query0, args_2=TracedLLM_response0, args_3=TracedLLM0_user_query1, args_4=TracedLLM_response1, args_5=TracedLLM0_user_query2)
    """

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

        self.system_prompt = trace.node(system_prompt, name='system_prompt',
                                        description=DEFAULT_SYSTEM_PROMPT_DESCRIPTION,
                                        trainable=trainable)
        # if system_prompt is already a node, then we have to override its trainable attribute
        self.system_prompt.trainable = trainable

        if llm is None:
            llm = LLM()
        assert isinstance(llm, AbstractModel), f"{llm} must be an instance of AbstractModel"
        self.llm = llm
        self.chat_history = ChatHistory()
        self.chat_history_on = chat_history_on

        current_llm_sessions = USED_TracedLLM.get()
        self.model_name = model_name if model_name else f"{self.__class__.__name__}{len(current_llm_sessions)}"
        current_llm_sessions.append(1)  # just a marker

    def forward(self, user_query: str, chat_history_on: Optional[bool] = None) -> str:
        """This function takes user_query as input, and returns the response from the LLM, with the system prompt prepended.
        This method will always save chat history.

        If chat_history_on is set to False, the chat history will not be included in the LLM input.
        If chat_history_on is None, it will use the class-level chat_history_on setting.
        If chat_history_on is True, the chat history will be included in the LLM input.

        Args:
            user_query: The user query to send to the LLM

        Returns:
            str: For direct pattern
        """
        chat_history_on = self.chat_history_on if chat_history_on is None else chat_history_on

        messages = [{"role": "system", "content": self.system_prompt.data}]
        if chat_history_on:
            messages.extend(self.chat_history.get_messages())
        messages.append({"role": "user", "content": user_query})

        response = self.llm(messages=messages)

        @trace.bundle(output_name=f"{self.model_name}_response")
        def call_llm(*messages) -> str:
            """Call the LLM model.
            Args:
                messages: All the conversation history so far, starting from system prompt, to alternating user/assistant messages, ending with the current user query.
            Returns:
                response from the LLM
            """
            return response.choices[0].message.content

        user_query_node = trace.node(user_query, name=f"{self.model_name}_user_query")
        arg_list = [self.system_prompt]
        if chat_history_on:
            arg_list += self.chat_history.get_messages_as_node(self.model_name)
        arg_list += [user_query_node]

        response_node = call_llm(*arg_list)

        # save to chat history
        self.chat_history.add(user_query_node, role="user")
        self.chat_history.add(response_node, role="assistant")

        return response_node

    def chat(self, user_query: str) -> str:
        return self.forward(user_query)
