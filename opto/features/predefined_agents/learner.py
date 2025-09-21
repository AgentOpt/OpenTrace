"""Common learner utilities for examples and documentation."""

from typing import Any
from opto import trace
from opto.utils.llm import LLM


def call_llm(llm, system_prompt: str, user_prompt_template: str, message: str) -> str:
    """Call LLM with system and user prompts.
    
    Args:
        llm: The LLM instance to use
        system_prompt: The system prompt for the LLM
        user_prompt_template: Template for the user prompt (must contain {message})
        message: The input message to format into the template
        
    Returns:
        The LLM response content
        
    Raises:
        ValueError: If user_prompt_template doesn't contain {message}
    """
    if '{message}' not in user_prompt_template:
        raise ValueError("user_prompt_template must contain '{message}'")
    
    response = llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt_template.format(message=message)}
        ]
    )
    return response.choices[0].message.content


@trace.model
class BasicLearner:
    """A reusable basic LLM agent for examples and tutorials.
    
    This class provides a standard implementation of an LLM-based learner
    that can be used across multiple examples and documentation notebooks.
    """
    
    def __init__(
        self, 
        system_prompt: str = "You're a helpful agent",
        user_prompt_template: str = "Query: {message}",
        llm: LLM = None
    ):
        """Initialize the learner.
        
        Args:
            system_prompt: The system prompt to guide LLM behavior
            user_prompt_template: Template for formatting user messages (must contain {message})
            llm: LLM instance to use (defaults to LLM())
        """
        self.system_prompt = trace.node(system_prompt, trainable=True)
        self.user_prompt_template = trace.node(user_prompt_template)
        self.llm = llm or LLM()
    
    @trace.bundle()
    def model(self, system_prompt: str, user_prompt_template: str, message: str) -> str:
        """Call the LLM model.
        
        Args:
            system_prompt: The system prompt to the agent. By tuning this prompt,
                we can control the behavior of the agent. For example, it can be used
                to provide instructions to the agent (such as how to reason about the
                problem, how to answer the question), or provide in-context examples
                of how to solve the problem.
            user_prompt_template: The user prompt template to the agent. It is used
                as formatting the input to the agent as user_prompt_template.format(message=message).
            message: The input to the agent. It can be a query, a task, a code, etc.
            
        Returns:
            The response from the agent.
        """
        return call_llm(self.llm, system_prompt, user_prompt_template, message)
    
    def forward(self, message: Any) -> Any:
        """Forward pass of the agent."""
        return self.model(self.system_prompt, self.user_prompt_template, message)


# Alias for backward compatibility
Learner = BasicLearner