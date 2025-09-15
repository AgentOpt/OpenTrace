import datasets
import numpy as np
from opto import trace, trainer
from opto.utils.llm import LLM, LiteLLM

from typing import Any


def call_llm(llm, system_prompt: str, user_prompt_template: str, message: str) -> str:
    if '{message}' not in user_prompt_template:
            raise ValueError("user_prompt_template must contain '{message}'")
    response = llm(
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt_template.format(message=message)}]
    )
    return response.choices[0].message.content


@trace.model
class Learner:
    """ A basic LLM agent. """

    def __init__(self, system_prompt: str = "You're a helpful agent",
                 user_prompt_template: str = "Query: {message}",
                 llm: LLM = None):
        self.system_prompt = trace.node(system_prompt, trainable=True)
        self.user_prompt_template = trace.node(user_prompt_template)
        self.llm = llm or LLM()

    @trace.bundle()
    def model(self, system_prompt: str, user_prompt_template: str, message: str) -> str:
        """Call the LLM model.

        Args:
            system_prompt: the system prompt to the agent. By tuning this prompt, we can control the behavior of the agent. For example, it can be used to provide instructions to the agent (such as how to reason about the problem, how to answer the question), or provide in-context examples of how to solve the problem.
            user_prompt_template: the user prompt template to the agent. It is used as formatting the input to the agent as user_prompt_template.format(message=message).
            message: the input to the agent. It can be a query, a task, a code, etc.
        Returns:
            The response from the agent.
        """
        return call_llm(self.llm, system_prompt, user_prompt_template, message)

    def forward(self, message: Any) -> Any:
        """ Forward pass of the agent. """
        return self.model(self.system_prompt, self.user_prompt_template, message)



def main():
    # set seed
    seed = 42
    num_epochs = 1
    batch_size = 3  # number of queries to sample from the training data
    test_frequency = -1

    num_threads = 10
    datasize = 5

    np.random.seed(seed)

    # In this example, we use the GSM8K dataset, which is a dataset of math word problems.
    # We will look the training error of the agent on a small portion of this dataset.
    train_dataset = datasets.load_dataset('BBEH/bbeh')['train'][:datasize]
    train_dataset = dict(inputs=train_dataset['input'], infos=train_dataset['target'])

    agent = Learner(llm=LLM())

    trainer.train(
        model=agent,
        train_dataset=train_dataset,
        # trainer kwargs
        num_epochs=num_epochs,
        batch_size=batch_size,
        test_frequency=test_frequency,
        num_threads=num_threads,
        verbose='output',
    )


if __name__ == "__main__":
    main()
