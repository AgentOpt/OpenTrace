import datasets
import numpy as np
from opto import trainer
from opto.utils.llm import LLM
from opto.features.predefined_agents import BasicLearner


def main():
    # set seed
    seed = 42
    num_epochs = 1
    batch_size = 1
    test_frequency = -1
    num_threads = 3

    np.random.seed(seed)

    # In this example, we use the GSM8K dataset, which is a dataset of math word problems.
    # We will look the training error of the agent on a small portion of this dataset.
    train_dataset = datasets.load_dataset('openai/gsm8k', 'main')['train'][:10]
    train_dataset = dict(inputs=train_dataset['question'], infos=train_dataset['answer'])

    agent = BasicLearner(llm=LLM())

    trainer.train(
        model=agent,
        train_dataset=train_dataset,
        # trainer kwargs
        num_epochs=num_epochs,
        batch_size=batch_size,
        test_frequency=test_frequency,
        test_dataset=train_dataset,
        num_threads=num_threads,
        verbose='output',
    )


if __name__ == "__main__":
    main()
