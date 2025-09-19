import datasets
import numpy as np
from opto import trainer
from opto.utils.llm import LLM
from opto.features.predefined_agents import BasicLearner



def main():
    # set seed
    seed = 42
    num_epochs = 1
    batch_size = 3  # number of queries to sample from the training data
    eval_frequency = -1

    num_threads = 10
    datasize = 5

    np.random.seed(seed)

    # In this example, we use the GSM8K dataset, which is a dataset of math word problems.
    # We will look the training error of the agent on a small portion of this dataset.
    train_dataset = datasets.load_dataset('BBEH/bbeh')['train'][:datasize]
    train_dataset = dict(inputs=train_dataset['input'], infos=train_dataset['target'])

    agent = BasicLearner(llm=LLM())

    trainer.train(
        model=agent,
        train_dataset=train_dataset,
        # trainer kwargs
        num_epochs=num_epochs,
        batch_size=batch_size,
        eval_frequency=eval_frequency,
        num_threads=num_threads,
        verbose='output',
    )


if __name__ == "__main__":
    main()
