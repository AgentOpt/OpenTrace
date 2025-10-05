import datasets
import numpy as np
from opto import trace
from opto.utils.llm import LLM
from opto.features.predefined_agents import BasicLearner
from opto.optimizers import OptoPrimeV2 as OptoPrime
from opto.features.priority_search import PrioritySearch as SearchAlgorithm
from opto.trainer.loggers import TensorboardLogger
from opto.trainer.guide import LLMJudge


def main():
    # set seed
    seed = 42
    num_epochs = 1
    batch_size = 3  # number of queries to sample from the training data
    num_batches = 2  # number of queries each optimizer sees
    num_proposals = 3  # number of proposals to generate for each query
    num_candidates = 2  # number of candidates for exploration
    score_range = (0, 1)  # range of the score for the guide
    test_frequency = -1
    num_test_samples = 2
    score_function = 'mean'

    num_threads = 10
    datasize = 5
    verbose = True

    np.random.seed(seed)

    # In this example, we use the BBEH dataset
    train_dataset = datasets.load_dataset('BBEH/bbeh')['train'][:datasize]
    train_dataset = dict(inputs=train_dataset['input'], infos=train_dataset['target'])

    agent = BasicLearner(llm=LLM())
    guide = LLMJudge(llm=LLM())
    optimizer = OptoPrime(agent.parameters(), llm=LLM())
    logger = TensorboardLogger(verbose=verbose)

    alg = SearchAlgorithm(
        agent=agent,
        optimizer=optimizer,
        logger=logger
    )

    alg.train(guide,
              train_dataset,
              num_epochs=num_epochs,
              batch_size=batch_size,
              test_frequency=test_frequency,
              test_dataset=test_dataset,
              num_threads=num_threads,
              num_batches=num_batches,
              num_proposals=num_proposals,
              num_candidates=num_candidates,
              score_range=score_range,
              num_test_samples=num_test_samples,
              score_function=score_function,
              verbose='output' if verbose else False)


if __name__ == "__main__":
    main()
