

from opto import trace
from opto.trainer.loader import DataLoader
from opto.trainer.algorithms import BasicSearchAlgorithm
from opto.optimizers import OptoPrimeV2
from opto.trainer.guide import Guide as _Guide
from opto.utils.llm import DummyLLM

import re, os
import numpy as np
import copy

@trace.bundle(trainable=True)
def fun(x):
    """ Some docstring. """
    return len(x), x.count('\n')

def test_saving_load():
    x = 'hello\nworld\n'
    a, b = fun(x)
    print(a, b)

    print(fun.parameters()[0].data)

    fun.parameters()[0]._data =fun.parameters()[0]._data.replace('len(x)', '"Hello"')

    a, b = fun(x)
    print(a, b)
    fun.save('fun.pkl')

    fun.load('fun.pkl')



    a, b = fun(x)
    print(a, b)

suggested_value = 5

def _llm_callable(messages, **kwargs):
    """
    A dummy LLM callable that simulates a response.
    """
    problem = messages[1]['content']

        # extract name from <variable name= name ... >
        name = re.findall(r"<variable name=\"\s*(.*?)\" type=.*>", problem)
        if name:
            name = name[0]
        else:
            name = "unknown"

        return f"""
        <reasoning> Dummy reasoning based on the input messages. </reasoning>
        <variable>
        <name> {name} </name>
        <value> {suggested_value} </value>
        </variable>
        """

     # Create a dummy LLM and an agent
    dummy_llm = DummyLLM(_llm_callable)
    agent = Agent()
    optimizer = OptoPrimeV2(
        agent.parameters(),
        llm=dummy_llm,
    )
    optimizer.objective = 'fake objective'
    algo = BasicSearchAlgorithm(
        agent,
        optimizer,
    )

    algo.train(
        guide=Guide(),
        train_dataset=dataset,
        batch_size=batch_size,
        num_threads=num_threads,
        num_candidates=num_candidates,
        num_proposals=num_proposals,
        verbose=False, #'output',
    )
    agent.param._data = 10 # to simulate a change in the agent's parameters

    algo.save('./test_algo')


    # Load the algorithm and check if it works
    agent = Agent()
    optimizer = OptoPrimeV2(
        agent.parameters(),
        llm=dummy_llm,
    )
    algo2 = BasicSearchAlgorithm(
        agent,
        optimizer,
    )
    algo2.load('./test_algo')

#     assert algo2.agent.param.data == 10, "Loaded agent's parameter does not match the saved one."
#     assert algo2.optimizer.objective == 'fake objective', "Loaded optimizer's objective does not match the saved one."

    os.remove('./test_algo')
    os.remove('./test_algo_agent.module')
    os.remove('./test_algo_optimizer.optimizer')
    os.remove('./test_algo_validate_guide.guide')