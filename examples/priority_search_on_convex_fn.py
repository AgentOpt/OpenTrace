import re
import sys
import string
import numpy as np
from opto.trace.utils import dedent


def np_random(seed: int | None = None) -> tuple[np.random.Generator, int]:
    """Returns a NumPy random number generator (RNG) along with seed value from the inputted seed.

    If ``seed`` is ``None`` then a **random** seed will be generated as the RNG's initial seed.
    This randomly selected seed is returned as the second value of the tuple.

    .. py:currentmodule:: gymnasium.Env

    This function is called in :meth:`reset` to reset an environment's initial RNG.

    Args:
        seed: The seed used to create the generator

    Returns:
        A NumPy-based Random Number Generator and generator seed

    Raises:
        Error: Seed must be a non-negative integer
    """
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        if isinstance(seed, int) is False:
            raise Exception(
                f"Seed must be a python integer, actual type: {type(seed)}"
            )
        else:
            raise Exception(
                f"Seed must be greater or equal to zero, actual value: {seed}"
            )

    seed_seq = np.random.SeedSequence(seed)
    np_seed = seed_seq.entropy
    rng = np.random.Generator(np.random.PCG64(seed_seq))
    return rng, np_seed


class LossLandscapeBase:
    def __init__(self, callable_func, x_low, x_high, min_y, optimal_sol,
                 feedback=0, seed=None, precision_digit=2, horizon=10):
        self.x_low = x_low
        self.x_high = x_high

        self._np_random = None
        self.stop_keywords = ['reach', 'stay', 'stop']

        self.callable_func = callable_func

        self.prev_x = None
        self.left_attempts = horizon
        self.min_y = min_y
        self.optimal_sol = optimal_sol
        self.precision_digit = precision_digit

        self.horizon = horizon

        self._seed = self.seed(seed)

        self.reward_range = (self.get_min_reward(), -self.min_y)

        # Note: currently we treat the first line as "instruction"
        self.docstring = dedent("""
        You are trying to minimize the output (y) of a function by choosing input (x). The goal is to choose x such that y is as small as possible.

        You get to observe y once you choose the value of x, where x is a 2-dimensional vector.
        This means x = [x1, x2], where x1 and x2 are real numbers.


        The range of x1 and x2 is [{}, {}].
        Please do not choose x outside of this range.

        Choose x within {} attempts.
        You can choose to stop at any time.

        Output format:
        x = [x1, x2]
        """)

        self.docstring = self.docstring.strip()
        self.docstring = self.docstring.format(self.x_low, self.x_high, self.horizon)

        self.called_reset = False

    def get_min_reward(self):
        x_range = [self.x_low, self.x_high]
        y_max = [self.callable_func(np.array([x_range[i], x_range[j]])) for i in range(2) for j in range(2)]
        y_max = max(y_max)
        return -y_max

    def get_optimal_solution(self):
        return self.optimal_sol

    def reset(self, **kwargs):
        if 'seed' in kwargs:
            self._seed = self.seed(kwargs['seed'])
        # we sample the initial state from the uniform distribution
        x = self.np_random.uniform(self.x_low, self.x_high, size=2)
        # we round the floating point precision to 2 decimal places
        x = np.round(x, self.precision_digit)
        self.prev_x = x

        y = self.callable_func(x)

        self.left_attempts = self.horizon

        obs = "x={}\nFunction outputs y = {}\nYou have {} attempts left!\n".format(x.tolist(), y, self.left_attempts)
        obs += "Please output the next x that will make this function output the smallest y.\n"
        obs += "Format: x = [x1, x2]\n"
        obs += "Output:"

        self.called_reset = True

        return obs

    def seed(self, seed=None):
        """Seed the PRNG of this space and possibly the PRNGs of subspaces."""
        self._np_random, seed = np_random(seed)
        return [seed]

    @property
    def np_random(self):
        """Lazily seed the PRNG since this is expensive and only needed if sampling from this space."""
        if self._np_random is None:
            self.seed()
        return self._np_random  # type: ignore  ## self.seed() call guarantees right type.

    def text_extract(self, text):
        # return np.array([x1, x2]), agent decides to stop
        for stop_word in self.stop_keywords:
            if stop_word in text:
                return None, True

        pattern = r'\[(-?\d+\.?\d*(?:e[-+]?\d+)?),\s*(-?\d+\.?\d*(?:e[-+]?\d+)?)\]'
        match = re.search(pattern, text)
        if match is None:
            return None, False
        else:
            numbers = [float(g) for g in match.groups()]
            return np.array(numbers), False

    def step(self, action):
        if not self.called_reset:
            raise Exception("must call env.reset() first before step()")

        x, stop = self.text_extract(action)
        feedback = ''
        if x is None and stop is False:
            feedback = f'You entered an invalid action: {action}' + f" Please enter a valid action within ({self.x_low, self.x_high})"

            return None, -1, True, {'feedback': feedback, 'success': False}

        if stop:
            success = np.abs(self.callable_func(self.prev_x) - self.min_y) < 1e-2
            feedback = f'You have chosen to stop at {self.prev_x}.'
            if success:
                feedback += ' You have reached the minimum!'
            else:
                feedback += ' You have not reached the minimum!'
            return None, float(self.callable_func(self.prev_x)), True, {'feedback': feedback, 'success': success}

        loss = self.callable_func(x)

        if np.abs(loss - self.min_y) < 1e-2:
            feedback = "Function outputs y: {}\nYou have reached the minimum!".format(self.min_y)
            return feedback, -self.min_y, True, {'feedback': feedback, "success": True}

        obs = "Function outputs y = {}\nYou have {} attempts left!\n".format(loss, self.left_attempts)
        obs += "Please output the next x that will make this function output the smallest y.\n"
        obs += "Format: x = [x1, x2]\n"
        obs += "Output:"

        self.prev_x = x
        self.left_attempts -= 1

        r = np.clip(float(-loss), self.get_min_reward(), -self.min_y)

        feedback += f"You chose {action}. Choose different numbers such that you can minimize y."

        return obs, r, False, {'feedback': feedback, 'success': False}


class Rosenbrock(LossLandscapeBase):
    def __init__(self, a=1, b=1, feedback=0, seed=None, horizon=10):  # b = 100
        # https://en.wikipedia.org/wiki/Rosenbrock_function
        # all of them are lambda functions that expect Numpy array of shape (2,)
        two_dim_rosenbrock = lambda x: (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2
        super().__init__(callable_func=two_dim_rosenbrock,
                         x_low=-5, x_high=10, min_y=0, optimal_sol=np.ones(2),
                         feedback=feedback, seed=seed, horizon=horizon)


class SixHumpCamel(LossLandscapeBase):
    def __init__(self, feedback=0, seed=None, horizon=10):
        func = lambda x: (4 - 2.1 * x[0] ** 2 + (x[0] ** 4) / 3) * x[0] ** 2 + x[0] * x[1] + (-4 + 4 * x[1] ** 2) * x[
            1] ** 2
        # note that SixHumpCamel has two global minima
        # also the range on x is x1 = [-3, 3], x2 = [-2, 2]
        # but we use x1 = [-2, 2], x2 = [-3, 3] for simplicity
        super().__init__(callable_func=func,
                         x_low=-2, x_high=2, min_y=-1.0316,
                         optimal_sol=[np.array([0.0898, -0.7126]), np.array([-0.0898, 0.7126])],
                         feedback=feedback, seed=seed, horizon=horizon, precision_digit=4)

# ============ Add testing code =============
import datasets
import numpy as np
from opto import trace
from opto.utils.llm import LLM, LiteLLM
from opto.optimizers import OptoPrimeV2 as OptoPrime
from opto.features.priority_search import PrioritySearch as SearchAlgorithm
from opto.trainer.guide import Guide
from opto.trainer.loggers import TensorboardLogger
from opto.trainer.guide import LLMJudge
from typing import Any
from opto import trainer
from typing import Tuple


class RewardGuide(Guide):
    def __init__(self, env):
        self.env = env

    def get_feedback(self, query: str, response: str, reference=None, **kwargs) -> Tuple[float, str]:
        # score, feedbak str
        obs, reward, done, info = self.env.step(response)

        return reward, obs + '\n\n' + info['feedback']

env = SixHumpCamel(horizon=200)
train_dataset = dict(inputs=[None], infos=[None])
instruction = env.reset()
initial_input = instruction.split("\n")[0].strip()
param = trace.node(initial_input, description='Input x into the hidden function to get y.', trainable=True)

guide = RewardGuide(env)
logger = TensorboardLogger(log_dir='./logs/priority_search_on_convex_fn')

trainer.train(
    model=param,
    # optimizer='OptoPrimeV2',  # by default, OPROv2 is used for single-node optimization
    algorithm=SearchAlgorithm,
    train_dataset=train_dataset,
    logger=logger,
    score_range=[-10, 10],
    # trainer kwargs
    num_epochs=3*4,
    batch_size=2,  # this is just for testing. effectively, this is the same batch_size=1 and num_proposals=4
    num_batches=2,
    verbose=False, #'output',
    guide=guide,
    num_candidates=4,
    num_proposals=2,
    memory_update_frequency=2,
    optimizer_kwargs={'objective':"You have a task of guessing two numbers. You should make sure your guess minimizes y.",
                     'memory_size': 10}
)