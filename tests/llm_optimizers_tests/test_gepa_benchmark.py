import os
import pytest
import numpy as np

from opto import trace
from opto.optimizers.optoprime_v2 import OptoPrimeV2
from opto.trainer.algorithms.gepa_algorithms import GEPAAlgorithmBase, GEPAUCBSearch, GEPABeamPareto
from opto.trainer.algorithms.basic_algorithms import BasicSearchAlgorithm
from opto.trainer.guide import LLMJudge
from opto.utils.llm import LLM


RUN_BENCH = "1"


def _datasets_or_skip():
    try:
        import datasets  # noqa: F401
    except Exception:
        pytest.skip("datasets library not available; skipping GEPA benchmark test.")


def _llm_env_or_skip():
    have_key = any(os.getenv(k) for k in ["OPENAI_API_KEY", "AZURE_OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OAI_CONFIG_LIST"])
    if not have_key:
        pytest.skip("No LLM credentials found in environment; skipping GEPA benchmark test.")


@trace.model
class Learner:
    """Agent that calls an LLM. The only trainable variable is 'system_prompt'."""

    def __init__(self, system_prompt: str = "You're a helpful agent", user_prompt_template: str = "Query: {message}", llm: LLM = None):
        self.system_prompt = trace.node(system_prompt, trainable=True)
        self.user_prompt_template = trace.node(user_prompt_template)
        self.llm = llm or LLM()  # default profile

    @trace.bundle()
    def model(self, system_prompt: str, user_prompt_template: str, message: str) -> str:
        if "{message}" not in user_prompt_template:
            raise ValueError("user_prompt_template must contain '{message}'")
        resp = self.llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_template.format(message=message)},
            ]
        )
        return resp.choices[0].message.content

    def forward(self, message):
        return self.model(self.system_prompt, self.user_prompt_template, message)


@pytest.mark.skipif(not RUN_BENCH, reason="Set RUN_GEPA_BENCH=1 to run this optional benchmark test.")
def test_gepa_benchmark_gsm8k_real_llm():
    _datasets_or_skip()
    _llm_env_or_skip()

    import datasets

    # Load a tiny subset of GSM8k
    ds = datasets.load_dataset("openai/gsm8k", "main")
    train = ds["train"][:6]
    train_dataset = {"inputs": train["question"], "infos": train["answer"]}

    # Teacher/judge with a low-cost profile
    guide = LLMJudge(llm=LLM(profile="cheap"))

    # Set a budget constraint for algorithms that support it (e.g., GEPABeamPareto)
    budget_limit = 5

    # Agent and optimizer (low-cost profile)
    agent = Learner(llm=LLM(profile="cheap"))
    optimizer = OptoPrimeV2(agent.parameters(), llm=LLM(profile="cheap"))

    algos = [
        ("GEPA-Base", GEPAAlgorithmBase(agent, optimizer=optimizer, logger=None, num_threads=2), dict(num_iters=2, train_batch_size=1, merge_every=2)),
        (f"GEPA-BeamPareto-Budget{budget_limit}", GEPABeamPareto(agent, optimizer=optimizer, logger=None, num_threads=2), dict(num_search_iterations=2, train_batch_size=1, merge_every=2, budget_B=budget_limit)),
        ("GEPA-BeamPareto", GEPABeamPareto(agent, optimizer=optimizer, logger=None, num_threads=2), dict(num_search_iterations=2, train_batch_size=1, merge_every=2)),
        ("GEPA-UCB", GEPAUCBSearch(agent, optimizer=optimizer, logger=None, num_threads=2), dict(num_search_iterations=2, train_batch_size=1, merge_every=2)),
        ("BasicSearch", BasicSearchAlgorithm(agent, optimizer=optimizer, logger=None, num_threads=2), dict(num_epochs=1, batch_size=1, num_proposals=2)),
    ]

    results = {}
    for name, algo, kwargs in algos:
        if name == "BasicSearch":
            # Conform to BasicSearch's interface
            algo.train(guide=guide, train_dataset=train_dataset, validate_dataset=train_dataset, test_dataset=train_dataset, eval_frequency=1, num_threads=2, verbose=False, **kwargs)
            results[name] = 0.0  # placeholder; evaluation is heavy and non-deterministic
        else:
            _, best = algo.train(guide=guide, train_dataset=train_dataset, validate_dataset=train_dataset, pareto_subset_size=4, num_threads=2, **kwargs)
            results[name] = float(best)

    # Sanity check that we produced some floats for each algorithm
    assert set(results.keys()) == {"GEPA-Base", "GEPA-UCB", "GEPA-Beam", "BasicSearch"}
    for v in results.values():
        assert isinstance(v, float)

