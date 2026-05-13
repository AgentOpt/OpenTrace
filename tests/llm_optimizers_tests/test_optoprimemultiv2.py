import os

import pytest

from opto.optimizers import OptoPrimeMultiV2
from opto.optimizers.optoprime_v2 import OptimizerPromptSymbolSetJSON
from opto.trace.nodes import GRAPH, node
from opto.utils.llm import LLM


def _test_model_name():
    if os.getenv("OPENAI_API_KEY"):
        return os.getenv("TRACE_OPENAI_TEST_MODEL", "gpt-4o-mini")
    if os.getenv("OPENROUTER_API_KEY"):
        return os.getenv("TRACE_OPENROUTER_TEST_MODEL", "openrouter/openai/gpt-4o-mini")
    return None


@pytest.fixture(autouse=True)
def clear_graph():
    GRAPH.clear()
    yield
    GRAPH.clear()


@pytest.mark.llm
def test_optoprimemultiv2_real_llm_json_smoke():
    model_name = _test_model_name()
    if model_name is None:
        pytest.skip("Set OPENAI_API_KEY or OPENROUTER_API_KEY to run this smoke test.")
    os.environ["TRACE_LITELLM_MODEL"] = model_name

    prompt = node("alpha", name="prompt", trainable=True, description="The optimized value should become the exact string BETA_TOKEN.")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=LLM(),
        optimizer_prompt_symbol_set=OptimizerPromptSymbolSetJSON(),
        num_responses=2,
        generation_technique="temperature_variation",
        selection_technique="best_of_n",
        max_tokens=512,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, "Please update the single trainable variable so its new value is exactly BETA_TOKEN.")
    updates = optimizer.step(bypassing=True)

    assert prompt in updates
    assert "BETA_TOKEN" in str(updates[prompt]).upper()
