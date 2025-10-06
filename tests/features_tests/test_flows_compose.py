import pytest
from unittest.mock import patch, Mock


# Mock LLM at module level to ensure no real API calls
@pytest.fixture(autouse=True)
def mock_llm_globally():
    """Automatically mock all LLM calls for all tests with structured responses.

    The dummy LLM returns an object with `.choices[0].message.content` so callers
    like `TracedLLM` and optimizers can read a string without hitting a network.
    You can override `dummy_llm.responses` in tests to control returned strings.
    """
    class _Choice:
        def __init__(self, content):
            self.message = type('m', (), {'content': content})

    class DummyLLM:
        def __init__(self):
            # Default to an endless stream of the same mocked response
            self.responses = ["Mocked LLM response"]
            self._idx = 0

        def __call__(self, *args, **kwargs):
            # Return a response-like object with choices
            if self._idx >= len(self.responses):
                # Repeat last if we run out
                content = self.responses[-1]
            else:
                content = self.responses[self._idx]
            self._idx += 1
            return type('r', (), {'choices': [_Choice(content)]})

    dummy_llm = DummyLLM()
    with patch('opto.utils.llm.LLM', return_value=dummy_llm):
        yield dummy_llm



def test_tracedllm_and_optoprimev2_prompt_with_mock_llm(mock_llm_globally):
    # Arrange custom fake responses for three chat turns
    mock_llm_globally.responses = [
        "I can't access your location.",
        "Noted: you're in Paris.",
        "You are in Paris.",
    ]

    from opto.features.flows.compose import TracedLLM
    from opto.optimizers.optoprime_v2 import OptoPrimeV2, OptimizerPromptSymbolSet2

    # Act: run the user-provided flow without any real LLM calls
    traced_llm = TracedLLM(system_prompt="Be a friendly personal assistant.", trainable=True, chat_history_on=True)
    output = traced_llm("Where am I?")
    output2 = traced_llm("I'm in Paris.")
    output3 = traced_llm("Where am I?")

    optimizer = OptoPrimeV2(
        traced_llm.parameters(),
        use_json_object_format=False,
        ignore_extraction_error=False,
        include_example=False,
        optimizer_prompt_symbol_set=OptimizerPromptSymbolSet2(),
        memory_size=5,
        initial_var_char_limit=500,
    )

    optimizer.zero_feedback()
    optimizer.backward(output3, "Don't mention that you don't have the ability to locate my location if I tell you where I am.")

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    assert "Your response:" in part2
    print(part2)

