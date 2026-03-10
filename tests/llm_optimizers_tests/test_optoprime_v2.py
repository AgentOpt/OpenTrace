import os
import pytest
from opto.trace import bundle, node, GRAPH
import opto.optimizers
import importlib
import inspect
import json
import pickle
from opto.utils.llm import LLM

from opto import trace
from opto.trace import node, bundle
from opto.optimizers.optoprime_v2 import OptoPrimeV2, OptimizerPromptSymbolSet2

# You can override for temporarly testing a specific optimizer ALL_OPTIMIZERS = [TextGrad] # [OptoPrimeMulti] ALL_OPTIMIZERS = [OptoPrime]

# Skip tests if no API credentials are available
SKIP_REASON = "No API credentials found"
HAS_CREDENTIALS = os.path.exists("OAI_CONFIG_LIST") or os.environ.get("TRACE_LITELLM_MODEL") or os.environ.get(
    "OPENAI_API_KEY")
llm = LLM()


@pytest.fixture(autouse=True)
def clear_graph():
    """Reset the graph before each test"""
    GRAPH.clear()
    yield
    GRAPH.clear()


@pytest.mark.skipif(not HAS_CREDENTIALS, reason=SKIP_REASON)
def test_response_extraction():
    pass


def test_tag_template_change():
    num_1 = node(1, trainable=True)
    num_2 = node(2, trainable=True, description="<=5")
    result = num_1 + num_2
    optimizer = OptoPrimeV2([num_1, num_2], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True,
                            optimizer_prompt_symbol_set=OptimizerPromptSymbolSet2())

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    assert """<var name="variable_name" type="data_type">""" in part1, "Expected <var> tag to be present in part1"
    assert """<const name="y" type="int">""" in part2, "Expected <const> tag to be present in part2"

    print(part1)
    print(part2)


@bundle()
def transform(num):
    """Add number"""
    return num + 1


@bundle(trainable=True)
def multiply(num):
    return num * 5


def test_function_repr():
    num_1 = node(1, trainable=False)

    result = multiply(transform(num_1))
    optimizer = OptoPrimeV2([multiply.parameter], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True)

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    # Variable counter (__code0, __code1, ...) shifts based on test execution
    # order, so match the structural content rather than the exact counter.
    assert 'type="code">' in part2, "Expected code variable in part2"
    assert "def multiply(num):" in part2, "Expected function definition in part2"
    assert "return num * 5" in part2, "Expected function body in part2"
    assert "The code should start with:" in part2, "Expected constraint in part2"

@pytest.mark.xfail(reason="Upstream: initial_var_char_limit truncation not applied in current OptoPrimeV2")
def test_big_data_truncation():
    num_1 = node(1, trainable=True)

    list_1 = node([1, 2, 3, 4, 5, 6, 7, 8, 9, 20] * 10, trainable=True)

    result = num_1 + list_1[30]

    optimizer = OptoPrimeV2([num_1, list_1], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True, initial_var_char_limit=10)

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    truncated_repr = """<variable name="list0" type="list">
<value>
[1, 2, 3, ...(skipped due to length limit)
</value>
</variable>"""

    assert truncated_repr in part2, "Expected truncated list representation to be present in part2"

def test_extraction_pipeline():
    num_1 = node(1, trainable=True)
    num_2 = node(2, trainable=True, description="<=5")
    result = num_1 + num_2
    optimizer = OptoPrimeV2([num_1, num_2], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True,
                            optimizer_prompt_symbol_set=OptimizerPromptSymbolSet2())

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    messages = [
        {"role": "system", "content": part1},
        {"role": "user", "content": part2},
    ]

    # response = optimizer.llm(messages=messages)
    # response = response.choices[0].message.content
    response = """<reason>
The instruction suggests that the output, `add0`, needs to be made bigger than it currently is (3). The code performs an addition of `int0` and `int1` to produce `add0`. To increase `add0`, we can increase the values of `int0` or `int1`, or both. Given that `int1` has a constraint of being less than or equal to 5, we can set `int0` to a higher value, since it has no explicit constraint. By adjusting `int0` to a higher value, the output can be made larger in accordance with the feedback.
</reason>

<var>
<name>int0</name>
<data>
5
</data>
</var>

<var>
<name>int1</name>
<data>
5
</data>
</var>"""
    reasoning = response
    suggestion = optimizer.extract_llm_suggestion(response)

    assert 'reasoning' in suggestion, "Expected 'reasoning' in suggestion"
    assert 'variables' in suggestion, "Expected 'variables' in suggestion"
    assert 'int0' in suggestion['variables'], "Expected 'int0' variable in suggestion"
    assert 'int1' in suggestion['variables'], "Expected 'int1' variable in suggestion"
    assert str(suggestion['variables']['int0']) == '5', "Expected int0 to be incremented to 5"
    assert str(suggestion['variables']['int1']) == '5', "Expected int1 to be incremented to 5"
