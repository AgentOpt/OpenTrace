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
from opto.optimizers.opro_v2 import OPROv2, OPROPromptSymbolSet

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
    optimizer = OPROv2([num_1, num_2], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True,
                            optimizer_prompt_symbol_set=OPROPromptSymbolSet())

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
    optimizer = OPROv2([multiply.parameter], use_json_object_format=False,
                            ignore_extraction_error=False,
                            include_example=True)

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    function_repr = """<solution name="__code0" type="code">
<value>
def multiply(num):
    return num * 5
</value>
<constraint>
The code should start with:
def multiply(num):
</constraint>
</solution>"""

    assert function_repr in part2, "Expected function representation to be present in part2"

def test_big_data_truncation():
    num_1 = node(1, trainable=True)

    list_1 = node([1, 2, 3, 4, 5, 6, 7, 8, 9, 20] * 10, trainable=True)

    result = num_1 + list_1[30]

    optimizer = OPROv2([num_1, list_1], use_json_object_format=False,
                            ignore_extraction_error=False, initial_var_char_limit=10)

    optimizer.zero_feedback()
    optimizer.backward(result, 'make this number bigger')

    summary = optimizer.summarize()
    part1, part2 = optimizer.construct_prompt(summary)

    part1 = optimizer.replace_symbols(part1, optimizer.prompt_symbols)
    part2 = optimizer.replace_symbols(part2, optimizer.prompt_symbols)

    truncated_repr = "[1, 2, 3, ...(skipped due to length limit)"

    assert truncated_repr in part2, "Expected truncated list representation to be present in part2"

def test_extraction_pipeline():
    num_1 = node(1, trainable=True)
    optimizer = OPROv2([num_1], use_json_object_format=False,
                       ignore_extraction_error=False,
                       include_example=True)

    @bundle()
    def propose_solution(x):
        """
        Propose a solution to the given prompt using the input.
        """
        return x + 1

    result = propose_solution(num_1)

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

    response = '```\n\n<reasoning>\nThe #Instruction requests a new solution that incorporates the given feedback into the proposed solution. The #Variables section includes an integer variable "int0" with the current value set to 1. The feedback states that this number should be made "bigger." Thus, the current value does not meet the feedback requirement, and I should change it to a larger integer value to comply with the feedback. A simple increment will suffice, so I will propose changing "int0" from 1 to 2.\n</reasoning>\n<variable>\n<name>int0</name>\n<value>\n2\n</value>\n</variable>\n\n```'
    reasoning = response
    suggestion = optimizer.extract_llm_suggestion(response)

    assert 'reasoning' in suggestion, "Expected 'reasoning' in suggestion"
    assert 'variables' in suggestion, "Expected 'variables' in suggestion"
    assert 'int0' in suggestion['variables'], "Expected 'int0' variable in suggestion"
    assert suggestion['variables']['int0'] == 2, "Expected int0 to be incremented to 2"
