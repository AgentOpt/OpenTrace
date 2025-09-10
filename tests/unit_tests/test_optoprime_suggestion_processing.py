import json
import re
import timeit

import pytest
from opto import trace
from opto.optimizers import OptoPrime
from opto.utils.llm import DummyLLM


def make_optimizer(params):
    return OptoPrime(parameters=params, llm=DummyLLM(lambda *args, **kwargs: ""))


def test_construct_update_dict_alias_and_type_conversion():
    trace.GRAPH.clear()
    param = trace.node(1, trainable=True)
    opt = make_optimizer([param])
    suggestion = {"int:0": "2"}
    update = opt.construct_update_dict(suggestion)
    assert update[param] == 2 and isinstance(update[param], int)


def test_construct_update_dict_none_data():
    trace.GRAPH.clear()
    param = trace.node(None, trainable=True)
    opt = make_optimizer([param])
    suggestion = {param.py_name: "value"}
    update = opt.construct_update_dict(suggestion)
    assert update[param] == "value"


def test_extract_llm_suggestion_missing_tag():
    trace.GRAPH.clear()
    dummy = trace.node(0, trainable=True)
    opt = make_optimizer([dummy])
    response = json.dumps({"param1": 5})
    suggestion = opt.extract_llm_suggestion(response)
    assert suggestion == {"param1": 5}


def test_extract_llm_suggestion_non_dict_suggestion():
    trace.GRAPH.clear()
    dummy = trace.node(0, trainable=True)
    opt = make_optimizer([dummy])
    response = json.dumps({"suggestion": "not a dict", "param1": 5})
    suggestion = opt.extract_llm_suggestion(response)
    assert suggestion == {"suggestion": "not a dict", "param1": 5}


def test_efficiency_construct_update_dict():
    def baseline_construct_update_dict(parameters, suggestion):
        update_dict = {}
        for node in parameters:
            if node.trainable and node.py_name in suggestion:
                try:
                    formatted_suggestion = suggestion[node.py_name]
                    update_dict[node] = type(node.data)(formatted_suggestion)
                except (ValueError, KeyError):
                    pass
        return update_dict

    trace.GRAPH.clear()
    params = [trace.node(i, trainable=True) for i in range(50)]
    suggestion = {p.py_name: i for i, p in enumerate(params)}
    opt = make_optimizer(params)

    t_base = timeit.timeit(lambda: baseline_construct_update_dict(params, suggestion), number=200)
    t_new = timeit.timeit(lambda: opt.construct_update_dict(suggestion), number=200)
    assert t_new <= t_base * 5


def test_efficiency_extract_llm_suggestion():
    def baseline_extract(response, suggestion_tag="suggestion"):
        suggestion = {}
        attempt_n = 0
        while attempt_n < 2:
            try:
                suggestion = json.loads(response)[suggestion_tag]
                break
            except json.JSONDecodeError:
                resp_list = re.findall(r"{.*}", response, re.DOTALL)
                if len(resp_list) > 0:
                    response = resp_list[0]
                attempt_n += 1
            except Exception:
                attempt_n += 1
        if not isinstance(suggestion, dict):
            suggestion = {}
        return suggestion

    trace.GRAPH.clear()
    dummy = trace.node(0, trainable=True)
    opt = make_optimizer([dummy])
    response = json.dumps({"suggestion": {"a": 1}})
    t_base = timeit.timeit(lambda: baseline_extract(response), number=2000)
    t_new = timeit.timeit(lambda: opt.extract_llm_suggestion(response), number=2000)
    assert t_new <= t_base * 5
