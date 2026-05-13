import json

import pytest

from opto.optimizers import OptoPrimeMultiV2
from opto.optimizers.optoprime_v2 import OptimizerPromptSymbolSetJSON
from opto.trace.nodes import GRAPH, node
from opto.utils.llm import DummyLLM, LLMFactory


def make_xml_candidate(var_name: str, value: str, reasoning: str = "update"):
    return f"<reasoning>{reasoning}</reasoning>\n<variable><name>{var_name}</name><value>{value}</value></variable>"


def make_json_candidate(var_name: str, value: str, reasoning: str = "update"):
    return json.dumps({"reasoning": reasoning, "suggestion": {var_name: value}})


class ScriptedResponder:
    def __init__(self, var_name: str, value: str, output_mode: str = "xml"):
        self.var_name = var_name
        self.value = value
        self.output_mode = output_mode
        self.calls = []

    def candidate(self, value: str | None = None):
        if self.output_mode == "json":
            return make_json_candidate(self.var_name, value or self.value)
        return make_xml_candidate(self.var_name, value or self.value)

    def __call__(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        messages = kwargs.get("messages", [])
        system_prompt = messages[0]["content"] if messages else ""
        if "expert persona" in system_prompt:
            return json.dumps(["Algorithm Expert", "Prompt Engineer", "Reviewer"])
        if "LLM profile:" in system_prompt:
            profile = system_prompt.split("LLM profile:", 1)[1].split("]", 1)[0].strip()
            return self.candidate(value=f"{profile}_value")
        if "Choose or synthesize the most promising candidate update" in system_prompt:
            return self.candidate(value="selected_by_best_of_n")
        if "synthesizing multiple candidate updates" in system_prompt:
            return self.candidate(value="selected_by_moa")
        if "PREVIOUS_CANDIDATE" in system_prompt:
            return self.candidate(value="self_refined_value")
        if "Generate a materially different new candidate" in system_prompt:
            return self.candidate(value="iterative_value")
        if "Provide your strongest candidate solution" in system_prompt:
            return self.candidate(value="expert_value")
        return self.candidate()


@pytest.fixture(autouse=True)
def clear_graph():
    GRAPH.clear()
    yield
    GRAPH.clear()


def test_optoprimemultiv2_default_xml_mode_does_not_force_json_object():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    responder = ScriptedResponder(prompt.py_name, "beta", output_mode="xml")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(responder),
        num_responses=2,
        generation_technique="temperature_variation",
        selection_technique="last_of_n",
        max_tokens=256,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, "Change the trainable variable to beta.")
    updates = optimizer.step(bypassing=True)

    assert updates[prompt] == "beta"
    assert len(optimizer.candidates) == 2
    assert optimizer.selected_candidate_details["variables"][prompt.py_name] == "beta"
    assert all("response_format" not in call["kwargs"] for call in responder.calls)


def test_optoprimemultiv2_json_symbol_set_uses_json_object_response_format():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    responder = ScriptedResponder(prompt.py_name, "beta", output_mode="json")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(responder),
        optimizer_prompt_symbol_set=OptimizerPromptSymbolSetJSON(),
        num_responses=1,
        generation_technique="temperature_variation",
        selection_technique="last_of_n",
        max_tokens=256,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, "Change the trainable variable to beta.")
    updates = optimizer.step(bypassing=True)

    assert updates[prompt] == "beta"
    assert any(call["kwargs"].get("response_format") == {"type": "json_object"} for call in responder.calls)


@pytest.mark.parametrize(
    "generation_technique, expected_value",
    [
        ("temperature_variation", "beta"),
        ("self_refinement", "self_refined_value"),
        ("iterative_alternatives", "iterative_value"),
        ("multi_experts", "expert_value"),
    ],
)
def test_optoprimemultiv2_generation_strategies(generation_technique, expected_value):
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    responder = ScriptedResponder(prompt.py_name, "beta", output_mode="xml")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(responder),
        num_responses=3,
        generation_technique=generation_technique,
        selection_technique="last_of_n",
        experts_list=["Algorithm Expert", "Prompt Engineer", "Reviewer"],
        max_tokens=256,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, f"Change the trainable variable to {expected_value}.")
    updates = optimizer.step(bypassing=True)

    assert updates[prompt] == expected_value
    assert optimizer.selected_candidate_details["valid"] is True


def test_optoprimemultiv2_falls_back_to_last_valid_candidate():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    responder = ScriptedResponder(prompt.py_name, "beta", output_mode="xml")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(responder),
        num_responses=2,
        generation_technique="temperature_variation",
        selection_technique="last_of_n",
        selector=lambda candidates: "not a parseable candidate",
        max_tokens=256,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, "Change the trainable variable to beta.")
    updates = optimizer.step(bypassing=True)

    assert updates[prompt] == "beta"
    assert optimizer.selected_candidate_details["valid"] is True


def test_optoprimemultiv2_selected_terminate_falls_back_to_last_valid_candidate():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    responder = ScriptedResponder(prompt.py_name, "beta", output_mode="xml")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(responder),
        num_responses=2,
        generation_technique="temperature_variation",
        selection_technique="last_of_n",
        selector=lambda candidates: "TERMINATE",
        max_tokens=256,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, "Change the trainable variable to beta.")
    updates = optimizer.step(bypassing=True)

    assert updates[prompt] == "beta"
    assert optimizer.selected_candidate_details["valid"] is True
    assert optimizer.selected_candidate_details["terminate"] is False


def test_optoprimemultiv2_plain_dict_suggestion_candidate_is_parsed():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(lambda *args, **kwargs: "TERMINATE"),
        max_tokens=256,
    )

    parsed = optimizer._parse_candidate(
        {
            "reasoning": "use the explicit suggestion mapping",
            "suggestion": {prompt.py_name: "beta"},
        }
    )

    assert parsed["valid"] is True
    assert parsed["variables"] == {prompt.py_name: "beta"}


def test_optoprimemultiv2_public_call_llm_returns_string_like_v2():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    responder = ScriptedResponder(prompt.py_name, "beta", output_mode="xml")
    optimizer = OptoPrimeMultiV2([prompt], llm=DummyLLM(responder), max_tokens=256)

    response = optimizer.call_llm("system", "user", max_tokens=32)

    assert isinstance(response, str)
    assert "<variable>" in response


def test_optoprimemultiv2_all_terminate_returns_no_update():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(lambda *args, **kwargs: "TERMINATE"),
        num_responses=2,
        generation_technique="temperature_variation",
        selection_technique="last_of_n",
        max_tokens=256,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, "No useful update exists.")
    updates = optimizer.step(bypassing=True)

    assert updates == {}
    assert optimizer.selected_candidate == "TERMINATE"


def test_optoprimemultiv2_selection_techniques():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    responder = ScriptedResponder(prompt.py_name, "beta", output_mode="xml")
    optimizer = OptoPrimeMultiV2([prompt], llm=DummyLLM(responder), max_tokens=256)
    candidate_a = make_xml_candidate(prompt.py_name, "value_a")
    candidate_b = make_xml_candidate(prompt.py_name, "value_b")

    assert optimizer._parse_candidate(
        optimizer.select_candidate([candidate_a, candidate_b], selection_technique="last_of_n", problem_summary="problem")
    )["variables"][prompt.py_name] == "value_b"
    assert optimizer._parse_candidate(
        optimizer.select_candidate([candidate_a, candidate_a, candidate_b], selection_technique="majority", problem_summary="problem")
    )["variables"][prompt.py_name] == "value_a"
    assert optimizer._parse_candidate(
        optimizer.select_candidate([candidate_a, candidate_b], selection_technique="best_of_n", problem_summary="problem")
    )["variables"][prompt.py_name] == "selected_by_best_of_n"
    assert optimizer._parse_candidate(
        optimizer.select_candidate([candidate_a, candidate_b], selection_technique="moa", problem_summary="problem")
    )["variables"][prompt.py_name] == "selected_by_moa"


def test_optoprimemultiv2_majority_duplicate_consensus_without_sklearn(monkeypatch):
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    responder = ScriptedResponder(prompt.py_name, "beta", output_mode="xml")
    optimizer = OptoPrimeMultiV2([prompt], llm=DummyLLM(responder), max_tokens=256)

    candidate_a = make_xml_candidate(prompt.py_name, "value_a")
    candidate_b = make_xml_candidate(prompt.py_name, "value_b")

    import builtins

    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name in {"numpy", "sklearn"} or name.startswith("sklearn"):
            raise ImportError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    selected = optimizer.select_candidate(
        [candidate_a, candidate_a, candidate_b],
        selection_technique="majority",
        problem_summary="problem",
    )
    assert optimizer._parse_candidate(selected)["variables"][prompt.py_name] == "value_a"


def test_optoprimemultiv2_multi_experts_json_mode_accepts_experts_object():
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")

    class ExpertObjectResponder(ScriptedResponder):
        def __call__(self, *args, **kwargs):
            self.calls.append({"args": args, "kwargs": kwargs})
            messages = kwargs.get("messages", [])
            system_prompt = messages[0]["content"] if messages else ""
            if "expert persona" in system_prompt:
                return json.dumps({"experts": ["Algorithm Expert", "Reviewer"]})
            if "Provide your strongest candidate solution" in system_prompt:
                return make_json_candidate(self.var_name, "expert_value")
            return make_json_candidate(self.var_name, "fallback")

    responder = ExpertObjectResponder(prompt.py_name, "beta", output_mode="json")
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(responder),
        optimizer_prompt_symbol_set=OptimizerPromptSymbolSetJSON(),
        num_responses=2,
        generation_technique="multi_experts",
        selection_technique="last_of_n",
        max_tokens=256,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, "Change the trainable variable.")
    updates = optimizer.step(bypassing=True)

    assert updates[prompt] == "expert_value"
    assert any(call["kwargs"].get("response_format") == {"type": "json_object"} for call in responder.calls)


def test_optoprimemultiv2_multi_llm_generation_uses_profiles(monkeypatch):
    prompt = node("alpha", name="prompt", trainable=True, description="simple prompt")
    default_responder = ScriptedResponder(prompt.py_name, "default_value", output_mode="xml")
    profile_calls = []

    def fake_get_llm(profile):
        profile_calls.append(profile)
        return DummyLLM(lambda *args, **kwargs: make_xml_candidate(prompt.py_name, f"{profile}_value"))

    monkeypatch.setattr(LLMFactory, "get_llm", fake_get_llm)
    optimizer = OptoPrimeMultiV2(
        [prompt],
        llm=DummyLLM(default_responder),
        llm_profiles=["cheap", "premium"],
        num_responses=3,
        generation_technique="multi_llm",
        selection_technique="last_of_n",
        max_tokens=256,
    )

    optimizer.zero_feedback()
    optimizer.backward(prompt, "Change the trainable variable.")
    updates = optimizer.step(bypassing=True)

    assert profile_calls == ["cheap", "premium"]
    assert len(optimizer.candidates) == 3
    assert updates[prompt] == "cheap_value"
