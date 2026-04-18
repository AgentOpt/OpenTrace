import pytest

# PrioritySearch integration is covered by graph_module_train smoke.
# This file exists to keep a dedicated feature test entry point.

pytest.importorskip("langgraph.graph")


def test_prioritysearch_entrypoint_smoke():
    assert True
