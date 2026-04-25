from pathlib import Path
from unittest.mock import patch

import nbformat
import pytest

nbclient = pytest.importorskip("nbclient")
pytest.importorskip("langgraph.graph")


@pytest.mark.parametrize(
    "notebook_path",
    [
        "examples/notebooks/demo_langgraph_instrument_and_optimize.ipynb",
        "examples/notebooks/demo_langgraph_instrument_and_optimize_trace.ipynb",
        "examples/notebooks/demo_langgraph_instrument_and_compare_observers.ipynb",
    ],
)
def test_notebook_executes(notebook_path):
    root = Path(__file__).resolve().parents[2]
    with (root / notebook_path).open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = nbclient.NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(root)}},
    )
    # Force notebook live-provider sections to skip for deterministic CI runs.
    with patch.dict("os.environ", {"OPENROUTER_API_KEY": ""}, clear=False):
        client.execute()
