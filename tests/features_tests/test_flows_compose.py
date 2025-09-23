import pytest
from unittest.mock import patch, Mock


# Mock LLM at module level to ensure no real API calls
@pytest.fixture(autouse=True)
def mock_llm_globally():
    """Automatically mock all LLM calls for all tests."""
    with patch('opto.utils.llm.LLM') as mock_llm_class:
        # Create a mock LLM instance that doesn't require API keys
        mock_llm_instance = Mock()
        mock_llm_instance.return_value = Mock()
        mock_llm_class.return_value = mock_llm_instance
        yield mock_llm_instance


@pytest.fixture(autouse=True)
def mock_trace_operators():
    """Mock trace operators to prevent any external dependencies."""
    with patch('opto.trace.operators.call_llm') as mock_call_llm:
        mock_call_llm.return_value = "Mocked LLM response"
        yield mock_call_llm


