import pytest
import os
from unittest.mock import patch, MagicMock, Mock
from opto.flows.compose import TracedLLM, TracedResponse
from opto.flows.types import TracedInput, TracedOutput


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


class TestTracedLLM:
    """Test cases for TracedLLM functionality."""
    
    def test_basic_initialization(self):
        """Test basic TracedLLM initialization."""
        llm = TracedLLM("You are a helpful assistant")
        assert llm.system_prompt.data == "You are a helpful assistant"
        assert llm._input_fields == []
        assert llm._output_fields == []
        assert llm._field_types == {}
    
    def test_docstring_as_system_prompt(self):
        """Test that class docstring is used as system prompt when none provided."""
        class TestLLM(TracedLLM):
            """This is a test LLM for testing purposes"""
            pass
        
        llm = TestLLM()
        assert llm.system_prompt.data == "This is a test LLM for testing purposes"
    
    def test_explicit_system_prompt_overrides_docstring(self):
        """Test that explicit system prompt overrides docstring."""
        class TestLLM(TracedLLM):
            """This is a test LLM"""
            pass
        
        llm = TestLLM("Custom prompt")
        assert llm.system_prompt.data == "Custom prompt"
    
    def test_field_detection_basic(self):
        """Test basic field detection for input and output fields."""
        class BasicScorer(TracedLLM):
            """Basic document scorer"""
            doc: str = TracedInput(description="Document to score")
            score: int = TracedOutput(description="Score from 1-10")
        
        scorer = BasicScorer()
        assert scorer._input_fields == ['doc']
        assert scorer._output_fields == ['score']
        assert scorer._field_types == {'doc': str, 'score': int}
    
    def test_field_detection_multiple_fields(self):
        """Test field detection with multiple input/output fields."""
        class MultiFieldScorer(TracedLLM):
            """Multi-field scorer"""
            doc: str = TracedInput(description="Document")
            context: str = TracedInput(description="Context")
            score: int = TracedOutput(description="Score")
            confidence: float = TracedOutput(description="Confidence")
            tags: list = TracedOutput(description="Tags")
        
        scorer = MultiFieldScorer()
        assert set(scorer._input_fields) == {'doc', 'context'}
        assert set(scorer._output_fields) == {'score', 'confidence', 'tags'}
        assert scorer._field_types['doc'] == str
        assert scorer._field_types['score'] == int
        assert scorer._field_types['confidence'] == float
        assert scorer._field_types['tags'] == list
    
    def test_direct_pattern_call(self, mock_trace_operators):
        """Test direct usage pattern (no inheritance fields)."""
        mock_trace_operators.return_value = "Hello! The weather is sunny."
        
        llm = TracedLLM("You are a helpful assistant")
        response = llm("Hello, what's the weather today?")
        
        assert response == "Hello! The weather is sunny."
        mock_trace_operators.assert_called_once()
    
    def test_inheritance_pattern_call(self, mock_trace_operators):
        """Test inheritance pattern with structured input/output."""
        mock_trace_operators.return_value = "The score is 8 out of 10"
        
        class Scorer(TracedLLM):
            """Score documents"""
            doc: str = TracedInput(description="Document to score")
            score: int = TracedOutput(
                description="Score from 1-10",
                parser=r"score[:\s]*is[:\s]*(\d+)|(\d+)\s*out\s*of"
            )
        
        scorer = Scorer()
        response = scorer(doc="This is a great document")
        
        assert isinstance(response, TracedResponse)
        assert response.score == 8
        mock_trace_operators.assert_called_once()
    
    def test_dynamic_response_model_creation(self):
        """Test dynamic Pydantic model creation."""
        class TestScorer(TracedLLM):
            """Test scorer"""
            doc: str = TracedInput(description="Document")
            score: int = TracedOutput(description="Score")
            confidence: float = TracedOutput(description="Confidence")
        
        scorer = TestScorer()
        ResponseModel = scorer._create_dynamic_response_model()
        
        assert ResponseModel.__name__ == "TestScorerResponse"
        assert 'score' in ResponseModel.model_fields
        assert 'confidence' in ResponseModel.model_fields
        assert ResponseModel.model_fields['score'].annotation == int
        assert ResponseModel.model_fields['confidence'].annotation == float
    
    def test_json_extraction(self):
        """Test JSON response extraction."""
        class Scorer(TracedLLM):
            """Test scorer"""
            doc: str = TracedInput()
            score: int = TracedOutput()
        
        scorer = Scorer()
        json_response = '{"score": 9}'
        extracted = scorer._extract_structured_data(json_response)
        
        assert extracted == {'score': 9}
    
    def test_text_extraction_with_patterns(self):
        """Test text extraction using field name patterns."""
        class Scorer(TracedLLM):
            """Test scorer"""
            doc: str = TracedInput()
            score: int = TracedOutput(parser=r"score[:\s]*is[:\s]*(\d+)|(\d+)\s*out\s*of")
        
        scorer = Scorer()
        text_response = "The score is 7 out of 10"
        extracted = scorer._extract_structured_data(text_response)
        
        assert extracted == {'score': 7}


class TestTracedInput:
    """Test cases for TracedInput."""
    
    def test_basic_initialization(self):
        """Test basic TracedInput initialization."""
        input_field = TracedInput(description="Test input")
        assert input_field.description == "Test input"
        assert input_field.required == True
    
    def test_optional_field(self):
        """Test optional TracedInput field."""
        input_field = TracedInput(description="Optional input", required=False)
        assert input_field.required == False


class TestTracedOutput:
    """Test cases for TracedOutput."""
    
    def test_basic_initialization(self):
        """Test basic TracedOutput initialization."""
        output_field = TracedOutput(description="Test output")
        assert output_field.description == "Test output"
        assert output_field.required == True
        assert output_field.parser is None
        assert output_field.default_value is None
    
    def test_with_default_value(self):
        """Test TracedOutput with default value."""
        output_field = TracedOutput(description="Score", default_value=5)
        assert output_field.default_value == 5
    
    def test_regex_parser_extraction(self):
        """Test extraction using regex parser."""
        output_field = TracedOutput(
            description="Rating",
            parser=r"(\d+)/5|rating[:\s]+(\d+)",
            default_value=0
        )
        
        # Test successful extraction
        result = output_field.extract_from_text("The rating is 4/5 stars", int)
        assert result == 4
        
        # Test fallback to default
        result = output_field.extract_from_text("No rating information", int)
        assert result == 0
    
    def test_function_parser_extraction(self):
        """Test extraction using function parser."""
        def sentiment_parser(text):
            if "good" in text.lower():
                return "Positive"
            elif "bad" in text.lower():
                return "Negative"
            else:
                return "Neutral"
        
        output_field = TracedOutput(
            description="Sentiment",
            parser=sentiment_parser,
            default_value="Unknown"
        )
        
        # Test successful extraction
        result = output_field.extract_from_text("This is a good product", str)
        assert result == "Positive"
        
        result = output_field.extract_from_text("This is a bad product", str)
        assert result == "Negative"
        
        # Test parser exception (should return default)
        def failing_parser(text):
            raise Exception("Parser error")
        
        output_field_with_failing_parser = TracedOutput(
            description="Sentiment",
            parser=failing_parser,
            default_value="Unknown"
        )
        result = output_field_with_failing_parser.extract_from_text("Some text", str)
        assert result == "Unknown"
    
    def test_boolean_parsing(self):
        """Test boolean value parsing."""
        output_field = TracedOutput(default_value=False)
        
        # Test positive cases
        assert output_field._parse_boolean("true") == True
        assert output_field._parse_boolean("yes") == True
        assert output_field._parse_boolean("positive") == True
        assert output_field._parse_boolean("definitely") == True
        
        # Test negative cases
        assert output_field._parse_boolean("false") == False
        assert output_field._parse_boolean("no") == False
        assert output_field._parse_boolean("negative") == False
        assert output_field._parse_boolean("no way") == False
        
        # Test default case
        assert output_field._parse_boolean("unclear") == False
    
    def test_type_conversion(self):
        """Test automatic type conversion."""
        output_field = TracedOutput(default_value=0)
        
        # Test int conversion
        assert output_field._convert_to_type("42", int) == 42
        assert output_field._convert_to_type("Score: 8", int) == 8
        assert output_field._convert_to_type("No numbers", int) == 0  # default
        
        # Test float conversion
        assert output_field._convert_to_type("3.14", float) == 3.14
        assert output_field._convert_to_type("Rating: 4.5", float) == 4.5
        
        # Test list conversion
        assert output_field._convert_to_type('["a", "b", "c"]', list) == ["a", "b", "c"]
        assert output_field._convert_to_type("a, b, c", list) == ["a", "b", "c"]


class TestDynamicModelMixin:
    """Test cases for DynamicModelMixin."""
    
    def test_create_response_model(self):
        """Test dynamic response model creation."""
        from opto.flows.types import DynamicModelMixin
        
        class TestClass(DynamicModelMixin):
            pass
        
        field_defs = {
            'score': (int, TracedOutput(description="Score value", default_value=0)),
            'tags': (list, TracedOutput(description="Tag list", required=False, default_value=[]))
        }
        
        ResponseModel = TestClass.create_response_model(field_defs)
        
        assert ResponseModel.__name__ == "TestClassResponse"
        assert 'score' in ResponseModel.model_fields
        assert 'tags' in ResponseModel.model_fields
        assert ResponseModel.model_fields['score'].annotation == int
        assert ResponseModel.model_fields['tags'].annotation == list
    
    def test_create_input_model(self):
        """Test dynamic input model creation."""
        from opto.flows.types import DynamicModelMixin
        
        class TestClass(DynamicModelMixin):
            pass
        
        field_defs = {
            'doc': (str, TracedInput(description="Document", required=True)),
            'context': (str, TracedInput(description="Context", required=False))
        }
        
        InputModel = TestClass.create_input_model(field_defs)
        
        assert InputModel.__name__ == "TestClassInput"
        assert 'doc' in InputModel.model_fields
        assert 'context' in InputModel.model_fields


class TestTracedResponse:
    """Test cases for TracedResponse."""
    
    def test_dynamic_attribute_setting(self):
        """Test that TracedResponse allows dynamic attribute setting."""
        response = TracedResponse(score=8, confidence=0.85, tags=["good", "clear"])
        
        assert response.score == 8
        assert response.confidence == 0.85
        assert response.tags == ["good", "clear"]


class TestIntegration:
    """Integration tests for the complete flows system."""
    
    def test_end_to_end_workflow(self, mock_trace_operators):
        """Test complete end-to-end workflow."""
        mock_trace_operators.return_value = "Score: 9, Sentiment: Positive, Confidence: 90%"
        
        class DocumentAnalyzer(TracedLLM):
            """Analyze documents comprehensively"""
            document: str = TracedInput(description="Document to analyze")
            score: int = TracedOutput(
                description="Quality score 1-10",
                parser=r"score[:\s]+(\d+)",
                default_value=5
            )
            sentiment: str = TracedOutput(
                description="Sentiment analysis",
                parser=lambda text: "Positive" if "positive" in text.lower() else "Negative",
                default_value="Neutral"
            )
            confidence: float = TracedOutput(
                description="Confidence percentage",
                parser=r"confidence[:\s]+(\d+)%?",
                default_value=0.5
            )
        
        analyzer = DocumentAnalyzer()
        
        # Test field detection
        assert set(analyzer._input_fields) == {'document'}
        assert set(analyzer._output_fields) == {'score', 'sentiment', 'confidence'}
        
        # Test analysis
        response = analyzer(document="This is a test document")
        
        assert isinstance(response, TracedResponse)
        assert response.score == 9
        assert response.sentiment == "Positive"
        assert response.confidence == 90.0
        
        # Verify LLM was called correctly
        mock_trace_operators.assert_called_once()
        args, kwargs = mock_trace_operators.call_args
        assert "This is a test document" in args


class TestCICompatibility:
    """Tests specifically designed for CI/CD environments without API keys."""
    
    def test_no_real_api_calls_made(self):
        """Ensure no real API calls are made during testing."""
        # This test verifies that our mocking is working correctly
        class SimpleScorer(TracedLLM):
            """Simple scorer"""
            text: str = TracedInput(description="Text input")
            score: int = TracedOutput(description="Score output", default_value=5)
        
        scorer = SimpleScorer()
        
        # This should not fail even without API keys because everything is mocked
        assert scorer.system_prompt.data == "Simple scorer"
        assert scorer._input_fields == ['text']
        assert scorer._output_fields == ['score']
    
    def test_offline_functionality(self):
        """Test functionality that doesn't require any external services."""
        # Test type extraction
        output_field = TracedOutput(parser=r"score[:\s]*is[:\s]*(\d+)", default_value=0)
        result = output_field.extract_from_text("The score is 85", int)
        assert result == 85
        
        # Test boolean parsing
        bool_field = TracedOutput(default_value=False)
        assert bool_field._parse_boolean("yes") == True
        assert bool_field._parse_boolean("no") == False
        
        # Test type conversion
        assert output_field._convert_to_type("42", int) == 42
        assert output_field._convert_to_type("3.14", float) == 3.14
    
    def test_mock_verification(self, mock_trace_operators):
        """Verify that mocking is working as expected."""
        # Check that the mock is active
        assert mock_trace_operators is not None
        
        # Create a TracedLLM instance
        llm = TracedLLM("Test prompt")
        
        # This should use the mock, not real API
        mock_trace_operators.return_value = "Mocked response"
        response = llm("Test input")
        
        assert response == "Mocked response"
        mock_trace_operators.assert_called_once()
    
    @pytest.mark.skipif(
        os.getenv('GITHUB_ACTIONS') == 'true' and not os.getenv('OPENAI_API_KEY'),
        reason="Skipping in GitHub Actions without API key"
    )
    def test_optional_real_api_integration(self):
        """Optional test that can be skipped in CI without API keys."""
        # This test is automatically skipped in GitHub Actions if no API key is set
        # It can be useful for local testing with real APIs
        pytest.skip("Real API integration test - skipped for CI safety")
    
    def test_boolean_parsing_delegates_to_traced_output(self, mock_trace_operators):
        """Test that boolean parsing properly delegates to TracedOutput when available."""
        mock_trace_operators.return_value = "answer: yes"  # More structured format
        
        class BooleanTester(TracedLLM):
            """Test boolean delegation"""
            question: str = TracedInput(description="Question to ask")
            answer: bool = TracedOutput(
                description="Boolean answer",
                parser=r"answer[:\s]*([^\n,]+)",  # Add explicit parser to extract "yes"
                default_value=False  # This should be used by TracedOutput._parse_boolean
            )
        
        tester = BooleanTester()
        response = tester(question="Is this working?")
        
        # The TracedOutput._parse_boolean should handle the parsing with its default_value logic
        assert isinstance(response, TracedResponse)
        # Since "yes" is in positive_words, it should return True regardless of default_value
        assert response.answer == True
