import pydantic
from pydantic import BaseModel, ValidationError, Field, create_model
import opto.trace as trace
from typing import Union, get_type_hints, Any, Dict
from opto.utils.llm import AbstractModel, LLM
from opto.flows.types import TracedInput, TracedOutput, DynamicModelMixin
import inspect
import json
import re


"""
TracedLLM:
1. special operations that supports specifying inputs (system_prompt, user_prompt) to LLM and parsing of outputs, wrap
   everything under one command.
2. Easy to use interface -- can be inherited by users.

Usage patterns:

1. Direct use: (only supports single input, single output) (signature: str -> str)
llm = TracedLLM("You are a helpful assistant.")
response = llm("Hello, what's the weather in France today?")

2. Inheritance:
class Scorer(TracedLLM):
   "This is a class that scores the response from LLM"
   doc: str = TracedInput(description="The document to score")
   score: int = TracedOutput(description="The score of the document")

scorer = Scorer()  # if a system prompt is passed in here, it will override the docstring.
response = scorer(doc="The response is ...")
print(response.score)
"""


class TracedResponse:
    """Dynamic response object that holds output field values."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class StructuredInputOutputMixin:
    """Mixin providing structured input/output parsing capabilities for TracedLLM."""
    
    def _detect_fields(self):
        """Detect TracedInput and TracedOutput fields from class annotations and defaults."""
        # Get type hints to extract the actual types
        type_hints = get_type_hints(self.__class__)
        
        # Look at class attributes and their default values
        for attr_name in dir(self.__class__):
            if not attr_name.startswith('_'):  # Skip private attributes
                attr_value = getattr(self.__class__, attr_name, None)
                if isinstance(attr_value, TracedInput):
                    self._input_fields.append(attr_name)
                    # Store the type annotation for this field
                    self._field_types[attr_name] = type_hints.get(attr_name, str)
                elif isinstance(attr_value, TracedOutput):
                    self._output_fields.append(attr_name)
                    # Store the type annotation for this field
                    self._field_types[attr_name] = type_hints.get(attr_name, str)
    
    def _create_dynamic_response_model(self) -> type[BaseModel]:
        """
        Create a dynamic Pydantic model for parsing LLM responses. We avoid creating an explicit signature by creating a dynamic model instead.
        The only disadvantage is nested-type parsing can be slightly more difficult, but that level of flexibility + nested LLM usage is rare and not a primary
        use case for Trace.
        """
        # Create field definitions for create_model
        field_definitions = {}
        
        for field_name in self._output_fields:
            field_type = self._field_types.get(field_name, str)
            # Get the description from the TracedOutput instance
            traced_output = getattr(self.__class__, field_name, None)
            description = getattr(traced_output, 'description', None) if traced_output else None
            
            # Create field definition tuple: (type, Field(...))
            field_definitions[field_name] = (field_type, Field(description=description))
        
        # Use Pydantic's create_model for dynamic model creation
        ResponseModel = create_model(
            f"{self.__class__.__name__}Response",
            **field_definitions
        )
        
        return ResponseModel
    
    # TODO: rewrite this part
    # TODO: 1. append at the end of the system prompt about generation instructions. XML based format with Markdown.
    # TODO: 2. extract by XML, put into a JSON string (allow nested XML parsing, such that the fields/response model can actually be nested)
    # TODO: 3. use the dynamic ResponseModel to do the parsing
    def _extract_structured_data(self, llm_response: str) -> Dict[str, Any]:
        """Extract structured data from LLM response - delegates to TracedOutput instances."""
        # Strategy 1: Try to parse as JSON if it looks like JSON
        llm_response_stripped = llm_response.strip()
        if llm_response_stripped.startswith('{') and llm_response_stripped.endswith('}'):
            try:
                json_data = json.loads(llm_response_stripped)
                # Validate that all fields are expected
                validated_data = {}
                for field_name, value in json_data.items():
                    if field_name in self._output_fields:
                        validated_data[field_name] = value
                    else:
                        print(f"Warning: Unexpected field '{field_name}' in JSON response, ignoring")
                return validated_data
            except json.JSONDecodeError:
                pass
        
        # Strategy 2: Delegate to each TracedOutput instance for parsing
        extracted_data = {}
        
        for field_name in self._output_fields:
            # Get the TracedOutput class variable
            traced_output = getattr(self.__class__, field_name, None)
            
            if traced_output and isinstance(traced_output, TracedOutput):
                # Delegate parsing to the TracedOutput instance
                field_type = self._field_types.get(field_name, str)
                try:
                    value = traced_output.extract_from_text(llm_response, field_type)
                    if value is not None:
                        extracted_data[field_name] = value
                except Exception as e:
                    print(f"Warning: Failed to extract field '{field_name}': {e}")
            else:
                print(f"Warning: Field '{field_name}' not properly defined as TracedOutput, ignoring")
        
        return extracted_data
    
    def _process_structured_inputs(self, **kwargs) -> TracedResponse:
        """Process structured inputs and return structured output with Pydantic parsing."""
        # Validate that all required input fields are provided
        missing_fields = [field for field in self._input_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required input field(s): {missing_fields}")
        
        # For now, use the first input field value as the user prompt
        # This will be expanded later with proper parsing/formatting
        user_prompt = kwargs[self._input_fields[0]]
        llm_response = self._call_llm(user_prompt)
        
        # Extract structured data from LLM response
        extracted_data = self._extract_structured_data(llm_response)
        
        # Create dynamic Pydantic model for validation
        ResponseModel = self._create_dynamic_response_model()
        
        try:
            # Use Pydantic to validate and parse the extracted data
            validated_response = ResponseModel(**extracted_data)
            
            # Convert to TracedResponse
            response_data = validated_response.model_dump()
            
        except ValidationError as e:
            # If Pydantic validation fails, include error info
            response_data = {}
            for output_field in self._output_fields:
                # Try to get individual field values, fall back to raw response
                response_data[output_field] = extracted_data.get(output_field, llm_response)
            
            response_data['_validation_errors'] = [str(error) for error in e.errors()]
            response_data['_raw_response'] = llm_response
        
        except Exception as e:
            # If extraction fails completely, return raw response
            response_data = {}
            for output_field in self._output_fields:
                response_data[output_field] = llm_response
            response_data['_extraction_error'] = str(e)
            response_data['_raw_response'] = llm_response
        
        return TracedResponse(**response_data)
    

@trace.model
class TracedLLM(StructuredInputOutputMixin, DynamicModelMixin):
    def __init__(self,
                 system_prompt: Union[str, None, trace.Node] = None,
                 llm: AbstractModel = None):
        """Initialize TracedLLM with a system prompt.

        Args:
            system_prompt: The system prompt to use for LLM calls. If None and the class has a docstring, the docstring will be used.
            llm: The LLM model to use for inference
        """
        # Use class docstring as system prompt if none provided
        if system_prompt is None:
            class_docstring = self.__class__.__doc__
            if class_docstring and class_docstring.strip():
                system_prompt = class_docstring.strip()
        
        self.system_prompt = trace.node(system_prompt)
        if llm is None:
            llm = LLM()
        assert isinstance(llm, AbstractModel), f"{llm} must be an instance of AbstractModel"
        self.llm = llm
        
        # Initialize fields for structured input/output
        self._input_fields = []
        self._output_fields = []
        self._field_types = {}  # Store type annotations for each field
        self._detect_fields()
    
    def forward(self, *args, **kwargs) -> Union[str, TracedResponse]:
        """Main function that handles both direct call and inheritance patterns.
        
        Args:
            *args: For direct pattern - single string argument
            **kwargs: For inheritance pattern - named input fields
            
        Returns:
            str: For direct pattern
            TracedResponse: For inheritance pattern with structured output fields
        """
        if self._input_fields:
            # Inheritance pattern: use named arguments
            return self._process_structured_inputs(**kwargs)
        else:
            # Direct pattern: single string argument
            if len(args) == 1 and isinstance(args[0], str):
                return self._call_llm(args[0])
            else:
                raise ValueError("Direct usage requires a single string argument")
    
    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM with user prompt and system prompt."""
        return trace.operators.call_llm(self.llm, self.system_prompt, user_prompt)
