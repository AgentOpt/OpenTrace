"""Types for opto flows."""
from pydantic import BaseModel, Field, create_model, ConfigDict
from typing import Any, Optional, Callable, Dict, Union, Type, List
import re
import json


class TracedInput(BaseModel):
    """Pydantic model for input fields in TracedLLM inheritance pattern."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    description: Optional[str] = "Input specified by the user for the LLM."
    required: bool = True


class TracedOutput(BaseModel):
    """Pydantic model for output fields in TracedLLM inheritance pattern."""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    description: Optional[str] = "Output from the LLM."
    parser: Optional[Union[Callable[[str], Any], str]] = None  # Can be function or regex pattern
    default_value: Optional[Any] = None
    required: bool = True
    
    def extract_from_text(self, text: str, field_type: Type) -> Any:
        """Extract value from text using parser (function or regex) or default logic."""
        if self.parser:
            if callable(self.parser):
                # Parser is a function
                try:
                    return self.parser(text)
                except:
                    return self.default_value
            elif isinstance(self.parser, str):
                # Parser is a regex pattern
                match = re.search(self.parser, text, re.IGNORECASE)
                if match:
                    # Find the first non-None group or use group(0)
                    extracted = None
                    for group in match.groups():
                        if group is not None:
                            extracted = group
                            break
                    if extracted is None:
                        extracted = match.group(0)
                    return self._convert_to_type(extracted, field_type)
                else:
                    return self.default_value
        
        # Fall back to default extraction logic
        return self._default_extract(text, field_type)
    
    def _convert_to_type(self, value: str, field_type: Type) -> Any:
        """Convert extracted string to target type."""
        # Default type conversion
        if field_type == int:
            numbers = re.findall(r'-?\d+', value)
            return int(numbers[0]) if numbers else self.default_value
        elif field_type == float:
            numbers = re.findall(r'-?\d+\.?\d*', value)
            return float(numbers[0]) if numbers else self.default_value
        elif field_type == bool:
            return self._parse_boolean(value)
        elif field_type == list:
            try:
                return json.loads(value)
            except:
                return [item.strip() for item in value.split(',')]
        else:
            return value
    
    def _default_extract(self, text: str, field_type: Type) -> Any:
        """Default extraction logic."""
        # If custom parser failed, return default value
        return self.default_value
    
    def _parse_boolean(self, text: str) -> bool:
        """Parse boolean from text."""
        text_lower = text.lower().strip()
        positive_words = ['true', 'yes', 'correct', 'positive', 'definitely', '1']
        negative_words = ['false', 'no', 'incorrect', 'negative', 'way', '0']
        
        if any(word in text_lower for word in positive_words):
            return True
        elif any(word in text_lower for word in negative_words):
            return False
        else:
            return self.default_value if self.default_value is not None else True


class DynamicModelMixin:
    """Mixin to provide dynamic model creation capabilities."""
    
    @classmethod
    def create_response_model(cls, field_definitions: Dict[str, tuple]) -> Type[BaseModel]:
        """Create a dynamic Pydantic model from field definitions.
        
        Args:
            field_definitions: Dict mapping field names to (type, TracedOutput) tuples
        
        Returns:
            Dynamically created Pydantic model class
        """
        pydantic_fields = {}
        
        for field_name, (field_type, traced_output) in field_definitions.items():
            # Create Pydantic field with metadata from TracedOutput
            field_kwargs = {}
            if traced_output.description:
                field_kwargs['description'] = traced_output.description
            if not traced_output.required:
                field_kwargs['default'] = traced_output.default_value
            
            pydantic_fields[field_name] = (field_type, Field(**field_kwargs))
        
        # Create the dynamic model
        return create_model(f"{cls.__name__}Response", **pydantic_fields)
    
    @classmethod 
    def create_input_model(cls, field_definitions: Dict[str, tuple]) -> Type[BaseModel]:
        """Create a dynamic input validation model."""
        pydantic_fields = {}
        
        for field_name, (field_type, traced_input) in field_definitions.items():
            field_kwargs = {}
            if traced_input.description:
                field_kwargs['description'] = traced_input.description
            if not traced_input.required:
                field_kwargs['default'] = None
                
            pydantic_fields[field_name] = (field_type, Field(**field_kwargs))
        
        return create_model(f"{cls.__name__}Input", **pydantic_fields)
