"""Types for opto flows."""
from pydantic import BaseModel, Field, create_model, ConfigDict
from typing import Any, Optional, Callable, Dict, Union, Type, List
import re
import json

class TraceObject:
    def __str__(self):
        # Any subclass that inherits this will be friendly to the optimizer
        raise NotImplementedError("Subclasses must implement __str__")