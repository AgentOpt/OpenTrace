"""Types for opto flows."""
from typing import List, Dict, Union
from pydantic import BaseModel, model_validator
from typing import Any, Optional, Callable, Dict, Union, Type, List
from dataclasses import dataclass
import re
import json
from opto.optimizers.utils import encode_image_to_base64, encode_numpy_to_base64
from opto import trace

class TraceObject:
    def __str__(self):
        # Any subclass that inherits this will be friendly to the optimizer
        raise NotImplementedError("Subclasses must implement __str__")


class MultiModalPayload(BaseModel):
    """
    A payload for multimodal content, particularly images.
    
    Supports three types of image inputs:
    1. URL (string starting with 'http://' or 'https://')
    2. Local file path (string path to image file)
    3. Numpy array (RGB image array)
    """
    image_bytes: Optional[str] = None  # Can be URL or base64-encoded data URL

    @classmethod
    def from_path(cls, path: str) -> "MultiModalPayload":
        """Create a payload by loading an image from a local file path."""
        data_url = encode_image_to_base64(path)
        return cls(image_bytes=data_url)
    
    @classmethod
    def from_url(cls, url: str) -> "MultiModalPayload":
        """Create a payload from an image URL."""
        return cls(image_bytes=url)
    
    @classmethod
    def from_array(cls, array: Any, format: str = "PNG") -> "MultiModalPayload":
        """Create a payload from a numpy array or array-like RGB image."""
        data_url = encode_numpy_to_base64(array, format=format)
        return cls(image_bytes=data_url)

    def load_image(self, path: str) -> None:
        """Mutate the current payload to include a new image from a file path."""
        self.image_bytes = encode_image_to_base64(path)
    
    def set_image(self, image: Union[str, Any], format: str = "PNG") -> None:
        """
        Set the image from various input formats.
        
        Args:
            image: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Local file path (string)
                - Numpy array or array-like RGB image
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
        """
        if isinstance(image, str):
            # Check if it's a URL
            if image.startswith('http://') or image.startswith('https://'):
                # Direct URL - litellm supports this
                self.image_bytes = image
            else:
                # Assume it's a local file path
                self.image_bytes = encode_image_to_base64(image)
        else:
            # Assume it's a numpy array or array-like object
            self.image_bytes = encode_numpy_to_base64(image, format=format)
    
    def get_content_block(self) -> Optional[Dict[str, Any]]:
        """
        Get the content block for the image in litellm format.
        
        Returns:
            Dict with format: {"type": "image_url", "image_url": {"url": ...}}
            or None if no image data is set
        """
        if self.image_bytes is None:
            return None
        
        return {
            "type": "image_url",
            "image_url": {
                "url": self.image_bytes
            }
        }

class QueryModel(BaseModel):
    # Expose "query" as already-normalized: always a List[Dict[str, Any]]
    query: List[Dict[str, Any]]
    multimodal_payload: Optional[MultiModalPayload] = None

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any):
        """
        Accepts:
          { "query": "hello" }
          { "query": "hello", "multimodal_payload": {"image_bytes": "..."} }
        And always produces:
          { "query": [ {text block}, maybe {image_url block} ], "multimodal_payload": ...}
        """
        if not isinstance(data, dict):
            raise TypeError("QueryModel input must be a dict")

        raw_query: Any = data.get("query")
        if isinstance(raw_query, trace.Node):
            assert isinstance(raw_query.data, (str, list)), "If using trace.Node, its data must be str"
            raw_query = raw_query.data

        # 1) Start with the text part
        if isinstance(raw_query, str):
            out: List[Dict[str, Any]] = [{"type": "text", "text": raw_query}]
        elif isinstance(raw_query, list):
            # Normalize each element in the list
            out = []
            for item in raw_query:
                if isinstance(item, str):
                    out.append({"type": "text", "text": item})
                elif isinstance(item, dict):
                    out.append(item)
                else:
                    raise TypeError("Elements of `query` list must be str or dict")
        else:
            raise TypeError("`query` must be a string or list")

        # 2) If we have an image, append an image block
        payload = data.get("multimodal_payload")
        image_bytes: Optional[str] = None
        if payload is not None:
            if isinstance(payload, dict):
                image_bytes = payload.get("image_bytes")
            else:
                # Could be already-parsed MultiModalPayload
                image_bytes = getattr(payload, "image_bytes", None)

        if image_bytes:
            out = out + [{
                "type": "image_url",
                "image_url": {"url": image_bytes}
            }]

        # 3) Write back normalized fields
        data["query"] = out
        return data
