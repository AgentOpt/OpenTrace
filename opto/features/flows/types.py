"""Types for opto flows."""
from typing import List, Dict, Union
from pydantic import BaseModel, model_validator
from typing import Any, Optional, Callable, Dict, Union, Type, List
from dataclasses import dataclass
import re
import json
from opto.optimizers.utils import encode_image_to_base64


class TraceObject:
    def __str__(self):
        # Any subclass that inherits this will be friendly to the optimizer
        raise NotImplementedError("Subclasses must implement __str__")


class MultiModalPayload(BaseModel):
    image_bytes: Optional[str] = None  # base64-encoded data URL

    @classmethod
    def from_path(cls, path: str) -> "MultiModalPayload":
        """Create a payload by loading an image from a local file path."""
        data_url = encode_image_to_base64(path)
        return cls(image_bytes=data_url)

    def load_image(self, path: str) -> None:
        """Mutate the current payload to include a new image."""
        self.image_bytes = encode_image_to_base64(path)

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

        raw_query: str = data.get("query")

        # 1) Start with the text part
        if isinstance(raw_query, str):
            out: List[Dict[str, Any]] = [{"type": "text", "text": raw_query}]
        else:
            raise TypeError("`query` must be a string or a list of dicts")

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
