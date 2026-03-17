"""
Flexible conversation manager for multi-turn LLM conversations.
Uses LiteLLM unified format for all providers (OpenAI, Anthropic, Google, etc.).

The class here follows this philosophy:
1. Every class is a data class (pickable/jsonable)
2. Most classes have `autocast` feature that takes in some data form and tries to automatically determine how to parse them into the right structured format. 

In order to support three types of data class construction methods:
1. Direct construction: `text = TextContent("Hello, world!")`
2. Build from a value: `text = TextContent.build("Hello, world!")`
3. Data class construction: `text = TextContent(text="Hello, world!")`

We use this approach:
`autocast()` method is the main automatic conversion method that determines how to parse the data. 
It will return a sequence of values that map to the fields of the data class.

In `__init__()` method, if `kwargs` are provided, we follow path 3 to construct the data class.
If not, we do autocast to construct the data class (path 1)

Alternatively, people can call `.build()` to construct the class. 
"""
from typing import List, Dict, Any, Optional, Literal, Union, Iterable, Tuple, TypeVar, Generic
from dataclasses import dataclass, field
import json
import base64
from pathlib import Path
import warnings

from PIL import Image
import io


# Default placeholder for images that cannot be rendered as text
DEFAULT_IMAGE_PLACEHOLDER = "\n[IMAGE]\n"

@dataclass
class ContentBlock:
    """Abstract base class for all content blocks."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the content block to a dictionary representation.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the content block
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def build(cls, value: Any, **kwargs) -> 'ContentBlock':
        """Build a content block from a value with auto-detection.
        
        Args:
            value: The value to build from (type depends on subclass)
            **kwargs: Additional keyword arguments for building
        
        Returns:
            ContentBlock: The built content block
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def is_empty(self) -> bool:
        """Check if the content block is empty (has no meaningful content).
        
        Returns:
            bool: True if the block is empty, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")

class ContentBlockList(list):
    """List of content blocks with automatic type conversion.
    
    Supports automatic conversion from:
    - str -> [TextContent(text=str)]
    - TextContent -> [TextContent]
    - ImageContent -> [ImageContent]
    - List[ContentBlock] -> ContentBlockList
    - None/empty -> []
    
    Note: This list can contain mixed types of ContentBlocks (text, images, PDFs, etc.).
    Type annotations like ContentBlockList[TextContent] are used for documentation
    purposes in specialized methods but don't restrict the actual content.
    """

    def __init__(self, content: Union[str, 'ContentBlock', List['ContentBlock'], None] = None):
        """Initialize ContentBlockList with automatic type conversion.
        
        Args:
            content: Can be a string (converted to TextContent), a single ContentBlock,
                    a list of ContentBlocks, or None (empty list).
        """
        super().__init__()
        if content is not None:
            self.extend(self._normalize(content))
    
    @staticmethod
    def _normalize(content: Union[str, 'ContentBlock', List['ContentBlock'], None]) -> List['ContentBlock']:
        """Normalize content to a list of ContentBlocks."""
        if content is None:
            return []
        if isinstance(content, str):
            return [TextContent(text=content)] if content else []
        if isinstance(content, list):
            return content
        # Single ContentBlock
        return [content]
    
    @classmethod
    def ensure(cls, content: Union[str, 'ContentBlock', List['ContentBlock'], None]) -> 'ContentBlockList':
        """Ensure content is a ContentBlockList with automatic conversion.
        
        Args:
            content: String, ContentBlock, list of ContentBlocks, or None
            
        Returns:
            ContentBlockList with the content
        """
        if isinstance(content, cls):
            return content
        return cls(content)
    
    def __getitem__(self, key: Union[int, slice]) -> Union['ContentBlock', 'ContentBlockList']:
        """Support indexing and slicing.
        
        Args:
            key: Integer index or slice object
            
        Returns:
            ContentBlock for single index, ContentBlockList for slices
        """
        if isinstance(key, slice):
            # Return a new ContentBlockList with the sliced items
            return ContentBlockList(list.__getitem__(self, key))
        else:
            # Return the single item for integer index
            return list.__getitem__(self, key)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": "list", "blocks": [b.to_dict() for b in self]}
    
    def append(self, item: Union[str, 'ContentBlock', 'ContentBlockList']) -> 'ContentBlockList':
        """Append a string or ContentBlock, merging consecutive text.
        
        Args:
            item: String (auto-converted to TextContent) or ContentBlock.
                  If the last item is TextContent and item is also text,
                  they are merged into a single TextContent.
        """
        if isinstance(item, str):
            # String: merge with last TextContent or create new one (with a separation mark " ")
            if self and isinstance(self[-1], TextContent):
                self[-1] = TextContent(text=self[-1].text + " " + item)
            else:
                super().append(TextContent(text=item))
        elif isinstance(item, TextContent):
            # TextContent: merge with last TextContent or add (with a separation mark " ")
            if self and isinstance(self[-1], TextContent):
                self[-1] = TextContent(text=self[-1].text + " " + item.text)
            else:
                super().append(item)
        elif isinstance(item, ContentBlockList):
            # we silently call extend here
            super().extend(item)
        else:
            # Other ContentBlock types (ImageContent, etc.): just add
            super().append(item)
        return self
    
    def extend(self, blocks: Union[str, 'ContentBlock', List[
        'ContentBlock'], 'ContentBlockList', None]) -> 'ContentBlockList':
        """Extend with blocks, merging consecutive TextContent.
        
        Args:
            blocks: String, ContentBlock, list of ContentBlocks, or None.
                    Strings are auto-converted. Consecutive text is merged.
        """
        normalized = self._normalize(blocks)
        for block in normalized:
            self.append(block)
        return self
    
    def __add__(self, other) -> 'ContentBlockList':
        """Concatenate content block lists with other content block lists or strings.
        
        Args:
            other: ContentBlockList, List[ContentBlock], or string to concatenate
        """
        if isinstance(other, (ContentBlockList, list)):
            result = ContentBlockList(list(self))
            result.extend(other)
            return result
        elif isinstance(other, str):
            result = ContentBlockList(list(self))
            result.append(TextContent(text=other))
            return result
        else:
            return NotImplemented
    
    def __radd__(self, other) -> 'ContentBlockList':
        """Right-side concatenation (when string is on the left).
        """
        if isinstance(other, str):
            result = ContentBlockList([TextContent(text=other)])
            result.extend(self)
            return result
        else:
            return NotImplemented

    def is_empty(self) -> bool:
        """Check if the content block list is empty."""
        if len(self) == 0:
            return True
        return all(block.is_empty() for block in self)
    
    def has_images(self) -> bool:
        """Check if the content block list contains any images."""
        return any(isinstance(block, ImageContent) for block in self)

    def has_text(self) -> bool:
        """Check if the content block list contains any text."""
        return any(isinstance(block, TextContent) for block in self)

    # --- Multimodal utilities ---
    @staticmethod
    def blocks_to_text(blocks: Iterable['ContentBlock'],
                       image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER) -> str:
        """Convert any iterable of ContentBlocks to text representation.
        
        This is a utility that can be used by composite classes containing
        multiple ContentBlockLists. Handles nested ContentBlockLists recursively.
        
        Args:
            blocks: Iterable of ContentBlock objects (may include nested ContentBlockLists)
            image_placeholder: Placeholder string for images (default: "[IMAGE]")
            
        Returns:
            str: Text representation where images are replaced with placeholder.
        """
        text_parts = []
        for block in blocks:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ImageContent):
                text_parts.append(image_placeholder)
            elif isinstance(block, ContentBlockList):
                # Recursively handle nested ContentBlockList
                nested_text = ContentBlockList.blocks_to_text(block, image_placeholder)
                if nested_text:
                    text_parts.append(nested_text)
        return " ".join(text_parts)
        
    def to_text(self, image_placeholder: str = DEFAULT_IMAGE_PLACEHOLDER) -> str:
        """Convert this list to text representation.
        
        Args:
            image_placeholder: Placeholder string for images (default: "[IMAGE]")
            
        Returns:
            str: Text representation where images are replaced with placeholder.
        """
        return self.blocks_to_text(self, image_placeholder)
    
    def __bool__(self) -> bool:
        """Check if there's any actual content (not just empty text).
        
        Returns:
            bool: True if content is non-empty (has images or non-whitespace text).
        """
        for block in self:
            if isinstance(block, ImageContent):
                return True
            if isinstance(block, TextContent) and block.text.strip():
                return True
        return False
    
    def __repr__(self) -> str:
        """Return text-only representation for logging.
        
        Images are represented as "[IMAGE]" placeholder.
        
        Returns:
            str: Text representation of the content.
        """
        return self.to_text()
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        try:
            from opto.utils.display.jupyter import render_content_block_list
            return render_content_block_list(self)
        except ImportError:
            # Fallback to text representation if display module unavailable
            return None
    
    def to_content_blocks(self) -> 'ContentBlockList':
        """Return self (for interface compatibility with composites).
        
        This allows ContentBlockList and classes that inherit from it
        to be used interchangeably with composite classes that have
        a to_content_blocks() method.
        
        Returns:
            ContentBlockList: Self reference.
        """
        return self
    
    def count_blocks(self) -> Dict[str, int]:
        """Count blocks by type, including nested structures.
        
        Recursively traverses the content block structure and counts
        each block type by its class name.
        
        Returns:
            Dict[str, int]: Dictionary mapping block class names to counts.
                           Example: {"TextContent": 3, "ImageContent": 1}
        """
        counts: Dict[str, int] = {}
        
        def _count_recursive(item: Any) -> None:
            """Recursively count blocks in nested structures."""
            if isinstance(item, ContentBlock):
                # Count this block
                class_name = item.__class__.__name__
                counts[class_name] = counts.get(class_name, 0) + 1
                
                # Check if this block has any attributes that might contain nested blocks
                if hasattr(item, '__dict__'):
                    for attr_value in item.__dict__.values():
                        if isinstance(attr_value, (ContentBlockList, list)):
                            for nested_item in attr_value:
                                _count_recursive(nested_item)
                        elif isinstance(attr_value, ContentBlock):
                            _count_recursive(attr_value)
            elif isinstance(item, (ContentBlockList, list)):
                # Recursively count items in lists
                for nested_item in item:
                    _count_recursive(nested_item)
        
        # Count all blocks in this list
        for block in self:
            _count_recursive(block)
        
        return counts
    
    def to_litellm_format(self, role: Optional[str] = None) -> List[Dict[str, Any]]:
        """Convert content blocks to LiteLLM Response API format.
        
        Args:
            role: Optional role context ("user" or "assistant") to determine the correct type.
                  If not provided, defaults to "user" for backward compatibility.
        
        Returns:
            List[Dict[str, Any]]: List of content block dictionaries in Response API format
        """
        if role is None:
            role = "user"
        
        content = []
        for block in self:
            # Skip empty content blocks
            if block.is_empty():
                continue
            
            # Handle different content block types
            if isinstance(block, TextContent):
                # Pass role context to TextContent for proper type selection
                content.append(block.to_litellm_format(role=role))
            elif isinstance(block, ImageContent):
                # ImageContent always uses input_image for user messages
                content.append(block.to_litellm_format())
            elif isinstance(block, PDFContent):
                # LiteLLM supports PDFs for providers like Claude
                # Use input_file type with PDF data URL for Response API
                if block.pdf_url:
                    warnings.warn("PDF URLs may not be supported by all providers through LiteLLM")
                    content.append({"type": "input_text", "text": f"[PDF: {block.pdf_url}]"})
                else:
                    # Encode as data URL for providers that support PDFs
                    data_url = f"data:application/pdf;base64,{block.pdf_data}"
                    content.append({"type": "input_file", "input_file": {"url": data_url}})
            elif isinstance(block, FileContent):
                # For file content, add as text or data URL based on type
                if block.is_binary:
                    data_url = f"data:{block.mime_type};base64,{block.file_data}"
                    content.append({"type": "input_file", "input_file": {"url": data_url}})
                else:
                    content.append({"type": "input_text", "text": f"[File: {block.filename}]\n{block.file_data}"})
            elif hasattr(block, 'to_litellm_format'):
                # Fallback: use block's own to_litellm_format method
                content.append(block.to_litellm_format())
            else:
                # Last resort: use to_dict()
                content.append(block.to_dict())
        
        return content


class Content(ContentBlockList):
    """Semantic wrapper providing multi-modal content for the optimizer agent.

    The goal is to provide a flexible interface for user to add mixed text and image content to the optimizer agent.

    Inherits all ContentBlockList functionality (append, extend, has_images,
    to_text, __bool__, __repr__, etc.) with a flexible constructor that
    supports multiple input patterns.

    Primary use cases:
    - Building problem context for the optimizer agent
    - Providing user feedback

    Creation patterns:
    - Variadic: Content("text", image, "more text")
    - Template: Content("See [IMAGE] here", images=[img])
    - Empty: Content()

    Examples:
        # Text-only content
        ctx = Content("Important background information")

        # Image content
        ctx = Content(ImageContent.build("diagram.png"))

        # Mixed content (variadic mode)
        ctx = Content(
            "Here's the diagram:",
            "diagram.png",  # auto-detected as image file
            "And the analysis."
        )

        # Template mode with placeholders
        ctx = Content(
            "Compare [IMAGE] with [IMAGE]:",
            images=[img1, img2]
        )

        # Manual building
        ctx = Content()
        ctx.append("Here's the relevant diagram:")
        ctx.append(ImageContent.build("diagram.png"))
    """

    def __init__(
            self,
            *args,
            images: Optional[List[Any]] = None,
            format: str = "PNG"
    ):
        """Initialize a Content from various input patterns.

        Supports two usage modes:

        **Mode 1: Variadic (images=None)**
        Pass any mix of text and image sources as arguments.
        Strings are auto-detected as text or image paths/URLs.

            Content("Hello", some_image, "World")
            Content("Check this:", "path/to/image.png")

        **Mode 2: Template (images provided)**
        Pass a template string with [IMAGE] placeholders and a list of images.

            Content(
                "Compare [IMAGE] with [IMAGE]",
                images=[img1, img2]
            )

        Args:
            *args: Variable arguments - text strings and/or image sources (Mode 1),
                   or a single template string (Mode 2)
            images: Optional list of images for template mode. When provided,
                    expects exactly one template string in args.
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG

        Raises:
            ValueError: In template mode, if placeholder count doesn't match image count,
                       or if args is not a single template string.
        """
        # Initialize empty list first
        super().__init__()

        # Build content based on mode
        if images is not None:
            if len(args) != 1 or not isinstance(args[0], str):
                raise ValueError(
                    "Template mode requires exactly one template string as the first argument. "
                    f"Got {len(args)} arguments."
                )
            self._build_from_template(args[0], images=images, format=format)
        elif args:
            self._build_from_variadic(*args)

    def _build_from_variadic(self, *args) -> None:
        """Populate self from variadic arguments.

        Each argument is either text (str) or an image source.
        Strings are auto-detected: if they look like image paths/URLs,
        they're converted to ImageContent; otherwise treated as text.

        Args:
            *args: Alternating text and image sources
            format: Image format for numpy arrays
        """
        for arg in args:
            # for Future expansion, we can check if the string is any special content type
            # by is_empty() on special ContentBlock subclasses
            image_content = ImageContent.build(arg)
            if not image_content.is_empty():
                self.append(image_content)
            else:
                self.append(arg)

    def _build_from_template(
            self,
            template: str,
            images: List[Any],
            format: str = "PNG"
    ) -> None:
        """Populate self from template with [IMAGE] placeholders.

        The template string contains [IMAGE] placeholders that are replaced
        by images from the images list in order.

        Args:
            template: Template string containing [IMAGE] placeholders
            images: List of image sources to insert at placeholders
            format: Image format for numpy arrays

        Raises:
            ValueError: If placeholder count doesn't match the number of images.
        """
        placeholder = DEFAULT_IMAGE_PLACEHOLDER

        # Count placeholders
        placeholder_count = template.count(placeholder)
        if placeholder_count != len(images):
            raise ValueError(
                f"Number of {placeholder} placeholders ({placeholder_count}) "
                f"does not match number of images ({len(images)})"
            )

        # Split template by placeholder and interleave with images
        parts = template.split(placeholder)

        for i, part in enumerate(parts):
            if part:  # Add text part if non-empty
                self.append(part)

            # Add image after each part except the last
            if i < len(images):
                image_content = ImageContent.build(images[i], format=format)
                if image_content is None:
                    raise ValueError(
                        f"Could not convert image at index {i} to ImageContent: {type(images[i])}"
                    )
                self.append(image_content)


class PromptTemplate:
    """Template for building ContentBlockLists with {placeholder} support.
    
    Similar to str.format(), but supports multimodal content (ContentBlockList).
    
    Return type depends on values:
    - All strings → returns str (backward compatible)
    - Any multimodal content → returns ContentBlockList
    
    Features:
    - Multiple placeholders: {a}, {b}, {c}
    - Escaping: {{ and }} for literal braces
    - Missing placeholders: left as-is in text
    - Extra kwargs: silently ignored (no error)
    - Nested templates: if value is PromptTemplate, formats it first
    - Mixed values: str, ContentBlockList, or objects with to_content_blocks()
    
    Examples:
        # Define template (can be class attribute)
        user_prompt_template = PromptTemplate('''
        Now you see problem instance:

        ================================
        {problem_instance}
        ================================
        ''')

        # Format with ContentBlockList (may contain images)
        content = user_prompt_template.format(
            problem_instance=problem.to_content_blocks()
        )
        # Returns ContentBlockList: [TextContent("Now you see..."), *problem_blocks, TextContent("===...")]

        # Multiple placeholders
        template = PromptTemplate("User: {user}\\nAssistant: {assistant}")
        result = template.format(user=user_blocks, assistant=assistant_blocks)

        # Nested templates
        outer = PromptTemplate("Header\\n{body}\\nFooter")
        inner = PromptTemplate("Content: {data}")
        result = outer.format(body=inner, data="some data")  # inner gets same kwargs

        # Escaping braces
        template = PromptTemplate('JSON example: {{"key": "{value}"}}')
        result = template.format(value="hello")  # {"key": "hello"}
        
        # Extra kwargs are ignored (no error)
        result = template.format(value="hello", unused_key="ignored")
        
        # Missing placeholders left as-is
        template = PromptTemplate("Hello {name}, score: {score}")
        result = template.format(name="Alice")  # "Hello Alice, score: {score}"
    """
    
    # Regex to find {placeholder} but not {{ or }}
    _PLACEHOLDER_PATTERN = None  # Lazy compiled
    
    def __init__(self, template: str):
        """Initialize with a template string.
        
        Args:
            template: Template string with {placeholder} syntax.
        """
        self.template = template
    
    @classmethod
    def _get_pattern(cls):
        """Lazily compile the placeholder regex pattern."""
        if cls._PLACEHOLDER_PATTERN is None:
            import re
            # Match {name} but not {{ or }}
            # Captures the placeholder name
            cls._PLACEHOLDER_PATTERN = re.compile(r'\{(\w+)\}')
        return cls._PLACEHOLDER_PATTERN
    
    def format(self, **kwargs) -> Union[str, 'ContentBlockList']:
        """Format the template with the given values.
        
        Similar to str.format(), but supports multimodal content.
        Extra kwargs are silently ignored.
        
        If all values are strings, returns a str (backward compatible).
        If any value is a ContentBlockList or multimodal, returns ContentBlockList.
        
        Args:
            **kwargs: Placeholder values. Each value can be:
                - str: inserted as text
                - ContentBlockList: blocks spliced in at that position
                - PromptTemplate: formatted first, then spliced in
                - Object with to_content_blocks(): method called, result spliced
                - Other: converted to str
        
        Returns:
            str: If all values are strings (backward compatible behavior).
            ContentBlockList: If any value is multimodal content.
        """
        # Check if all values are simple strings - if so, use simple string formatting
        pattern = self._get_pattern()
        placeholder_names = set(pattern.findall(self.template))
        
        # Only check values for placeholders that exist in the template
        relevant_values = {k: v for k, v in kwargs.items() if k in placeholder_names}
        
        if all(isinstance(v, str) for v in relevant_values.values()):
            # All strings: use simple string replacement, return str
            # Handle escaping and missing placeholders
            result = self.template.replace("{{", "\x00LBRACE\x00").replace("}}", "\x00RBRACE\x00")
            
            for name in placeholder_names:
                placeholder = "{" + name + "}"
                if name in kwargs:
                    result = result.replace(placeholder, kwargs[name])
                # Missing placeholders left as-is
            
            result = result.replace("\x00LBRACE\x00", "{").replace("\x00RBRACE\x00", "}")
            return result
        
        # Multimodal content: build ContentBlockList
        result = ContentBlockList()
        
        # Handle escaping: replace {{ with a sentinel, }} with another
        LBRACE_SENTINEL = "\x00LBRACE\x00"
        RBRACE_SENTINEL = "\x00RBRACE\x00"
        
        text = self.template.replace("{{", LBRACE_SENTINEL).replace("}}", RBRACE_SENTINEL)
        
        last_end = 0
        
        for match in pattern.finditer(text):
            # Add text before this placeholder
            prefix = text[last_end:match.start()]
            if prefix:
                # Restore escaped braces in prefix
                prefix = prefix.replace(LBRACE_SENTINEL, "{").replace(RBRACE_SENTINEL, "}")
                result.append(prefix)
            
            # Get placeholder name and value
            placeholder_name = match.group(1)
            
            if placeholder_name in kwargs:
                value = kwargs[placeholder_name]
                # Convert value to ContentBlockList and splice in
                content = self._value_to_content(value, **kwargs)
                result.extend(content)
            else:
                # Missing placeholder: leave as-is (restore original {name})
                result.append("{" + placeholder_name + "}")
            
            last_end = match.end()
        
        # Add remaining text after last placeholder
        suffix = text[last_end:]
        if suffix:
            suffix = suffix.replace(LBRACE_SENTINEL, "{").replace(RBRACE_SENTINEL, "}")
            result.append(suffix)
        
        return result
    
    def _value_to_content(self, value, **kwargs) -> 'ContentBlockList':
        """Convert a value to ContentBlockList.
        
        Args:
            value: The value to convert
            **kwargs: Passed to nested PromptTemplate.render()
        
        Returns:
            ContentBlockList: The value as content blocks.
        """
        if isinstance(value, ContentBlockList):
            return value
        elif isinstance(value, PromptTemplate):
            # Nested template: format it with the same kwargs
            return value.format(**kwargs)
        elif hasattr(value, 'to_content_blocks'):
            # Object with to_content_blocks method (e.g., ProblemInstance)
            return value.to_content_blocks()
        elif isinstance(value, str):
            return ContentBlockList(value)
        else:
            # Fallback: convert to string
            return ContentBlockList(str(value))
    
    def __repr__(self) -> str:
        """Return a preview of the template."""
        preview = self.template[:50] + "..." if len(self.template) > 50 else self.template
        return f"PromptTemplate({preview!r})"


@dataclass
class TextContent(ContentBlock):
    """Text content block"""
    type: Literal["text"] = "text"
    text: str = ""

    def __init__(self, text: str = ""):
        super().__init__(text=text)

    def is_empty(self) -> bool:
        """Check if the text content is empty."""
        return not self.text

    @classmethod
    def build(cls, value: Any = "", **kwargs) -> 'TextContent':
        """Build a text content block from a value.
        
        Args:
            value: String or any value to convert to text
            **kwargs: Unused, for compatibility with base class
        
        Returns:
            TextContent: Text content block with the value as text
        """
        if isinstance(value, str):
            return cls(text=value)
        return cls(text=str(value))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {"type": self.type, "text": self.text}
    
    def to_litellm_format(self, role: str = "user") -> Dict[str, Any]:
        """Convert to LiteLLM/OpenAI Response API compatible format.
        
        Args:
            role: The role context ("user" or "assistant") to determine the correct type
        
        Returns dict in format: 
            - {"type": "input_text", "text": "..."} for user messages
            - {"type": "output_text", "text": "..."} for assistant messages
        """
        text_type = "input_text" if role == "user" else "output_text"
        return {"type": text_type, "text": self.text}
    
    def __add__(self, other) -> 'TextContent':
        """Concatenate text content with strings or other TextContent objects.
        
        Args:
            other: String or TextContent to concatenate
            
        Returns:
            TextContent: New TextContent with concatenated text
        """
        if isinstance(other, str):
            return TextContent(text=self.text + " " + other)
        elif isinstance(other, TextContent):
            return TextContent(text=self.text + " " + other.text)
        else:
            return NotImplemented
    
    def __radd__(self, other) -> 'TextContent':
        """Right-side concatenation (when string is on the left).
        
        Args:
            other: String to concatenate
            
        Returns:
            TextContent: New TextContent with concatenated text
        """
        if isinstance(other, str):
            return TextContent(text=other + " " + self.text)
        else:
            return NotImplemented


@dataclass
class ImageContent(ContentBlock):
    """Image content block - supports URLs, base64, file paths, and numpy arrays.

    OpenAI uses base64 encoded images in the image_data field and recombine it into a base64 string of the format `"image_url": f"data:image/jpeg;base64,{base64_image}"` when sending to the API.
    Gemini uses raw bytes in the image_bytes field:
    ```
    types.Part.from_bytes(
        data=image_bytes,
        mime_type='image/jpeg',
      )
    ```
    
    Supports multiple ways to create an ImageContent:
    1. Direct instantiation with image_url or image_data
    2. from_file/from_path: Load from local file path
    3. from_url: Create from HTTP/HTTPS URL
    4. from_array: Create from numpy array or array-like RGB image
    5. from_value: Auto-detect and create from various formats
    """
    type: Literal["image"] = "image"
    image_url: Optional[str] = None
    image_data: Optional[str] = None  # base64 encoded
    image_bytes: Optional[bytes] = None
    media_type: str = "image/jpeg"  # image/jpeg, image/png, image/gif, image/webp
    detail: Optional[str] = None  # OpenAI: "auto", "low", "high"

    def __init__(self, value: Any = None, format: str = "PNG", **kwargs):
        """Initialize ImageContentBlock with auto-detection of input type.
        
        Args:
            value: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Data URL string (starting with 'data:image/')
                - Local file path (string)
                - Numpy array or array-like RGB image
                - PIL Image object
                - Raw bytes
                - None (empty image)
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
            **kwargs: Direct field values (image_url, image_data, media_type, detail)
        """
        # If explicit field values are provided, use them directly
        if kwargs:
            kwargs.setdefault('type', 'image')
            kwargs.setdefault('media_type', 'image/jpeg')
            super().__init__(**kwargs)
        else:
            # Use autocast to detect and convert the value
            value_dict = self.autocast(value, format=format)
            super().__init__(**value_dict)

    def __str__(self) -> str:
        # Truncate image_data and image_bytes for readability
        image_data_str = f"{self.image_data[:10]}..." if self.image_data and len(self.image_data) > 10 else self.image_data
        image_bytes_str = f"{str(self.image_bytes[:10])}..." if self.image_bytes and len(self.image_bytes) > 10 else self.image_bytes
        return f"ImageContent(image_url={self.image_url}, image_data={image_data_str}, image_bytes={image_bytes_str}, media_type={self.media_type})"
    
    def __repr__(self) -> str:
        # Truncate image_data and image_bytes for readability
        image_data_str = f"{self.image_data[:10]}..." if self.image_data and len(self.image_data) > 10 else self.image_data
        image_bytes_str = f"{str(self.image_bytes[:10])}..." if self.image_bytes and len(self.image_bytes) > 10 else self.image_bytes
        return f"ImageContent(image_url={self.image_url}, image_data={image_data_str}, image_bytes={image_bytes_str}, media_type={self.media_type})"

    def is_empty(self) -> bool:
        """Check if the image content is empty (no URL or data)."""
        return not self.image_url and not self.image_data and not self.image_bytes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (not LiteLLM format).
        
        For LiteLLM format, use to_litellm_format() instead.
        """
        result = {
            "type": self.type,
            "media_type": self.media_type
        }
        if self.image_url:
            result["image_url"] = self.image_url
        if self.image_data:
            result["image_data"] = self.image_data
        if self.image_bytes:
            result["image_bytes"] = self.image_bytes
        if self.detail:
            result["detail"] = self.detail
        return result
    
    def to_litellm_format(self) -> Dict[str, Any]:
        """Convert to LiteLLM Response API compatible format.
        
        Returns dict in format:
        {"type": "input_image", "image_url": {"url": "..."}}
        """
        # Determine the URL to use
        if self.image_url:
            url = self.image_url
        elif self.image_data:
            # Convert base64 data to data URL
            url = f"data:{self.media_type};base64,{self.image_data}"
        elif self.image_bytes:
            # Convert bytes to base64 and then to data URL
            import base64
            b64_data = base64.b64encode(self.image_bytes).decode('utf-8')
            url = f"data:{self.media_type};base64,{b64_data}"
        else:
            # Empty image
            return {"type": "input_image", "image_url": ""}
        
        # Build the result in Response API format
        result = {
            "type": "input_image",
            "image_url": url
        }
        
        # Add detail if specified (OpenAI-specific)
        if self.detail:
            result["detail"] = self.detail
            
        return result

    @classmethod
    def from_file(cls, filepath: str, media_type: Optional[str] = None):
        """Load image from file path."""
        path = Path(filepath)
        if not media_type:
            ext_to_type = {
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.png': 'image/png',
                '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = ext_to_type.get(path.suffix.lower(), 'image/jpeg')

        with open(filepath, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        return cls(image_data=image_data, media_type=media_type)

    @classmethod
    def from_path(cls, filepath: str, media_type: Optional[str] = None):
        """Load image from file path. Alias for from_file."""
        return cls.from_file(filepath, media_type)

    @classmethod
    def from_url(cls, url: str, media_type: str = "image/jpeg"):
        """Create ImageContent from an HTTP/HTTPS URL.
        
        Args:
            url: HTTP or HTTPS URL pointing to an image
            media_type: MIME type of the image (default: image/jpeg)
        """
        return cls(image_url=url, media_type=media_type)

    @classmethod
    def from_array(cls, array: Any, format: str = "PNG"):
        """Create ImageContent from a numpy array or array-like RGB image.
        
        Args:
            array: numpy array representing an image (H, W, C) with values in [0, 255] or [0, 1]
            format: Image format (PNG, JPEG, etc.). Default: PNG
        
        Returns:
            ImageContent with base64-encoded image data
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for from_array. Install with: pip install numpy")
        
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("Pillow is required for from_array. Install with: pip install Pillow")
        
        import io
        
        # Convert to numpy array if not already
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Normalize to [0, 255] if needed
        if array.dtype == np.float32 or array.dtype == np.float64:
            if array.max() <= 1.0:
                array = (array * 255).astype(np.uint8)
            else:
                array = array.astype(np.uint8)
        elif array.dtype != np.uint8:
            array = array.astype(np.uint8)
        
        # Convert to PIL Image and encode
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format=format.upper())
        buffer.seek(0)
        
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        media_type = f"image/{format.lower()}"
        
        return cls(image_data=image_data, media_type=media_type)

    @classmethod
    def from_pil(cls, image: Any, format: str = "PNG"):
        """Create ImageContent from a PIL Image.
        
        Args:
            image: PIL Image object
            format: Image format (PNG, JPEG, etc.). Default: PNG
        
        Returns:
            ImageContent with base64-encoded image data
        """
        import io
        
        buffer = io.BytesIO()
        img_format = image.format or format.upper()
        image.save(buffer, format=img_format)
        buffer.seek(0)
        
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        media_type = f"image/{img_format.lower()}"
        
        return cls(image_data=image_data, media_type=media_type)

    @classmethod
    def from_bytes(cls, data: bytes, media_type: str = "image/jpeg"):
        """Create ImageContent from raw image bytes.
        
        Args:
            data: Raw image bytes
            media_type: MIME type of the image (default: image/jpeg)
        
        Returns:
            ImageContent with base64-encoded data
        """
        image_data = base64.b64encode(data).decode('utf-8')
        return cls(image_data=image_data, media_type=media_type)

    @classmethod
    def from_base64(cls, b64_data: str, media_type: str = "image/jpeg"):
        """Create ImageContent from base64-encoded string.
        
        Args:
            b64_data: Base64-encoded image data (without data URL prefix)
            media_type: MIME type of the image (default: image/jpeg)
        
        Returns:
            ImageContent with the provided base64 data
        """
        return cls(image_data=b64_data, media_type=media_type)

    @classmethod
    def from_data_url(cls, data_url: str):
        """Create ImageContent from a data URL (data:image/...;base64,...).
        
        Args:
            data_url: Data URL string in format data:image/<type>;base64,<data>
        
        Returns:
            ImageContent with extracted base64 data and media type
        """
        try:
            header, b64_data = data_url.split(',', 1)
            media_type = header.split(':')[1].split(';')[0]  # e.g., "image/png"
            return cls(image_data=b64_data, media_type=media_type)
        except (ValueError, IndexError):
            # Fallback: assume the whole thing is base64 data
            return cls(image_data=data_url.split(',')[-1], media_type="image/jpeg")

    @staticmethod
    def autocast(value: Any, format: str = "PNG") -> Dict[str, Any]:
        """Auto-detect value type and return image field values.
        
        Args:
            value: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Data URL string (starting with 'data:image/')
                - Local file path (string)
                - Numpy array or array-like RGB image
                - PIL Image object
                - Raw bytes
                - None (empty image)
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
        
        Returns:
            Dictionary with keys: image_url, image_data, image_bytes, media_type
        """
        # Handle None or empty
        if value is None:
            return {"image_url": None, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}
        
        # Handle ImageContentBlock instance
        if isinstance(value, ImageContent):
            return {
                "image_url": value.image_url, 
                "image_data": value.image_data, 
                "image_bytes": value.image_bytes,
                "media_type": value.media_type
            }
        
        # Handle string inputs
        if isinstance(value, str):
            if not value.strip():
                return {"image_url": None, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}
            
            # Data URL
            if value.startswith('data:image/'):
                try:
                    header, b64_data = value.split(',', 1)
                    media_type = header.split(':')[1].split(';')[0]
                    return {"image_url": None, "image_data": b64_data, "image_bytes": None, "media_type": media_type}
                except (ValueError, IndexError):
                    return {"image_url": None, "image_data": value.split(',')[-1], "image_bytes": None, "media_type": "image/jpeg"}
            
            # HTTP/HTTPS URL
            if value.startswith('http://') or value.startswith('https://'):
                return {"image_url": value, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}
            
            # File path - only check if string is reasonable length (< 4096 chars)
            # Long strings are clearly not file paths and would cause OS errors
            if len(value) < 4096:
                path = Path(value)
                try:
                    if path.exists():
                        ext_to_type = {
                            '.jpg': 'image/jpeg',
                            '.jpeg': 'image/jpeg',
                            '.png': 'image/png',
                            '.gif': 'image/gif',
                            '.webp': 'image/webp'
                        }
                        media_type = ext_to_type.get(path.suffix.lower(), 'image/jpeg')
                        with open(value, 'rb') as f:
                            image_data = base64.b64encode(f.read()).decode('utf-8')
                        return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": media_type}
                except (OSError, IOError):
                    # Not a valid file path, continue to other checks
                    pass
                    
        # Handle bytes - store as base64 for portability
        if isinstance(value, bytes):
            image_data = base64.b64encode(value).decode('utf-8')
            return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": "image/jpeg"}
        
        # Handle PIL Image
        try:
            from PIL import Image
            if isinstance(value, Image.Image):
                import io
                buffer = io.BytesIO()
                img_format = value.format or format.upper()
                value.save(buffer, format=img_format)
                buffer.seek(0)
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                media_type = f"image/{img_format.lower()}"
                return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": media_type}
        except ImportError:
            pass
        
        # Handle numpy array or array-like
        try:
            import numpy as np
            if isinstance(value, np.ndarray) or hasattr(value, '__array__'):
                try:
                    from PIL import Image
                except ImportError:
                    raise ImportError("Pillow is required for array conversion. Install with: pip install Pillow")
                
                import io
                
                if not isinstance(value, np.ndarray):
                    value = np.array(value)
                
                # Normalize to [0, 255] if needed
                if value.dtype == np.float32 or value.dtype == np.float64:
                    if value.max() <= 1.0:
                        value = (value * 255).astype(np.uint8)
                    else:
                        value = value.astype(np.uint8)
                elif value.dtype != np.uint8:
                    value = value.astype(np.uint8)
                
                image = Image.fromarray(value)
                buffer = io.BytesIO()
                image.save(buffer, format=format.upper())
                buffer.seek(0)
                
                image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                media_type = f"image/{format.lower()}"
                return {"image_url": None, "image_data": image_data, "image_bytes": None, "media_type": media_type}
        except ImportError:
            pass
        
        return {"image_url": None, "image_data": None, "image_bytes": None, "media_type": "image/jpeg"}

    @classmethod
    def build(cls, value: Any, format: str = "PNG") -> 'ImageContent':
        """Auto-detect format and create ImageContent from various input types.
        
        Args:
            value: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Data URL string (starting with 'data:image/')
                - Local file path (string)
                - Numpy array or array-like RGB image
                - PIL Image object
                - Raw bytes
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
        
        Returns:
            ImageContent or None if the value cannot be converted
        """
        # Handle ImageContentBlock instance directly
        if isinstance(value, cls):
            return value
        
        value_dict = cls.autocast(value, format=format)
        return cls(**value_dict)

    def set_image(self, image: Any, format: str = "PNG") -> None:
        """Set the image from various input formats (mutates self).
        
        Args:
            image: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Data URL string (starting with 'data:image/')
                - Local file path (string)
                - Numpy array or array-like RGB image
                - PIL Image object
                - Raw bytes
            format: Image format for numpy arrays (PNG, JPEG, etc.). Default: PNG
        """
        result = ImageContent.build(image, format=format)
        if result:
            self.image_url = result.image_url
            self.image_data = result.image_data
            # Only copy image_bytes if it was explicitly set (e.g., from Google API)
            if result.image_bytes:
                self.image_bytes = result.image_bytes
            self.media_type = result.media_type

    def as_image(self) -> Image.Image:
        """Convert the image to a PIL Image.
        
        Fetches the image from URL if necessary (including HTTP/HTTPS URLs).
        
        Returns:
            PIL Image object
            
        Raises:
            ValueError: If no image data is available
            requests.RequestException: If fetching from URL fails
        """
        # Try to get image bytes from any available source
        image_bytes = self.get_bytes()
        
        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))
        elif self.image_url:
            if self.image_url.startswith(('http://', 'https://')):
                # Fetch image from URL
                try:
                    import requests
                    response = requests.get(self.image_url, timeout=30)
                    response.raise_for_status()
                    return Image.open(io.BytesIO(response.content))
                except ImportError:
                    # Fallback to urllib if requests is not available
                    from urllib.request import urlopen
                    with urlopen(self.image_url, timeout=30) as response:
                        return Image.open(io.BytesIO(response.read()))
            else:
                # If it's a local file path
                return Image.open(self.image_url)
        else:
            raise ValueError("No image data available to convert to PIL Image")

    def show(self) -> Image.Image:
        """A convenience alias for as_image()"""
        return self.as_image()
    
    def get_bytes(self) -> Optional[bytes]:
        """Get raw image bytes.
        
        Returns image_bytes if available, otherwise decodes image_data from base64.
        
        Returns:
            Raw image bytes or None if no image data available
        """
        if self.image_bytes:
            return self.image_bytes
        elif self.image_data:
            return base64.b64decode(self.image_data)
        return None
    
    def get_base64(self) -> Optional[str]:
        """Get base64-encoded image data.
        
        Returns image_data if available, otherwise encodes image_bytes to base64.
        
        Returns:
            Base64-encoded string or None if no image data available
        """
        if self.image_data:
            return self.image_data
        elif self.image_bytes:
            return base64.b64encode(self.image_bytes).decode('utf-8')
        return None
    
    def ensure_bytes(self) -> None:
        """Ensure image_bytes is populated (converts from image_data if needed)."""
        if not self.image_bytes and self.image_data:
            self.image_bytes = base64.b64decode(self.image_data)
    
    def ensure_base64(self) -> None:
        """Ensure image_data is populated (converts from image_bytes if needed)."""
        if not self.image_data and self.image_bytes:
            self.image_data = base64.b64encode(self.image_bytes).decode('utf-8')

@dataclass
class PDFContent(ContentBlock):
    """PDF content block"""
    type: Literal["pdf"] = "pdf"
    pdf_url: Optional[str] = None
    pdf_data: Optional[str] = None  # base64 encoded
    filename: Optional[str] = None

    def __post_init__(self):
        # Ensure type is always "pdf" (fixes issue when user passes positional arg)
        object.__setattr__(self, 'type', 'pdf')

    def is_empty(self) -> bool:
        """Check if the PDF content is empty (no URL or data)."""
        return not self.pdf_url and not self.pdf_data

    @classmethod
    def build(cls, value: Any, **kwargs) -> 'PDFContent':
        """Build a PDF content block from a value.
        
        Args:
            value: Can be:
                - URL string (starting with 'http://' or 'https://')
                - Local file path (string)
                - Raw bytes
            **kwargs: Unused, for compatibility with base class
        
        Returns:
            PDFContent or None if the value cannot be converted
        """
        if isinstance(value, str):
            # HTTP/HTTPS URL
            if value.startswith('http://') or value.startswith('https://'):
                return cls(pdf_url=value)
            # Assume it's a file path
            if Path(value).exists():
                return cls.from_file(value)
            return None
        
        # Handle bytes
        if isinstance(value, bytes):
            pdf_data = base64.b64encode(value).decode('utf-8')
            return cls(pdf_data=pdf_data)
        
        return None

    def to_dict(self) -> Dict[str, Any]:
        if self.pdf_url:
            return {
                "type": "document",
                "source": {"type": "url", "url": self.pdf_url},
                "filename": self.filename
            }
        else:
            return {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": self.pdf_data
                },
                "filename": self.filename
            }

    @classmethod
    def from_file(cls, filepath: str):
        """Load PDF from file"""
        path = Path(filepath)
        with open(filepath, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')

        return cls(pdf_data=pdf_data, filename=path.name)


@dataclass
class FileContent(ContentBlock):
    """Generic file content block (for code, data files, etc.)"""
    file_data: str  # Could be text content or base64 for binary
    filename: str
    type: Literal["file"] = "file"
    mime_type: str = "text/plain"
    is_binary: bool = False

    @classmethod
    def build(cls, value: Any, **kwargs) -> 'FileContent':
        """Build a file content block from a value.
        
        Args:
            value: Can be:
                - Local file path (string)
                - Tuple of (filename, content) where content is str or bytes
            **kwargs: Additional arguments like mime_type
        
        Returns:
            FileContent or None if the value cannot be converted
        """
        mime_type = kwargs.get('mime_type')
        
        if isinstance(value, str):
            # Assume it's a file path
            if Path(value).exists():
                return cls.from_file(value, mime_type=mime_type)
            return None
        
        # Handle tuple of (filename, content)
        if isinstance(value, tuple) and len(value) == 2:
            filename, content = value
            if isinstance(content, bytes):
                file_data = base64.b64encode(content).decode('utf-8')
                is_binary = True
            else:
                file_data = str(content)
                is_binary = False
            return cls(
                file_data=file_data,
                filename=filename,
                mime_type=mime_type or 'application/octet-stream',
                is_binary=is_binary
            )
        
        return None

    def is_empty(self) -> bool:
        """Check if the file content is empty (no data)."""
        return not self.file_data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "filename": self.filename,
            "mime_type": self.mime_type,
            "file_data": self.file_data,
            "is_binary": self.is_binary
        }

    @classmethod
    def from_file(cls, filepath: str, mime_type: Optional[str] = None):
        """Load file from disk"""
        path = Path(filepath)

        # Try to read as text first
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                file_data = f.read()
                is_binary = False
        except UnicodeDecodeError:
            # Fall back to binary
            with open(filepath, 'rb') as f:
                file_data = base64.b64encode(f.read()).decode('utf-8')
                is_binary = True

        if not mime_type:
            # Simple mime type detection
            ext_to_type = {
                '.py': 'text/x-python',
                '.js': 'text/javascript',
                '.json': 'application/json',
                '.csv': 'text/csv',
                '.txt': 'text/plain',
                '.md': 'text/markdown',
                '.html': 'text/html',
            }
            mime_type = ext_to_type.get(path.suffix.lower(), 'application/octet-stream')

        return cls(
            file_data=file_data,
            filename=path.name,
            mime_type=mime_type,
            is_binary=is_binary
        )

# Union type alias for common content types (for type hints)
# Note: ContentBlock remains the abstract base class for inheritance
ContentBlockUnion = Union[TextContent, ImageContent, PDFContent, FileContent]


@dataclass
class ToolCall(ContentBlock):
    """Represents a tool call made by the LLM"""
    id: str
    type: str  # "function", "web_search", etc.
    name: Optional[str] = None  # function name
    arguments: Optional[Dict[str, Any]] = None  # function arguments

    def is_empty(self) -> bool:
        """Check if the tool call is empty (no id)."""
        return not self.id

    def to_dict(self) -> Dict[str, Any]:
        result = {"id": self.id, "type": self.type}
        if self.name:
            result["name"] = self.name
        if self.arguments:
            result["arguments"] = self.arguments
        return result


@dataclass
class ToolResult(ContentBlock):
    """Represents the result of a tool execution"""
    tool_call_id: str
    content: str  # Result as string (can be JSON stringified)
    is_error: bool = False

    def is_empty(self) -> bool:
        """Check if the tool result is empty (no tool_call_id)."""
        return not self.tool_call_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
            "is_error": self.is_error
        }


@dataclass
class ToolDefinition(ContentBlock):
    """Defines a tool that the LLM can use"""
    type: str  # "function", "web_search", "file_search", etc.
    name: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: bool = False  # OpenAI strict mode
    # Provider-specific fields
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        """Check if the tool definition is empty (no type)."""
        return not self.type

    def to_dict(self) -> Dict[str, Any]:
        result = {"type": self.type}
        if self.name:
            result["name"] = self.name
        if self.description:
            result["description"] = self.description
        if self.parameters:
            result["parameters"] = self.parameters
        if self.strict:
            result["strict"] = self.strict
        result.update(self.extra)
        return result

@dataclass
class UserTurn:
    """Represents a user message turn in the conversation"""
    role: str = "user"

    content: ContentBlockList = field(default_factory=ContentBlockList)
    tools: List[ToolDefinition] = field(default_factory=list)

    # Provider-specific settings
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None

    # Metadata
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, content=None, tools=None, **kwargs):
        """
        Initialize UserTurn with content and tools.
        
        Four ways to initialize:
        1. Empty: UserTurn() - creates empty turn with defaults
        2. Copy: UserTurn(existing_turn) - creates a copy of an existing UserTurn
        3. Positional args: UserTurn(content, tools) - pass content and/or tools
        4. Keyword args: UserTurn(content=..., tools=..., temperature=...) - explicit fields
        
        Args:
            content: ContentBlockList, list of content blocks, UserTurn (for copying), or None
            tools: List of ToolDefinition or None
            **kwargs: Additional fields (temperature, max_tokens, top_p, timestamp, metadata)
        """
        self.output_contains_image = False

        # Handle copy constructor: UserTurn(existing_turn)
        if isinstance(content, UserTurn):
            source = content
            self.role = source.role
            self.content = ContentBlockList(source.content)  # Deep copy the content list
            self.tools = list(source.tools)  # Shallow copy the tools list
            self.temperature = source.temperature
            self.max_tokens = source.max_tokens
            self.top_p = source.top_p
            self.timestamp = source.timestamp
            self.metadata = dict(source.metadata)  # Copy the metadata dict
            return
        
        # Handle content
        if content is None:
            content = ContentBlockList()
        elif not isinstance(content, ContentBlockList):
            # If it's a list, wrap it in ContentBlockList
            content = ContentBlockList(content) if isinstance(content, list) else ContentBlockList([content])
        
        # Handle tools
        if tools is None:
            tools = []
        
        # Set all fields
        self.role = kwargs.get('role', "user")
        self.content = content
        self.tools = tools
        self.temperature = kwargs.get('temperature', None)
        self.max_tokens = kwargs.get('max_tokens', None)
        self.top_p = kwargs.get('top_p', None)
        self.timestamp = kwargs.get('timestamp', None)
        self.metadata = kwargs.get('metadata', {})

    def add_text(self, text: str) -> 'UserTurn':
        """Add text content"""
        self.content.append(TextContent(text=text))
        return self

    def add_image(self, url: Optional[str] = None, data: Optional[str] = None,
                  media_type: str = "image/jpeg") -> 'UserTurn':
        """Add image content"""
        self.content.append(ImageContent(
            image_url=url,
            image_data=data,
            media_type=media_type
        ))
        return self

    def add_image_file(self, filepath: str) -> 'UserTurn':
        """Add image from file"""
        self.content.append(ImageContent.from_file(filepath))
        return self

    def add_pdf(self, url: Optional[str] = None, data: Optional[str] = None) -> 'UserTurn':
        """Add PDF content"""
        self.content.append(PDFContent(pdf_url=url, pdf_data=data))
        return self

    def add_pdf_file(self, filepath: str) -> 'UserTurn':
        """Add PDF from file"""
        self.content.append(PDFContent.from_file(filepath))
        return self

    def add_file(self, filepath: str, mime_type: Optional[str] = None) -> 'UserTurn':
        """Add file from disk"""
        self.content.append(FileContent.from_file(filepath, mime_type))
        return self

    def add_tool(self, tool: ToolDefinition) -> 'UserTurn':
        """Add a tool definition"""
        self.tools.append(tool)
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": "user",
            "content": [c.to_dict() for c in self.content],
            "tools": [t.to_dict() for t in self.tools] if self.tools else None,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "metadata": self.metadata
        }

    def enable_image_generation(self):
        self.output_contains_image = True
    
    def __repr__(self) -> str:
        """Safe string representation that handles missing attributes."""
        content_preview = str(self.content)[:50] + "..." if len(str(self.content)) > 50 else str(self.content)
        parts = [f"UserTurn(content={content_preview!r}"]
        
        # Safely add optional fields if they exist
        tools = getattr(self, 'tools', [])
        if tools:
            parts.append(f", tools={len(tools)} tool(s)")
        temperature = getattr(self, 'temperature', None)
        if temperature is not None:
            parts.append(f", temperature={temperature}")
        
        parts.append(")")
        return "".join(parts)

    def to_litellm_format(self) -> Dict[str, Any]:
        """Convert to LiteLLM Response API format (OpenAI Response API compatible)"""
        return {
            "role": "user",
            "content": self.content.to_litellm_format(role="user")
        }
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks with glassmorphism design."""
        try:
            from opto.utils.display.jupyter import render_user_turn
            return render_user_turn(self)
        except ImportError:
            # Fallback to text representation if display module unavailable
            return None


@dataclass
class Turn:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


@dataclass
class AssistantTurn(Turn):
    """Represents an assistant message turn in the conversation"""
    role: str = "assistant"
    content: ContentBlockList = field(default_factory=ContentBlockList)

    # Tool usage (Option B: Everything in AssistantTurn)
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)

    # Provider-specific features
    reasoning: Optional[str] = None  # OpenAI reasoning/thinking
    finish_reason: Optional[str] = None  # "stop", "length", "tool_calls", etc.

    # Token usage
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # Metadata
    model: Optional[str] = None
    timestamp: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, *args, **kwargs):
        """
        Initialize AssistantTurn from a raw response or with explicit fields.
        
        Three ways to initialize:
        1. Empty: AssistantTurn() - creates empty turn with defaults
        2. From raw response: AssistantTurn(response) - autocasts the response
        3. With fields: AssistantTurn(role="assistant", content=[...]) - explicit fields
        """
        if len(args) == 1 and isinstance(args[0], AssistantTurn):
            # Case: Copy constructor - create a copy of another AssistantTurn
            other = args[0]
            super().__init__(
                role=other.role,
                content=ContentBlockList(other.content),
                tool_calls=list(other.tool_calls),
                tool_results=list(other.tool_results),
                reasoning=other.reasoning,
                finish_reason=other.finish_reason,
                prompt_tokens=other.prompt_tokens,
                completion_tokens=other.completion_tokens,
                model=other.model,
                timestamp=other.timestamp,
                metadata=dict(other.metadata)
            )
            return

        if len(args) > 0 and len(kwargs) == 0:
            # Case 2: Single positional arg - autocast from raw response
            value_dict = self.autocast(args[0])
            super().__init__(**value_dict)
        elif len(kwargs) > 0:
            # Case 3: Keyword arguments - use them directly
            super().__init__(**kwargs)
        else:
            # Case 1: No arguments - initialize with defaults
            super().__init__(
                role="assistant",
                content=ContentBlockList(),
                tool_calls=[],
                tool_results=[],
                reasoning=None,
                finish_reason=None,
                prompt_tokens=None,
                completion_tokens=None,
                model=None,
                timestamp=None,
                metadata={}
            )

    @staticmethod
    def from_google_genai(value: Any) -> Dict[str, Any]:
        """Parse a Google GenAI response into a dictionary of AssistantTurn fields.
        
        Supports both the legacy generate_content API and the new Interactions API.
        
        Args:
            value: Raw response from Google GenAI API
            
        Returns:
            Dict[str, Any]: Dictionary with keys corresponding to AssistantTurn fields
        """
        # Initialize the result dictionary with default values
        result = {
            "role": "assistant",
            "content": ContentBlockList(),
            "tool_calls": [],
            "tool_results": [],
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model": None,
            "timestamp": None,
            "metadata": {}
        }
        
        # Check if this is a normalized response (from our GoogleGenAILLM)
        if hasattr(value, 'raw_response'):
            raw_response = value.raw_response
        else:
            raw_response = value
        
        # Handle Interactions API format (new)
        if hasattr(raw_response, 'outputs'):
            # This is an Interaction object
            interaction = raw_response
            
            # Extract text from outputs
            if interaction.outputs and len(interaction.outputs) > 0:
                for output in interaction.outputs:
                    if hasattr(output, 'text') and output.text:
                        result["content"].append(TextContent(text=output.text))
                    # Handle other output types if they exist
                    elif hasattr(output, 'content'):
                        # Content could be a list of parts
                        if isinstance(output.content, list):
                            for part in output.content:
                                if hasattr(part, 'text') and part.text:
                                    result["content"].append(TextContent(text=part.text))
                        else:
                            result["content"].append(TextContent(text=str(output.content)))
            
            # Extract model info
            if hasattr(interaction, 'model'):
                result["model"] = interaction.model
            
            # Extract status as finish_reason
            if hasattr(interaction, 'status'):
                result["finish_reason"] = interaction.status
            
            # Extract token usage from Interactions API
            if hasattr(interaction, 'usage'):
                usage = interaction.usage
                if hasattr(usage, 'input_tokens'):
                    result["prompt_tokens"] = usage.input_tokens
                elif hasattr(usage, 'prompt_token_count'):
                    result["prompt_tokens"] = usage.prompt_token_count
                    
                if hasattr(usage, 'output_tokens'):
                    result["completion_tokens"] = usage.output_tokens
                elif hasattr(usage, 'candidates_token_count'):
                    result["completion_tokens"] = usage.candidates_token_count
            
            # Extract interaction ID as metadata
            if hasattr(interaction, 'id'):
                result["metadata"]['interaction_id'] = interaction.id
        
        # Handle legacy generate_content API format
        else:
            # Extract thinking/reasoning (for Gemini 2.5+ models)
            if hasattr(raw_response, 'thoughts') and raw_response.thoughts:
                # Gemini's thinking budget feature
                result["reasoning"] = str(raw_response.thoughts)
            
            # Extract model info
            if hasattr(raw_response, 'model_version'):
                result["model"] = raw_response.model_version
            
            # Extract token usage (if available)
            if hasattr(raw_response, 'usage_metadata'):
                usage = raw_response.usage_metadata
                if hasattr(usage, 'prompt_token_count'):
                    result["prompt_tokens"] = usage.prompt_token_count
                if hasattr(usage, 'candidates_token_count'):
                    result["completion_tokens"] = usage.candidates_token_count
            
            # Handle multimodal content from Gemini (candidates with parts)
            content_extracted = False
            if hasattr(raw_response, 'candidates') and raw_response.candidates:
                candidate = raw_response.candidates[0]
                
                # Extract from parts (supports multimodal responses with text and images)
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        # Handle text parts
                        if hasattr(part, 'text') and part.text:
                            result["content"].append(TextContent(text=part.text))
                            content_extracted = True
                        # Handle inline data (images, generated images, etc.)
                        elif hasattr(part, 'inline_data'):
                            # Try to extract image data, preferring direct inline_data access
                            inline = part.inline_data
                            image_bytes = None
                            image_data = None
                            media_type = 'image/jpeg'

                            
                            # Extract from inline_data Blob (most reliable method)
                            # Google's Blob.data should be raw bytes
                            if hasattr(inline, 'data'):
                                data = inline.data
                                # Check if it's bytes or string
                                if isinstance(data, bytes):
                                    # Store raw bytes for Gemini compatibility
                                    # (Gemini prefers raw bytes when sending images)
                                    image_bytes = data
                                elif isinstance(data, str):
                                    # Already base64-encoded string
                                    image_data = data
                                    # Don't decode to bytes - keep as base64 for portability
                            
                            if hasattr(inline, 'mime_type'):
                                media_type = inline.mime_type
                            
                            # If we got the data, create ImageContent
                            # Store image_bytes only if we got raw bytes from Google
                            if image_data or image_bytes:
                                result["content"].append(ImageContent(
                                    image_data=image_data,
                                    image_bytes=image_bytes if isinstance(data, bytes) else None,
                                    media_type=media_type
                                ))
                                content_extracted = True
                
                # Extract finish reason
                if hasattr(candidate, 'finish_reason'):
                    result["finish_reason"] = str(candidate.finish_reason)
            
            # Fallback: Extract simple text content if no candidates/parts were found
            if not content_extracted:
                if hasattr(raw_response, 'text'):
                    result["content"].append(TextContent(text=raw_response.text))
                elif hasattr(value, 'choices'):
                    # Fallback to normalized format
                    result["content"].append(TextContent(text=value.choices[0].message.content))
        
        return result
    
    @staticmethod
    def from_litellm_openai_response_api(value: Any) -> Dict[str, Any]:
        """Parse a LiteLLM/OpenAI-style response into a dictionary of AssistantTurn fields.
        
        Handles both formats:
        - New Responses API: Has 'output' field with ResponseOutputMessage objects
        - Legacy Completion API: Has 'choices' field with message objects
        
        Args:
            value: Response from LiteLLM/OpenAI API (Responses API or Completion API)
            
        Returns:
            Dict[str, Any]: Dictionary with keys corresponding to AssistantTurn fields
        """
        # Initialize the result dictionary with default values
        result = {
            "role": "assistant",
            "content": ContentBlockList(),
            "tool_calls": [],
            "tool_results": [],
            "reasoning": None,
            "finish_reason": None,
            "prompt_tokens": None,
            "completion_tokens": None,
            "model": None,
            "timestamp": None,
            "metadata": {}
        }
        
        # Handle Bedrock Converse API format (has 'output' field with 'message')
        # Check both attribute-based and dict-based access for robustness
        is_bedrock = False
        bedrock_output = None
        bedrock_value = value  # Keep reference to the original value for later access
        
        # Try attribute-based access first
        if hasattr(value, 'output'):
            output_val = value.output
            
            if hasattr(output_val, 'message'):
                is_bedrock = True
                bedrock_output = output_val
            # Also check dict-based access on the output attribute
            elif isinstance(output_val, dict) and 'message' in output_val:
                is_bedrock = True
                bedrock_output = output_val
        
        # If not found, try dict-based access on value itself
        if not is_bedrock and isinstance(value, dict) and 'output' in value:
            output_val = value['output']
            if isinstance(output_val, dict) and 'message' in output_val:
                is_bedrock = True
                bedrock_output = output_val
                bedrock_value = value  # Use the dict directly
        
        if is_bedrock and bedrock_output is not None:
            # Bedrock Converse API format detected
            # Get message with dict or attr access
            message = bedrock_output.get('message') if isinstance(bedrock_output, dict) else (bedrock_output.message if hasattr(bedrock_output, 'message') else None)
            
            if message:
                # Extract role
                if isinstance(message, dict):
                    result["role"] = message.get('role', 'assistant')
                elif hasattr(message, 'role'):
                    result["role"] = message.role
                
                # Extract content
                content_list = message.get('content') if isinstance(message, dict) else (message.content if hasattr(message, 'content') else None)
                
                if content_list:
                    for content_item in content_list:
                        # Handle text content (dict or attr)
                        text_val = None
                        if isinstance(content_item, dict):
                            text_val = content_item.get('text')
                        elif hasattr(content_item, 'text'):
                            text_val = content_item.text
                        
                        if text_val:
                            result["content"].append(TextContent(text=text_val))
            
            # Extract finish reason from stopReason (check both value and bedrock_value)
            stop_reason = None
            if isinstance(bedrock_value, dict):
                stop_reason = bedrock_value.get('stopReason')
            elif hasattr(bedrock_value, 'stopReason'):
                stop_reason = bedrock_value.stopReason
            if stop_reason:
                result["finish_reason"] = stop_reason
            
            # Extract token usage (check both value and bedrock_value)
            usage = None
            if isinstance(bedrock_value, dict):
                usage = bedrock_value.get('usage')
            elif hasattr(bedrock_value, 'usage'):
                usage = bedrock_value.usage
            
            if usage:
                if isinstance(usage, dict):
                    result["prompt_tokens"] = usage.get('inputTokens')
                    result["completion_tokens"] = usage.get('outputTokens')
                else:
                    if hasattr(usage, 'inputTokens'):
                        result["prompt_tokens"] = usage.inputTokens
                    if hasattr(usage, 'outputTokens'):
                        result["completion_tokens"] = usage.outputTokens
        
        # Handle Responses API format (new format with 'output' field)
        # The output field is a list of output items (messages, image generation calls, etc.)
        # NOTE: LiteLLM may set value.object to 'chat.completion' or 'response' depending on the provider
        elif hasattr(value, 'output') and hasattr(value, 'object'):
            # Extract metadata
            if hasattr(value, 'id'):
                result["metadata"]['response_id'] = value.id
            if hasattr(value, 'created_at'):
                result["timestamp"] = str(value.created_at)
            
            # Extract model info
            if hasattr(value, 'model'):
                result["model"] = value.model
            
            # Extract status as finish_reason
            if hasattr(value, 'status'):
                result["finish_reason"] = value.status
            
            # Extract content from output (list of output items)
            if value.output and len(value.output) > 0:
                for output_item in value.output:
                    # Handle ImageGenerationCall
                    if hasattr(output_item, 'type') and output_item.type == 'image_generation_call':
                        # Extract generated image
                        if hasattr(output_item, 'result') and output_item.result:
                            # Determine media type from output_format
                            media_type = 'image/jpeg'  # default
                            if hasattr(output_item, 'output_format'):
                                format_map = {
                                    'png': 'image/png',
                                    'jpeg': 'image/jpeg',
                                    'jpg': 'image/jpeg',
                                    'webp': 'image/webp',
                                    'gif': 'image/gif'
                                }
                                media_type = format_map.get(output_item.output_format.lower(), 'image/jpeg')
                            
                            # Add image to content
                            result["content"].append(ImageContent(
                                image_data=output_item.result,
                                media_type=media_type
                            ))
                            
                            # Store additional metadata about the image generation
                            if hasattr(output_item, 'revised_prompt') and output_item.revised_prompt:
                                if 'image_generation' not in result["metadata"]:
                                    result["metadata"]['image_generation'] = []
                                result["metadata"]['image_generation'].append({
                                    'id': output_item.id if hasattr(output_item, 'id') else None,
                                    'revised_prompt': output_item.revised_prompt,
                                    'size': output_item.size if hasattr(output_item, 'size') else None,
                                    'quality': output_item.quality if hasattr(output_item, 'quality') else None,
                                    'status': output_item.status if hasattr(output_item, 'status') else None
                                })
                    
                    # Handle ResponseOutputMessage
                    elif hasattr(output_item, 'type') and output_item.type == 'message':
                        # Extract role
                        if hasattr(output_item, 'role'):
                            result["role"] = output_item.role
                        
                        # Extract status for this message
                        if hasattr(output_item, 'status') and not result["finish_reason"]:
                            result["finish_reason"] = output_item.status
                        
                        # Extract content items
                        if hasattr(output_item, 'content') and output_item.content:
                            for content_item in output_item.content:
                                # Handle text content
                                if hasattr(content_item, 'type') and content_item.type == 'output_text':
                                    if hasattr(content_item, 'text') and content_item.text:
                                        result["content"].append(TextContent(text=content_item.text))
                                # Handle other content types as they become available
                                elif hasattr(content_item, 'text') and content_item.text:
                                    result["content"].append(TextContent(text=str(content_item.text)))
            
            # Extract reasoning (for models with reasoning capabilities)
            if hasattr(value, 'reasoning'):
                reasoning_parts = []
                if isinstance(value.reasoning, dict):
                    if value.reasoning.get('summary'):
                        reasoning_parts.append(f"Summary: {value.reasoning['summary']}")
                    if value.reasoning.get('effort'):
                        reasoning_parts.append(f"Effort: {value.reasoning['effort']}")
                    if reasoning_parts:
                        result["reasoning"] = "\n".join(reasoning_parts)
                elif value.reasoning:
                    result["reasoning"] = str(value.reasoning)
            
            # Extract token usage (Responses API format)
            if hasattr(value, 'usage'):
                if hasattr(value.usage, 'input_tokens'):
                    result["prompt_tokens"] = value.usage.input_tokens
                if hasattr(value.usage, 'output_tokens'):
                    result["completion_tokens"] = value.usage.output_tokens
        
        # Handle legacy Completion API format (has 'choices' field)
        elif hasattr(value, 'choices') and len(value.choices) > 0:
            choice = value.choices[0]
            message = choice.message if hasattr(choice, 'message') else choice
            
            # Extract text content
            if hasattr(message, 'content') and message.content:
                result["content"].append(TextContent(text=str(message.content)))
            
            # Extract tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tc in message.tool_calls:
                    tool_call = ToolCall(
                        id=tc.id if hasattr(tc, 'id') else None,
                        type=tc.type if hasattr(tc, 'type') else "function",
                        name=tc.function.name if hasattr(tc, 'function') else tc.name,
                        arguments=json.loads(tc.function.arguments) if hasattr(tc, 'function') and hasattr(tc.function, 'arguments') else {}
                    )
                    result["tool_calls"].append(tool_call)
            
            # Extract finish reason
            if hasattr(choice, 'finish_reason'):
                result["finish_reason"] = choice.finish_reason
            
            # Extract reasoning/thinking (for OpenAI o1/o3 models)
            if hasattr(message, 'reasoning') and message.reasoning:
                result["reasoning"] = message.reasoning
            
            # Extract token usage (Completion API format)
            if hasattr(value, 'usage'):
                if hasattr(value.usage, 'prompt_tokens'):
                    result["prompt_tokens"] = value.usage.prompt_tokens
                if hasattr(value.usage, 'completion_tokens'):
                    result["completion_tokens"] = value.usage.completion_tokens
            
            # Extract model info
            if hasattr(value, 'model'):
                result["model"] = value.model
        
        return result
    
    @staticmethod
    def autocast(value: Any) -> Dict[str, Any]:
        """Automatically parse a response from any API into a dictionary of AssistantTurn fields.
        
        Automatically detects the response format and uses the appropriate parser:
        - Google GenAI (generate_content or Interactions API)
        - LiteLLM/OpenAI Responses API (new format with 'output' field)
        - LiteLLM/OpenAI Completion API (legacy format with 'choices' field)
        
        Args:
            value: Raw response from any supported API
            
        Returns:
            Dict[str, Any]: Dictionary with keys corresponding to AssistantTurn fields
        """
        
        # Check if this is a normalized response (from our GoogleGenAILLM)
        raw_response = value.raw_response if hasattr(value, 'raw_response') else value
        
        # Detect Google GenAI format (Interactions API or generate_content)
        # Google GenAI has 'outputs' (Google Interactions API) or 'candidates' (generate_content)
        # Note: 'outputs' is for Google's Interactions API, 'output' is for LiteLLM Responses API
        if hasattr(raw_response, 'outputs') or \
           (hasattr(raw_response, 'candidates') and not hasattr(value, 'choices')) or \
           hasattr(raw_response, 'usage_metadata'):
            return AssistantTurn.from_google_genai(value)
        
        # Detect LiteLLM/OpenAI/Bedrock format (Responses API, Completion API, or Bedrock Converse)
        # Responses API has 'output' field and object='response'
        # Completion API has 'choices' field
        # Bedrock Converse API has 'output' field with nested 'message'
        # Check both attribute and dict-based access
        has_output = hasattr(value, 'output') or (isinstance(value, dict) and 'output' in value)
        has_choices = hasattr(value, 'choices') or (isinstance(value, dict) and 'choices' in value)
        
        if has_output or has_choices:
            return AssistantTurn.from_litellm_openai_response_api(value)
        
        # Fallback: if has 'text' attribute, might be a simple Google response
        elif hasattr(raw_response, 'text'):
            return AssistantTurn.from_google_genai(value)
        
        # Default to empty result if format is not recognized
        else:
            return {
                "role": "assistant",
                "content": ContentBlockList(),
                "tool_calls": [],
                "tool_results": [],
                "reasoning": None,
                "finish_reason": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "model": None,
                "timestamp": None,
                "metadata": {}
            }

    def add_text(self, text: str) -> 'AssistantTurn':
        """Add text content"""
        self.content.append(text)
        return self

    def add_image(self, url: Optional[str] = None, data: Optional[str] = None,
                  media_type: str = "image/jpeg") -> 'AssistantTurn':
        """Add image content (some models can generate images)"""
        self.content.append(ImageContent(
            image_url=url,
            image_data=data,
            media_type=media_type
        ))
        return self

    def add_tool_call(self, tool_call: ToolCall) -> 'AssistantTurn':
        """Add a tool call"""
        self.tool_calls.append(tool_call)
        return self

    def add_tool_result(self, result: ToolResult) -> 'AssistantTurn':
        """Add a tool result"""
        self.tool_results.append(result)
        return self

    def to_text(self) -> str:
        """Get all text content concatenated. Images will be presented as placeholder text."""
        return self.content.to_text()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "role": self.role,
            "content": [c.to_dict() for c in self.content],
            "tool_calls": [tc.to_dict() for tc in self.tool_calls] if self.tool_calls else None,
            "tool_results": [tr.to_dict() for tr in self.tool_results] if self.tool_results else None,
            "reasoning": self.reasoning,
            "finish_reason": self.finish_reason,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "model": self.model,
            "metadata": self.metadata
        }

    def get_text(self) -> ContentBlockList:
        """Get all text content blocks.
        
        Returns:
            ContentBlockList: List containing only TextContent blocks
        """
        text_blocks = ContentBlockList()
        for block in self.content:
            if isinstance(block, TextContent):
                text_blocks.append(block)
        return text_blocks
    
    def get_images(self) -> ContentBlockList:
        """Get all image content blocks.
        
        Returns:
            ContentBlockList: List containing only ImageContent blocks
        """
        image_blocks = ContentBlockList()
        for block in self.content:
            if isinstance(block, ImageContent):
                image_blocks.append(block)
        return image_blocks

    def __repr__(self) -> str:
        """Safe string representation that handles missing attributes."""
        content_preview = str(self.content)[:50] + "..." if len(str(self.content)) > 50 else str(self.content)
        parts = [f"AssistantTurn(content={content_preview!r}"]
        
        # Safely add optional fields if they exist
        if hasattr(self, 'model') and self.model:
            parts.append(f", model={self.model!r}")
        if hasattr(self, 'prompt_tokens') and self.prompt_tokens:
            parts.append(f", prompt_tokens={self.prompt_tokens}")
        if hasattr(self, 'completion_tokens') and self.completion_tokens:
            parts.append(f", completion_tokens={self.completion_tokens}")
        
        parts.append(")")
        return "".join(parts)
    
    def to_litellm_format(self) -> Dict[str, Any]:
        """Convert to LiteLLM Response API format (OpenAI Response API compatible)"""
        result = {"role": self.role}

        # Handle content blocks (text, images, etc.) - delegate to ContentBlockList
        result["content"] = self.content.to_litellm_format(role=self.role)

        if self.tool_calls:
            result["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments) if tc.arguments else "{}"
                    }
                }
                for tc in self.tool_calls
            ]

        return result
    
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks with glassmorphism design."""
        try:
            from opto.utils.display.jupyter import render_assistant_turn
            return render_assistant_turn(self)
        except ImportError:
            # Fallback to text representation if display module unavailable
            return None


@dataclass
class Chat:
    """Manages conversation history across multiple turns using LiteLLM unified format"""
    turns: List[Union[UserTurn, AssistantTurn]] = field(default_factory=list)
    system_prompt: Optional[str] = None
    protected_rounds: int = 0  # Initial rounds to never truncate (task definition)

    def add_user_turn(self, turn: Union[str, ContentBlockList, 'TextContent', 'ImageContent', 'Content', UserTurn]) -> 'Chat':
        """Add a user turn
        
        Args:
            turn: Can be:
                - str: Plain text message
                - ContentBlockList: List of content blocks
                - TextContent: Single text content block
                - ImageContent: Single image content block
                - Content: Multi-modal content wrapper
                - UserTurn: Complete user turn object
        
        Returns:
            Chat: Self for method chaining
            
        Raises:
            TypeError: If turn is not one of the accepted types
        """
        # Accept UserTurn directly
        if isinstance(turn, UserTurn):
            self.turns.append(turn)
            return self

        assert isinstance(
            turn, (str, ContentBlockList, TextContent, ImageContent, Content)
        ), "turn must be a string, ContentBlockList, TextContent, ImageContent, or Content object"
        user_turn = UserTurn(content=turn)
        self.turns.append(user_turn)
        return self

    def add_assistant_turn(self, turn: AssistantTurn) -> 'Chat':
        """Add an assistant turn. AssistantTurn parses the response from the LLM."""
        assert isinstance(turn, AssistantTurn), "turn must be an AssistantTurn object"
        self.turns.append(turn)
        return self

    def get_last_user_turn(self) -> Optional[UserTurn]:
        """Get the most recent user turn"""
        for turn in reversed(self.turns):
            if isinstance(turn, UserTurn):
                return turn
        return None

    def get_last_assistant_turn(self) -> Optional[AssistantTurn]:
        """Get the most recent assistant turn"""
        for turn in reversed(self.turns):
            if isinstance(turn, AssistantTurn):
                return turn
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        return {
            "system_prompt": self.system_prompt,
            "protected_rounds": self.protected_rounds,
            "turns": [turn.to_dict() for turn in self.turns]
        }

    def to_litellm_format(
        self, 
        n: int = -1,
        truncate_strategy: Literal["from_start", "from_end"] = "from_start",
        protected_rounds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert to LiteLLM messages format (OpenAI-compatible, works with all providers)
        
        Args:
            n: Number of historical rounds (user+assistant pairs) to include.
               -1 means all history (default: -1).
               The current (potentially incomplete) round is always included.
            truncate_strategy: How to truncate when n is specified:
                - "from_start": Remove oldest rounds, keep the most recent n rounds (default)
                - "from_end": Remove newest rounds, keep the oldest n rounds
            protected_rounds: Number of initial rounds to never truncate (task definition).
                If None, uses self.protected_rounds. These rounds count towards n, so
                if n=5 and protected_rounds=1, you get 1 protected + 4 truncatable rounds.
        
        Returns:
            List of message dictionaries in LiteLLM format
        """
        # Determine protected rounds
        n_protected = protected_rounds if protected_rounds is not None else self.protected_rounds
        protected_turns = n_protected * 2  # Each round = user + assistant
        
        # Apply truncation to turns
        if n == -1:
            selected_turns = self.turns
        else:
            # Protected rounds count towards N
            # So if N=5 and protected_rounds=1, we keep 1 protected + 4 from truncatable
            remaining_rounds = max(0, n - n_protected)
            
            # Split into protected and truncatable turns
            protected_part = self.turns[:protected_turns]
            truncatable_part = self.turns[protected_turns:]
            
            # remaining_rounds = number of rounds (pairs) from the truncatable part
            # Each round = 2 turns (user + assistant)
            # Plus include current incomplete round (if last turn is user, +1)
            has_incomplete_round = len(truncatable_part) > 0 and isinstance(truncatable_part[-1], UserTurn)
            n_turns = remaining_rounds * 2 + (1 if has_incomplete_round else 0)
            
            if truncate_strategy == "from_start":
                # Keep last n_turns from truncatable part (remove from start)
                truncated_part = truncatable_part[-n_turns:] if n_turns > 0 else []
            elif truncate_strategy == "from_end":
                # Keep first n_turns from truncatable part (remove from end)
                truncated_part = truncatable_part[:n_turns] if n_turns > 0 else []
            else:
                raise ValueError(f"Unknown truncate_strategy: {truncate_strategy}. Use 'from_start' or 'from_end'")
            
            # Combine protected + truncated
            selected_turns = protected_part + truncated_part
        
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        for turn in selected_turns:
            messages.append(turn.to_litellm_format())

            # Add tool results as separate messages in LiteLLM/OpenAI format
            if isinstance(turn, AssistantTurn) and turn.tool_results:
                for result in turn.tool_results:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": result.tool_call_id,
                        "content": result.content
                    })

        return messages
    
    def to_messages(
        self,
        n: int = -1,
        truncate_strategy: Literal["from_start", "from_end"] = "from_start",
        protected_rounds: Optional[int] = None,
        model_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Smart message format conversion that auto-detects the appropriate format.
        
        This method automatically chooses between Gemini format and LiteLLM format based on
        the model name. Detection priority:
        1. If model_name argument is provided and contains "gemini", uses Gemini format
        2. Otherwise, checks if any AssistantTurn has a model name containing "gemini"
        3. If no Gemini model detected, uses LiteLLM format (default)
        
        Note: This detection may not work for custom LLM backends with Gemini model names.
        In such cases, call to_gemini_format() or to_litellm_format() explicitly.
        
        Args:
            n: Number of historical rounds (user+assistant pairs) to include.
               -1 means all history (default: -1).
               The current (potentially incomplete) round is always included.
            truncate_strategy: How to truncate when n is specified:
                - "from_start": Remove oldest rounds, keep the most recent n rounds (default)
                - "from_end": Remove newest rounds, keep the oldest n rounds
            protected_rounds: Number of initial rounds to never truncate (task definition).
                If None, uses self.protected_rounds. Counts towards n.
            model_name: Optional model name to use for format detection. If provided and
                contains "gemini" (case-insensitive), forces Gemini format.
        
        Returns:
            List of message dictionaries in the appropriate format
        
        Example:
            # Automatically uses Gemini format if model is Gemini
            history = ConversationHistory()
            history.system_prompt = "You are helpful."
            history.add_user_turn(UserTurn().add_text("Hello"))
            
            # Force Gemini format by providing model name
            messages = history.to_messages(model_name="gemini-2.5-flash")
            
            # Or be explicit:
            messages = history.to_gemini_format()  # Force Gemini format
            messages = history.to_litellm_format()  # Force LiteLLM format
        """
        # Check if model_name argument indicates Gemini (highest priority)
        use_gemini_format = False
        if model_name and 'gemini' in model_name.lower():
            use_gemini_format = True
        else:
            # Check if any AssistantTurn has a Gemini model
            for turn in self.turns:
                if isinstance(turn, AssistantTurn) and turn.model:
                    if 'gemini' in turn.model.lower():
                        use_gemini_format = True
                        break
        
        # Use the appropriate format
        if use_gemini_format:
            return self.to_gemini_format(
                n=n,
                truncate_strategy=truncate_strategy,
                protected_rounds=protected_rounds
            )
        else:
            return self.to_litellm_format(
                n=n,
                truncate_strategy=truncate_strategy,
                protected_rounds=protected_rounds
            )
    
    def to_gemini_format(
        self,
        n: int = -1,
        truncate_strategy: Literal["from_start", "from_end"] = "from_start",
        protected_rounds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Convert to Google Gemini format (messages with 'model' role instead of 'assistant')
        
        This method converts the conversation history to a format compatible with Google's
        Gemini API. The main differences from LiteLLM format are:
        - Uses 'model' instead of 'assistant' for role names
        - Content is structured as 'parts' (list of text/image parts)
        - System message (if present) remains as first message with role='system'
        
        The GoogleGenAILLM class will extract the system message and convert it to
        system_instruction when making the API call.
        
        Args:
            n: Number of historical rounds (user+assistant pairs) to include.
               -1 means all history (default: -1).
               The current (potentially incomplete) round is always included.
            truncate_strategy: How to truncate when n is specified:
                - "from_start": Remove oldest rounds, keep the most recent n rounds (default)
                - "from_end": Remove newest rounds, keep the oldest n rounds
            protected_rounds: Number of initial rounds to never truncate (task definition).
                If None, uses self.protected_rounds. These rounds count towards n.
        
        Returns:
            List of message dictionaries in Gemini format with 'role' and 'parts'.
            System message (if present) is included as first message with role='system'.
        
        Example:
            from opto.utils.llm import LLM
            from opto.utils.backbone import ConversationHistory, UserTurn
            
            # Create conversation
            history = ConversationHistory()
            history.system_prompt = "You are a helpful assistant."
            history.add_user_turn(UserTurn().add_text("Hello!"))
            
            # Convert to Gemini format
            messages = history.to_gemini_format()
            
            # Use with GoogleGenAILLM
            llm = LLM(model="gemini-2.5-flash")
            response = llm(messages=messages)
        """
        # Get the LiteLLM format messages first (handles truncation logic)
        litellm_messages = self.to_litellm_format(
            n=n,
            truncate_strategy=truncate_strategy,
            protected_rounds=protected_rounds
        )
        
        # Convert messages to Google GenAI format
        gemini_messages = []
        
        for msg in litellm_messages:
            role = msg.get('role')
            content = msg.get('content')
            
            # Keep system messages as-is (will be extracted by GoogleGenAILLM)
            if role == 'system':
                gemini_messages.append({'role': 'system', 'content': content})
                continue
            
            # Map roles: user -> user, assistant -> model
            if role == 'assistant':
                role = 'model'
            elif role == 'tool':
                # Skip tool messages for now - Gemini handles these differently
                # TODO: Handle tool results properly if needed
                continue
            
            # Handle content (can be string or list of content blocks)
            if isinstance(content, str):
                gemini_messages.append({'role': role, 'parts': [{'text': content}]})
            elif isinstance(content, list):
                # Convert content blocks to parts
                parts = []
                for block in content:
                    if block.get('type') == 'text':
                        parts.append({'text': block.get('text', '')})
                    elif block.get('type') == 'image':
                        # Handle image URLs
                        image_url = block.get('image_url', '')
                        if image_url.startswith('data:'):
                            # Extract base64 data
                            import re
                            match = re.match(r'data:([^;]+);base64,(.+)', image_url)
                            if match:
                                mime_type, data = match.groups()
                                parts.append({'inline_data': {'mime_type': mime_type, 'data': data}})
                        else:
                            # External URL
                            parts.append({'file_data': {'file_uri': image_url}})
                if parts:
                    gemini_messages.append({'role': role, 'parts': parts})
        
        return gemini_messages

    def save_to_file(self, filepath: str):
        """Save conversation history to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filepath: str) -> 'Chat':
        """Load conversation history from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # This is a simplified loader - you'd want more robust deserialization
        history = cls(
            system_prompt=data.get('system_prompt'),
            protected_rounds=data.get('protected_rounds', 0)
        )

        # Note: Full deserialization would require reconstructing objects from dicts
        # This is left as an exercise since it depends on your exact needs

        return history

    def clear(self):
        """Clear all turns from history"""
        self.turns.clear()

    def get_token_count_estimate(self) -> int:
        """Rough estimate of token count (actual count requires tokenizer)"""
        total = 0
        for turn in self.turns:
            if isinstance(turn, (UserTurn, AssistantTurn)):
                for block in turn.content:
                    if isinstance(block, TextContent):
                        # Very rough estimate: ~4 chars per token
                        total += len(block.text) // 4
        return total
    
    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks with glassmorphism design."""
        try:
            from opto.utils.display.jupyter import render_chat
            return render_chat(self)
        except ImportError:
            # Fallback to text representation if display module unavailable
            return None
