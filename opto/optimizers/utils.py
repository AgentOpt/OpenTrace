import base64
import mimetypes
import io
from typing import Dict, Any, Union, Optional, List
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

import opto.trace as trace


def is_bedrock_model(model_name: str) -> bool:
    """
    Check if a model name represents an AWS Bedrock model.
    
    Bedrock models in LiteLLM can be represented as:
    - "bedrock/us.anthropic.claude-3-7-sonnet-20250219-v1:0" (with bedrock/ prefix)
    - "us.anthropic.claude-3-7-sonnet-20250219-v1:0" (region prefix only)
    
    Args:
        model_name: The model name string to check
        
    Returns:
        True if the model is a Bedrock model, False otherwise
    """
    if model_name is None:
        return False

    if model_name.startswith('bedrock/'):
        return True
    # Check for AWS region prefixes (us-east-1, eu-west-1, ap-northeast-1, etc.)
    return any(model_name.startswith(f'{region}.') for region in ('us', 'eu', 'ap'))


def print_color(message, color=None, logger=None):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
    }
    print(
        f"{colors.get(color, '')}{message}\033[0m"
    )  # Default to no color if invalid color is provided

    if logger is not None:
        logger.log(message)


def truncate_expression(value, limit):
    # https://stackoverflow.com/questions/1436703/what-is-the-difference-between-str-and-repr
    value = str(value)
    if len(value) > limit:
        return value[:limit] + "...(skipped due to length limit)"
    return value


def extract_top_level_blocks(text: str, tag: str):
    """Extract all top-level <tag>...</tag> blocks from text."""
    blocks = []
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    stack = []
    start = None
    i = 0
    while i < len(text):
        if text.startswith(start_tag, i):
            if not stack:
                start = i + len(start_tag)
            stack.append(i)
            i += len(start_tag)
        elif text.startswith(end_tag, i):
            if stack:
                stack.pop()
                if not stack and start is not None:
                    blocks.append(text[start:i])
                    start = None
            i += len(end_tag)
        else:
            i += 1
    return blocks


def extract_first_top_level_block(text: str, tag: str):
    blocks = extract_top_level_blocks(text, tag)
    return blocks[0] if blocks else None


def strip_nested_blocks(text: str, tag: str) -> str:
    """Remove all nested <tag>...</tag> blocks from text, leaving only the top-level text."""
    result = ''
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    stack = []
    i = 0
    last = 0
    while i < len(text):
        if text.startswith(start_tag, i):
            if not stack:
                result += text[last:i]
            stack.append(i)
            i += len(start_tag)
        elif text.startswith(end_tag, i):
            if stack:
                stack.pop()
                if not stack:
                    last = i + len(end_tag)
            i += len(end_tag)
        else:
            i += 1
    if not stack:
        result += text[last:]
    return result.strip()


def extract_reasoning_and_remainder(text: str, tag: str = "reasoning"):
    """Extract reasoning and the remainder of the text after reasoning block (if closed). Strip whitespace only if properly closed."""
    start_tag = f'<{tag}>'
    end_tag = f'</{tag}>'
    start = text.find(start_tag)
    if start == -1:
        return '', text
    start += len(start_tag)
    end = text.find(end_tag, start)
    if end == -1:
        # If not properly closed, don't strip whitespace to preserve original formatting
        return text[start:], ''
    return text[start:end].strip(), text[end + len(end_tag):]


def extract_xml_like_data(text: str, reasoning_tag: str = "reasoning",
                          improved_variable_tag: str = "variable",
                          name_tag: str = "name",
                          value_tag: str = "value") -> Dict[str, Any]:
    """
    Extract thinking content and improved variables from text containing XML-like tags.

    Args:
        text (str): Text containing <reasoning> and <variable> tags

    Returns:
        Dict containing:
        - 'reasoning': content of <reasoning> element
        - 'variables': dict mapping variable names to their values
    """
    result = {
        'reasoning': '',
        'variables': {}
    }

    # Extract reasoning and the remainder of the text
    reasoning, remainder = extract_reasoning_and_remainder(text, reasoning_tag)
    result['reasoning'] = reasoning

    # Only parse variables from the remainder (i.e., after a closed reasoning tag)
    variable_blocks = extract_top_level_blocks(remainder, improved_variable_tag)
    for var_block in variable_blocks:
        name_block = extract_first_top_level_block(var_block, name_tag)
        value_block = extract_first_top_level_block(var_block, value_tag)
        # Only add if both name and value tags are present and name is non-empty after stripping
        if name_block is not None and value_block is not None:
            var_name = name_block.strip()
            var_value = value_block.strip() if value_block is not None else ''
            if var_name:  # Only require name to be non-empty, value can be empty
                result['variables'][var_name] = var_value
    return result


class MultiModalPayload:
    """
    A payload for multimodal content, particularly images.

    Supports three types of image inputs:
    1. URL (string starting with 'http://' or 'https://')
    2. Local file path (string path to image file)
    3. Numpy array (RGB image array)
    """
    image_data: Optional[str] = None  # Can be URL or base64 data URL

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
                self.image_data = image
            else:
                # Assume it's a local file path
                self.image_data = encode_image_to_base64(image)
        else:
            # Assume it's a numpy array or array-like object
            self.image_data = encode_numpy_to_base64(image, format=format)

    def get_content_block(self) -> Optional[Dict[str, Any]]:
        """
        Get the content block for the image in litellm format.

        Returns:
            Dict with format: {"type": "image_url", "image_url": {"url": ...}}
            or None if no image data is set
        """
        if self.image_data is None:
            return None

        return {
            "type": "image_url",
            "image_url": {
                "url": self.image_data
            }
        }

def encode_image_to_base64(path: str) -> str:
    """Encode a local image file to base64 data URL."""
    # Read binary
    with open(path, "rb") as f:
        image_bytes = f.read()
    # Guess MIME type from file extension
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        # fallback
        mime_type = "image/jpeg"
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    data_url = f"data:{mime_type};base64,{b64}"
    return data_url


def encode_numpy_to_base64(array, format: str = "PNG") -> str:
    """
    Encode a numpy array to base64 data URL.
    
    Args:
        array: numpy array representing an image (H, W, C) with values in [0, 255] or [0, 1]
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        Base64 encoded data URL string
    """
    if not NUMPY_AVAILABLE:
        raise ImportError("numpy is required to encode numpy arrays. Install it with: pip install numpy")
    
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required to encode numpy arrays. Install it with: pip install Pillow")
    
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
    
    # Convert to PIL Image
    image = Image.fromarray(array)
    
    # Save to bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format=format.upper())
    buffer.seek(0)
    
    # Encode to base64
    image_bytes = buffer.getvalue()
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Determine MIME type
    mime_type = f"image/{format.lower()}"
    data_url = f"data:{mime_type};base64,{b64}"
    
    return data_url

class ChatHistory:
    def __init__(self, max_turn=50, auto_summary=False):
        """Initialize chat history for multi-turn conversation.

        Args:
            max_turn: Maximum number of conversation turns to keep in history.

            auto_summary: Whether to automatically summarize old messages
        """
        self.messages: List[Dict[str, Any]] = []
        self.max_len = max_turn * 2
        self.auto_summary = auto_summary

    def __len__(self):
        return len(self.messages)

    def add(self, content: Union[trace.Node, str], role):
        """Add a message to history with role validation.

        Args:
            content: The content of the message
            role: The role of the message ("user" or "assistant")
        """
        if role not in ["user", "assistant"]:
            raise ValueError(f"Invalid role '{role}'. Must be 'user' or 'assistant'.")

        # Check for alternating user/assistant pattern
        if len(self.messages) > 0:
            last_msg = self.messages[-1]
            if last_msg["role"] == role:
                print(f"Warning: Adding consecutive {role} messages. Consider alternating user/assistant messages.")

        self.messages.append({"role": role, "content": content})
        self._trim_history()

    def append(self, message: Dict[str, Any]):
        """Append a message directly to history."""
        if "role" not in message or "content" not in message:
            raise ValueError("Message must have 'role' and 'content' fields.")
        self.add(message["content"], message["role"])

    def __iter__(self):
        return iter(self.messages)

    def get_messages(self) -> List[Dict[str, str]]:
        messages = []
        for message in self.messages:
            if isinstance(message['content'], trace.Node):
                messages.append({"role": message["role"], "content": message["content"].data})
            else:
                messages.append(message)
        return messages

    def get_messages_as_node(self, llm_name="") -> List[trace.Node]:
        node_list = []
        for message in self.messages:
            # If user query is a node and has other computation attached, we can't rename it
            if isinstance(message['content'], trace.Node):
                node_list.append(message['content'])
            else:
                role = message["role"]
                content = message["content"]
                name = f"{llm_name}_{role}" if llm_name else f"{role}"
                if role == 'user':
                    name += "_query"
                elif role == 'assistant':
                    name += "_response"
                node_list.append(trace.node(content, name=name))

        return node_list

    def _trim_history(self):
        """Trim history to max_len while preserving first user message."""
        if len(self.messages) <= self.max_len:
            return

        # Find first user message index
        first_user_idx = None
        for i, msg in enumerate(self.messages):
            if msg["role"] == "user":
                first_user_idx = i
                break

        # Keep first user message
        protected_messages = []
        if first_user_idx is not None:
            first_user_msg = self.messages[first_user_idx]
            protected_messages.append(first_user_msg)

        # Calculate how many recent messages we can keep
        remaining_slots = self.max_len - len(protected_messages)
        if remaining_slots > 0:
            # Get recent messages
            recent_messages = self.messages[-remaining_slots:]
            # Avoid duplicating first user message
            if first_user_idx is not None:
                first_user_msg = self.messages[first_user_idx]
                recent_messages = [msg for msg in recent_messages if msg != first_user_msg]

            self.messages = protected_messages + recent_messages
        else:
            self.messages = protected_messages