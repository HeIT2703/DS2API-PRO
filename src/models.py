from dataclasses import dataclass, field
from typing import Any, List, Optional, Union


@dataclass
class Citation:
    """Represents a single search citation returned by DeepSeek."""
    title: str
    url: str
    snippet: str = ""


@dataclass
class FileInfo:
    """Information about an uploaded file."""
    id: str
    status: str
    file_name: str
    file_size: int
    preview_url: Optional[str] = None
    token_usage: Optional[int] = None


@dataclass
class ChatResponse:
    """
    Represents a complete response from a DeepSeek chat completion.
    
    You can cast this object to a string (e.g., str(response)) to get just the text answer.
    """
    text: str = ""
    thinking: str = ""
    thinking_elapsed: Optional[float] = None
    message_id: Optional[Union[int, str]] = None
    token_usage: Optional[int] = None
    citations: List[Citation] = field(default_factory=list)
    raw_text: str = ""

    def __str__(self) -> str:
        """Returns the final text answer when cast to a string."""
        return self.text


@dataclass
class StreamEvent:
    """
    Represents a single event chunk in a streaming response.
    
    event_type can be:
    - "THINK_TEXT": content is a chunk of thinking text (str).
    - "RESPONSE_TEXT": content is a chunk of answer text (str).
    - "SEARCH_RESULTS": content is a list of Citation objects.
    - "TOKEN_USAGE": content is the token usage count (int).
    - "THINKING_DONE": thinking has finished, content is the elapsed time in seconds (float).
    - "FINISHED": the stream has finished entirely.
    """
    event_type: str
    content: Any = None
    message_id: Optional[Union[int, str]] = None
