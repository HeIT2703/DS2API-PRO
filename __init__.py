"""Public package exports for the DS2API web chat client."""

from .client import DeepSeekClient
from .async_client import AsyncDeepSeekClient
from .src.models import ChatResponse, StreamEvent, Citation, FileInfo
from .src.exceptions import (
    APIRequestError,
    AuthenticationError,
    DeepSeekError,
    FileProcessingTimeoutError,
    JSONParseError,
    PoWSolverError,
    ValidationError,
)


__all__ = [
    "DeepSeekClient",
    "AsyncDeepSeekClient",
    "ChatResponse",
    "StreamEvent",
    "Citation",
    "FileInfo",
    "DeepSeekError",
    "AuthenticationError",
    "APIRequestError",
    "FileProcessingTimeoutError",
    "JSONParseError",
    "PoWSolverError",
    "ValidationError",
]
