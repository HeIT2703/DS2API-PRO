import re
from typing import Optional


_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")
_SENSITIVE_FIELD_RE = re.compile(
    r"(?i)(\"?(?:authorization|cookie|set-cookie|token|access_token|refresh_token|"
    r"api[_-]?key|secret|password|signature|userToken)\"?\s*[:=]\s*)"
    r"(\"[^\"]*\"|[^\\s,;}]*)"
)
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def redact_sensitive_text(text: Optional[str], limit: int = 500) -> Optional[str]:
    """Return a short, log-safe preview with common secrets and PII redacted."""
    if text is None:
        return None

    preview = str(text).replace("\r", "\\r").replace("\n", "\\n")
    preview = _BEARER_RE.sub(r"\1<redacted>", preview)
    preview = _SENSITIVE_FIELD_RE.sub(r"\1<redacted>", preview)
    preview = _EMAIL_RE.sub("<redacted-email>", preview)

    if len(preview) > limit:
        return f"{preview[:limit]}...<truncated>"
    return preview


class DeepSeekError(Exception):
    """Base exception for all DeepSeek API errors."""


class AuthenticationError(DeepSeekError):
    """Raised when authentication fails (token, hif headers, etc.)."""


class PoWSolverError(DeepSeekError):
    """Raised when the Proof-of-Work solver fails or WASM is missing."""


class ValidationError(DeepSeekError, ValueError):
    """Raised when caller-provided input is invalid."""


class APIRequestError(DeepSeekError):
    """Raised when the DeepSeek API returns an HTTP error or code != 0."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
        request_id: Optional[str] = None,
        api_code: Optional[int] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id
        self.api_code = api_code

    @property
    def response_preview(self) -> Optional[str]:
        return redact_sensitive_text(self.response_body)

    def __str__(self):
        base = super().__str__()
        if self.status_code is not None:
            base += f" [Status: {self.status_code}]"
        if self.api_code is not None:
            base += f" [API code: {self.api_code}]"
        if self.request_id:
            base += f" [Request ID: {self.request_id}]"
        preview = self.response_preview
        if preview:
            base += f" [Body: {preview}]"
        return base


class JSONParseError(DeepSeekError):
    """Raised when parsing API response or SSE stream fails."""


class FileProcessingTimeoutError(APIRequestError):
    """Raised when an uploaded file does not become ready before the timeout."""
