from .http_client import DeepSeekHTTPClient
from .validation import ensure_bool, ensure_positive_int, ensure_string


MAX_SESSION_FETCH_COUNT = 100


class SessionAPI:
    def __init__(self, http_client: DeepSeekHTTPClient):
        self.http = http_client

    def create(self, model_type: str = "default") -> dict:
        """Create a new chat session."""
        model_type = ensure_string(model_type, "model_type", max_length=64)
        return self.http.post("/api/v0/chat_session/create", json_data={"model_type": model_type})

    def list(self, count: int = 20) -> dict:
        """List chat sessions."""
        count = ensure_positive_int(count, "count", max_value=MAX_SESSION_FETCH_COUNT)
        return self.http.get("/api/v0/chat_session/fetch_page", params={"count": count})

    def delete(self, session_id: str) -> dict:
        """Delete a chat session."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        return self.http.post("/api/v0/chat_session/delete", json_data={"chat_session_id": session_id})

    def delete_all(self) -> dict:
        """Delete all chat sessions."""
        return self.http.post("/api/v0/chat_session/delete_all")

    def update_pinned(self, session_id: str, pinned: bool) -> dict:
        """Pin or unpin a chat session."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        pinned = ensure_bool(pinned, "pinned")
        return self.http.post(
            "/api/v0/chat_session/update_pinned",
            json_data={"chat_session_id": session_id, "pinned": pinned}
        )

    def update_title(self, session_id: str, title: str) -> dict:
        """Update the title of a chat session."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        title = ensure_string(title, "title", allow_empty=True, max_length=500)
        return self.http.post(
            "/api/v0/chat_session/update_title",
            json_data={"chat_session_id": session_id, "title": title}
        )
