from .http_client import DeepSeekHTTPClient


class UserAPI:
    def __init__(self, http_client: DeepSeekHTTPClient):
        self.http = http_client

    def get_current(self) -> dict:
        """Get current user information."""
        return self.http.get("/api/v0/users/current")

    def logout_all_sessions(self) -> dict:
        """Log out from all sessions."""
        return self.http.post("/api/v0/users/logout_all_sessions")

    def get_client_settings(self) -> dict:
        """Get client settings."""
        return self.http.get("/api/v0/client/settings")
