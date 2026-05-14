import logging
import sys
import threading
from typing import Generator, Iterable, Mapping, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .src.models import ChatResponse, StreamEvent

import requests

from .src.http_client import (
    DEFAULT_ACCEPT_LANGUAGE,
    DEFAULT_APP_VERSION,
    DEFAULT_CLIENT_LOCALE,
    DEFAULT_CLIENT_PLATFORM,
    DEFAULT_CLIENT_TIMEZONE_OFFSET,
    DEFAULT_CLIENT_VERSION,
    DEFAULT_RETRY_METHODS,
    DEFAULT_RETRY_STATUSES,
    DEFAULT_STREAM_TIMEOUT,
    DEFAULT_TIMEOUT,
    DEFAULT_USER_AGENT,
    DeepSeekHTTPClient,
    Timeout,
)
from .src.pow_solver import PoWSolver
from .src.api_user import UserAPI
from .src.api_session import SessionAPI
from .src.api_chat import ChatAPI, MAX_PROMPT_LENGTH, MAX_REF_FILES
from .src.api_file import FileAPI
from .src.exceptions import APIRequestError, ValidationError
from .src.validation import ensure_bool, ensure_string, ensure_string_list


logger = logging.getLogger(__name__)


# User-facing model aliases and their DeepSeek web payload values.
MODEL_INSTANT = "default"
MODEL_EXPERT = "expert"


class DeepSeekClient:
    """
    Main entry point for the unofficial chat.deepseek.com web client.

    Minimal usage (only token required):

        client = DeepSeekClient(token="your_token")
        answer = client.ask("Hello!")

    Multi-turn with context memory:

        sid = client.new_chat()
        r1 = client.send(sid, "My name is Vu")
        r2 = client.send(sid, "What's my name?")  # Remembers!

    Expert model with DeepThink:

        answer = client.ask("Prove sqrt(2) is irrational", model="expert", thinking=True)
    """

    def __init__(
        self,
        token: Optional[str] = None,
        hif_dliq: Optional[str] = None,
        hif_leim: Optional[str] = None,
        cookies: Optional[Mapping[str, str]] = None,
        base_url: str = DeepSeekHTTPClient.BASE_URL,
        timeout: Timeout = DEFAULT_TIMEOUT,
        stream_timeout: Timeout = DEFAULT_STREAM_TIMEOUT,
        max_retries: int = 2,
        session: Optional[requests.Session] = None,
        authorization: Optional[str] = None,
        default_headers: Optional[Mapping[str, str]] = None,
        user_agent: Optional[str] = DEFAULT_USER_AGENT,
        accept_language: Optional[str] = DEFAULT_ACCEPT_LANGUAGE,
        app_version: Optional[str] = DEFAULT_APP_VERSION,
        client_locale: Optional[str] = DEFAULT_CLIENT_LOCALE,
        client_platform: Optional[str] = DEFAULT_CLIENT_PLATFORM,
        client_timezone_offset: Optional[Union[str, int]] = DEFAULT_CLIENT_TIMEZONE_OFFSET,
        client_version: Optional[str] = DEFAULT_CLIENT_VERSION,
        proxies: Optional[Mapping[str, str]] = None,
        verify: Union[bool, str] = True,
        cert: Optional[Union[str, Tuple[str, str]]] = None,
        trust_env: Optional[bool] = None,
        retry_methods: Iterable[str] = DEFAULT_RETRY_METHODS,
        retry_statuses: Iterable[int] = DEFAULT_RETRY_STATUSES,
        retry_backoff_factor: float = 0.5,
        request_options: Optional[Mapping[str, object]] = None,
        max_upload_size_bytes: Optional[int] = None,
        pow_max_difficulty: Optional[int] = None,
        http_client: Optional[DeepSeekHTTPClient] = None,
        pow_solver: Optional[PoWSolver] = None,
    ):
        if http_client is not None:
            self.http = http_client
        else:
            self.http = DeepSeekHTTPClient(
                token=token,
                hif_dliq=hif_dliq,
                hif_leim=hif_leim,
                cookies=dict(cookies) if cookies else None,
                base_url=base_url,
                timeout=timeout,
                stream_timeout=stream_timeout,
                max_retries=max_retries,
                session=session,
                authorization=authorization,
                default_headers=default_headers,
                user_agent=user_agent,
                accept_language=accept_language,
                app_version=app_version,
                client_locale=client_locale,
                client_platform=client_platform,
                client_timezone_offset=client_timezone_offset,
                client_version=client_version,
                proxies=proxies,
                verify=verify,
                cert=cert,
                trust_env=trust_env,
                retry_methods=retry_methods,
                retry_statuses=retry_statuses,
                retry_backoff_factor=retry_backoff_factor,
                request_options=request_options,
            )

        self.pow_solver = pow_solver or PoWSolver(max_difficulty=pow_max_difficulty)

        # Sub-API modules (for power users)
        self.user = UserAPI(self.http)
        self.session = SessionAPI(self.http)
        self.chat = ChatAPI(self.http, self.pow_solver)
        self.file = FileAPI(self.http, self.pow_solver, max_upload_size_bytes=max_upload_size_bytes)

        # Internal state: tracks model_type and last message_id per session
        self._session_model: dict[str, str] = {}
        self._last_message_id: dict[str, Union[int, str]] = {}
        self._session_locks: dict[str, threading.RLock] = {}
        self._state_lock = threading.RLock()

    # One-shot chat.

    def ask(
        self,
        prompt: str,
        model: str = "instant",
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
        print_to_stdout: bool = False,
    ) -> "ChatResponse":
        """
        Send a one-shot message and return the full response object.

        Automatically creates a temporary session, sends the prompt,
        collects the full response, and deletes the session.

        Args:
            prompt: The message to send.
            model: "instant" (DeepSeek-V3) or "expert" (DeepSeek-R1).
            thinking: Enable DeepThink when supported by the backend.
            search: Enable web search.
            ref_file_ids: File IDs to attach (from client.file.upload_file()).
            print_to_stdout: If True, also print chunks to stdout as they arrive.

        Returns:
            A ChatResponse object containing text, thinking, citations, etc.
        """
        model_type = self._resolve_model_type(model)
        prompt, thinking, search, ref_file_ids, _ = self._validate_completion_inputs(
            prompt=prompt,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
        )
        session_id = self._create_temp_session(model_type)
        try:
            response = self._collect_stream_with_id(
                session_id=session_id,
                prompt=prompt,
                model_type=model_type,
                thinking_enabled=thinking,
                search_enabled=search,
                ref_file_ids=ref_file_ids,
                parent_message_id=None,
                print_to_stdout=print_to_stdout,
            )
            return response
        finally:
            self._delete_session_quietly(session_id)

    def ask_stream(
        self,
        prompt: str,
        model: str = "instant",
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
    ) -> Generator["StreamEvent", None, None]:
        """
        Send a one-shot message and stream the response events.

        Creates a temporary session, yields StreamEvent chunks, and deletes
        the session when the generator finishes.
        """
        model_type = self._resolve_model_type(model)
        prompt, thinking, search, ref_file_ids, _ = self._validate_completion_inputs(
            prompt=prompt,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
        )
        def _generator():
            session_id = self._create_temp_session(model_type)
            try:
                yield from self.chat.completion(
                    session_id=session_id,
                    prompt=prompt,
                    model_type=model_type,
                    thinking_enabled=thinking,
                    search_enabled=search,
                    ref_file_ids=ref_file_ids,
                )
            finally:
                self._delete_session_quietly(session_id)

        return _generator()

    # Managed multi-turn chat.

    def new_chat(self, model: str = "instant") -> str:
        """
        Create a managed chat session and return its session ID.

        The model type is fixed for the lifetime of this session.
        Use send() or send_stream() for multi-turn conversations.

        Args:
            model: "instant" (DeepSeek-V3) or "expert" (DeepSeek-R1).

        Returns:
            The session ID string.
        """
        model_type = self._resolve_model_type(model)
        session_id = self._create_temp_session(model_type)
        with self._state_lock:
            self._session_model[session_id] = model_type
            self._session_locks.setdefault(session_id, threading.RLock())
        return session_id

    def adopt_chat(self, session_id: str, model: str = "instant", last_message_id: Optional[Union[int, str]] = None) -> str:
        """
        Register an existing chat session for high-level conversation methods.

        This keeps high-level sends explicit: sessions are either created with
        new_chat() or deliberately adopted with the model/context state the
        caller wants this client instance to use. last_message_id may be an
        integer or string ID returned by the DeepSeek web stream.
        """
        session_id = ensure_string(session_id, "session_id", max_length=128)
        model_type = self._resolve_model_type(model)
        last_message_id = self._validate_parent_message_id(last_message_id)
        with self._state_lock:
            self._session_model[session_id] = model_type
            if last_message_id is None:
                self._last_message_id.pop(session_id, None)
            else:
                self._last_message_id[session_id] = last_message_id
            self._session_locks.setdefault(session_id, threading.RLock())
        return session_id

    def delete_chat(self, session_id: str) -> dict:
        """Delete a chat session and clear any local conversation state."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        try:
            return self.session.delete(session_id)
        finally:
            self._drop_session_state(session_id)

    def send(
        self,
        session_id: str,
        prompt: str,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
        parent_message_id: Optional[Union[int, str]] = None,
        print_to_stdout: bool = False,
    ) -> "ChatResponse":
        """
        Send a message in an existing session and return the full ChatResponse.

        Model type is determined by the session (set at new_chat()).
        Context is automatically chained between turns via parent_message_id.

        Args:
            session_id: Session ID from new_chat().
            prompt: The message to send.
            thinking: Enable DeepThink for this message.
            search: Enable web search for this message.
            ref_file_ids: File IDs to attach.
            parent_message_id: Optional int or string message ID to branch from.
            print_to_stdout: Print chunks to stdout as they arrive.
        """
        session_id = ensure_string(session_id, "session_id", max_length=128)
        prompt, thinking, search, ref_file_ids, parent_message_id = self._validate_completion_inputs(
            prompt=prompt,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
            parent_message_id=parent_message_id,
        )
        model_type = self._require_session_model(session_id)

        with self._session_lock(session_id):
            # Auto-chain: use tracked parent if caller didn't provide one.
            if parent_message_id is None:
                with self._state_lock:
                    parent_message_id = self._last_message_id.get(session_id)

            response = self._collect_stream_with_id(
                session_id=session_id,
                prompt=prompt,
                model_type=model_type,
                thinking_enabled=thinking,
                search_enabled=search,
                ref_file_ids=ref_file_ids,
                parent_message_id=parent_message_id,
                print_to_stdout=print_to_stdout,
            )

            if response.message_id is not None:
                with self._state_lock:
                    self._last_message_id[session_id] = response.message_id

        return response

    def send_stream(
        self,
        session_id: str,
        prompt: str,
        thinking: bool = False,
        search: bool = False,
        ref_file_ids: Optional[list] = None,
        parent_message_id: Optional[Union[int, str]] = None,
    ) -> Generator["StreamEvent", None, None]:
        """
        Send a message in an existing session and stream response events.

        Context is automatically chained between turns unless parent_message_id
        is provided.
        """
        session_id = ensure_string(session_id, "session_id", max_length=128)
        prompt, thinking, search, ref_file_ids, parent_message_id = self._validate_completion_inputs(
            prompt=prompt,
            thinking=thinking,
            search=search,
            ref_file_ids=ref_file_ids,
            parent_message_id=parent_message_id,
        )
        model_type = self._require_session_model(session_id)

        def _generator():
            with self._session_lock(session_id):
                effective_parent = parent_message_id
                if effective_parent is None:
                    with self._state_lock:
                        effective_parent = self._last_message_id.get(session_id)

                for event in self.chat.completion(
                    session_id=session_id,
                    prompt=prompt,
                    model_type=model_type,
                    thinking_enabled=thinking,
                    search_enabled=search,
                    parent_message_id=effective_parent,
                    ref_file_ids=ref_file_ids,
                ):
                    if event.event_type == "MESSAGE_ID" and event.content is not None:
                        with self._state_lock:
                            self._last_message_id[session_id] = event.content
                    yield event

        return _generator()

    def regenerate(self, session_id: str, message_id: Union[int, str], print_to_stdout: bool = False) -> "ChatResponse":
        """Regenerate a response."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        message_id = self._validate_message_id(message_id)
        self._require_session_model(session_id)
        with self._session_lock(session_id):
            response = self._consume_stream(self.chat.regenerate(session_id, str(message_id)), print_to_stdout)
            if response.message_id is not None:
                with self._state_lock:
                    self._last_message_id[session_id] = response.message_id
        return response

    def edit_message(self, session_id: str, message_id: Union[int, str], prompt: str, print_to_stdout: bool = False) -> "ChatResponse":
        """Edit a message and stream the new response."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        message_id = self._validate_message_id(message_id)
        prompt = ensure_string(prompt, "prompt", max_length=MAX_PROMPT_LENGTH)
        self._require_session_model(session_id)
        with self._session_lock(session_id):
            response = self._consume_stream(self.chat.edit_message(session_id, str(message_id), prompt), print_to_stdout)
            if response.message_id is not None:
                with self._state_lock:
                    self._last_message_id[session_id] = response.message_id
        return response

    def continue_response(self, session_id: str, message_id: Union[int, str], print_to_stdout: bool = False) -> "ChatResponse":
        """Continue a truncated response."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        message_id = self._validate_message_id(message_id)
        self._require_session_model(session_id)
        with self._session_lock(session_id):
            response = self._consume_stream(self.chat.continue_response(session_id, str(message_id)), print_to_stdout)
            if response.message_id is not None:
                with self._state_lock:
                    self._last_message_id[session_id] = response.message_id
        return response

    def stop(self, session_id: str) -> dict:
        """Stop an active generation stream in a session."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        self._require_session_model(session_id)
        return self.chat.stop_stream(session_id)

    # Raw API pass-through for advanced callers.

    @property
    def base_url(self) -> str:
        return self.http.base_url

    @property
    def default_headers(self) -> dict:
        return self.http.default_headers

    @property
    def cookies(self):
        return self.http.session.cookies

    @property
    def requests_session(self) -> requests.Session:
        return self.http.session

    def get(self, path: str, params: Optional[dict] = None, headers: Optional[dict] = None, **kwargs) -> dict:
        """Call a raw GET endpoint."""
        return self.http.get(path, params=params, headers=headers, **kwargs)

    def post(self, path: str, json_data: Optional[dict] = None, files=None, headers: Optional[dict] = None, **kwargs) -> dict:
        """Call a raw POST endpoint."""
        return self.http.post(path, json_data=json_data, files=files, headers=headers, **kwargs)

    def post_stream(self, path: str, json_data: dict, headers: Optional[dict] = None, **kwargs) -> requests.Response:
        """Open a raw streaming POST endpoint."""
        return self.http.post_stream(path, json_data=json_data, headers=headers, **kwargs)

    def request(self, method: str, path: str, **kwargs) -> requests.Response:
        """Send a raw HTTP request."""
        return self.http.request(method, path, **kwargs)

    def set_default_header(self, name: str, value: str) -> None:
        self.http.set_default_header(name, value)

    def remove_default_header(self, name: str) -> None:
        self.http.remove_default_header(name)

    def set_cookie(self, name: str, value: str) -> None:
        self.http.set_cookie(name, value)

    def clear_cookies(self) -> None:
        self.http.clear_cookies()

    # Internal helpers.

    @staticmethod
    def _resolve_model_type(model: str) -> str:
        """Resolve user-friendly model name to API value."""
        mapping = {
            "instant": MODEL_INSTANT,
            "default": MODEL_INSTANT,
            "expert": MODEL_EXPERT,
        }
        key = model.lower().strip()
        if key not in mapping:
            raise ValidationError(f"Unknown model '{model}'. Use 'instant' or 'expert'.")
        return mapping[key]

    @staticmethod
    def _validate_parent_message_id(parent_message_id):
        if parent_message_id is None:
            return None
        if isinstance(parent_message_id, bool):
            raise ValidationError("parent_message_id must be a non-empty string or positive integer.")
        if isinstance(parent_message_id, int):
            if parent_message_id <= 0:
                raise ValidationError("parent_message_id must be a non-empty string or positive integer.")
            return parent_message_id
        return ensure_string(parent_message_id, "parent_message_id", max_length=128)

    @staticmethod
    def _validate_message_id(message_id):
        if isinstance(message_id, bool):
            raise ValidationError("message_id must be a non-empty string or positive integer.")
        if isinstance(message_id, int):
            if message_id <= 0:
                raise ValidationError("message_id must be a non-empty string or positive integer.")
            return message_id
        return ensure_string(message_id, "message_id", max_length=128)

    @classmethod
    def _validate_completion_inputs(
        cls,
        *,
        prompt: str,
        thinking: bool,
        search: bool,
        ref_file_ids: Optional[list],
        parent_message_id: Optional[Union[int, str]] = None,
    ):
        return (
            ensure_string(prompt, "prompt", max_length=MAX_PROMPT_LENGTH),
            ensure_bool(thinking, "thinking"),
            ensure_bool(search, "search"),
            ensure_string_list(ref_file_ids, "ref_file_ids", max_items=MAX_REF_FILES),
            cls._validate_parent_message_id(parent_message_id),
        )

    @staticmethod
    def extract_session_id(api_response: dict) -> str:
        """Extract session ID from a raw session.create() API response."""
        biz = api_response.get("data", {}).get("biz_data", {})
        sid = (
            biz.get("id")
            or (biz.get("chat_session") or {}).get("id")
            or biz.get("chat_session_id")
        )
        if not sid:
            raise APIRequestError(f"Could not extract session ID. Keys: {list(biz.keys())}")
        return sid

    def _create_temp_session(self, model_type: str = MODEL_INSTANT) -> str:
        """Create a session and return just the session ID string."""
        resp = self.session.create(model_type=model_type)
        return self.extract_session_id(resp)

    def _require_session_model(self, session_id: str) -> str:
        with self._state_lock:
            model_type = self._session_model.get(session_id)
        if model_type is None:
            raise ValidationError(
                "Unknown session_id. Use new_chat() to create a managed session, "
                "adopt_chat() to register an existing session, or client.chat for raw calls."
            )
        return model_type

    def _session_lock(self, session_id: str) -> threading.RLock:
        with self._state_lock:
            return self._session_locks.setdefault(session_id, threading.RLock())

    def _drop_session_state(self, session_id: str) -> None:
        with self._state_lock:
            self._last_message_id.pop(session_id, None)
            self._session_model.pop(session_id, None)
            self._session_locks.pop(session_id, None)

    def _delete_session_quietly(self, session_id: str) -> None:
        """Delete a session without masking the caller's primary result/error."""
        try:
            self.session.delete(session_id)
        except Exception as exc:
            logger.warning(
                "Failed to delete DeepSeek chat session during cleanup",
                extra={"session_id": session_id, "error_type": exc.__class__.__name__},
            )
        finally:
            self._drop_session_state(session_id)

    def _consume_stream(self, stream: Generator["StreamEvent", None, None], print_to_stdout: bool) -> "ChatResponse":
        from .src.models import ChatResponse
        
        response = ChatResponse()
        
        for event in stream:
            if event.event_type == "MESSAGE_ID":
                response.message_id = event.content
            elif event.event_type == "THINK_TEXT":
                response.thinking += event.content
                response.raw_text += event.content
                if print_to_stdout:
                    sys.stdout.write(event.content)
                    sys.stdout.flush()
            elif event.event_type == "RESPONSE_TEXT":
                response.text += event.content
                response.raw_text += event.content
                if print_to_stdout:
                    sys.stdout.write(event.content)
                    sys.stdout.flush()
            elif event.event_type == "THINKING_DONE":
                response.thinking_elapsed = event.content
            elif event.event_type == "SEARCH_RESULTS":
                response.citations.extend(event.content)
            elif event.event_type == "TOKEN_USAGE":
                response.token_usage = event.content

        if print_to_stdout:
            print()
            
        return response

    def _collect_stream_with_id(
        self,
        session_id: str,
        prompt: str,
        model_type: str,
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        ref_file_ids: Optional[list] = None,
        parent_message_id: Optional[Union[int, str]] = None,
        print_to_stdout: bool = False,
    ) -> "ChatResponse":
        """Consume a chat stream and return the full ChatResponse."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        prompt, thinking_enabled, search_enabled, ref_file_ids, parent_message_id = self._validate_completion_inputs(
            prompt=prompt,
            thinking=thinking_enabled,
            search=search_enabled,
            ref_file_ids=ref_file_ids,
            parent_message_id=parent_message_id,
        )
        model_type = ensure_string(model_type, "model_type", max_length=64)
        payload = {
            "chat_session_id": session_id,
            "parent_message_id": parent_message_id,
            "prompt": prompt,
            "model_type": model_type,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "ref_file_ids": ref_file_ids or [],
            "preempt": False,
        }

        stream = self.chat._stream_structured("/api/v0/chat/completion", payload)
        return self._consume_stream(stream, print_to_stdout)

    def completion_text(self, session_id: str, prompt: str, print_to_stdout: bool = True, **kwargs) -> str:
        """Send a message and return only the final answer text."""
        return self.send(session_id, prompt, print_to_stdout=print_to_stdout, **kwargs).text

    def close(self) -> None:
        """Close underlying network resources."""
        self.http.close()
        with self._state_lock:
            self._last_message_id.clear()
            self._session_model.clear()
            self._session_locks.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
