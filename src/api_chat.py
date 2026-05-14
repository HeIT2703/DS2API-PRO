import base64
import json
import logging
from typing import Generator, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .models import StreamEvent

import requests

from .exceptions import APIRequestError, JSONParseError, PoWSolverError, ValidationError, redact_sensitive_text
from .http_client import DeepSeekHTTPClient
from .pow_solver import PoWSolver
from .validation import ensure_api_path, ensure_bool, ensure_optional_message_id, ensure_string, ensure_string_list


MAX_PROMPT_LENGTH = 200_000
MAX_REF_FILES = 100


logger = logging.getLogger(__name__)


class ChatAPI:
    def __init__(self, http_client: DeepSeekHTTPClient, pow_solver: PoWSolver):
        self.http = http_client
        self.pow = pow_solver

    def _create_pow_challenge(self, target_path: str = "/api/v0/chat/completion") -> dict:
        """Request a PoW challenge from the server."""
        target_path = ensure_api_path(target_path, "target_path", max_length=256)
        resp = self.http.post("/api/v0/chat/create_pow_challenge", json_data={"target_path": target_path})
        challenge = resp.get("data", {}).get("biz_data", {}).get("challenge")
        if not isinstance(challenge, dict):
            raise APIRequestError("Malformed PoW challenge response: missing data.biz_data.challenge.")
        return challenge

    def _solve_and_encode_pow(self, challenge_data: dict) -> str:
        """Solve the PoW challenge and return the base64 encoded header string."""
        if not isinstance(challenge_data, dict):
            raise APIRequestError("Malformed PoW challenge: expected object.")

        required_fields = ("challenge", "salt", "difficulty", "expire_at", "signature")
        missing = [field for field in required_fields if field not in challenge_data]
        if missing:
            raise APIRequestError(f"Malformed PoW challenge: missing {', '.join(missing)}.")

        challenge = ensure_string(challenge_data["challenge"], "challenge", max_length=4096)
        salt = ensure_string(challenge_data["salt"], "salt", max_length=1024)
        difficulty = challenge_data["difficulty"]
        expire_at = challenge_data["expire_at"]
        signature = ensure_string(challenge_data["signature"], "signature", max_length=4096)
        algorithm = ensure_string(challenge_data.get("algorithm", "DeepSeekHashV1"), "algorithm", max_length=64)
        target_path = ensure_api_path(
            challenge_data.get("target_path", "/api/v0/chat/completion"),
            "target_path",
            max_length=256,
        )

        if algorithm != "DeepSeekHashV1":
            raise PoWSolverError(f"Unsupported PoW algorithm: {algorithm}")

        nonce = self.pow.solve(challenge, salt, expire_at, difficulty)

        solved = {
            "algorithm": algorithm,
            "challenge": challenge,
            "salt": salt,
            "answer": nonce,
            "signature": signature,
            "target_path": target_path,
        }
        encoded = json.dumps(solved, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return base64.b64encode(encoded).decode("ascii")

    def _stream_structured(self, target_path: str, payload: dict) -> Generator["StreamEvent", None, None]:
        from .models import StreamEvent, Citation

        target_path = ensure_api_path(target_path, "target_path", max_length=256)
        challenge = self._create_pow_challenge(target_path=target_path)
        pow_header = self._solve_and_encode_pow(challenge)

        resp = self.http.post_stream(
            target_path,
            json_data=payload,
            headers={"x-ds-pow-response": pow_header}
        )

        current_mode = "RESPONSE"  # Can be "THINK" or "RESPONSE"
        emitted_any = False
        unknown_chunks = 0

        try:
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].lstrip()
                if not data:
                    continue
                if data == "[DONE]":
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError as exc:
                    preview = redact_sensitive_text(data, limit=200)
                    raise JSONParseError(f"Malformed SSE chunk from {target_path}: {preview}") from exc
                if not isinstance(chunk, dict):
                    raise JSONParseError(
                        f"Malformed SSE chunk from {target_path}: expected object, got {type(chunk).__name__}."
                    )

                api_code = chunk.get("code")
                if api_code not in (None, 0):
                    msg = redact_sensitive_text(str(chunk.get("msg") or chunk.get("message") or "unknown"))
                    raise APIRequestError(f"Streaming API returned error code {api_code}: {msg}", api_code=api_code)

                emitted_before_chunk = emitted_any

                # Extract initial message_id
                v = chunk.get("v")
                if isinstance(v, dict) and "response" in v:
                    msg_id = v["response"].get("message_id")
                    if msg_id is not None:
                        emitted_any = True
                        yield StreamEvent("MESSAGE_ID", msg_id)

                def process_operation(operation, value, path):
                    nonlocal current_mode, emitted_any
                    
                    if operation is None and path == "" and isinstance(value, str):
                        emitted_any = True
                        yield StreamEvent(f"{current_mode}_TEXT", value)
                    elif operation == "APPEND" and path.endswith("content") and isinstance(value, str):
                        emitted_any = True
                        yield StreamEvent(f"{current_mode}_TEXT", value)
                    elif path.endswith("elapsed_secs") and isinstance(value, (int, float)):
                        emitted_any = True
                        yield StreamEvent("THINKING_DONE", float(value))
                    elif path.endswith("results") and isinstance(value, list):
                        citations = []
                        for res in value:
                            if isinstance(res, dict):
                                citations.append(Citation(
                                    title=res.get("title", ""),
                                    url=res.get("url", ""),
                                    snippet=res.get("snippet", "")
                                ))
                        if citations:
                            emitted_any = True
                            yield StreamEvent("SEARCH_RESULTS", citations)
                    elif path.endswith("status") and value == "FINISHED":
                        emitted_any = True
                        yield StreamEvent("FINISHED")
                    elif path.endswith("accumulated_token_usage") and isinstance(value, int):
                        emitted_any = True
                        yield StreamEvent("TOKEN_USAGE", value)

                o = chunk.get("o")
                v = chunk.get("v")
                p = chunk.get("p", "")

                # Check for initial fragment type
                if isinstance(v, dict) and "response" in v:
                    frags = v["response"].get("fragments", [])
                    for f in frags:
                        if isinstance(f, dict):
                            f_type = f.get("type")
                            if f_type == "THINK":
                                current_mode = "THINK"
                            elif f_type == "RESPONSE":
                                current_mode = "RESPONSE"
                            content = f.get("content")
                            if content:
                                emitted_any = True
                                yield StreamEvent(f"{current_mode}_TEXT", content)

                # Check for new fragment appended
                elif o == "APPEND" and "fragments" in p and isinstance(v, list):
                    for f in v:
                        if isinstance(f, dict):
                            f_type = f.get("type")
                            if f_type == "THINK":
                                current_mode = "THINK"
                            elif f_type == "RESPONSE":
                                current_mode = "RESPONSE"
                            content = f.get("content")
                            if content:
                                emitted_any = True
                                yield StreamEvent(f"{current_mode}_TEXT", content)

                # Process main payload
                elif o == "BATCH" and isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            yield from process_operation(item.get("o"), item.get("v"), item.get("p", ""))
                else:
                    yield from process_operation(o, v, p)

                if emitted_any == emitted_before_chunk:
                    unknown_chunks += 1
                    logger.debug(
                        "DeepSeek stream chunk produced no recognized events",
                        extra={"target_path": target_path, "chunk_keys": sorted(chunk.keys())},
                    )

            if not emitted_any:
                raise APIRequestError(
                    f"Streaming response ended without any recognized events for {target_path}."
                )
            if unknown_chunks:
                logger.debug(
                    "DeepSeek stream ignored unrecognized chunks",
                    extra={"target_path": target_path, "unknown_chunks": unknown_chunks},
                )

        except requests.RequestException as exc:
            raise APIRequestError(f"Streaming response interrupted for {target_path}: {exc.__class__.__name__}") from exc
        finally:
            resp.close()

    def completion(
        self,
        session_id: str,
        prompt: str,
        model_type: str = "default",
        thinking_enabled: bool = False,
        search_enabled: bool = False,
        parent_message_id: Optional[Union[int, str]] = None,
        ref_file_ids: Optional[list] = None,
    ) -> Generator["StreamEvent", None, None]:
        """Send a chat message and stream the response."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        prompt = ensure_string(prompt, "prompt", max_length=MAX_PROMPT_LENGTH)
        model_type = ensure_string(model_type, "model_type", max_length=64)
        parent_message_id = ensure_optional_message_id(parent_message_id, "parent_message_id", max_length=128)
        ref_file_ids = ensure_string_list(ref_file_ids, "ref_file_ids", max_items=MAX_REF_FILES)
        thinking_enabled = ensure_bool(thinking_enabled, "thinking_enabled")
        search_enabled = ensure_bool(search_enabled, "search_enabled")

        payload = {
            "chat_session_id": session_id,
            "parent_message_id": parent_message_id,
            "model_type": model_type,
            "prompt": prompt,
            "ref_file_ids": ref_file_ids or [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "preempt": False,
        }
        yield from self._stream_structured("/api/v0/chat/completion", payload)

    def continue_response(self, session_id: str, message_id: Union[int, str]) -> Generator["StreamEvent", None, None]:
        """Continue a truncated response."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        message_id = ensure_optional_message_id(message_id, "message_id", max_length=128)
        if message_id is None:
            raise ValidationError("message_id must be a non-empty string or positive integer.")
        payload = {"chat_session_id": session_id, "message_id": message_id}
        yield from self._stream_structured("/api/v0/chat/continue", payload)

    def regenerate(self, session_id: str, message_id: Union[int, str]) -> Generator["StreamEvent", None, None]:
        """Regenerate a response."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        message_id = ensure_optional_message_id(message_id, "message_id", max_length=128)
        if message_id is None:
            raise ValidationError("message_id must be a non-empty string or positive integer.")
        payload = {"chat_session_id": session_id, "message_id": message_id}
        yield from self._stream_structured("/api/v0/chat/regenerate", payload)

    def edit_message(self, session_id: str, message_id: Union[int, str], prompt: str) -> Generator["StreamEvent", None, None]:
        """Edit a message and stream the new response."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        message_id = ensure_optional_message_id(message_id, "message_id", max_length=128)
        if message_id is None:
            raise ValidationError("message_id must be a non-empty string or positive integer.")
        prompt = ensure_string(prompt, "prompt", max_length=MAX_PROMPT_LENGTH)
        payload = {"chat_session_id": session_id, "message_id": message_id, "prompt": prompt}
        yield from self._stream_structured("/api/v0/chat/edit_message", payload)

    def get_history(self, session_id: str) -> dict:
        """Get chat history for a session."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        return self.http.post("/api/v0/chat/history_messages", json_data={"chat_session_id": session_id})

    def stop_stream(self, session_id: str) -> dict:
        """Stop an active stream."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        return self.http.post("/api/v0/chat/stop_stream", json_data={"chat_session_id": session_id})

    def resume_stream(self, session_id: str) -> dict:
        """Resume a stream."""
        session_id = ensure_string(session_id, "session_id", max_length=128)
        return self.http.post("/api/v0/chat/resume_stream", json_data={"chat_session_id": session_id})

    def submit_feedback(self, message_id: Union[int, str], feedback: str) -> dict:
        """Submit feedback for a message (like/dislike)."""
        message_id = ensure_optional_message_id(message_id, "message_id", max_length=128)
        if message_id is None:
            raise ValidationError("message_id must be a non-empty string or positive integer.")
        feedback = ensure_string(feedback, "feedback", max_length=64)
        return self.http.post(
            "/api/v0/chat/message_feedback",
            json_data={"message_id": message_id, "feedback": feedback}
        )
