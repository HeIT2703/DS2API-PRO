"""Local HTTP server for DS2API.

Run from this folder:

    python server.py --host 127.0.0.1 --port 8000

Set DEEPSEEK_USER_TOKEN or pass it as an Authorization bearer token.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlparse

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from DS2API import DeepSeekClient
    from DS2API.src.exceptions import AuthenticationError, DeepSeekError, ValidationError
else:
    from . import DeepSeekClient
    from .src.exceptions import AuthenticationError, DeepSeekError, ValidationError


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
MODEL_ALIASES = {
    "default": "instant",
    "instant": "instant",
    "expert": "expert",
    "deepseek-chat": "instant",
    "deepseek-v3": "instant",
    "deepseek-reasoner": "expert",
    "deepseek-r1": "expert",
}


class RequestError(Exception):
    def __init__(self, status: HTTPStatus, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _normalize_model(model: Any) -> str:
    if model is None:
        return "instant"
    if not isinstance(model, str):
        raise RequestError(HTTPStatus.BAD_REQUEST, "model must be a string")
    key = model.strip().lower()
    if key not in MODEL_ALIASES:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            "model must be one of: instant, expert, deepseek-chat, deepseek-reasoner",
        )
    return MODEL_ALIASES[key]


def _coerce_bool(value: Any, name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    raise RequestError(HTTPStatus.BAD_REQUEST, f"{name} must be a boolean")


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    raise RequestError(HTTPStatus.BAD_REQUEST, "message content must be a string or text parts")


def _messages_to_prompt(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise RequestError(HTTPStatus.BAD_REQUEST, "messages must be a non-empty list")

    parts: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            raise RequestError(HTTPStatus.BAD_REQUEST, "each message must be an object")
        role = str(message.get("role") or "user").strip() or "user"
        text = _extract_text(message.get("content", ""))
        if text:
            parts.append(f"{role}: {text}" if len(messages) > 1 else text)

    prompt = "\n".join(parts).strip()
    if not prompt:
        raise RequestError(HTTPStatus.BAD_REQUEST, "messages must contain text")
    return prompt


def _header_value(headers: Mapping[str, str], name: str) -> Optional[str]:
    return headers.get(name) or headers.get(name.lower()) or headers.get(name.upper())


def _bearer_token(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    scheme, _, token = value.strip().partition(" ")
    if scheme.lower() in {"bearer", "bear"} and token.strip():
        return token.strip()
    return None


def _resolve_token(headers: Mapping[str, str]) -> str:
    token = os.environ.get("DEEPSEEK_USER_TOKEN")
    if token:
        return token

    for header_name in ("Authorization", "Authentication"):
        token = _bearer_token(_header_value(headers, header_name))
        if token:
            return token

    raise RequestError(
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "Set DEEPSEEK_USER_TOKEN or pass Authorization/Authentication: Bearer <token>.",
    )


def _response_payload(response: Any) -> dict[str, Any]:
    return {
        "text": response.text,
        "thinking": response.thinking,
        "thinking_elapsed": response.thinking_elapsed,
        "message_id": response.message_id,
        "token_usage": response.token_usage,
        "citations": [asdict(citation) for citation in response.citations],
        "raw_text": response.raw_text,
    }


def _openai_payload(response: Any, model: str) -> dict[str, Any]:
    created = int(time.time())
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text,
                },
                "finish_reason": "stop",
            }
        ],
    }
    if response.token_usage is not None:
        payload["usage"] = {
            "prompt_tokens": 0,
            "completion_tokens": response.token_usage,
            "total_tokens": response.token_usage,
        }
    return payload


def make_handler(
    client_factory: Callable[..., DeepSeekClient] = DeepSeekClient,
    token_resolver: Callable[[Mapping[str, str]], str] = _resolve_token,
) -> type[BaseHTTPRequestHandler]:
    class DS2APIServerHandler(BaseHTTPRequestHandler):
        server_version = "DS2APIHTTP/0.1"

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                self._send_json({"status": "ok"})
                return
            if path == "/v1/models":
                self._send_json(
                    {
                        "object": "list",
                        "data": [
                            {"id": "deepseek-chat", "object": "model"},
                            {"id": "deepseek-reasoner", "object": "model"},
                            {"id": "instant", "object": "model"},
                            {"id": "expert", "object": "model"},
                        ],
                    }
                )
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")

        def do_POST(self) -> None:
            try:
                path = urlparse(self.path).path
                payload = self._read_json()
                if path == "/ask":
                    self._handle_ask(payload)
                    return
                if path == "/v1/chat/completions":
                    self._handle_openai_chat_completion(payload)
                    return
                raise RequestError(HTTPStatus.NOT_FOUND, "Not found")
            except RequestError as exc:
                self._send_error(exc.status, exc.message)
            except ValidationError as exc:
                self._send_error(HTTPStatus.BAD_REQUEST, str(exc))
            except AuthenticationError as exc:
                self._send_error(HTTPStatus.UNAUTHORIZED, str(exc))
            except DeepSeekError as exc:
                self._send_error(HTTPStatus.BAD_GATEWAY, str(exc))

        def _handle_ask(self, payload: Mapping[str, Any]) -> None:
            prompt = payload.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                raise RequestError(HTTPStatus.BAD_REQUEST, "prompt must be a non-empty string")
            model = _normalize_model(payload.get("model"))
            thinking = _coerce_bool(payload.get("thinking"), "thinking")
            search = _coerce_bool(payload.get("search"), "search")
            ref_file_ids = payload.get("ref_file_ids")

            token = token_resolver(self.headers)
            with client_factory(token=token) as client:
                response = client.ask(
                    prompt,
                    model=model,
                    thinking=thinking,
                    search=search,
                    ref_file_ids=ref_file_ids,
                )
            self._send_json(_response_payload(response))

        def _handle_openai_chat_completion(self, payload: Mapping[str, Any]) -> None:
            if payload.get("stream"):
                raise RequestError(HTTPStatus.BAD_REQUEST, "stream=true is not supported by this local server")

            requested_model = payload.get("model") or "deepseek-chat"
            model = _normalize_model(requested_model)
            prompt = _messages_to_prompt(payload.get("messages"))
            thinking_default = str(requested_model).strip().lower() in {"deepseek-reasoner", "deepseek-r1"}
            thinking = _coerce_bool(payload.get("thinking"), "thinking", default=thinking_default)
            search = _coerce_bool(payload.get("search"), "search")

            token = token_resolver(self.headers)
            with client_factory(token=token) as client:
                response = client.ask(prompt, model=model, thinking=thinking, search=search)
            self._send_json(_openai_payload(response, str(requested_model)))

        def _read_json(self) -> Mapping[str, Any]:
            content_length = self.headers.get("Content-Length")
            if not content_length:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Request body must be JSON")
            try:
                length = int(content_length)
            except ValueError as exc:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from exc

            raw_body = self.rfile.read(length)
            try:
                payload = json.loads(raw_body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise RequestError(HTTPStatus.BAD_REQUEST, "Request body must be valid JSON") from exc
            if not isinstance(payload, Mapping):
                raise RequestError(HTTPStatus.BAD_REQUEST, "Request body must be a JSON object")
            return payload

        def _send_json(self, payload: Mapping[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = _json_bytes(payload)
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, status: HTTPStatus, message: str) -> None:
            self._send_json({"error": {"message": message, "type": status.phrase}}, status=status)

    return DS2APIServerHandler


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), make_handler())
    print(f"DS2API local server listening on http://{host}:{port}")
    print("POST /ask or /v1/chat/completions. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping DS2API local server.")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local HTTP server for DS2API.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Bind host. Defaults to {DEFAULT_HOST}.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Bind port. Defaults to {DEFAULT_PORT}.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_server(host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
