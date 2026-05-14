"""Live smoke tests for the chat.deepseek.com-backed client.

Run manually with:

    $env:DS2API_LIVE_TOKEN = "..."
    python tests/live_smoke.py

The script avoids printing the token, full response bodies, or account details.
It is intentionally not part of the default unittest discovery because it uses
the live DeepSeek web service and consumes account quota.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from DS2API import DeepSeekClient
from DS2API.src.exceptions import APIRequestError, DeepSeekError


@dataclass
class Result:
    name: str
    ok: bool
    detail: str
    elapsed: float


def _summarize_error(exc: BaseException) -> str:
    bits = [exc.__class__.__name__]
    status_code = getattr(exc, "status_code", None)
    api_code = getattr(exc, "api_code", None)
    request_id = getattr(exc, "request_id", None)
    if status_code is not None:
        bits.append(f"status={status_code}")
    if api_code is not None:
        bits.append(f"api_code={api_code}")
    if request_id:
        bits.append(f"request_id={request_id}")
    message = str(exc).splitlines()[0]
    if len(message) > 240:
        message = message[:237] + "..."
    bits.append(message)
    return " | ".join(bits)


def _run(name: str, func: Callable[[], str]) -> Result:
    start = time.monotonic()
    try:
        detail = func()
        return Result(name=name, ok=True, detail=detail, elapsed=time.monotonic() - start)
    except Exception as exc:
        detail = _summarize_error(exc)
        if not isinstance(exc, DeepSeekError):
            detail += "\n" + "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return Result(name=name, ok=False, detail=detail, elapsed=time.monotonic() - start)


def _text_ok(value: str, min_len: int = 1) -> bool:
    return isinstance(value, str) and len(value.strip()) >= min_len


def main() -> int:
    token = os.environ.get("DS2API_LIVE_TOKEN")
    if not token:
        print("SKIP: DS2API_LIVE_TOKEN is not set.")
        return 0

    client = DeepSeekClient(token=token, timeout=(10, 45), stream_timeout=(10, 180))
    created_sessions: list[str] = []
    uploaded_file_id: Optional[str] = None

    def cleanup_sessions() -> None:
        for session_id in reversed(created_sessions):
            try:
                client.delete_chat(session_id)
            except APIRequestError:
                pass

    def auth_user() -> str:
        data = client.user.get_current()
        biz = data.get("data", {}).get("biz_data", {})
        keys = sorted(k for k in biz.keys() if k not in {"email", "mobile", "phone", "name"})
        return f"authenticated; user_keys={keys[:8]}"

    def one_shot_instant() -> str:
        response = client.ask(
            "Return only this exact marker: DS2API_LIVE_INSTANT_OK",
            model="instant",
        )
        if not _text_ok(response.text):
            raise AssertionError("instant one-shot returned empty response text")
        marker_seen = "DS2API_LIVE_INSTANT_OK" in response.text
        return f"text_len={len(response.text)} marker_seen={marker_seen} message_id={bool(response.message_id)}"

    def session_context() -> str:
        session_id = client.new_chat(model="instant")
        created_sessions.append(session_id)
        code = "BLUE-7319"
        first = client.send(session_id, f"Remember this code for the next message: {code}. Reply with OK only.")
        second = client.send(session_id, "What code did I ask you to remember? Reply with only the code.")
        if code not in second.text:
            raise AssertionError("session context did not preserve the code word")
        return (
            f"first_len={len(first.text)} second_len={len(second.text)} "
            f"message_chain={bool(first.message_id and second.message_id)}"
        )

    def streaming_instant() -> str:
        chunks = []
        event_types = []
        for event in client.ask_stream("Count from 1 to 3, separated by commas.", model="instant"):
            event_types.append(event.event_type)
            if event.event_type == "RESPONSE_TEXT" and event.content:
                chunks.append(str(event.content))
        text = "".join(chunks)
        if not _text_ok(text):
            raise AssertionError("stream returned no response text")
        return f"text_len={len(text)} events={sorted(set(event_types))}"

    def expert_thinking() -> str:
        response = client.ask(
            "Solve 12 * 7. Return the final number clearly.",
            model="expert",
            thinking=True,
        )
        if not _text_ok(response.text):
            raise AssertionError("expert thinking returned empty response text")
        return (
            f"text_len={len(response.text)} thinking_len={len(response.thinking)} "
            f"thinking_elapsed={response.thinking_elapsed}"
        )

    def web_search() -> str:
        response = client.ask(
            "Use web search if available: what is the official domain of DeepSeek? Answer briefly.",
            model="instant",
            search=True,
        )
        if not _text_ok(response.text):
            raise AssertionError("search request returned empty response text")
        return f"text_len={len(response.text)} citations={len(response.citations)}"

    def file_upload_and_attach() -> str:
        nonlocal uploaded_file_id
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "ds2api-live-note.txt"
            file_path.write_text("The live attachment marker is DS2API_FILE_OK.", encoding="utf-8")
            upload = client.file.upload_file(str(file_path), wait_ready=True, timeout=45)
        info = client.file.extract_file_info(upload)
        uploaded_file_id = info.id
        if not info.id:
            raise AssertionError("upload returned no file id")

        response = client.ask(
            "Read the attached text file and return the marker string only.",
            model="instant",
            ref_file_ids=[info.id],
        )
        if "DS2API_FILE_OK" not in response.text:
            raise AssertionError("file attachment response did not include marker")
        return f"file_status={info.status} file_id_present={bool(info.id)} text_len={len(response.text)}"

    tests: list[tuple[str, Callable[[], str]]] = [
        ("auth_user", auth_user),
        ("one_shot_instant", one_shot_instant),
        ("session_context", session_context),
        ("streaming_instant", streaming_instant),
        ("expert_thinking", expert_thinking),
        ("web_search", web_search),
        ("file_upload_and_attach", file_upload_and_attach),
    ]

    results = [_run(name, func) for name, func in tests]
    cleanup_sessions()
    client.close()

    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"{status} {result.name} ({result.elapsed:.1f}s): {result.detail}")

    failed = [result for result in results if not result.ok]
    if uploaded_file_id:
        print("INFO uploaded_file_id_present=True")
    print(f"SUMMARY passed={len(results) - len(failed)} failed={len(failed)} total={len(results)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
