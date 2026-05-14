import asyncio
import base64
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from DS2API.async_client import AsyncDeepSeekClient
from DS2API.client import DeepSeekClient, MODEL_EXPERT, MODEL_INSTANT
from DS2API.src.api_chat import ChatAPI, MAX_REF_FILES
from DS2API.src.api_file import FileAPI
from DS2API.src.exceptions import (
    APIRequestError,
    AuthenticationError,
    FileProcessingTimeoutError,
    JSONParseError,
    PoWSolverError,
    ValidationError,
)
from DS2API.src.http_client import DeepSeekHTTPClient
from DS2API.src.models import ChatResponse, Citation, StreamEvent


def sse(payload):
    return "data: " + json.dumps(payload, separators=(",", ":"))


class FakePow:
    def __init__(self, nonce=7):
        self.nonce = nonce
        self.calls = []

    def solve(self, challenge, salt, expire_at, difficulty):
        self.calls.append((challenge, salt, expire_at, difficulty))
        return self.nonce


class FakeStreamResponse:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def iter_lines(self, decode_unicode=True):
        yield from self.lines

    def close(self):
        self.closed = True


class FakeChatHTTP:
    def __init__(self, lines=None, challenge=None):
        self.lines = lines or []
        self.challenge = challenge or {
            "algorithm": "DeepSeekHashV1",
            "challenge": "target",
            "salt": "salt",
            "difficulty": 1,
            "expire_at": 123,
            "signature": "sig",
            "target_path": "/api/v0/chat/completion",
        }
        self.post_calls = []
        self.post_stream_calls = []
        self.response = None

    def post(self, path, json_data=None, files=None, headers=None):
        self.post_calls.append({"path": path, "json_data": json_data, "files": files, "headers": headers})
        if path == "/api/v0/chat/create_pow_challenge":
            challenge = dict(self.challenge)
            if json_data and "target_path" in json_data:
                challenge["target_path"] = json_data["target_path"]
            return {"code": 0, "data": {"biz_data": {"challenge": challenge}}}
        return {"code": 0, "data": {"biz_data": {}}}

    def post_stream(self, path, json_data, headers=None):
        self.post_stream_calls.append({"path": path, "json_data": json_data, "headers": headers})
        self.response = FakeStreamResponse(self.lines)
        return self.response


class FakeSessionAPI:
    def __init__(self, session_id="session-1", delete_raises=False):
        self.session_id = session_id
        self.delete_raises = delete_raises
        self.created = []
        self.deleted = []

    def create(self, model_type="default"):
        self.created.append(model_type)
        return {"data": {"biz_data": {"id": self.session_id}}}

    def delete(self, session_id):
        self.deleted.append(session_id)
        if self.delete_raises:
            raise RuntimeError("delete failed")
        return {"code": 0}


class FakeChatAPI:
    def __init__(self, events=None, error=None):
        self.events = events or []
        self.error = error
        self.structured_calls = []
        self.completion_calls = []
        self.regenerate_calls = []
        self.edit_calls = []
        self.continue_calls = []
        self.stop_calls = []

    def _iter_events(self):
        if self.error:
            raise self.error
        yield from self.events

    def _stream_structured(self, path, payload):
        self.structured_calls.append({"path": path, "payload": payload})
        yield from self._iter_events()

    def completion(self, session_id, prompt, model_type="default", thinking_enabled=False,
                   search_enabled=False, parent_message_id=None, ref_file_ids=None):
        self.completion_calls.append({
            "session_id": session_id,
            "prompt": prompt,
            "model_type": model_type,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "parent_message_id": parent_message_id,
            "ref_file_ids": ref_file_ids,
        })
        yield from self._iter_events()

    def regenerate(self, session_id, message_id):
        self.regenerate_calls.append((session_id, message_id))
        yield from self._iter_events()

    def edit_message(self, session_id, message_id, prompt):
        self.edit_calls.append((session_id, message_id, prompt))
        yield from self._iter_events()

    def continue_response(self, session_id, message_id):
        self.continue_calls.append((session_id, message_id))
        yield from self._iter_events()

    def stop_stream(self, session_id):
        self.stop_calls.append(session_id)
        return {"code": 0}


def make_client(events=None, error=None, session_api=None):
    client = DeepSeekClient(http_client=object(), pow_solver=FakePow())
    client.session = session_api or FakeSessionAPI()
    client.chat = FakeChatAPI(events=events, error=error)
    return client


class ClientFacadeContractTests(unittest.TestCase):
    def test_ask_creates_session_sends_payload_and_cleans_up(self):
        citation = Citation(title="DeepSeek", url="https://www.deepseek.com", snippet="source")
        events = [
            StreamEvent("MESSAGE_ID", 101),
            StreamEvent("THINK_TEXT", "reason "),
            StreamEvent("THINKING_DONE", 1.25),
            StreamEvent("SEARCH_RESULTS", [citation]),
            StreamEvent("TOKEN_USAGE", 42),
            StreamEvent("RESPONSE_TEXT", "answer"),
        ]
        session_api = FakeSessionAPI(session_id="temp-1")
        client = make_client(events=events, session_api=session_api)

        response = client.ask(
            "hello",
            model="expert",
            thinking=True,
            search=True,
            ref_file_ids=["file-1"],
        )

        self.assertEqual(session_api.created, [MODEL_EXPERT])
        self.assertEqual(session_api.deleted, ["temp-1"])
        call = client.chat.structured_calls[0]
        self.assertEqual(call["path"], "/api/v0/chat/completion")
        self.assertEqual(call["payload"]["chat_session_id"], "temp-1")
        self.assertEqual(call["payload"]["model_type"], MODEL_EXPERT)
        self.assertEqual(call["payload"]["thinking_enabled"], True)
        self.assertEqual(call["payload"]["search_enabled"], True)
        self.assertEqual(call["payload"]["ref_file_ids"], ["file-1"])
        self.assertIsNone(call["payload"]["parent_message_id"])
        self.assertEqual(response.message_id, 101)
        self.assertEqual(response.thinking, "reason ")
        self.assertEqual(response.thinking_elapsed, 1.25)
        self.assertEqual(response.token_usage, 42)
        self.assertEqual(response.citations, [citation])
        self.assertEqual(response.text, "answer")

    def test_ask_cleans_up_when_stream_raises_without_masking_original_error(self):
        session_api = FakeSessionAPI(session_id="temp-2", delete_raises=True)
        original = APIRequestError("stream failed", api_code=500)
        client = make_client(error=original, session_api=session_api)

        with self.assertLogs("DS2API.client", level="WARNING") as logs:
            with self.assertRaises(APIRequestError) as caught:
                client.ask("hello")

        self.assertIs(caught.exception, original)
        self.assertEqual(session_api.deleted, ["temp-2"])
        self.assertIn("Failed to delete DeepSeek chat session", "\n".join(logs.output))

    def test_ask_stream_cleans_up_after_generator_is_exhausted(self):
        session_api = FakeSessionAPI(session_id="stream-1")
        client = make_client(
            events=[StreamEvent("RESPONSE_TEXT", "a")],
            session_api=session_api,
        )

        events = list(client.ask_stream("hello", model="instant", ref_file_ids=["f1"]))

        self.assertEqual([event.content for event in events], ["a"])
        self.assertEqual(session_api.deleted, ["stream-1"])
        self.assertEqual(client.chat.completion_calls[0]["ref_file_ids"], ["f1"])

    def test_multi_turn_send_auto_chains_message_ids(self):
        client = make_client(events=[StreamEvent("MESSAGE_ID", 11), StreamEvent("RESPONSE_TEXT", "first")])
        session_id = client.new_chat(model="expert")

        first = client.send(session_id, "first prompt")

        self.assertEqual(first.message_id, 11)
        self.assertEqual(client.chat.structured_calls[0]["payload"]["parent_message_id"], None)

        client.chat.events = [StreamEvent("MESSAGE_ID", 12), StreamEvent("RESPONSE_TEXT", "second")]
        second = client.send(session_id, "second prompt")

        self.assertEqual(second.message_id, 12)
        self.assertEqual(client.chat.structured_calls[1]["payload"]["parent_message_id"], 11)
        self.assertEqual(client.chat.structured_calls[1]["payload"]["model_type"], MODEL_EXPERT)

    def test_send_stream_updates_last_message_id_as_events_are_consumed(self):
        client = make_client(events=[StreamEvent("MESSAGE_ID", 44), StreamEvent("RESPONSE_TEXT", "stream")])
        session_id = client.new_chat()

        list(client.send_stream(session_id, "stream me"))

        self.assertEqual(client._last_message_id[session_id], 44)

    def test_regenerate_edit_continue_and_stop_delegate_to_chat_api(self):
        client = make_client(events=[StreamEvent("MESSAGE_ID", 77), StreamEvent("RESPONSE_TEXT", "ok")])
        client.adopt_chat("s1")

        self.assertEqual(client.regenerate("s1", 1).message_id, 77)
        self.assertEqual(client.edit_message("s1", 2, "new").text, "ok")
        self.assertEqual(client.continue_response("s1", 3).text, "ok")
        self.assertEqual(client.stop("s1"), {"code": 0})

        self.assertEqual(client.chat.regenerate_calls, [("s1", "1")])
        self.assertEqual(client.chat.edit_calls, [("s1", "2", "new")])
        self.assertEqual(client.chat.continue_calls, [("s1", "3")])
        self.assertEqual(client.chat.stop_calls, ["s1"])

    def test_completion_text_returns_plain_text_for_text_helper_callers(self):
        client = make_client(events=[StreamEvent("MESSAGE_ID", 1), StreamEvent("RESPONSE_TEXT", "plain")])
        client.adopt_chat("s1")

        result = client.completion_text("s1", "hello", print_to_stdout=False)

        self.assertEqual(result, "plain")
        self.assertIsInstance(result, str)

    def test_model_aliases_and_unknown_model_validation(self):
        self.assertEqual(DeepSeekClient._resolve_model_type(" instant "), MODEL_INSTANT)
        self.assertEqual(DeepSeekClient._resolve_model_type("default"), MODEL_INSTANT)
        self.assertEqual(DeepSeekClient._resolve_model_type("expert"), MODEL_EXPERT)

        with self.assertRaises(ValidationError):
            DeepSeekClient._resolve_model_type("deepseek-chat")

    def test_extract_session_id_supports_known_response_shapes(self):
        self.assertEqual(DeepSeekClient.extract_session_id({"data": {"biz_data": {"id": "a"}}}), "a")
        self.assertEqual(
            DeepSeekClient.extract_session_id({"data": {"biz_data": {"chat_session": {"id": "b"}}}}),
            "b",
        )
        self.assertEqual(DeepSeekClient.extract_session_id({"data": {"biz_data": {"chat_session_id": "c"}}}), "c")

        with self.assertRaises(APIRequestError):
            DeepSeekClient.extract_session_id({"data": {"biz_data": {}}})


class ChatStreamContractTests(unittest.TestCase):
    def test_structured_stream_extracts_modes_citations_usage_finish_and_closes(self):
        lines = [
            sse({
                "v": {
                    "response": {
                        "message_id": 55,
                        "fragments": [
                            {"type": "THINK", "content": "think1 "},
                            {"type": "RESPONSE", "content": "answer1 "},
                        ],
                    }
                }
            }),
            sse({
                "o": "APPEND",
                "p": "/response/fragments",
                "v": [
                    {"type": "THINK", "content": "think2 "},
                    {"type": "RESPONSE", "content": "answer2 "},
                ],
            }),
            sse({
                "o": "BATCH",
                "v": [
                    {"o": "APPEND", "p": "/response/fragments/1/content", "v": "answer3"},
                    {"p": "/response/thinking/elapsed_secs", "v": 2.5},
                    {"p": "/search/results", "v": [{"title": "t", "url": "u", "snippet": "s"}]},
                    {"p": "/response/accumulated_token_usage", "v": 99},
                    {"p": "/response/status", "v": "FINISHED"},
                ],
            }),
            "data: [DONE]",
        ]
        http = FakeChatHTTP(lines=lines)
        pow_solver = FakePow(nonce=1234)
        chat = ChatAPI(http, pow_solver)

        events = list(chat.completion(
            "session-1",
            "hello",
            model_type="expert",
            thinking_enabled=True,
            search_enabled=True,
            ref_file_ids=["file-1"],
        ))

        self.assertTrue(http.response.closed)
        self.assertEqual(pow_solver.calls, [("target", "salt", 123, 1)])
        self.assertEqual(http.post_stream_calls[0]["json_data"]["model_type"], "expert")
        self.assertEqual(http.post_stream_calls[0]["json_data"]["thinking_enabled"], True)
        self.assertEqual(http.post_stream_calls[0]["json_data"]["search_enabled"], True)
        self.assertEqual(http.post_stream_calls[0]["json_data"]["ref_file_ids"], ["file-1"])

        pow_header = http.post_stream_calls[0]["headers"]["x-ds-pow-response"]
        decoded = json.loads(base64.b64decode(pow_header).decode("utf-8"))
        self.assertEqual(decoded["answer"], 1234)
        self.assertEqual(decoded["target_path"], "/api/v0/chat/completion")

        self.assertEqual(
            [(event.event_type, event.content) for event in events[:6]],
            [
                ("MESSAGE_ID", 55),
                ("THINK_TEXT", "think1 "),
                ("RESPONSE_TEXT", "answer1 "),
                ("THINK_TEXT", "think2 "),
                ("RESPONSE_TEXT", "answer2 "),
                ("RESPONSE_TEXT", "answer3"),
            ],
        )
        self.assertEqual(events[6].event_type, "THINKING_DONE")
        self.assertEqual(events[6].content, 2.5)
        self.assertEqual(events[7].event_type, "SEARCH_RESULTS")
        self.assertEqual(events[7].content, [Citation(title="t", url="u", snippet="s")])
        self.assertEqual(events[8].event_type, "TOKEN_USAGE")
        self.assertEqual(events[8].content, 99)
        self.assertEqual(events[9].event_type, "FINISHED")

    def test_streaming_api_error_chunk_raises_and_closes_response(self):
        http = FakeChatHTTP(lines=[sse({"code": 1001, "msg": "bad token"})])
        chat = ChatAPI(http, FakePow())

        with self.assertRaises(APIRequestError) as caught:
            list(chat.completion("session-1", "hello"))

        self.assertEqual(caught.exception.api_code, 1001)
        self.assertTrue(http.response.closed)

    def test_malformed_stream_json_raises_parse_error_and_closes_response(self):
        http = FakeChatHTTP(lines=["data: {not-json"])
        chat = ChatAPI(http, FakePow())

        with self.assertRaises(JSONParseError):
            list(chat.completion("session-1", "hello"))

        self.assertTrue(http.response.closed)

    def test_official_openai_delta_chunks_raise_clear_web_stream_error(self):
        http = FakeChatHTTP(lines=[
            sse({"choices": [{"delta": {"content": "hello"}}]}),
            "data: [DONE]",
        ])
        chat = ChatAPI(http, FakePow())

        with self.assertRaises(APIRequestError) as caught:
            list(chat.completion("session-1", "hello"))

        self.assertIn("without any recognized events", str(caught.exception))

    def test_chat_input_validation_fails_before_network_or_pow(self):
        chat = ChatAPI(FakeChatHTTP(), FakePow())

        invalid_calls = [
            lambda: list(chat.completion("", "hello")),
            lambda: list(chat.completion("session-1", "")),
            lambda: list(chat.completion("session-1", "hello", thinking_enabled="yes")),
            lambda: list(chat.completion("session-1", "hello", ref_file_ids=["x"] * (MAX_REF_FILES + 1))),
            lambda: list(chat.continue_response("session-1", "")),
            lambda: list(chat.regenerate("", "message-1")),
            lambda: list(chat.edit_message("session-1", "message-1", "")),
        ]

        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValidationError):
                    call()

    def test_pow_challenge_validation_reports_clear_errors(self):
        chat = ChatAPI(FakeChatHTTP(), FakePow())

        with self.assertRaises(APIRequestError):
            chat._solve_and_encode_pow({"challenge": "c"})

        with self.assertRaises(PoWSolverError):
            chat._solve_and_encode_pow({
                "algorithm": "Other",
                "challenge": "c",
                "salt": "s",
                "difficulty": 1,
                "expire_at": 1,
                "signature": "sig",
            })


class FakeFileHTTP:
    def __init__(self, challenge=None, upload_result=None, fetch_result=None, fetch_error=None):
        self.challenge = challenge or {
            "algorithm": "DeepSeekHashV1",
            "challenge": "target",
            "salt": "salt",
            "difficulty": 1,
            "expire_at": 123,
            "signature": "sig",
        }
        self.upload_result = upload_result or {"code": 0, "data": {"biz_data": {"id": "file-1", "status": "PENDING"}}}
        self.fetch_result = fetch_result or {
            "code": 0,
            "data": {"biz_data": {"files": [{"id": "file-1", "status": "READY", "file_name": "a.txt"}]}},
        }
        self.fetch_error = fetch_error
        self.post_calls = []
        self.get_calls = []

    def post(self, path, json_data=None, files=None, headers=None):
        self.post_calls.append({"path": path, "json_data": json_data, "files": files, "headers": headers})
        if path == "/api/v0/chat/create_pow_challenge":
            return {"code": 0, "data": {"biz_data": {"challenge": self.challenge}}}
        if path == FileAPI.UPLOAD_PATH:
            return self.upload_result
        raise AssertionError(f"Unexpected POST path: {path}")

    def get(self, path, params=None):
        self.get_calls.append({"path": path, "params": params})
        if self.fetch_error:
            raise self.fetch_error
        return self.fetch_result


class FileAPIContractTests(unittest.TestCase):
    def test_upload_solves_pow_sends_file_headers_and_updates_ready_status(self):
        http = FakeFileHTTP()
        api = FileAPI(http, FakePow(nonce=88), max_upload_size_bytes=1024)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("hello", encoding="utf-8")

            result = api.upload_file(str(path), wait_ready=True, timeout=1)

        upload_call = http.post_calls[1]
        self.assertEqual(upload_call["path"], FileAPI.UPLOAD_PATH)
        self.assertEqual(upload_call["headers"]["x-file-size"], "5")
        self.assertIn("x-ds-pow-response", upload_call["headers"])
        self.assertEqual(upload_call["files"]["file"][0], "sample.txt")
        self.assertEqual(http.get_calls, [{"path": "/api/v0/file/fetch_files", "params": {"file_ids": "file-1"}}])
        self.assertEqual(result["data"]["biz_data"]["status"], "READY")

    def test_upload_wait_ready_false_does_not_poll(self):
        http = FakeFileHTTP()
        api = FileAPI(http, FakePow())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("hello", encoding="utf-8")

            result = api.upload_file(str(path), wait_ready=False)

        self.assertEqual(result["data"]["biz_data"]["id"], "file-1")
        self.assertEqual(http.get_calls, [])

    def test_upload_rejects_oversized_file_before_pow_or_http_upload(self):
        http = FakeFileHTTP()
        api = FileAPI(http, FakePow(), max_upload_size_bytes=1)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("hello", encoding="utf-8")

            with self.assertRaises(ValidationError):
                api.upload_file(str(path))

        self.assertEqual(http.post_calls, [])

    def test_upload_polling_errors_are_not_swallowed(self):
        http = FakeFileHTTP(fetch_error=RuntimeError("network down"))
        api = FileAPI(http, FakePow())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("hello", encoding="utf-8")

            with self.assertRaises(APIRequestError) as caught:
                api.upload_file(str(path), wait_ready=True, timeout=1)

        self.assertIn("Failed to poll uploaded file status", str(caught.exception))

    def test_upload_timeout_raises_specific_file_processing_error(self):
        http = FakeFileHTTP(fetch_result={
            "code": 0,
            "data": {"biz_data": {"files": [{"id": "file-1", "status": "PENDING"}]}},
        })
        api = FileAPI(http, FakePow())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("hello", encoding="utf-8")

            with patch("DS2API.src.api_file.time.sleep", lambda _: None):
                with self.assertRaises(FileProcessingTimeoutError) as caught:
                    api.upload_file(str(path), wait_ready=True, timeout=0.01)

        self.assertIn("did not become ready", str(caught.exception))

    def test_fetch_files_rejects_empty_input(self):
        api = FileAPI(FakeFileHTTP(), FakePow())

        with self.assertRaises(ValidationError):
            api.fetch_files([])

    def test_file_pow_challenge_validation_reports_clear_errors(self):
        bad_http = FakeFileHTTP(challenge={"challenge": "target"})
        api = FileAPI(bad_http, FakePow())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("hello", encoding="utf-8")

            with self.assertRaises(APIRequestError):
                api.upload_file(str(path), wait_ready=False)

        unsupported_http = FakeFileHTTP(challenge={
            "algorithm": "Other",
            "challenge": "target",
            "salt": "salt",
            "difficulty": 1,
            "expire_at": 123,
            "signature": "sig",
        })
        api = FileAPI(unsupported_http, FakePow())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.txt"
            path.write_text("hello", encoding="utf-8")

            with self.assertRaises(PoWSolverError):
                api.upload_file(str(path), wait_ready=False)

    def test_fetch_files_and_extract_file_info_cover_list_and_direct_shapes(self):
        http = FakeFileHTTP()
        api = FileAPI(http, FakePow())

        response = api.fetch_files(["file-1", "file-2"])

        self.assertEqual(response["code"], 0)
        self.assertEqual(http.get_calls[0]["params"], {"file_ids": "file-1,file-2"})

        direct = FileAPI.extract_file_info({
            "data": {"biz_data": {
                "id": "direct",
                "status": "READY",
                "file_name": "doc.pdf",
                "file_size": 10,
                "signed_preview_url": "https://preview",
                "token_usage": 3,
            }}
        })
        nested = FileAPI.extract_file_info({
            "data": {"biz_data": {"files": [{
                "id": "nested",
                "status": "READY",
                "file_name": "image.png",
                "file_size": 20,
            }]}}
        })

        self.assertEqual(direct.id, "direct")
        self.assertEqual(direct.preview_url, "https://preview")
        self.assertEqual(direct.token_usage, 3)
        self.assertEqual(nested.id, "nested")


class FakeHTTPResponse:
    def __init__(self, ok=True, status_code=200, headers=None, text='{"code":0}', json_data=None, json_error=None):
        self.ok = ok
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text
        self._json_data = json_data
        self._json_error = json_error
        self.closed = False

    def json(self):
        if self._json_error:
            raise self._json_error
        if self._json_data is not None:
            return self._json_data
        return json.loads(self.text)

    def close(self):
        self.closed = True


class RecordingSession(requests.Session):
    def __init__(self, response=None, error=None):
        super().__init__()
        self.response = response or FakeHTTPResponse(json_data={"code": 0, "data": {"biz_data": {}}})
        self.error = error
        self.calls = []
        self.closed = False

    def request(self, method, url, **kwargs):
        self.calls.append({"method": method, "url": url, "kwargs": kwargs})
        if self.error:
            raise self.error
        return self.response

    def close(self):
        self.closed = True
        super().close()


class HTTPClientContractTests(unittest.TestCase):
    def test_handle_response_surfaces_http_api_and_business_errors(self):
        client = DeepSeekHTTPClient(token="token", max_retries=0)

        with self.assertLogs("DS2API.src.http_client", level="WARNING"):
            with self.assertRaises(APIRequestError) as http_error:
                client._handle_response(
                    FakeHTTPResponse(
                        ok=False,
                        status_code=429,
                        headers={"x-request-id": "rid-1"},
                        text='{"token":"secret-token"}',
                    ),
                    "GET /x",
                )
        self.assertEqual(http_error.exception.status_code, 429)
        self.assertEqual(http_error.exception.request_id, "rid-1")
        self.assertNotIn("secret-token", str(http_error.exception))

        with self.assertRaises(APIRequestError) as api_error:
            client._handle_response(FakeHTTPResponse(json_data={"code": 4001, "msg": "bad"}), "POST /x")
        self.assertEqual(api_error.exception.api_code, 4001)

        with self.assertRaises(APIRequestError) as biz_error:
            client._handle_response(
                FakeHTTPResponse(json_data={"code": 0, "data": {"biz_code": 3001, "biz_msg": "rate limited"}}),
                "POST /x",
            )
        self.assertEqual(biz_error.exception.api_code, 3001)

        client.close()

    def test_handle_response_rejects_invalid_json_and_non_object_json(self):
        client = DeepSeekHTTPClient(token="token", max_retries=0)

        with self.assertRaises(JSONParseError):
            client._handle_response(
                FakeHTTPResponse(text="not json", json_error=ValueError("bad json")),
                "GET /bad-json",
            )

        with self.assertRaises(JSONParseError):
            client._handle_response(FakeHTTPResponse(json_data=[]), "GET /array")

        client.close()

    def test_request_normalizes_network_errors(self):
        session = RecordingSession(error=requests.Timeout("boom"))
        client = DeepSeekHTTPClient(token="token", session=session, max_retries=0)

        with self.assertLogs("DS2API.src.http_client", level="WARNING"):
            with self.assertRaises(APIRequestError) as caught:
                client.get("/api/v0/ping")

        self.assertIn("Network error", str(caught.exception))

    def test_files_request_removes_json_content_type_for_multipart_boundary(self):
        session = RecordingSession()
        client = DeepSeekHTTPClient(token="token", session=session, max_retries=0)

        client.post("/api/v0/file/upload_file", files={"file": ("a.txt", b"hi", "text/plain")})

        headers = session.calls[0]["kwargs"]["headers"]
        self.assertNotIn("content-type", {key.lower() for key in headers})

    def test_constructor_validation_rejects_bad_auth_timeout_retry_and_trust_env(self):
        invalid_constructors = [
            lambda: DeepSeekHTTPClient(),
            lambda: DeepSeekHTTPClient(token=" "),
            lambda: DeepSeekHTTPClient(token="token", timeout=0),
            lambda: DeepSeekHTTPClient(token="token", timeout=(1, 0)),
            lambda: DeepSeekHTTPClient(token="token", max_retries=True),
            lambda: DeepSeekHTTPClient(token="token", retry_statuses=[0]),
            lambda: DeepSeekHTTPClient(token="token", retry_backoff_factor=-1),
            lambda: DeepSeekHTTPClient(token="token", trust_env="false"),
        ]

        for constructor in invalid_constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises((AuthenticationError, ValidationError)):
                    constructor()


class FakeSyncClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.closed = False
        FakeSyncClient.instances.append(self)

    def ask(self, **kwargs):
        self.calls.append(("ask", kwargs))
        return ChatResponse(text="async answer")

    def adopt_chat(self, session_id, model="instant", last_message_id=None):
        self.calls.append(("adopt_chat", {
            "session_id": session_id,
            "model": model,
            "last_message_id": last_message_id,
        }))
        return session_id

    def delete_chat(self, session_id):
        self.calls.append(("delete_chat", {"session_id": session_id}))
        return {"code": 0}

    def send(self, **kwargs):
        self.calls.append(("send", kwargs))
        return ChatResponse(text="async send")

    def ask_stream(self, **kwargs):
        self.calls.append(("ask_stream", kwargs))
        yield StreamEvent("RESPONSE_TEXT", "chunk")

    def send_stream(self, **kwargs):
        self.calls.append(("send_stream", kwargs))
        yield StreamEvent("RESPONSE_TEXT", "send chunk")

    def close(self):
        self.closed = True


class BlockingStream:
    def __init__(self):
        self.closed = threading.Event()
        self.yielded = False

    def __iter__(self):
        return self

    def __next__(self):
        if not self.yielded:
            self.yielded = True
            return StreamEvent("RESPONSE_TEXT", "first")
        if self.closed.wait(timeout=5):
            raise StopIteration
        raise RuntimeError("stream was not closed")

    def close(self):
        self.closed.set()


class BlockingSyncClient(FakeSyncClient):
    def ask_stream(self, **kwargs):
        self.calls.append(("ask_stream", kwargs))
        stream = BlockingStream()
        self.active_stream = stream
        return stream


class AsyncClientContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        FakeSyncClient.instances = []

    async def test_async_client_passes_constructor_options_and_file_ids(self):
        with patch("DS2API.async_client.DeepSeekClient", FakeSyncClient):
            client = AsyncDeepSeekClient(
                token="token",
                timeout=(1, 2),
                max_retries=0,
                default_headers={"x-test": "yes"},
            )
            result = await client.ask("hello", ref_file_ids=["file-1"], search=True)

        sync = FakeSyncClient.instances[0]
        self.assertEqual(sync.kwargs["token"], "token")
        self.assertEqual(sync.kwargs["timeout"], (1, 2))
        self.assertEqual(sync.kwargs["max_retries"], 0)
        self.assertEqual(sync.kwargs["default_headers"], {"x-test": "yes"})
        self.assertEqual(result.text, "async answer")
        self.assertEqual(sync.calls[0][0], "ask")
        self.assertEqual(sync.calls[0][1]["ref_file_ids"], ["file-1"])
        self.assertEqual(sync.calls[0][1]["search"], True)

    async def test_async_client_passes_adopt_and_parent_message_id(self):
        with patch("DS2API.async_client.DeepSeekClient", FakeSyncClient):
            client = AsyncDeepSeekClient(token="token")
            await client.adopt_chat("session-1", model="expert", last_message_id=10)
            await client.send("session-1", "hello", parent_message_id=10)
            await client.delete_chat("session-1")

        sync = FakeSyncClient.instances[0]
        self.assertEqual(sync.calls, [
            ("adopt_chat", {
                "session_id": "session-1",
                "model": "expert",
                "last_message_id": 10,
            }),
            ("send", {
                "session_id": "session-1",
                "prompt": "hello",
                "thinking": False,
                "search": False,
                "ref_file_ids": None,
                "parent_message_id": 10,
                "print_to_stdout": False,
            }),
            ("delete_chat", {"session_id": "session-1"}),
        ])

    async def test_async_stream_wrapper_yields_items_and_propagates_kwargs(self):
        with patch("DS2API.async_client.DeepSeekClient", FakeSyncClient):
            client = AsyncDeepSeekClient(token="token")
            events = [event async for event in client.ask_stream("hello", ref_file_ids=["file-1"])]

        sync = FakeSyncClient.instances[0]
        self.assertEqual(events, [StreamEvent("RESPONSE_TEXT", "chunk")])
        self.assertEqual(sync.calls[0][0], "ask_stream")
        self.assertEqual(sync.calls[0][1]["ref_file_ids"], ["file-1"])

    async def test_async_close_closes_underlying_sync_client(self):
        with patch("DS2API.async_client.DeepSeekClient", FakeSyncClient):
            client = AsyncDeepSeekClient(token="token")
            await client.close()

        self.assertTrue(FakeSyncClient.instances[0].closed)

    async def test_async_stream_closes_sync_generator_when_consumer_stops_early(self):
        with patch("DS2API.async_client.DeepSeekClient", BlockingSyncClient):
            client = AsyncDeepSeekClient(token="token")
            stream = client.ask_stream("hello")
            await stream.__anext__()
            await stream.aclose()
            sync = BlockingSyncClient.instances[0]
            self.assertTrue(sync.active_stream.closed.wait(timeout=1))


class OfficialDeepSeekCapabilityGapTests(unittest.TestCase):
    def test_official_api_capabilities_not_implemented_by_this_web_client_yet(self):
        missing_from_web_wrapper = {
            "list_models",
            "get_balance",
            "chat_completions",
            "fim_completion",
            "anthropic_messages",
            "create_api_key",
        }

        actually_missing = {name for name in missing_from_web_wrapper if not hasattr(DeepSeekClient, name)}

        self.assertEqual(actually_missing, missing_from_web_wrapper)

    def test_current_client_exposes_expected_web_chat_modules(self):
        class MinimalHTTP:
            base_url = "https://chat.deepseek.com"

        client = DeepSeekClient(http_client=MinimalHTTP(), pow_solver=FakePow())

        self.assertTrue(hasattr(client, "user"))
        self.assertTrue(hasattr(client, "session"))
        self.assertTrue(hasattr(client, "chat"))
        self.assertTrue(hasattr(client, "file"))
        self.assertEqual(client.base_url, "https://chat.deepseek.com")


if __name__ == "__main__":
    unittest.main()
