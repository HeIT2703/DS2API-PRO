import sys
import unittest
from pathlib import Path

import requests

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from DS2API.client import DeepSeekClient
from DS2API.src.api_chat import ChatAPI
from DS2API.src.api_file import FileAPI
from DS2API.src.exceptions import APIRequestError, JSONParseError, ValidationError
from DS2API.src.http_client import DeepSeekHTTPClient
from DS2API.src.validation import ensure_api_path


class FakeStreamResponse:
    def __init__(self, lines):
        self.lines = lines
        self.closed = False

    def iter_lines(self, decode_unicode=True):
        yield from self.lines

    def close(self):
        self.closed = True


class FakeHTTP:
    def __init__(self, lines):
        self.lines = lines
        self.response = None

    def post(self, path, json_data=None, files=None, headers=None):
        return {
            "code": 0,
            "data": {
                "biz_data": {
                    "challenge": {
                        "algorithm": "DeepSeekHashV1",
                        "challenge": "target",
                        "salt": "salt",
                        "difficulty": 1,
                        "expire_at": 1,
                        "signature": "sig",
                        "target_path": json_data["target_path"],
                    }
                }
            },
        }

    def post_stream(self, path, json_data, headers=None):
        self.response = FakeStreamResponse(self.lines)
        return self.response


class FakePow:
    def solve(self, challenge, salt, expire_at, difficulty):
        return 42


class FakeJSONResponse:
    ok = True
    status_code = 200
    headers = {}
    text = '{"code":0}'

    def json(self):
        return {"code": 0, "data": {"biz_data": {}}}


class RecordingSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.last_request = None

    def request(self, method, url, **kwargs):
        self.last_request = (method, url, kwargs)
        return FakeJSONResponse()


class HardeningTests(unittest.TestCase):
    def test_api_error_string_redacts_sensitive_body(self):
        err = APIRequestError(
            "failed",
            status_code=401,
            response_body='{"token":"secret-token","authorization":"Bearer secret","email":"user@example.com"}',
        )

        rendered = str(err)

        self.assertNotIn("secret-token", rendered)
        self.assertNotIn("Bearer secret", rendered)
        self.assertNotIn("user@example.com", rendered)
        self.assertIn("<redacted", rendered)

    def test_api_path_validation_rejects_absolute_urls(self):
        with self.assertRaises(ValidationError):
            ensure_api_path("https://example.com/api")

        with self.assertRaises(ValidationError):
            ensure_api_path("//example.com/api")

    def test_http_header_validation_rejects_newlines(self):
        client = DeepSeekHTTPClient(token="token")

        with self.assertRaises(ValidationError):
            client._headers({"x-test": "bad\nvalue"})

        client.close()

    def test_stream_parser_raises_on_malformed_json_and_closes_response(self):
        http = FakeHTTP(["data: {not-json"])
        chat = ChatAPI(http, FakePow())

        with self.assertRaises(JSONParseError):
            list(chat.completion("session-id", "hello"))

        self.assertTrue(http.response.closed)

    def test_stream_parser_extracts_supported_chunk_formats(self):
        http = FakeHTTP([
            'data: {"v":"hello"}',
            'data: {"v":{"response":{"message_id":123,"fragments":[{"type":"RESPONSE","content":" hi"}]}}}',
            "data: [DONE]",
        ])
        chat = ChatAPI(http, FakePow())

        events = list(chat._stream_structured("/api/v0/chat/completion", {"prompt": "hello"}))

        self.assertEqual([event.event_type for event in events], ["RESPONSE_TEXT", "MESSAGE_ID", "RESPONSE_TEXT"])
        self.assertEqual([event.content for event in events], ["hello", 123, " hi"])

    def test_upload_file_rejects_missing_path_before_http_call(self):
        api = FileAPI(http_client=object(), pow_solver=FakePow())

        with self.assertRaises(ValidationError):
            api.upload_file("missing-file.txt")

    def test_client_exposes_low_level_controls(self):
        session = RecordingSession()
        pow_solver = FakePow()
        client = DeepSeekClient(
            token="token",
            session=session,
            pow_solver=pow_solver,
            user_agent="CustomAgent/1.0",
            accept_language="vi-VN",
            app_version="9.9.9",
            client_locale="vi_VN",
            client_timezone_offset=420,
            default_headers={"x-custom": "yes"},
            trust_env=False,
            proxies={"https": "http://proxy.local:8080"},
            verify=False,
            request_options={"allow_redirects": False},
            max_retries=0,
            max_upload_size_bytes=123,
        )

        result = client.get("/api/v0/ping", request_options={"verify": True})

        self.assertEqual(result["code"], 0)
        self.assertIs(client.pow_solver, pow_solver)
        self.assertIs(client.requests_session, session)
        self.assertFalse(session.trust_env)
        self.assertEqual(client.file.max_upload_size_bytes, 123)

        method, url, kwargs = session.last_request
        self.assertEqual(method, "GET")
        self.assertEqual(url, "https://chat.deepseek.com/api/v0/ping")
        self.assertTrue(kwargs["verify"])
        self.assertFalse(kwargs["allow_redirects"])
        self.assertEqual(kwargs["proxies"], {"https": "http://proxy.local:8080"})

        headers = kwargs["headers"]
        self.assertEqual(headers["authorization"], "Bearer token")
        self.assertEqual(headers["user-agent"], "CustomAgent/1.0")
        self.assertEqual(headers["accept-language"], "vi-VN")
        self.assertEqual(headers["x-app-version"], "9.9.9")
        self.assertEqual(headers["x-client-locale"], "vi_VN")
        self.assertEqual(headers["x-client-timezone-offset"], "420")
        self.assertEqual(headers["x-custom"], "yes")

    def test_client_accepts_raw_authorization_and_header_mutation(self):
        session = RecordingSession()
        client = DeepSeekClient(
            authorization="Custom auth-value",
            session=session,
            pow_solver=FakePow(),
            max_retries=0,
        )

        client.set_default_header("x-mode", "debug")
        response = client.request("POST", "/api/v0/raw", json_data={"ok": True})
        client.set_cookie("k", "v")
        client.remove_default_header("X-MODE")

        self.assertIsInstance(response, FakeJSONResponse)
        method, _, kwargs = session.last_request
        self.assertEqual(method, "POST")
        self.assertEqual(kwargs["headers"]["authorization"], "Custom auth-value")
        self.assertEqual(kwargs["headers"]["x-mode"], "debug")
        self.assertEqual(client.cookies.get("k"), "v")
        self.assertNotIn("x-mode", client.default_headers)


if __name__ == "__main__":
    unittest.main()
