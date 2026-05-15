import json
import os
import sys
import threading
import unittest
from pathlib import Path
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from unittest.mock import patch

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from DS2API.server import _messages_to_prompt, _normalize_model, _resolve_token, make_handler
from DS2API.src.models import ChatResponse


class FakeClient:
    calls = []

    def __init__(self, token):
        self.token = token

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def ask(self, prompt, model="instant", thinking=False, search=False, ref_file_ids=None):
        self.calls.append(
            {
                "token": self.token,
                "prompt": prompt,
                "model": model,
                "thinking": thinking,
                "search": search,
                "ref_file_ids": ref_file_ids,
            }
        )
        return ChatResponse(text="server-ok", message_id=123, token_usage=9)


class ServerHelperTests(unittest.TestCase):
    def test_normalize_model_accepts_openai_style_aliases(self):
        self.assertEqual(_normalize_model("deepseek-chat"), "instant")
        self.assertEqual(_normalize_model("deepseek-reasoner"), "expert")

    def test_messages_to_prompt_preserves_roles_for_transcript(self):
        prompt = _messages_to_prompt(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ]
        )
        self.assertEqual(prompt, "system: Be brief.\nuser: Hello")

    def test_resolve_token_accepts_authentication_bearer_header(self):
        with patch.dict(os.environ, {}, clear=True):
            token = _resolve_token({"Authentication": "Bearer header-token"})

        self.assertEqual(token, "header-token")

    def test_resolve_token_accepts_authentication_bear_header(self):
        with patch.dict(os.environ, {}, clear=True):
            token = _resolve_token({"Authentication": "Bear header-token"})

        self.assertEqual(token, "header-token")

    def test_resolve_token_accepts_authorization_bearer_header(self):
        with patch.dict(os.environ, {}, clear=True):
            token = _resolve_token({"Authorization": "Bearer header-token"})

        self.assertEqual(token, "header-token")


class ServerEndpointTests(unittest.TestCase):
    def setUp(self):
        FakeClient.calls = []
        handler = make_handler(
            client_factory=FakeClient,
            token_resolver=lambda headers: "test-token",
        )
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=1)

    def post_json(self, path, payload):
        request = Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_ask_endpoint_calls_client(self):
        response = self.post_json(
            "/ask",
            {"prompt": "ping", "model": "expert", "thinking": True, "search": True},
        )

        self.assertEqual(response["text"], "server-ok")
        self.assertEqual(FakeClient.calls[0]["prompt"], "ping")
        self.assertEqual(FakeClient.calls[0]["model"], "expert")
        self.assertEqual(FakeClient.calls[0]["thinking"], True)
        self.assertEqual(FakeClient.calls[0]["search"], True)

    def test_openai_chat_completion_endpoint_returns_openai_shape(self):
        response = self.post_json(
            "/v1/chat/completions",
            {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

        self.assertEqual(response["object"], "chat.completion")
        self.assertEqual(response["choices"][0]["message"]["content"], "server-ok")
        self.assertEqual(FakeClient.calls[0]["prompt"], "ping")
        self.assertEqual(FakeClient.calls[0]["model"], "instant")


if __name__ == "__main__":
    unittest.main()
