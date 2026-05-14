import sys
import unittest
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_PARENT))

from DS2API.client import DeepSeekClient, MODEL_EXPERT, MODEL_INSTANT
from DS2API.src.exceptions import APIRequestError, ValidationError
from DS2API.src.models import StreamEvent


class FakePow:
    def solve(self, challenge, salt, expire_at, difficulty):
        return 1


class IncrementingSessionAPI:
    def __init__(self):
        self.created = []
        self.deleted = []
        self.next_id = 1

    def create(self, model_type="default"):
        session_id = f"session-{self.next_id}"
        self.next_id += 1
        self.created.append({"session_id": session_id, "model_type": model_type})
        return {"data": {"biz_data": {"id": session_id}}}

    def delete(self, session_id):
        self.deleted.append(session_id)
        return {"code": 0}


class RecordingChatAPI:
    def __init__(self, event_batches=None, error=None):
        self.event_batches = list(event_batches or [])
        self.error = error
        self.structured_calls = []
        self.completion_calls = []

    def _next_events(self):
        if self.error:
            raise self.error
        if self.event_batches:
            yield from self.event_batches.pop(0)
            return
        yield from [StreamEvent("MESSAGE_ID", 999), StreamEvent("RESPONSE_TEXT", "ok")]

    def _stream_structured(self, path, payload):
        self.structured_calls.append({"path": path, "payload": payload})
        yield from self._next_events()

    def completion(
        self,
        session_id,
        prompt,
        model_type="default",
        thinking_enabled=False,
        search_enabled=False,
        parent_message_id=None,
        ref_file_ids=None,
    ):
        self.completion_calls.append({
            "session_id": session_id,
            "prompt": prompt,
            "model_type": model_type,
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "parent_message_id": parent_message_id,
            "ref_file_ids": ref_file_ids,
        })
        yield from self._next_events()


def make_client(event_batches=None, error=None):
    client = DeepSeekClient(http_client=object(), pow_solver=FakePow())
    client.session = IncrementingSessionAPI()
    client.chat = RecordingChatAPI(event_batches=event_batches, error=error)
    return client


class WebChatCoreFeatureTests(unittest.TestCase):
    def test_one_shot_ask_is_isolated_between_calls(self):
        client = make_client(event_batches=[
            [StreamEvent("MESSAGE_ID", 101), StreamEvent("RESPONSE_TEXT", "first")],
            [StreamEvent("MESSAGE_ID", 202), StreamEvent("RESPONSE_TEXT", "second")],
        ])

        first = client.ask("first prompt", model="instant")
        second = client.ask("second prompt", model="instant")

        self.assertEqual(first.text, "first")
        self.assertEqual(second.text, "second")
        self.assertEqual(
            client.session.created,
            [
                {"session_id": "session-1", "model_type": MODEL_INSTANT},
                {"session_id": "session-2", "model_type": MODEL_INSTANT},
            ],
        )
        self.assertEqual(client.session.deleted, ["session-1", "session-2"])
        self.assertEqual(client.chat.structured_calls[0]["payload"]["parent_message_id"], None)
        self.assertEqual(client.chat.structured_calls[1]["payload"]["parent_message_id"], None)
        self.assertEqual(client._last_message_id, {})
        self.assertEqual(client._session_model, {})

    def test_new_chat_session_keeps_model_and_context_chain(self):
        client = make_client(event_batches=[
            [StreamEvent("MESSAGE_ID", 11), StreamEvent("RESPONSE_TEXT", "hello")],
            [StreamEvent("MESSAGE_ID", 12), StreamEvent("RESPONSE_TEXT", "remembered")],
        ])

        session_id = client.new_chat(model="expert")
        first = client.send(session_id, "My project name is DS2API.")
        second = client.send(session_id, "What is my project name?")

        self.assertEqual(session_id, "session-1")
        self.assertEqual(first.message_id, 11)
        self.assertEqual(second.message_id, 12)
        self.assertEqual(client._session_model[session_id], MODEL_EXPERT)
        self.assertEqual(client.chat.structured_calls[0]["payload"]["parent_message_id"], None)
        self.assertEqual(client.chat.structured_calls[1]["payload"]["parent_message_id"], 11)
        self.assertEqual(client.chat.structured_calls[0]["payload"]["model_type"], MODEL_EXPERT)
        self.assertEqual(client.chat.structured_calls[1]["payload"]["model_type"], MODEL_EXPERT)

    def test_adopt_chat_registers_existing_session_explicitly(self):
        client = make_client(event_batches=[
            [StreamEvent("MESSAGE_ID", 31), StreamEvent("RESPONSE_TEXT", "adopted")]
        ])

        session_id = client.adopt_chat("existing-session", model="expert", last_message_id=30)
        response = client.send(session_id, "continue")

        self.assertEqual(response.text, "adopted")
        self.assertEqual(client.chat.structured_calls[0]["payload"]["chat_session_id"], "existing-session")
        self.assertEqual(client.chat.structured_calls[0]["payload"]["model_type"], MODEL_EXPERT)
        self.assertEqual(client.chat.structured_calls[0]["payload"]["parent_message_id"], 30)

    def test_delete_chat_clears_local_state(self):
        client = make_client(event_batches=[
            [StreamEvent("MESSAGE_ID", 41), StreamEvent("RESPONSE_TEXT", "first")]
        ])
        session_id = client.new_chat()
        client.send(session_id, "hello")

        result = client.delete_chat(session_id)

        self.assertEqual(result, {"code": 0})
        self.assertEqual(client.session.deleted, [session_id])
        self.assertNotIn(session_id, client._session_model)
        self.assertNotIn(session_id, client._last_message_id)

        with self.assertRaises(ValidationError):
            client.send(session_id, "after delete")

    def test_instant_and_expert_modes_plus_thinking_and_search_flags(self):
        client = make_client(event_batches=[
            [StreamEvent("MESSAGE_ID", 1), StreamEvent("RESPONSE_TEXT", "instant")],
            [StreamEvent("MESSAGE_ID", 2), StreamEvent("RESPONSE_TEXT", "expert")],
        ])

        client.ask("quick answer", model="instant", thinking=False, search=True)
        client.ask("deep answer", model="expert", thinking=True, search=False)

        instant_payload = client.chat.structured_calls[0]["payload"]
        expert_payload = client.chat.structured_calls[1]["payload"]
        self.assertEqual(instant_payload["model_type"], MODEL_INSTANT)
        self.assertEqual(instant_payload["thinking_enabled"], False)
        self.assertEqual(instant_payload["search_enabled"], True)
        self.assertEqual(expert_payload["model_type"], MODEL_EXPERT)
        self.assertEqual(expert_payload["thinking_enabled"], True)
        self.assertEqual(expert_payload["search_enabled"], False)

    def test_file_attachment_payload_works_for_one_shot_and_session_chat(self):
        client = make_client(event_batches=[
            [StreamEvent("MESSAGE_ID", 1), StreamEvent("RESPONSE_TEXT", "one-shot file")],
            [StreamEvent("MESSAGE_ID", 2), StreamEvent("RESPONSE_TEXT", "session file")],
        ])

        client.ask("summarize this", ref_file_ids=["file-a"])
        session_id = client.new_chat(model="instant")
        client.send(session_id, "continue with file", ref_file_ids=["file-b"], search=True)

        one_shot_payload = client.chat.structured_calls[0]["payload"]
        session_payload = client.chat.structured_calls[1]["payload"]
        self.assertEqual(one_shot_payload["ref_file_ids"], ["file-a"])
        self.assertEqual(session_payload["ref_file_ids"], ["file-b"])
        self.assertEqual(session_payload["search_enabled"], True)

    def test_streaming_one_shot_uses_same_feature_payload_and_cleans_up(self):
        client = make_client(event_batches=[
            [StreamEvent("MESSAGE_ID", 7), StreamEvent("RESPONSE_TEXT", "chunk")]
        ])

        events = list(client.ask_stream(
            "stream this",
            model="expert",
            thinking=True,
            search=True,
            ref_file_ids=["file-stream"],
        ))

        self.assertEqual([event.content for event in events], [7, "chunk"])
        self.assertEqual(client.session.deleted, ["session-1"])
        call = client.chat.completion_calls[0]
        self.assertEqual(call["model_type"], MODEL_EXPERT)
        self.assertEqual(call["thinking_enabled"], True)
        self.assertEqual(call["search_enabled"], True)
        self.assertEqual(call["ref_file_ids"], ["file-stream"])

    def test_session_send_stream_chains_integer_message_id(self):
        client = make_client(event_batches=[
            [StreamEvent("MESSAGE_ID", 81), StreamEvent("RESPONSE_TEXT", "first")],
            [StreamEvent("MESSAGE_ID", 82), StreamEvent("RESPONSE_TEXT", "second")],
        ])
        session_id = client.new_chat()

        list(client.send_stream(session_id, "first stream"))
        list(client.send_stream(session_id, "second stream"))

        self.assertEqual(client.chat.completion_calls[0]["parent_message_id"], None)
        self.assertEqual(client.chat.completion_calls[1]["parent_message_id"], 81)
        self.assertEqual(client._last_message_id[session_id], 82)


class WebChatErrorBehaviorTests(unittest.TestCase):
    def test_high_level_validation_errors_are_specific_and_do_not_create_sessions(self):
        invalid_cases = [
            ("empty prompt", lambda client: client.ask(""), "prompt must not be empty"),
            ("bad model", lambda client: client.ask("hello", model="reasoner"), "Unknown model"),
            ("bad thinking flag", lambda client: client.ask("hello", thinking="yes"), "thinking must be a boolean"),
            ("bad search flag", lambda client: client.ask("hello", search=1), "search must be a boolean"),
            ("bad ref files shape", lambda client: client.ask("hello", ref_file_ids="file-id"), "ref_file_ids must be a list of strings"),
        ]

        for name, action, expected_message in invalid_cases:
            with self.subTest(name=name):
                client = make_client()
                with self.assertRaises(ValidationError) as caught:
                    action(client)
                self.assertIn(expected_message, str(caught.exception))
                self.assertEqual(client.session.created, [])
                self.assertEqual(client.chat.structured_calls, [])

    def test_send_validation_errors_are_specific_and_do_not_stream(self):
        client = make_client()
        session_id = client.new_chat()

        invalid_cases = [
            ("empty session", lambda: client.send("", "hello"), "session_id must not be empty"),
            ("empty prompt", lambda: client.send(session_id, ""), "prompt must not be empty"),
            ("bad parent", lambda: client.send(session_id, "hello", parent_message_id=0), "parent_message_id"),
            ("bad file item", lambda: client.send(session_id, "hello", ref_file_ids=[123]), "ref_file_ids[0] must be a string"),
        ]

        for name, action, expected_message in invalid_cases:
            with self.subTest(name=name):
                with self.assertRaises(ValidationError) as caught:
                    action()
                self.assertIn(expected_message, str(caught.exception))

        self.assertEqual(client.chat.structured_calls, [])

    def test_unknown_session_errors_are_specific_and_do_not_stream(self):
        client = make_client()

        with self.assertRaises(ValidationError) as caught:
            client.send("not-managed", "hello")

        self.assertIn("Unknown session_id", str(caught.exception))
        self.assertIn("adopt_chat", str(caught.exception))
        self.assertEqual(client.chat.structured_calls, [])

    def test_expert_with_file_server_rejection_stays_clear_and_cleans_up(self):
        error = APIRequestError(
            "Streaming API returned error code 422: expert mode does not accept this file",
            api_code=422,
        )
        client = make_client(error=error)

        with self.assertRaises(APIRequestError) as caught:
            client.ask("read this", model="expert", ref_file_ids=["file-1"])

        self.assertIs(caught.exception, error)
        self.assertIn("expert mode does not accept this file", str(caught.exception))
        self.assertIn("[API code: 422]", str(caught.exception))
        self.assertEqual(client.session.deleted, ["session-1"])


if __name__ == "__main__":
    unittest.main()
