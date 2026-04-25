import os
import unittest
from unittest.mock import patch

from coding_agent.config import load_settings
from coding_agent.model_client import (
    OpenAIChatClient,
    ModelResponse,
    parse_chat_completion_response,
)


class OpenAIChatClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_sends_openai_compatible_chat_completion_request(self):
        calls = []

        async def post_json(url, payload, headers, timeout):
            calls.append(
                {
                    "url": url,
                    "payload": payload,
                    "headers": headers,
                    "timeout": timeout,
                }
            )
            return {
                "choices": [
                    {
                        "message": {
                            "content": "done",
                            "tool_calls": [],
                        }
                    }
                ]
            }

        client = OpenAIChatClient(_settings(), post_json=post_json, timeout=5.0)
        response = await client.complete(
            [{"role": "user", "content": "hello"}],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "get_files_info",
                        "description": "",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        self.assertEqual(response, ModelResponse("done"))
        self.assertEqual(
            calls[0]["url"],
            "http://localhost:11434/v1/chat/completions",
        )
        self.assertEqual(calls[0]["payload"]["model"], "coding-qwen-gpu")
        self.assertEqual(calls[0]["payload"]["tool_choice"], "auto")
        self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer ollama")
        self.assertEqual(calls[0]["timeout"], 5.0)

    async def test_omits_tool_choice_when_no_tools_are_available(self):
        calls = []

        async def post_json(url, payload, headers, timeout):
            calls.append(payload)
            return {"choices": [{"message": {"content": "done"}}]}

        client = OpenAIChatClient(_settings(), post_json=post_json)
        await client.complete([{"role": "user", "content": "hello"}], [])

        self.assertNotIn("tool_choice", calls[0])


class ParseChatCompletionResponseTests(unittest.TestCase):
    def test_parses_tool_calls(self):
        parsed = parse_chat_completion_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_files_info",
                                        "arguments": '{"directory": "."}',
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        )

        self.assertIsNone(parsed.content)
        self.assertEqual(len(parsed.tool_calls), 1)
        self.assertEqual(parsed.tool_calls[0].id, "call_1")
        self.assertEqual(parsed.tool_calls[0].name, "get_files_info")
        self.assertEqual(parsed.tool_calls[0].arguments, '{"directory": "."}')

    def test_rejects_response_without_choices(self):
        with self.assertRaisesRegex(RuntimeError, "choices"):
            parse_chat_completion_response({})


def _settings():
    with patch.dict(os.environ, {}, clear=True):
        return load_settings()


if __name__ == "__main__":
    unittest.main()
