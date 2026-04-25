import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from coding_agent.agent import mcp_tool_to_openai_tool, run_agent
from coding_agent.config import load_settings
from coding_agent.model_client import ModelResponse, ModelToolCall


class FakeModelClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete(self, messages, tools):
        self.calls.append(
            {
                "messages": [dict(message) for message in messages],
                "tools": [dict(tool) for tool in tools],
            }
        )
        return self.responses.pop(0)


class FakeToolClient:
    def __init__(self):
        self.calls = []
        self.tools = [
            {
                "name": "get_files_info",
                "description": "List files",
                "inputSchema": {
                    "type": "object",
                    "properties": {"directory": {"type": "string"}},
                },
            }
        ]

    async def list_tools(self):
        return self.tools

    async def get_system_prompt(self):
        return "system prompt"

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return f"{name} result"


class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_direct_final_answer(self):
        model = FakeModelClient([ModelResponse("done")])
        tools = FakeToolClient()

        result = await run_agent(
            "hello",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "done")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(tools.calls, [])
        self.assertEqual(model.calls[0]["messages"][0]["role"], "system")
        self.assertEqual(model.calls[0]["messages"][1]["content"], "hello")

    async def test_calls_tool_then_returns_final_answer(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="get_files_info",
                            arguments='{"directory": "."}',
                        ),
                    ),
                ),
                ModelResponse("files checked"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "list files",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "files checked")
        self.assertEqual(tools.calls, [("get_files_info", {"directory": "."})])
        second_messages = model.calls[1]["messages"]
        self.assertEqual(second_messages[-1]["role"], "tool")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_1")
        self.assertIn("get_files_info result", second_messages[-1]["content"])

    async def test_coerces_plain_json_tool_request_from_model_content(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    '{"name": "get_files_info", "arguments": {"directory": "."}}'
                ),
                ModelResponse("files checked"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "list files",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "files checked")
        self.assertEqual(tools.calls, [("get_files_info", {"directory": "."})])
        second_messages = model.calls[1]["messages"]
        self.assertEqual(second_messages[-2]["role"], "assistant")
        self.assertIsNone(second_messages[-2]["content"])
        self.assertEqual(
            second_messages[-2]["tool_calls"][0]["function"]["name"],
            "get_files_info",
        )
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_text_0")

    async def test_coerces_tool_request_with_unquoted_name_value(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    '{"name": get_files_info, "arguments": {"directory": "."}}'
                ),
                ModelResponse("files checked"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "list files",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "files checked")
        self.assertEqual(tools.calls, [("get_files_info", {"directory": "."})])

    async def test_calls_multiple_tools_from_one_assistant_turn(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="get_files_info",
                            arguments='{"directory": "."}',
                        ),
                        ModelToolCall(
                            id="call_2",
                            name="get_file_content",
                            arguments='{"file_path": "main.py"}',
                        ),
                    ),
                ),
                ModelResponse("done"),
            ]
        )
        tools = FakeToolClient()

        await run_agent(
            "inspect",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(
            tools.calls,
            [
                ("get_files_info", {"directory": "."}),
                ("get_file_content", {"file_path": "main.py"}),
            ],
        )
        second_messages = model.calls[1]["messages"]
        self.assertEqual(second_messages[-2]["tool_call_id"], "call_1")
        self.assertEqual(second_messages[-1]["tool_call_id"], "call_2")

    async def test_malformed_tool_arguments_become_tool_error(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="get_files_info",
                            arguments="{not-json",
                        ),
                    ),
                ),
                ModelResponse("recovered"),
            ]
        )
        tools = FakeToolClient()

        await run_agent(
            "bad args",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(tools.calls, [])
        tool_message = model.calls[1]["messages"][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertIn("invalid JSON arguments", tool_message["content"])

    async def test_fails_when_max_iterations_is_reached(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="get_files_info",
                            arguments='{"directory": "."}',
                        ),
                    ),
                ),
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_2",
                            name="get_files_info",
                            arguments='{"directory": "."}',
                        ),
                    ),
                ),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "Maximum iterations"):
            await run_agent(
                "loop",
                settings=replace(_settings(), max_iterations=2),
                model_client=model,
                tool_client=FakeToolClient(),
            )

    def test_converts_mcp_tool_schema_to_openai_tool_schema(self):
        converted = mcp_tool_to_openai_tool(
            {
                "name": "read",
                "description": "Read a file",
                "inputSchema": {"type": "object", "properties": {}},
            }
        )

        self.assertEqual(converted["type"], "function")
        self.assertEqual(converted["function"]["name"], "read")
        self.assertEqual(converted["function"]["parameters"]["type"], "object")


def _settings():
    with patch.dict(os.environ, {}, clear=True):
        return load_settings()


if __name__ == "__main__":
    unittest.main()
