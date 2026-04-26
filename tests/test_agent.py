import asyncio
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from coding_agent.agent import format_agent_trace, mcp_tool_to_openai_tool, run_agent
from coding_agent.config import load_settings
from coding_agent.workflow import AgentResult
from coding_agent.model_client import ModelResponse, ModelToolCall

try:
    import langgraph  # noqa: F401
except ModuleNotFoundError:
    LANGGRAPH_AVAILABLE = False
else:
    LANGGRAPH_AVAILABLE = True


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
            },
            {
                "name": "get_file_content",
                "description": "Read a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                },
            },
            {
                "name": "write_file",
                "description": "Write a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
            {
                "name": "run_python_file",
                "description": "Run a Python file",
                "inputSchema": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                },
            },
        ]

    async def list_tools(self):
        return self.tools

    async def get_system_prompt(self):
        return "system prompt"

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return f"{name} result"


class ErrorToolClient(FakeToolClient):
    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return "Error: file not found"


@unittest.skipIf(not LANGGRAPH_AVAILABLE, "langgraph is not installed")
class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 1.0

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

    async def test_coerces_first_json_tool_request_when_model_adds_extra_text(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    '{"name": "get_files_info", "arguments": {"directory": "."}}\n'
                    '{"name": "missing_tool", "arguments": {}}'
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

    async def test_returns_failure_when_max_iterations_is_reached(self):
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

        result = await run_agent(
            "loop",
            settings=replace(_settings(), max_iterations=2),
            model_client=model,
            tool_client=FakeToolClient(),
        )

        self.assertIn("Maximum iterations", result.content)
        self.assertIn("Maximum iterations", result.blocked_reason)

    async def test_print_file_content_stops_after_single_read(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="get_file_content",
                            arguments='{"file_path": "lorem.txt"}',
                        ),
                    ),
                ),
                ModelResponse("should not be used"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "print lorem.txt file content",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "get_file_content result")
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(tools.calls, [("get_file_content", {"file_path": "lorem.txt"})])
        event_types = [event["type"] for event in result.events]
        self.assertIn("intent_classified", event_types)
        self.assertIn("tool_executed", event_types)
        self.assertNotIn("tool_blocked", event_types)

    async def test_blocks_write_file_for_read_only_prompt(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="write_file",
                            arguments='{"file_path": "content.txt", "content": "new_text"}',
                        ),
                    ),
                ),
                ModelResponse("I will answer without writing."),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "print lorem.txt file content",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "I will answer without writing.")
        self.assertEqual(tools.calls, [])
        blocked_events = [
            event for event in result.events if event["type"] == "tool_blocked"
        ]
        self.assertEqual(len(blocked_events), 1)
        self.assertEqual(blocked_events[0]["tool"], "write_file")
        self.assertIsNone(result.blocked_reason)

    async def test_blocks_run_python_file_for_read_only_prompt(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="run_python_file",
                            arguments='{"file_path": "main.py"}',
                        ),
                    ),
                ),
                ModelResponse("I will answer without running code."),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "show main.py content",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "I will answer without running code.")
        self.assertEqual(tools.calls, [])
        blocked_events = [
            event for event in result.events if event["type"] == "tool_blocked"
        ]
        self.assertEqual(blocked_events[0]["tool"], "run_python_file")

    async def test_explicit_write_prompt_allows_write_file(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="write_file",
                            arguments='{"file_path": "content.txt", "content": "new_text"}',
                        ),
                    ),
                ),
                ModelResponse("wrote it"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "write new_text to content.txt",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "wrote it")
        self.assertEqual(
            tools.calls,
            [("write_file", {"file_path": "content.txt", "content": "new_text"})],
        )

    async def test_explicit_run_prompt_allows_run_python_file(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="run_python_file",
                            arguments='{"file_path": "main.py"}',
                        ),
                    ),
                ),
                ModelResponse("ran it"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "run main.py",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "ran it")
        self.assertEqual(tools.calls, [("run_python_file", {"file_path": "main.py"})])

    async def test_repeated_unsafe_tool_proposal_stops_blocked(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="write_file",
                            arguments='{"file_path": "content.txt", "content": "one"}',
                        ),
                    ),
                ),
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_2",
                            name="write_file",
                            arguments='{"file_path": "content.txt", "content": "two"}',
                        ),
                    ),
                ),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "print lorem.txt file content",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(tools.calls, [])
        self.assertIsNotNone(result.blocked_reason)
        self.assertIn("Blocked unsafe tool use", result.content)

    async def test_tool_error_is_recorded_and_routed_back_to_model(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="get_file_content",
                            arguments='{"file_path": "missing.txt"}',
                        ),
                    ),
                ),
                ModelResponse("missing.txt was not found"),
            ]
        )
        tools = ErrorToolClient()

        result = await run_agent(
            "print missing.txt file content",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "missing.txt was not found")
        self.assertEqual(tools.calls, [("get_file_content", {"file_path": "missing.txt"})])
        tool_events = [
            event for event in result.events if event["type"] == "tool_executed"
        ]
        self.assertEqual(tool_events[0]["result"], "Error: file not found")
        self.assertEqual(result.iterations, 2)

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

    def test_formats_agent_trace_from_events(self):
        lines = format_agent_trace(
            AgentResult(
                content="done",
                messages=(),
                iterations=1,
                tool_call_count=1,
                events=(
                    {
                        "type": "user_input",
                        "content": "list files",
                    },
                    {
                        "type": "intent_classified",
                        "intent": "read_only",
                        "risk_level": "safe",
                    },
                    {
                        "type": "tool_executed",
                        "tool": "get_files_info",
                        "arguments": {"directory": "."},
                        "result": "- README.md: file_size=12 bytes, is_dir=False",
                    },
                ),
            )
        )

        self.assertEqual(
            lines,
            [
                "Agent trace",
                "User input: list files",
                "Intent: read_only",
                "Model turns: 1",
                "Tool calls: 1",
                "",
                "Event 1: user_input content=list files",
                "",
                "Event 2: intent_classified intent=read_only risk=safe",
                "",
                'Event 3: tool_executed get_files_info({"directory": "."}) -> - README.md: file_size=12 bytes, is_dir=False',
            ],
        )


def _settings():
    with patch.dict(os.environ, {}, clear=True):
        return load_settings()


if __name__ == "__main__":
    unittest.main()
