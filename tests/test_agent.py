import asyncio
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from coding_agent.agent import format_agent_trace, mcp_tool_to_openai_tool, run_agent
from coding_agent.config import load_settings
from coding_agent.model_client import ModelResponse, ModelToolCall
from coding_agent.tracing import RecordingTracer
from coding_agent.workflow import AgentResult, classify_intent
from prompts import build_final_response_prompt

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
                "name": "search_files",
                "description": "Search file names",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "directory": {"type": "string"},
                    },
                },
            },
            {
                "name": "grep_files",
                "description": "Search file contents",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "directory": {"type": "string"},
                    },
                },
            },
            {
                "name": "append_file",
                "description": "Append to a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                },
            },
            {
                "name": "replace_in_file",
                "description": "Replace exact text in a file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "old_text": {"type": "string"},
                        "new_text": {"type": "string"},
                    },
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
        if name == "append_file":
            return f'Successfully appended to "{arguments.get("file_path")}"'
        if name == "replace_in_file":
            return f'Successfully replaced 1 occurrence(s) in "{arguments.get("file_path")}"'
        if name == "write_file":
            return f'Successfully wrote to "{arguments.get("file_path")}"'
        return f"{name} result"


class ErrorToolClient(FakeToolClient):
    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return "Error: file not found"


@unittest.skipIf(not LANGGRAPH_AVAILABLE, "langgraph is not installed")
class AgentLoopTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        asyncio.get_running_loop().slow_callback_duration = 1.0

    def test_classifies_add_prompt_as_write_intent(self):
        self.assertEqual(
            classify_intent("add 'this is a new line' to lrem.txt"),
            "write",
        )
        self.assertEqual(classify_intent("uv add langgraph"), "dependency")

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

    async def test_tracing_records_workflow_model_and_tool_spans(self):
        tracer = RecordingTracer()
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
                    usage={"prompt_tokens": 10, "completion_tokens": 3},
                ),
                ModelResponse("files checked"),
            ]
        )

        result = await run_agent(
            "list files",
            settings=_settings(),
            model_client=model,
            tool_client=FakeToolClient(),
            tracer=tracer,
        )

        self.assertEqual(result.content, "files checked")
        record_names = [record.name for record in tracer.records]
        self.assertIn("coding_agent.workflow", record_names)
        self.assertIn("workflow.model_step", record_names)
        self.assertIn("model.chat_completion", record_names)
        self.assertIn("workflow.policy_gate", record_names)
        self.assertIn("workflow.tool_executor", record_names)
        self.assertIn("mcp.call_tool", record_names)
        workflow_record = _first_record(tracer, "coding_agent.workflow")
        self.assertEqual(workflow_record.output["status"], "completed")
        llm_record = _first_record(tracer, "model.chat_completion")
        self.assertEqual(llm_record.span_type, "llm")
        self.assertEqual(llm_record.usage["prompt_tokens"], 10)
        tool_record = _first_record(tracer, "mcp.call_tool")
        self.assertEqual(tool_record.input["tool"], "get_files_info")
        self.assertTrue(tool_record.output["success"])

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

    async def test_tracing_records_policy_block_with_sanitized_arguments(self):
        tracer = RecordingTracer()
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="write_file",
                            arguments='{"file_path": "content.txt", "content": "secret text"}',
                        ),
                    ),
                ),
                ModelResponse("I will answer without writing."),
            ]
        )

        result = await run_agent(
            "print lorem.txt file content",
            settings=_settings(),
            model_client=model,
            tool_client=FakeToolClient(),
            tracer=tracer,
        )

        self.assertEqual(result.content, "I will answer without writing.")
        block = _first_record(tracer, "policy.blocked_tool")
        self.assertEqual(block.span_type, "guardrail")
        self.assertEqual(block.input["tool"], "write_file")
        self.assertIsInstance(block.input["arguments"]["content"], dict)
        self.assertIn("not allowed", block.output["reason"])

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

    async def test_explicit_add_prompt_allows_append_file(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="append_file",
                            arguments='{"file_path": "lorem.txt", "content": "new_text"}',
                        ),
                    ),
                ),
                ModelResponse("added it"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "add new_text to lorem.txt",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "added it")
        self.assertEqual(result.tool_call_count, 1)
        self.assertEqual(
            tools.calls,
            [("append_file", {"file_path": "lorem.txt", "content": "new_text"})],
        )
        event_types = [event["type"] for event in result.events]
        self.assertIn("write_completed", event_types)
        sent_tool_names = _sent_tool_names(model.calls[0])
        self.assertIn("append_file", sent_tool_names)
        self.assertNotIn("write_file", sent_tool_names)
        self.assertIn(
            "Executor node instructions",
            model.calls[0]["messages"][0]["content"],
        )
        self.assertIn("Preferred tools: append_file", model.calls[0]["messages"][0]["content"])

    async def test_tracing_records_write_completion_span(self):
        tracer = RecordingTracer()
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="append_file",
                            arguments='{"file_path": "notes.txt", "content": "hello"}',
                        ),
                    ),
                ),
                ModelResponse("added it"),
            ]
        )

        result = await run_agent(
            "add hello to notes.txt",
            settings=_settings(),
            model_client=model,
            tool_client=FakeToolClient(),
            tracer=tracer,
        )

        self.assertEqual(result.content, "added it")
        write_record = _first_record(tracer, "workflow.write_completed")
        self.assertEqual(write_record.output["tool"], "append_file")
        self.assertEqual(write_record.output["file_path"], "notes.txt")

    async def test_add_prompt_blocks_broad_write_then_allows_append(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="write_file",
                            arguments='{"file_path": "lorem.txt", "content": "new_text"}',
                        ),
                    ),
                ),
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_2",
                            name="append_file",
                            arguments='{"file_path": "lorem.txt", "content": "new_text"}',
                        ),
                    ),
                ),
                ModelResponse("added it safely"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "add new_text to lorem.txt",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "added it safely")
        self.assertEqual(
            tools.calls,
            [("append_file", {"file_path": "lorem.txt", "content": "new_text"})],
        )
        blocked_events = [
            event for event in result.events if event["type"] == "tool_blocked"
        ]
        self.assertEqual(blocked_events[0]["tool"], "write_file")
        self.assertIn("append_file", blocked_events[0]["reason"])
        self.assertIn("Use append_file", model.calls[1]["messages"][-1]["content"])
        self.assertIn("Repair node instructions", model.calls[1]["messages"][0]["content"])

    async def test_explicit_replace_prompt_prefers_replace_in_file(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="replace_in_file",
                            arguments=(
                                '{"file_path": "lorem.txt", "old_text": "old", '
                                '"new_text": "new"}'
                            ),
                        ),
                    ),
                ),
                ModelResponse("replaced it"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "replace old with new in lorem.txt",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "replaced it")
        self.assertEqual(
            tools.calls,
            [
                (
                    "replace_in_file",
                    {"file_path": "lorem.txt", "old_text": "old", "new_text": "new"},
                )
            ],
        )
        sent_tool_names = _sent_tool_names(model.calls[0])
        self.assertIn("replace_in_file", sent_tool_names)
        self.assertNotIn("write_file", sent_tool_names)
        self.assertIn(
            "Preferred tools: replace_in_file",
            model.calls[0]["messages"][0]["content"],
        )

    async def test_replace_prompt_blocks_unplanned_write_path(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="replace_in_file",
                            arguments=(
                                '{"file_path": "other.txt", "old_text": "old", '
                                '"new_text": "new"}'
                            ),
                        ),
                    ),
                ),
                ModelResponse("I will not edit the wrong file."),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "replace old with new in expected.txt",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "I will not edit the wrong file.")
        self.assertEqual(tools.calls, [])
        blocked_events = [
            event for event in result.events if event["type"] == "tool_blocked"
        ]
        self.assertIn("plan only allows", blocked_events[0]["reason"])

    async def test_read_only_prompt_allows_search_and_grep_tools(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="search_files",
                            arguments='{"pattern": "agent"}',
                        ),
                        ModelToolCall(
                            id="call_2",
                            name="grep_files",
                            arguments='{"pattern": "TODO", "directory": "."}',
                        ),
                    ),
                ),
                ModelResponse("searched"),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "search files for agent and grep TODO",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "searched")
        self.assertEqual(
            tools.calls,
            [
                ("search_files", {"pattern": "agent"}),
                ("grep_files", {"pattern": "TODO", "directory": "."}),
            ],
        )
        sent_tool_names = _sent_tool_names(model.calls[0])
        self.assertIn("search_files", sent_tool_names)
        self.assertIn("grep_files", sent_tool_names)
        self.assertNotIn("append_file", sent_tool_names)
        self.assertNotIn("write_file", sent_tool_names)

    async def test_explicit_run_prompt_allows_run_python_file(self):
        tracer = RecordingTracer()
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
            tracer=tracer,
        )

        self.assertEqual(result.content, "ran it")
        self.assertEqual(tools.calls, [("run_python_file", {"file_path": "main.py"})])
        tool_record = _first_record(tracer, "mcp.call_tool")
        self.assertEqual(tool_record.input["tool"], "run_python_file")
        self.assertTrue(tool_record.output["success"])

    async def test_empty_final_after_tool_uses_latest_tool_result(self):
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
                ModelResponse(""),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "run main.py",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, "run_python_file result")
        review_events = [
            event for event in result.events if event["type"] == "review_completed"
        ]
        self.assertEqual(review_events[-1]["reason"], "fallback_tool_result")

    async def test_empty_final_after_write_uses_write_completion(self):
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id="call_1",
                            name="append_file",
                            arguments='{"file_path": "notes.txt", "content": "hello"}',
                        ),
                    ),
                ),
                ModelResponse(""),
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "add hello to notes.txt",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
        )

        self.assertEqual(result.content, 'Completed: Successfully appended to "notes.txt"')

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

    async def test_max_tool_calls_stops_runaway_tool_use(self):
        tracer = RecordingTracer()
        model = FakeModelClient(
            [
                ModelResponse(
                    None,
                    (
                        ModelToolCall(
                            id=f"call_{index}",
                            name="search_files",
                            arguments='{"pattern": "agent"}',
                        ),
                    ),
                )
                for index in range(1, 6)
            ]
        )
        tools = FakeToolClient()

        result = await run_agent(
            "search files for agent",
            settings=_settings(),
            model_client=model,
            tool_client=tools,
            tracer=tracer,
        )

        self.assertEqual(
            tools.calls,
            [
                ("search_files", {"pattern": "agent"}),
                ("search_files", {"pattern": "agent"}),
                ("search_files", {"pattern": "agent"}),
                ("search_files", {"pattern": "agent"}),
            ],
        )
        self.assertIn("Maximum tool calls (4) reached", result.content)
        self.assertIn("Maximum tool calls (4) reached", result.blocked_reason)
        self.assertEqual(model.calls[-1]["tools"], [])
        block = _first_record(tracer, "policy.blocked_tool")
        self.assertIn("Maximum tool calls", block.output["reason"])

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

    def test_final_response_prompt_summarizes_terminal_context(self):
        prompt = build_final_response_prompt(
            user_prompt="add hello to notes.txt",
            intent="write",
            blocked_reason=None,
            tool_events=[
                {
                    "type": "tool_executed",
                    "tool": "append_file",
                    "result": 'Successfully appended to "notes.txt"',
                }
            ],
            write_events=[
                {
                    "type": "write_completed",
                    "tool": "append_file",
                    "file_path": "notes.txt",
                }
            ],
        )

        self.assertIn("Final response node instructions", prompt)
        self.assertIn("Do not request or imply any additional tool calls", prompt)
        self.assertIn("Observed tool results: 1", prompt)
        self.assertIn("Observed successful writes: 1", prompt)


def _settings():
    with patch.dict(os.environ, {}, clear=True):
        return load_settings()


def _sent_tool_names(model_call):
    return {
        tool["function"]["name"]
        for tool in model_call["tools"]
    }


def _first_record(tracer, name):
    for record in tracer.records:
        if record.name == name:
            return record
    raise AssertionError(f"Missing trace record: {name}")


if __name__ == "__main__":
    unittest.main()
