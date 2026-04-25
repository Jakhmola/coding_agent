from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

from coding_agent.config import Settings, load_settings
from coding_agent.logging import configure_logging
from coding_agent.mcp_client import McpClient
from coding_agent.model_client import ModelResponse, ModelToolCall, OpenAIChatClient


JsonObject = dict[str, Any]


class ModelClient(Protocol):
    async def complete(
        self,
        messages: list[JsonObject],
        tools: list[JsonObject],
    ) -> ModelResponse:
        ...


class ToolClient(Protocol):
    async def list_tools(self) -> list[JsonObject]:
        ...

    async def get_system_prompt(self) -> str:
        ...

    async def call_tool(self, name: str, arguments: JsonObject) -> str:
        ...


@dataclass(frozen=True)
class AgentResult:
    content: str
    messages: tuple[JsonObject, ...]
    iterations: int


async def run_agent(
    user_prompt: str,
    *,
    settings: Settings | None = None,
    model_client: ModelClient | None = None,
    tool_client: ToolClient | None = None,
) -> AgentResult:
    settings = settings or load_settings()
    model_client = model_client or OpenAIChatClient(settings)
    tool_client = tool_client or McpClient(settings)

    system_prompt = await tool_client.get_system_prompt()
    tools = [
        mcp_tool_to_openai_tool(tool)
        for tool in await tool_client.list_tools()
    ]
    tool_names = {
        tool["function"]["name"]
        for tool in tools
        if isinstance(tool.get("function"), dict)
        and isinstance(tool["function"].get("name"), str)
    }
    messages: list[JsonObject] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for iteration in range(1, settings.max_iterations + 1):
        response = _coerce_text_tool_call(
            await model_client.complete(messages, tools),
            tool_names,
        )
        assistant_message = _assistant_message(response)
        messages.append(assistant_message)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_result = await _call_tool(tool_client, tool_call)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": tool_result,
                    }
                )
            continue

        if response.content is not None:
            return AgentResult(
                content=response.content,
                messages=tuple(messages),
                iterations=iteration,
            )

        raise RuntimeError("Model returned neither final content nor tool calls")

    raise RuntimeError(f"Maximum iterations ({settings.max_iterations}) reached")


def mcp_tool_to_openai_tool(tool: JsonObject) -> JsonObject:
    input_schema = (
        tool.get("inputSchema")
        or tool.get("input_schema")
        or {"type": "object", "properties": {}}
    )
    return {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description") or "",
            "parameters": input_schema,
        },
    }


def _assistant_message(response: ModelResponse) -> JsonObject:
    message: JsonObject = {
        "role": "assistant",
        "content": response.content,
    }
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            }
            for tool_call in response.tool_calls
        ]
    return message


def _coerce_text_tool_call(
    response: ModelResponse,
    tool_names: set[str],
) -> ModelResponse:
    if response.tool_calls or response.content is None:
        return response

    content = response.content.strip()
    if content.startswith("```"):
        content = _strip_code_fence(content)

    payload = _parse_text_tool_payload(content)
    if payload is None:
        return response

    name = payload.get("name")
    arguments = payload.get("arguments", {})
    if not isinstance(name, str) or name not in tool_names:
        return response

    if isinstance(arguments, str):
        arguments_json = arguments
    else:
        arguments_json = json.dumps(arguments)

    return ModelResponse(
        content=None,
        tool_calls=(
            ModelToolCall(
                id="call_text_0",
                name=name,
                arguments=arguments_json,
            ),
        ),
    )


def _parse_text_tool_payload(content: str) -> JsonObject | None:
    decoder = json.JSONDecoder()
    try:
        payload, _ = decoder.raw_decode(content)
    except json.JSONDecodeError:
        fixed_content = re.sub(
            r'("name"\s*:\s*)([A-Za-z_][A-Za-z0-9_.-]*)',
            r'\1"\2"',
            content,
            count=1,
        )
        if fixed_content == content:
            return None
        try:
            payload, _ = decoder.raw_decode(fixed_content)
        except json.JSONDecodeError:
            return None

    if not isinstance(payload, dict):
        return None
    return payload


def _strip_code_fence(content: str) -> str:
    lines = content.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return content


async def _call_tool(tool_client: ToolClient, tool_call: ModelToolCall) -> str:
    try:
        arguments = json.loads(tool_call.arguments or "{}")
    except json.JSONDecodeError as exc:
        return f"Error: invalid JSON arguments for {tool_call.name}: {exc.msg}"

    if not isinstance(arguments, dict):
        return f"Error: arguments for {tool_call.name} must be a JSON object"

    return await tool_client.call_tool(tool_call.name, arguments)


def format_agent_trace(messages: tuple[JsonObject, ...]) -> list[str]:
    lines: list[str] = []
    user_message = _first_message_with_role(messages, "user")
    model_turns = sum(1 for message in messages if message.get("role") == "assistant")
    tool_calls = _count_tool_calls(messages)

    lines.append("Agent trace")
    if user_message is not None:
        lines.append(
            f"User input: {_single_line(str(user_message.get('content', '')), limit=500)}"
        )
    lines.append(f"Model turns: {model_turns}")
    lines.append(f"Tool calls: {tool_calls}")

    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        turn_number = sum(
            1
            for prior_message in messages[: index + 1]
            if prior_message.get("role") == "assistant"
        )
        lines.append("")
        lines.append(f"Model turn {turn_number}")
        lines.append("  Sent to model:")
        sent_summaries = _summarize_model_input(tuple(messages[:index]))
        if sent_summaries:
            lines.extend(f"    - {summary}" for summary in sent_summaries)
        else:
            lines.append("    - no prior messages")

        lines.append("  Model response:")
        lines.extend(
            f"    - {summary}" for summary in _summarize_assistant_response(message)
        )

        tool_call_lines = _summarize_tool_calls_after_response(
            message,
            tuple(messages[index + 1 :]),
        )
        if tool_call_lines:
            lines.append("  Tool execution:")
            lines.extend(f"    - {summary}" for summary in tool_call_lines)
            lines.append("  Sent back to model next turn:")
            lines.extend(
                f"    - {summary}"
                for summary in _summarize_tool_results_after_response(
                    message,
                    tuple(messages[index + 1 :]),
                )
            )

    return lines


def _first_message_with_role(
    messages: tuple[JsonObject, ...],
    role: str,
) -> JsonObject | None:
    for message in messages:
        if message.get("role") == role:
            return message
    return None


def _count_tool_calls(messages: tuple[JsonObject, ...]) -> int:
    count = 0
    for message in messages:
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            count += len(tool_calls)
    return count


def _summarize_model_input(messages: tuple[JsonObject, ...]) -> list[str]:
    summaries: list[str] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            summaries.append("system prompt")
        elif role == "user":
            summaries.append(
                f"user: {_single_line(str(message.get('content', '')), limit=240)}"
            )
        elif role == "assistant":
            summaries.extend(_summarize_assistant_response(message, prefix="assistant"))
        elif role == "tool":
            name = message.get("name") or message.get("tool_call_id") or "tool"
            content = _single_line(str(message.get("content", "")), limit=240)
            summaries.append(f"tool result from {name}: {content}")
    return summaries


def _summarize_assistant_response(
    message: JsonObject,
    *,
    prefix: str = "assistant",
) -> list[str]:
    summaries: list[str] = []
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        summaries.append(f"{prefix}: {_single_line(content, limit=500)}")

    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for tool_call in tool_calls:
            name, arguments = _tool_call_name_and_arguments(tool_call)
            if name is not None:
                summaries.append(f"{prefix} requested tool: {name}({arguments})")

    if not summaries:
        summaries.append(f"{prefix}: <empty response>")
    return summaries


def _summarize_tool_calls_after_response(
    assistant_message: JsonObject,
    later_messages: tuple[JsonObject, ...],
) -> list[str]:
    summaries: list[str] = []
    tool_results = _tool_results_until_next_assistant(later_messages)
    tool_calls = assistant_message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return summaries

    for tool_call in tool_calls:
        name, arguments = _tool_call_name_and_arguments(tool_call)
        if name is None or not isinstance(tool_call, dict):
            continue
        result = tool_results.get(tool_call.get("id"))
        result_preview = ""
        if result is not None:
            result_preview = f" -> {_single_line(str(result.get('content', '')), limit=240)}"
        summaries.append(f"{name}({arguments}){result_preview}")
    return summaries


def _summarize_tool_results_after_response(
    assistant_message: JsonObject,
    later_messages: tuple[JsonObject, ...],
) -> list[str]:
    summaries: list[str] = []
    tool_results = _tool_results_until_next_assistant(later_messages)
    tool_calls = assistant_message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return summaries

    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        result = tool_results.get(tool_call.get("id"))
        if result is None:
            continue
        name = result.get("name") or _tool_call_name_and_arguments(tool_call)[0]
        content = _single_line(str(result.get("content", "")), limit=240)
        summaries.append(f"tool message for {name}: {content}")
    return summaries


def _tool_results_until_next_assistant(
    messages: tuple[JsonObject, ...],
) -> dict[Any, JsonObject]:
    results: dict[Any, JsonObject] = {}
    for message in messages:
        if message.get("role") == "assistant":
            break
        if message.get("role") == "tool":
            results[message.get("tool_call_id")] = message
    return results


def _tool_call_name_and_arguments(tool_call: Any) -> tuple[str | None, str]:
    if not isinstance(tool_call, dict):
        return None, "{}"
    function = tool_call.get("function")
    if not isinstance(function, dict):
        return None, "{}"
    name = function.get("name")
    arguments = function.get("arguments", "{}")
    if not isinstance(name, str):
        return None, "{}"
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, sort_keys=True)
    return name, _format_jsonish(arguments)


def _format_jsonish(value: str) -> str:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError:
        return value
    return json.dumps(parsed, sort_keys=True)


def _single_line(value: str, *, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv()


async def async_main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the MCP-backed coding agent loop")
    parser.add_argument("prompt", nargs="+", help="Prompt to send to the agent")
    parser.add_argument("--verbose", action="store_true", help="Print loop metadata")
    args = parser.parse_args(argv)

    _load_dotenv_if_available()
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_json)
    _quiet_chatty_client_logs()

    result = await run_agent(" ".join(args.prompt), settings=settings)
    if args.verbose:
        for line in format_agent_trace(result.messages):
            print(line)
        print("")
        print("Final answer:")
    print(result.content)
    return 0


def _quiet_chatty_client_logs() -> None:
    for logger_name in ("httpx", "mcp.client.streamable_http"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
