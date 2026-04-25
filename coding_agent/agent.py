from __future__ import annotations

import asyncio
import json
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
    try:
        payload = json.loads(content)
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
            payload = json.loads(fixed_content)
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

    result = await run_agent(" ".join(args.prompt), settings=settings)
    if args.verbose:
        print(f"Iterations: {result.iterations}")
    print(result.content)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
