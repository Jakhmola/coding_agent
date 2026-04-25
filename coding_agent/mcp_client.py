from __future__ import annotations

import json
from typing import Any

from coding_agent.config import Settings


JsonObject = dict[str, Any]


class McpClient:
    """Thin Streamable HTTP MCP client used by the agent loop."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def list_tools(self) -> list[JsonObject]:
        async with _client_session(self._settings.mcp_url) as session:
            result = await session.list_tools()
            return [_dump_model(tool) for tool in result.tools]

    async def get_system_prompt(self) -> str:
        async with _client_session(self._settings.mcp_url) as session:
            result = await session.get_prompt("coding_agent_system_prompt")
            return _prompt_text(result)

    async def call_tool(self, name: str, arguments: JsonObject) -> str:
        async with _client_session(self._settings.mcp_url) as session:
            result = await session.call_tool(name, arguments)
            return _tool_result_text(result)


def _require_mcp_client() -> tuple[Any, Any]:
    try:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MCP SDK is required for MCP client calls. "
            "Install project dependencies before running the agent loop."
        ) from exc
    return ClientSession, streamablehttp_client


class _client_session:
    def __init__(self, url: str) -> None:
        self._url = url
        self._http_context: Any = None
        self._session_context: Any = None
        self._session: Any = None

    async def __aenter__(self) -> Any:
        ClientSession, streamablehttp_client = _require_mcp_client()
        self._http_context = streamablehttp_client(self._url)
        read_stream, write_stream, _ = await self._http_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()
        return self._session

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._session_context is not None:
            await self._session_context.__aexit__(exc_type, exc, tb)
        if self._http_context is not None:
            await self._http_context.__aexit__(exc_type, exc, tb)


def _dump_model(value: Any) -> JsonObject:
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
    elif hasattr(value, "dict"):
        dumped = value.dict()
    elif isinstance(value, dict):
        dumped = value
    else:
        dumped = dict(value)

    if not isinstance(dumped, dict):
        raise RuntimeError("MCP object did not serialize to a JSON object")
    return dumped


def _prompt_text(result: Any) -> str:
    messages = getattr(result, "messages", None)
    if not messages:
        return ""

    parts: list[str] = []
    for message in messages:
        content = getattr(message, "content", None)
        text = getattr(content, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def _tool_result_text(result: Any) -> str:
    content_items = getattr(result, "content", None)
    if not content_items:
        return ""

    parts: list[str] = []
    for item in content_items:
        text = getattr(item, "text", None)
        if isinstance(text, str):
            parts.append(text)
        else:
            parts.append(json.dumps(_dump_model(item), sort_keys=True))
    return "\n".join(parts)
