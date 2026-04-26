from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from coding_agent.config import Settings


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class ModelToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class ModelResponse:
    content: str | None
    tool_calls: tuple[ModelToolCall, ...] = ()
    usage: JsonObject | None = None


class JsonPoster(Protocol):
    async def __call__(
        self,
        url: str,
        payload: JsonObject,
        headers: dict[str, str],
        timeout: float,
    ) -> JsonObject:
        ...


class OpenAIChatClient:
    """Small OpenAI-compatible chat completions client."""

    def __init__(
        self,
        settings: Settings,
        *,
        post_json: JsonPoster | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._settings = settings
        self._post_json = post_json or _httpx_post_json
        self._timeout = timeout

    async def complete(
        self,
        messages: list[JsonObject],
        tools: list[JsonObject],
    ) -> ModelResponse:
        payload: JsonObject = {
            "model": self._settings.model_name,
            "messages": messages,
            "tools": tools,
        }
        if tools:
            payload["tool_choice"] = "auto"

        headers = {
            "Authorization": f"Bearer {self._settings.openai_api_key}",
            "Content-Type": "application/json",
        }
        url = self._settings.openai_base_url.rstrip("/") + "/chat/completions"
        response = await self._post_json(url, payload, headers, self._timeout)
        return parse_chat_completion_response(response)


async def _httpx_post_json(
    url: str,
    payload: JsonObject,
    headers: dict[str, str],
    timeout: float,
) -> JsonObject:
    try:
        import httpx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "httpx is required for OpenAI-compatible model calls. "
            "Install project dependencies before running the agent loop."
        ) from exc

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("Model response must be a JSON object")
        return data


def parse_chat_completion_response(response: JsonObject) -> ModelResponse:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("Model response did not include any choices")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise RuntimeError("Model choice must be a JSON object")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError("Model choice did not include a message")

    content = message.get("content")
    if content is not None and not isinstance(content, str):
        content = str(content)

    parsed_calls: list[ModelToolCall] = []
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        raise RuntimeError("Model tool_calls must be a list")

    for index, tool_call in enumerate(tool_calls):
        if not isinstance(tool_call, dict):
            raise RuntimeError("Model tool call must be a JSON object")
        function = tool_call.get("function")
        if not isinstance(function, dict):
            raise RuntimeError("Model tool call did not include function metadata")

        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise RuntimeError("Model tool call function name is required")

        arguments = function.get("arguments", "{}")
        if not isinstance(arguments, str):
            arguments = "{}"

        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call_{index}"

        parsed_calls.append(
            ModelToolCall(id=call_id, name=name, arguments=arguments)
        )

    usage = response.get("usage")
    if not isinstance(usage, dict):
        usage = None

    return ModelResponse(
        content=content,
        tool_calls=tuple(parsed_calls),
        usage=usage,
    )
