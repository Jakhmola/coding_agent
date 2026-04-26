from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypedDict
from uuid import uuid4

from coding_agent.config import Settings, load_settings
from coding_agent.mcp_client import McpClient
from coding_agent.model_client import ModelResponse, ModelToolCall, OpenAIChatClient


JsonObject = dict[str, Any]
Intent = Literal["read_only", "write", "run", "git", "dependency", "unknown"]
RiskLevel = Literal["safe", "side_effect", "dangerous"]
Route = Literal["model_step", "final_response", "failure_response"]


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


class CodingAgentState(TypedDict):
    request_id: str
    user_prompt: str
    intent: Intent
    risk_level: RiskLevel
    plan: JsonObject | None
    messages: list[JsonObject]
    pending_tool_calls: list[JsonObject]
    final_answer: str | None
    blocked_reason: str | None
    turn_count: int
    events: list[JsonObject]


@dataclass(frozen=True)
class AgentResult:
    content: str
    messages: tuple[JsonObject, ...]
    iterations: int
    events: tuple[JsonObject, ...] = ()
    tool_call_count: int = 0
    blocked_reason: str | None = None


async def run_workflow(
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

    graph = _build_graph(
        settings=settings,
        model_client=model_client,
        tool_client=tool_client,
        system_prompt=system_prompt,
        tools=tools,
        tool_names=tool_names,
    )
    final_state = await graph.ainvoke(_initial_state(user_prompt))
    final_answer = final_state.get("final_answer") or final_state.get("blocked_reason")
    if final_answer is None:
        final_answer = "No final answer was produced."

    events = tuple(final_state["events"])
    return AgentResult(
        content=final_answer,
        messages=tuple(final_state["messages"]),
        iterations=final_state["turn_count"],
        events=events,
        tool_call_count=sum(1 for event in events if event.get("type") == "tool_executed"),
        blocked_reason=final_state.get("blocked_reason"),
    )


def _build_graph(
    *,
    settings: Settings,
    model_client: ModelClient,
    tool_client: ToolClient,
    system_prompt: str,
    tools: list[JsonObject],
    tool_names: set[str],
) -> Any:
    try:
        from langgraph.graph import END, START, StateGraph
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LangGraph is required for the internal workflow. "
            "Install project dependencies before running the agent."
        ) from exc

    async def intake(state: CodingAgentState) -> JsonObject:
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": state["user_prompt"]},
            ],
            "events": _append_event(
                state,
                "user_input",
                content=state["user_prompt"],
            ),
        }

    async def intent_classifier(state: CodingAgentState) -> JsonObject:
        intent = classify_intent(state["user_prompt"])
        risk_level = _risk_for_intent(intent)
        return {
            "intent": intent,
            "risk_level": risk_level,
            "events": _append_event(
                state,
                "intent_classified",
                intent=intent,
                risk_level=risk_level,
            ),
        }

    async def planner(state: CodingAgentState) -> JsonObject:
        allowed_tools = sorted(_allowed_tools_for_intent(state["intent"]))
        plan = {
            "goal": state["user_prompt"],
            "intent": state["intent"],
            "allowed_tools": allowed_tools,
            "stopping_condition": _stopping_condition(state["user_prompt"], state["intent"]),
        }
        return {
            "plan": plan,
            "events": _append_event(state, "plan_created", **plan),
        }

    async def model_step(state: CodingAgentState) -> JsonObject:
        response = _coerce_text_tool_call(
            await model_client.complete(state["messages"], tools),
            tool_names,
        )
        assistant_message = _assistant_message(response)
        pending_tool_calls = [
            _tool_call_to_dict(tool_call)
            for tool_call in response.tool_calls
        ]
        events = _append_event_to_list(
            state["events"],
            "model_responded",
            turn=state["turn_count"] + 1,
            content=response.content,
            tool_calls=pending_tool_calls,
        )
        return {
            "messages": [*state["messages"], assistant_message],
            "pending_tool_calls": pending_tool_calls,
            "turn_count": state["turn_count"] + 1,
            "final_answer": response.content if not response.tool_calls else None,
            "events": events,
        }

    async def policy_gate(state: CodingAgentState) -> JsonObject:
        events = state["events"]
        messages = state["messages"]
        allowed_tool_calls: list[JsonObject] = []
        blocked_messages: list[JsonObject] = []
        blocked_reason = state["blocked_reason"]

        for tool_call in state["pending_tool_calls"]:
            name = str(tool_call.get("name", ""))
            call_id = str(tool_call.get("id", ""))
            raw_arguments = str(tool_call.get("arguments", "{}"))
            arguments, parse_error = _parse_tool_arguments(raw_arguments)

            if parse_error is not None:
                reason = f"invalid JSON arguments for {name}: {parse_error}"
                events = _append_event_to_list(
                    events,
                    "tool_blocked",
                    tool=name,
                    arguments=raw_arguments,
                    reason=reason,
                )
                blocked_messages.append(_tool_message(call_id, name, f"Error: {reason}"))
                continue

            if not _is_tool_allowed(name, state["intent"]):
                reason = f'{name} is not allowed for {state["intent"]} intent'
                events = _append_event_to_list(
                    events,
                    "tool_blocked",
                    tool=name,
                    arguments=arguments,
                    reason=reason,
                )
                blocked_messages.append(_tool_message(call_id, name, f"Policy blocked: {reason}"))
                if _blocked_event_count(events) >= 2:
                    blocked_reason = f"Blocked unsafe tool use: {reason}"
                continue

            allowed_tool_calls.append({**tool_call, "parsed_arguments": arguments})
            events = _append_event_to_list(
                events,
                "tool_requested",
                tool=name,
                arguments=arguments,
            )

        if blocked_messages:
            messages = [*messages, *blocked_messages]

        return {
            "messages": messages,
            "pending_tool_calls": allowed_tool_calls,
            "blocked_reason": blocked_reason,
            "events": events,
        }

    async def tool_executor(state: CodingAgentState) -> JsonObject:
        events = state["events"]
        messages = state["messages"]
        for tool_call in state["pending_tool_calls"]:
            name = str(tool_call["name"])
            call_id = str(tool_call["id"])
            arguments = dict(tool_call["parsed_arguments"])
            result = await tool_client.call_tool(name, arguments)
            messages = [*messages, _tool_message(call_id, name, result)]
            events = _append_event_to_list(
                events,
                "tool_executed",
                tool=name,
                arguments=arguments,
                result=result,
            )

        return {
            "messages": messages,
            "pending_tool_calls": [],
            "events": events,
        }

    async def progress_reviewer(state: CodingAgentState) -> JsonObject:
        events = state["events"]
        final_answer = state["final_answer"]
        blocked_reason = state["blocked_reason"]
        route: Route = "model_step"
        reason = "continue"

        if blocked_reason is not None:
            route = "failure_response"
            reason = "blocked"
        elif final_answer is not None:
            route = "final_response"
            reason = "model_final"
        elif _should_stop_after_file_read(state):
            final_answer = _last_successful_tool_result(state, "get_file_content")
            route = "final_response"
            reason = "read_only_file_content"
        elif state["turn_count"] >= settings.max_iterations:
            blocked_reason = f"Maximum iterations ({settings.max_iterations}) reached"
            route = "failure_response"
            reason = "max_iterations"

        events = _append_event_to_list(
            events,
            "review_completed",
            route=route,
            reason=reason,
        )
        return {
            "final_answer": final_answer,
            "blocked_reason": blocked_reason,
            "events": events,
        }

    async def route_next(state: CodingAgentState) -> JsonObject:
        return {}

    async def final_response(state: CodingAgentState) -> JsonObject:
        content = state["final_answer"] or ""
        return {
            "events": _append_event(state, "final_answer", content=content),
        }

    async def failure_response(state: CodingAgentState) -> JsonObject:
        reason = state["blocked_reason"] or "The agent could not complete the task."
        return {
            "final_answer": reason,
            "events": _append_event(state, "failure", reason=reason),
        }

    def route_from_state(state: CodingAgentState) -> Route:
        if state["blocked_reason"] is not None:
            return "failure_response"
        if state["final_answer"] is not None:
            return "final_response"
        return "model_step"

    graph = StateGraph(CodingAgentState)
    graph.add_node("intake", intake)
    graph.add_node("intent_classifier", intent_classifier)
    graph.add_node("planner", planner)
    graph.add_node("model_step", model_step)
    graph.add_node("policy_gate", policy_gate)
    graph.add_node("tool_executor", tool_executor)
    graph.add_node("progress_reviewer", progress_reviewer)
    graph.add_node("route_next", route_next)
    graph.add_node("final_response", final_response)
    graph.add_node("failure_response", failure_response)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "intent_classifier")
    graph.add_edge("intent_classifier", "planner")
    graph.add_edge("planner", "model_step")
    graph.add_edge("model_step", "policy_gate")
    graph.add_edge("policy_gate", "tool_executor")
    graph.add_edge("tool_executor", "progress_reviewer")
    graph.add_edge("progress_reviewer", "route_next")
    graph.add_conditional_edges(
        "route_next",
        route_from_state,
        {
            "model_step": "model_step",
            "final_response": "final_response",
            "failure_response": "failure_response",
        },
    )
    graph.add_edge("final_response", END)
    graph.add_edge("failure_response", END)
    return graph.compile()


def _initial_state(user_prompt: str) -> CodingAgentState:
    return {
        "request_id": str(uuid4()),
        "user_prompt": user_prompt,
        "intent": "unknown",
        "risk_level": "safe",
        "plan": None,
        "messages": [],
        "pending_tool_calls": [],
        "final_answer": None,
        "blocked_reason": None,
        "turn_count": 0,
        "events": [],
    }


def classify_intent(user_prompt: str) -> Intent:
    prompt = user_prompt.lower()
    if _contains_word_any(prompt, ("install", "dependency", "package", "pip")) or _contains_any(prompt, ("uv add",)):
        return "dependency"
    if _contains_word_any(prompt, ("write", "create", "save", "edit", "modify", "delete", "overwrite", "add", "append", "insert")):
        return "write"
    if _contains_word_any(prompt, ("run", "execute", "test", "pytest", "unittest")):
        return "run"
    if _contains_word_any(prompt, ("commit", "push", "branch", "status", "diff", "merge")):
        return "git"
    if _contains_word_any(prompt, ("print", "show", "read", "cat", "display", "list", "summarize")):
        return "read_only"
    return "unknown"


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


def format_agent_trace(result: AgentResult) -> list[str]:
    lines = [
        "Agent trace",
        f"User input: {_single_line(_event_value(result.events, 'user_input', 'content'), limit=500)}",
        f"Intent: {_event_value(result.events, 'intent_classified', 'intent')}",
        f"Model turns: {result.iterations}",
        f"Tool calls: {result.tool_call_count}",
    ]

    for index, event in enumerate(result.events, start=1):
        lines.append("")
        lines.append(f"Event {index}: {_format_event(event)}")

    return lines


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


def _tool_call_to_dict(tool_call: ModelToolCall) -> JsonObject:
    return {
        "id": tool_call.id,
        "name": tool_call.name,
        "arguments": tool_call.arguments,
    }


def _tool_message(call_id: str, name: str, content: str) -> JsonObject:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


def _parse_tool_arguments(raw_arguments: str) -> tuple[JsonObject, str | None]:
    try:
        arguments = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError as exc:
        return {}, exc.msg

    if not isinstance(arguments, dict):
        return {}, "arguments must be a JSON object"
    return arguments, None


def _allowed_tools_for_intent(intent: Intent) -> set[str]:
    allowed = {"get_files_info", "get_file_content"}
    if intent == "write":
        allowed.add("write_file")
    if intent == "run":
        allowed.add("run_python_file")
    return allowed


def _is_tool_allowed(name: str, intent: Intent) -> bool:
    return name in _allowed_tools_for_intent(intent)


def _risk_for_intent(intent: Intent) -> RiskLevel:
    if intent == "read_only":
        return "safe"
    if intent in {"write", "run", "git", "dependency"}:
        return "side_effect"
    return "safe"


def _stopping_condition(user_prompt: str, intent: Intent) -> str:
    if intent == "read_only" and _is_direct_file_content_request(user_prompt):
        return "stop after successful file read"
    return "stop when the model has enough evidence for a final answer"


def _should_stop_after_file_read(state: CodingAgentState) -> bool:
    return (
        state["intent"] == "read_only"
        and _is_direct_file_content_request(state["user_prompt"])
        and _last_successful_tool_result(state, "get_file_content") is not None
    )


def _is_direct_file_content_request(user_prompt: str) -> bool:
    prompt = user_prompt.lower()
    if _contains_any(prompt, ("summarize", "summary", "explain", "analyze", "tell me", "what ", "why ", "how ")):
        return False
    return _contains_any(prompt, ("print", "cat", "display", "file content", "contents of", "show"))


def _last_successful_tool_result(state: CodingAgentState, tool_name: str) -> str | None:
    for event in reversed(state["events"]):
        if event.get("type") == "tool_executed" and event.get("tool") == tool_name:
            result = event.get("result")
            if isinstance(result, str) and not result.startswith("Error:"):
                return result
    return None


def _blocked_event_count(events: list[JsonObject]) -> int:
    return sum(1 for event in events if event.get("type") == "tool_blocked")


def _append_event(state: CodingAgentState, event_type: str, **payload: Any) -> list[JsonObject]:
    return _append_event_to_list(state["events"], event_type, **payload)


def _append_event_to_list(
    events: list[JsonObject],
    event_type: str,
    **payload: Any,
) -> list[JsonObject]:
    return [*events, {"type": event_type, **payload}]


def _event_value(
    events: tuple[JsonObject, ...],
    event_type: str,
    key: str,
) -> str:
    for event in events:
        if event.get("type") == event_type:
            value = event.get(key)
            return "" if value is None else str(value)
    return ""


def _format_event(event: JsonObject) -> str:
    event_type = str(event.get("type", "event"))
    if event_type == "user_input":
        return f"user_input content={_single_line(str(event.get('content', '')), limit=240)}"
    if event_type == "intent_classified":
        return f"intent_classified intent={event.get('intent')} risk={event.get('risk_level')}"
    if event_type == "plan_created":
        return f"plan_created allowed_tools={event.get('allowed_tools')}"
    if event_type == "model_responded":
        content = _single_line(str(event.get("content") or ""), limit=160)
        tool_calls = event.get("tool_calls") or []
        return f"model_responded content={content!r} tool_calls={_format_tool_calls(tool_calls)}"
    if event_type == "tool_requested":
        return f"tool_requested {event.get('tool')}({_format_json(event.get('arguments'))})"
    if event_type == "tool_blocked":
        return f"tool_blocked {event.get('tool')} reason={event.get('reason')}"
    if event_type == "tool_executed":
        result = _single_line(str(event.get("result", "")), limit=240)
        return f"tool_executed {event.get('tool')}({_format_json(event.get('arguments'))}) -> {result}"
    if event_type == "review_completed":
        return f"review_completed route={event.get('route')} reason={event.get('reason')}"
    if event_type == "final_answer":
        return f"final_answer content={_single_line(str(event.get('content', '')), limit=240)}"
    if event_type == "failure":
        return f"failure reason={event.get('reason')}"
    return f"{event_type} {_format_json(event)}"


def _format_tool_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list) or not tool_calls:
        return "[]"
    parts = []
    for tool_call in tool_calls:
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            raw_arguments = str(tool_call.get("arguments", "{}"))
            arguments, _ = _parse_tool_arguments(raw_arguments)
            parts.append(f"{name}({_format_json(arguments)})")
    return "[" + ", ".join(parts) + "]"


def _format_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True)
    except TypeError:
        return str(value)


def _contains_any(value: str, needles: tuple[str, ...]) -> bool:
    return any(needle in value for needle in needles)


def _contains_word_any(value: str, words: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(word)}\b", value) for word in words)


def _single_line(value: str, *, limit: int) -> str:
    compact = " ".join(value.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."
