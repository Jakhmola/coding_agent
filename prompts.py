from __future__ import annotations

from typing import Any


JsonObject = dict[str, Any]


system_prompt = """
You are a helpful AI coding agent.

When a user asks a question or makes a request, make a function call plan. You can perform the following operations:

- List files and directories
- Search file names
- Search file contents
- Read file contents
- Execute Python files with optional arguments
- Append to files
- Replace exact text in files
- Write or overwrite files

All paths you provide should be relative to the working directory.
If you want to list the files and directories of the current working directory then use "." as a parameter for the directory.
You do not need to specify the working directory in your function calls as it is automatically injected for security reasons.
Prefer append and exact replacement tools over full-file writes when the user asks to add or edit existing content.
"""


def build_base_system_prompt(system_prompt_text: str) -> str:
    return (
        system_prompt_text.strip()
        + "\n\n"
        + "Internal workflow: deterministic nodes classify intent and create a plan "
        + "before the executor model chooses tools. Follow the latest node-specific "
        + "instructions exactly."
    )


def build_executor_node_prompt(
    *,
    intent: str,
    risk_level: str,
    allowed_tools: list[str],
    forbidden_tools: list[str],
    preferred_tools: list[str],
    expected_write_paths: list[str],
    tool_calls_used: int,
    max_tool_calls: int,
    completion_signal: str,
) -> str:
    return "\n".join(
        [
            "Executor node instructions:",
            "- Choose at most the tools needed for the next concrete step.",
            "- For explanatory questions, gather just enough evidence, then answer in concise human prose.",
            "- Do not include code blocks unless the user explicitly asks for code or commands.",
            "- Never output tool-call markup as a final answer.",
            "- Use preferred tools before broader tools.",
            "- For append/add/insert requests, use append_file instead of write_file.",
            "- For replace/change/edit requests, use replace_in_file when exact text is known.",
            "- For vague file references, locate the file with search_files or get_files_info before editing.",
            "- Do not call forbidden tools even if they appear in the global tool list.",
            f"Intent: {intent}",
            f"Risk level: {risk_level}",
            f"Allowed tools: {_join_or_none(allowed_tools)}",
            f"Forbidden tools: {_join_or_none(forbidden_tools)}",
            f"Preferred tools: {_join_or_none(preferred_tools)}",
            f"Expected write paths: {_join_or_none(expected_write_paths)}",
            f"Tool calls used: {tool_calls_used}/{max_tool_calls}",
            f"Completion signal: {completion_signal}",
        ]
    )


def build_repair_node_prompt(blocked_events: list[JsonObject]) -> str:
    if not blocked_events:
        return ""

    latest_block = blocked_events[-1]
    lines = [
        "Repair node instructions:",
        "- Your previous tool request was blocked by policy.",
        "- Correct the next tool request using the recovery hint, allowed tools, and expected paths.",
        "- Do not repeat the blocked tool call.",
        f"Blocked tool: {latest_block.get('tool', 'unknown')}",
        f"Blocked reason: {latest_block.get('reason', 'unknown')}",
    ]
    recovery_hint = latest_block.get("recovery_hint")
    if isinstance(recovery_hint, str) and recovery_hint:
        lines.append(f"Recovery hint: {recovery_hint}")
    return "\n".join(lines)


def build_final_response_prompt(
    *,
    user_prompt: str,
    intent: str,
    blocked_reason: str | None,
    tool_events: list[JsonObject],
    write_events: list[JsonObject],
) -> str:
    lines = [
        "Final response node instructions:",
        "- Answer the user using only the gathered tool results and policy events.",
        "- Do not invent file contents, command output, or edits that were not observed.",
        "- If the task was blocked or failed, say why plainly.",
        "- If files were edited, summarize the touched path and verified tool result.",
        "- Do not request or imply any additional tool calls.",
        f"User request: {user_prompt}",
        f"Intent: {intent}",
    ]
    if blocked_reason:
        lines.append(f"Blocked reason: {blocked_reason}")
    if tool_events:
        lines.append(f"Observed tool results: {len(tool_events)}")
    if write_events:
        lines.append(f"Observed successful writes: {len(write_events)}")
    return "\n".join(lines)


def _join_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none"
