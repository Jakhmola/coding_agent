from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from coding_agent.config import Settings, load_settings
from coding_agent.logging import configure_logging, get_logger
from functions.append_file import append_file
from functions.get_file_content import get_file_content
from functions.get_files_info import get_files_info
from functions.grep_files import grep_files
from functions.replace_in_file import replace_in_file
from functions.run_python_file import run_python_file
from functions.search_files import search_files
from functions.write_file import write_file
from prompts import system_prompt

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    FastMCP = None


logger = get_logger(__name__)


def _require_mcp() -> None:
    if FastMCP is None:
        raise RuntimeError(
            "MCP SDK is not installed. Install project dependencies before running "
            "the MCP server."
        )


def _dump_model(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def build_mcp_server(settings: Settings | None = None) -> Any:
    _require_mcp()
    settings = settings or load_settings()
    policy = settings.workspace_policy

    server = FastMCP(
        "coding-agent",
        instructions=system_prompt.strip(),
        host=settings.mcp_host,
        port=settings.mcp_port,
        streamable_http_path=settings.mcp_path,
        log_level=settings.log_level.upper(),
    )

    @server.prompt(
        name="coding_agent_system_prompt",
        description="System prompt used by the coding agent.",
    )
    def coding_agent_system_prompt() -> str:
        return system_prompt.strip()

    @server.tool(
        name="get_files_info",
        description="List files in a workspace directory with size and directory metadata.",
    )
    def mcp_get_files_info(directory: str = ".") -> str:
        return get_files_info(policy.root, directory, policy=policy)

    @server.tool(
        name="get_file_content",
        description="Read a workspace file subject to configured size and read limits.",
    )
    def mcp_get_file_content(file_path: str) -> str:
        return get_file_content(policy.root, file_path, policy=policy)

    @server.tool(
        name="search_files",
        description="Recursively search workspace file and directory names by glob or substring.",
    )
    def mcp_search_files(
        pattern: str,
        directory: str = ".",
        max_results: int = 50,
    ) -> str:
        return search_files(
            policy.root,
            pattern,
            directory=directory,
            max_results=max_results,
            policy=policy,
        )

    @server.tool(
        name="grep_files",
        description="Search file contents by regex within workspace limits.",
    )
    def mcp_grep_files(
        pattern: str,
        directory: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> str:
        return grep_files(
            policy.root,
            pattern,
            directory=directory,
            file_pattern=file_pattern,
            case_sensitive=case_sensitive,
            max_results=max_results,
            policy=policy,
        )

    @server.tool(
        name="append_file",
        description="Append content to a workspace file without replacing existing content.",
    )
    def mcp_append_file(
        file_path: str,
        content: str,
        add_trailing_newline: bool = True,
    ) -> str:
        return append_file(
            policy.root,
            file_path,
            content,
            add_trailing_newline=add_trailing_newline,
            policy=policy,
        )

    @server.tool(
        name="replace_in_file",
        description="Replace exact text in a workspace file, failing if match count differs.",
    )
    def mcp_replace_in_file(
        file_path: str,
        old_text: str,
        new_text: str,
        expected_replacements: int = 1,
    ) -> str:
        return replace_in_file(
            policy.root,
            file_path,
            old_text,
            new_text,
            expected_replacements=expected_replacements,
            policy=policy,
        )

    @server.tool(
        name="write_file",
        description="Write content to a workspace file when writes are enabled.",
    )
    def mcp_write_file(file_path: str, content: str) -> str:
        return write_file(policy.root, file_path, content, policy=policy)

    @server.tool(
        name="run_python_file",
        description="Run a Python file in the workspace using the configured policy.",
    )
    def mcp_run_python_file(file_path: str) -> str:
        return run_python_file(policy.root, file_path, policy=policy)

    return server


async def list_server_capabilities(settings: Settings | None = None) -> dict[str, Any]:
    server = build_mcp_server(settings)
    tools = await server.list_tools()
    prompts = await server.list_prompts()
    return {
        "tools": [_dump_model(tool) for tool in tools],
        "prompts": [_dump_model(prompt) for prompt in prompts],
    }


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(".env")


def main() -> None:
    _load_dotenv_if_available()
    settings = load_settings()
    configure_logging(settings.log_level, settings.log_json)

    parser = argparse.ArgumentParser(description="Coding agent MCP server")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered MCP tools and prompts without starting the server.",
    )
    args = parser.parse_args()

    if args.list:
        capabilities = asyncio.run(list_server_capabilities(settings))
        print(json.dumps(capabilities, indent=2, sort_keys=True))
        return

    logger.info(
        "starting_mcp_server",
        extra={
            "service": "mcp-server",
            "host": settings.mcp_host,
            "port": settings.mcp_port,
            "path": settings.mcp_path,
        },
    )
    server = build_mcp_server(settings)
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
