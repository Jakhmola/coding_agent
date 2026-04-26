from __future__ import annotations

import asyncio
import logging

from coding_agent.config import Settings, load_settings
from coding_agent.logging import configure_logging
from coding_agent.tracing import Tracer
from coding_agent.workflow import (
    AgentResult,
    ModelClient,
    ToolClient,
    format_agent_trace,
    mcp_tool_to_openai_tool,
    run_workflow,
)


async def run_agent(
    user_prompt: str,
    *,
    settings: Settings | None = None,
    model_client: ModelClient | None = None,
    tool_client: ToolClient | None = None,
    tracer: Tracer | None = None,
) -> AgentResult:
    return await run_workflow(
        user_prompt,
        settings=settings,
        model_client=model_client,
        tool_client=tool_client,
        tracer=tracer,
    )


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
        for line in format_agent_trace(result):
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
