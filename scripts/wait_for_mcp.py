from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coding_agent.config import load_settings


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ModuleNotFoundError:
        return
    load_dotenv(".env")


def main() -> int:
    parser = argparse.ArgumentParser(description="Wait for the local MCP server port.")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    _load_dotenv_if_available()
    settings = load_settings()
    host = "localhost" if settings.mcp_host == "0.0.0.0" else settings.mcp_host
    deadline = time.monotonic() + args.timeout
    last_error: OSError | None = None

    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, settings.mcp_port), timeout=1.0):
                print(f"MCP server is listening on {host}:{settings.mcp_port}")
                return 0
        except OSError as exc:
            last_error = exc
            time.sleep(0.5)

    print(f"ERROR: MCP server did not start on {host}:{settings.mcp_port}: {last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
