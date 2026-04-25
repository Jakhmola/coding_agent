from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from coding_agent import constants
from coding_agent.workspace_policy import WorkspacePolicy


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _get_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc

    if value < 0:
        raise ValueError(f"{name} must be zero or greater")
    return value


def _get_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    values = tuple(part.strip() for part in raw_value.split(",") if part.strip())
    return values or default


@dataclass(frozen=True)
class OpikSettings:
    enabled: bool = constants.DEFAULT_OPIK_ENABLED
    api_key: str | None = None
    workspace: str | None = None
    project_name: str = constants.DEFAULT_OPIK_PROJECT_NAME
    url: str = constants.DEFAULT_OPIK_URL


@dataclass(frozen=True)
class Settings:
    model_name: str
    gemini_model: str
    openai_base_url: str
    openai_api_key: str
    max_iterations: int
    mcp_host: str
    mcp_port: int
    mcp_path: str
    a2a_host: str
    a2a_port: int
    log_level: str
    log_json: bool
    workspace_policy: WorkspacePolicy
    opik: OpikSettings

    @property
    def mcp_url(self) -> str:
        host = "localhost" if self.mcp_host == "0.0.0.0" else self.mcp_host
        return f"http://{host}:{self.mcp_port}{self.mcp_path}"

    @property
    def a2a_url(self) -> str:
        host = "localhost" if self.a2a_host == "0.0.0.0" else self.a2a_host
        return f"http://{host}:{self.a2a_port}"


def load_settings() -> Settings:
    workspace_policy = WorkspacePolicy(
        root=Path(os.environ.get("WORKSPACE_DIR", constants.DEFAULT_WORKSPACE_DIR)),
        read_only=_get_bool("WORKSPACE_READ_ONLY", constants.DEFAULT_WORKSPACE_READ_ONLY),
        allowed_commands=_get_csv("ALLOWED_COMMANDS", constants.DEFAULT_ALLOWED_COMMANDS),
        tool_timeout_seconds=_get_int(
            "TOOL_TIMEOUT_SECONDS", constants.DEFAULT_TOOL_TIMEOUT_SECONDS
        ),
        max_file_read_chars=_get_int(
            "MAX_FILE_READ_CHARS", constants.DEFAULT_MAX_FILE_READ_CHARS
        ),
        max_file_size_bytes=_get_int(
            "MAX_FILE_SIZE_BYTES", constants.DEFAULT_MAX_FILE_SIZE_BYTES
        ),
        max_tool_output_chars=_get_int(
            "MAX_TOOL_OUTPUT_CHARS", constants.DEFAULT_MAX_TOOL_OUTPUT_CHARS
        ),
    )

    opik = OpikSettings(
        enabled=_get_bool("OPIK_ENABLED", constants.DEFAULT_OPIK_ENABLED),
        api_key=os.environ.get("OPIK_API_KEY"),
        workspace=os.environ.get("OPIK_WORKSPACE"),
        project_name=os.environ.get(
            "OPIK_PROJECT_NAME", constants.DEFAULT_OPIK_PROJECT_NAME
        ),
        url=os.environ.get("OPIK_URL", constants.DEFAULT_OPIK_URL),
    )

    return Settings(
        model_name=os.environ.get("MODEL_NAME", constants.DEFAULT_MODEL_NAME),
        gemini_model=os.environ.get("GEMINI_MODEL", constants.DEFAULT_GEMINI_MODEL),
        openai_base_url=os.environ.get(
            "OPENAI_BASE_URL", constants.DEFAULT_OPENAI_BASE_URL
        ),
        openai_api_key=os.environ.get("OPENAI_API_KEY", constants.DEFAULT_OPENAI_API_KEY),
        max_iterations=_get_int("MAX_ITERATIONS", constants.DEFAULT_MAX_ITERATIONS),
        mcp_host=os.environ.get("MCP_HOST", constants.DEFAULT_MCP_HOST),
        mcp_port=_get_int("MCP_PORT", constants.DEFAULT_MCP_PORT),
        mcp_path=os.environ.get("MCP_PATH", constants.DEFAULT_MCP_PATH),
        a2a_host=os.environ.get("A2A_HOST", constants.DEFAULT_A2A_HOST),
        a2a_port=_get_int("A2A_PORT", constants.DEFAULT_A2A_PORT),
        log_level=os.environ.get("LOG_LEVEL", constants.DEFAULT_LOG_LEVEL),
        log_json=_get_bool("LOG_JSON", constants.DEFAULT_LOG_JSON),
        workspace_policy=workspace_policy,
        opik=opik,
    )

