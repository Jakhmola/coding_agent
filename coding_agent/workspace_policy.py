from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from coding_agent import constants


@dataclass(frozen=True)
class WorkspacePolicy:
    """Runtime limits and permissions for tools that touch the workspace."""

    root: Path
    read_only: bool = constants.DEFAULT_WORKSPACE_READ_ONLY
    allowed_commands: tuple[str, ...] = constants.DEFAULT_ALLOWED_COMMANDS
    tool_timeout_seconds: int = constants.DEFAULT_TOOL_TIMEOUT_SECONDS
    max_file_read_chars: int = constants.DEFAULT_MAX_FILE_READ_CHARS
    max_file_size_bytes: int = constants.DEFAULT_MAX_FILE_SIZE_BYTES
    max_tool_output_chars: int = constants.DEFAULT_MAX_TOOL_OUTPUT_CHARS

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    @property
    def resolved_root(self) -> Path:
        return self.root.expanduser().resolve()

    def resolve_path(self, relative_path: str) -> Path:
        candidate = (self.resolved_root / relative_path).resolve()
        try:
            candidate.relative_to(self.resolved_root)
        except ValueError as exc:
            raise ValueError(
                f'Path "{relative_path}" is outside the permitted workspace'
            ) from exc
        return candidate

    def ensure_write_allowed(self) -> None:
        if self.read_only:
            raise PermissionError("Workspace is configured as read-only")

    def ensure_command_allowed(self, command: str) -> None:
        if command not in self.allowed_commands:
            raise PermissionError(f'Command "{command}" is not allowed')
