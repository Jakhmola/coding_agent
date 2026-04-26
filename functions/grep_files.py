from fnmatch import fnmatch
from pathlib import Path
import re

from coding_agent.workspace_policy import WorkspacePolicy


IGNORED_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "coverage",
    "dist",
    "htmlcov",
    "node_modules",
    "target",
}


def _policy(working_directory: str | Path, policy: WorkspacePolicy | None) -> WorkspacePolicy:
    return policy or WorkspacePolicy(root=Path(working_directory))


def _normalize_pattern(pattern: str) -> str:
    if any(char in pattern for char in "*?[]"):
        return pattern
    return f"*{pattern}*"


def _clamp_max_results(max_results: int) -> int:
    return min(max(max_results, 1), 100)


def _truncate_line(line: str, limit: int = 240) -> str:
    stripped = line.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 3] + "..."


def _is_ignored_path(relative_path: Path) -> bool:
    return any(part in IGNORED_DIRECTORY_NAMES for part in relative_path.parts)


def _truncate_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"\n[...Output truncated at {max_chars} characters]"


def grep_files(
    working_directory: str | Path,
    pattern: str,
    directory: str = ".",
    file_pattern: str = "*",
    case_sensitive: bool = False,
    max_results: int = 50,
    policy: WorkspacePolicy | None = None,
) -> str:
    workspace_policy = _policy(working_directory, policy)

    try:
        search_root = workspace_policy.resolve_path(directory)
        if not search_root.is_dir():
            return f'Error: "{directory}" is not a directory'

        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return f"Error: Invalid regex pattern: {exc}"

        normalized_file_pattern = _normalize_pattern(file_pattern).lower()
        limit = _clamp_max_results(max_results)
        matches: list[str] = []

        for entry in sorted(search_root.rglob("*")):
            if not entry.is_file():
                continue
            try:
                relative_path = entry.resolve().relative_to(workspace_policy.resolved_root)
            except ValueError:
                continue
            if _is_ignored_path(relative_path):
                continue

            relative_text = relative_path.as_posix()
            if not fnmatch(entry.name.lower(), normalized_file_pattern) and not fnmatch(
                relative_text.lower(),
                normalized_file_pattern,
            ):
                continue
            if entry.stat().st_size > workspace_policy.max_file_size_bytes:
                continue

            try:
                content = entry.read_text(errors="replace")
            except UnicodeDecodeError:
                continue

            for line_number, line in enumerate(content.splitlines(), start=1):
                if regex.search(line):
                    matches.append(
                        f"{relative_text}:{line_number}: {_truncate_line(line)}"
                    )
                    if len(matches) >= limit:
                        matches.append(f"[...Results truncated at {limit} matches]")
                        return _truncate_output(
                            "\n".join(matches),
                            workspace_policy.max_tool_output_chars,
                        )

        if not matches:
            return f'No matches for "{pattern}" in "{directory}"'
        return _truncate_output(
            "\n".join(matches),
            workspace_policy.max_tool_output_chars,
        )
    except ValueError:
        return f'Error: Cannot grep "{directory}" as it is outside the permitted working directory'
    except Exception as e:
        return f"Error grepping files: {e}"
