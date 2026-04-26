from fnmatch import fnmatch
from pathlib import Path

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


def _is_ignored_path(relative_path: Path) -> bool:
    return any(part in IGNORED_DIRECTORY_NAMES for part in relative_path.parts)


def _truncate_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"\n[...Output truncated at {max_chars} characters]"


def search_files(
    working_directory: str | Path,
    pattern: str,
    directory: str = ".",
    max_results: int = 50,
    policy: WorkspacePolicy | None = None,
) -> str:
    workspace_policy = _policy(working_directory, policy)

    try:
        search_root = workspace_policy.resolve_path(directory)
        if not search_root.is_dir():
            return f'Error: "{directory}" is not a directory'

        normalized_pattern = _normalize_pattern(pattern).lower()
        limit = _clamp_max_results(max_results)
        matches: list[str] = []

        for entry in sorted(search_root.rglob("*")):
            try:
                relative_path = entry.resolve().relative_to(workspace_policy.resolved_root)
            except ValueError:
                continue
            if _is_ignored_path(relative_path):
                continue

            relative_text = relative_path.as_posix()
            if not fnmatch(entry.name.lower(), normalized_pattern) and not fnmatch(
                relative_text.lower(),
                normalized_pattern,
            ):
                continue

            stat = entry.stat()
            matches.append(
                f"- {relative_text}: file_size={stat.st_size} bytes, is_dir={entry.is_dir()}"
            )
            if len(matches) >= limit:
                matches.append(f"[...Results truncated at {limit} matches]")
                break

        if not matches:
            return f'No files matched "{pattern}" in "{directory}"'
        return _truncate_output(
            "\n".join(matches),
            workspace_policy.max_tool_output_chars,
        )
    except ValueError:
        return f'Error: Cannot search "{directory}" as it is outside the permitted working directory'
    except Exception as e:
        return f"Error searching files: {e}"
