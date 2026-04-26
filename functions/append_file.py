from pathlib import Path

from coding_agent.workspace_policy import WorkspacePolicy


def _policy(working_directory: str | Path, policy: WorkspacePolicy | None) -> WorkspacePolicy:
    return policy or WorkspacePolicy(root=Path(working_directory))


def append_file(
    working_directory: str | Path,
    file_path: str,
    content: str,
    add_trailing_newline: bool = True,
    policy: WorkspacePolicy | None = None,
) -> str:
    workspace_policy = _policy(working_directory, policy)

    try:
        workspace_policy.ensure_write_allowed()
        full_path = workspace_policy.resolve_path(file_path)

        if full_path.exists() and not full_path.is_file():
            return f'Error: "{file_path}" is not a regular file'

        append_content = content
        if add_trailing_newline and not append_content.endswith("\n"):
            append_content += "\n"

        if full_path.exists() and full_path.stat().st_size > 0:
            with full_path.open("rb") as existing_file:
                existing_file.seek(-1, 2)
                last_byte = existing_file.read(1)
            if last_byte != b"\n" and not append_content.startswith("\n"):
                append_content = "\n" + append_content

        existing_size = full_path.stat().st_size if full_path.exists() else 0
        if existing_size + len(append_content.encode()) > workspace_policy.max_file_size_bytes:
            return f'Error: Appending to "{file_path}" would exceed the maximum allowed size'

        full_path.parent.mkdir(parents=True, exist_ok=True)
        with full_path.open("a") as f:
            _ = f.write(append_content)

        return (
            f'Successfully appended to "{file_path}" '
            f"({len(append_content)} characters appended)"
        )
    except ValueError:
        return f'Error: Cannot append to "{file_path}" as it is outside the permitted working directory'
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"
