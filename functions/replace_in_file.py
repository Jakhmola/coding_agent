from pathlib import Path

from coding_agent.workspace_policy import WorkspacePolicy


def _policy(working_directory: str | Path, policy: WorkspacePolicy | None) -> WorkspacePolicy:
    return policy or WorkspacePolicy(root=Path(working_directory))


def replace_in_file(
    working_directory: str | Path,
    file_path: str,
    old_text: str,
    new_text: str,
    expected_replacements: int = 1,
    policy: WorkspacePolicy | None = None,
) -> str:
    workspace_policy = _policy(working_directory, policy)

    try:
        workspace_policy.ensure_write_allowed()
        full_path = workspace_policy.resolve_path(file_path)

        if not full_path.is_file():
            return f'Error: File not found or is not a regular file: "{file_path}"'
        if not old_text:
            return "Error: old_text must not be empty"
        if expected_replacements < 1:
            return "Error: expected_replacements must be 1 or greater"
        if full_path.stat().st_size > workspace_policy.max_file_size_bytes:
            return f'Error: File "{file_path}" exceeds the maximum allowed size'

        original_content = full_path.read_text()
        actual_replacements = original_content.count(old_text)
        if actual_replacements != expected_replacements:
            return (
                f'Error: Expected {expected_replacements} replacement(s) in "{file_path}" '
                f"but found {actual_replacements}; no changes were made"
            )

        updated_content = original_content.replace(
            old_text,
            new_text,
            expected_replacements,
        )
        if len(updated_content.encode()) > workspace_policy.max_file_size_bytes:
            return f'Error: Updated content for "{file_path}" exceeds the maximum allowed size'

        full_path.write_text(updated_content)
        return (
            f'Successfully replaced {actual_replacements} occurrence(s) '
            f'in "{file_path}"'
        )
    except ValueError:
        return f'Error: Cannot edit "{file_path}" as it is outside the permitted working directory'
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"
