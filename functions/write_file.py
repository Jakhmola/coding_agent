from pathlib import Path

from coding_agent.workspace_policy import WorkspacePolicy


try:
    from google.genai import types
except ModuleNotFoundError:
    types = None


def _policy(working_directory: str | Path, policy: WorkspacePolicy | None) -> WorkspacePolicy:
    return policy or WorkspacePolicy(root=Path(working_directory))


def write_file(
    working_directory: str | Path,
    file_path: str,
    content: str,
    policy: WorkspacePolicy | None = None,
) -> str:
    workspace_policy = _policy(working_directory, policy)

    try:
        workspace_policy.ensure_write_allowed()
        full_path = workspace_policy.resolve_path(file_path)

        if len(content.encode()) > workspace_policy.max_file_size_bytes:
            return f'Error: Content for "{file_path}" exceeds the maximum allowed size'

        full_path.parent.mkdir(parents=True, exist_ok=True)

        with full_path.open("w") as f:
            _ = f.write(content)

        return f'Successfully wrote to "{file_path}" ({len(content)} characters written)'
    except ValueError:
        return f'Error: Cannot write to "{file_path}" as it is outside the permitted working directory'
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


schema_write_file = (
    types.FunctionDeclaration(
        name="write_file",
        description="Writes content to a file within the working directory. Creates the file if it doesn't exist.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(
                    type=types.Type.STRING,
                    description="The file path to write the content to, relative to the working directory.",
                ),
                "content": types.Schema(
                    type=types.Type.STRING,
                    description="The content you want to write into a file.",
                ),
            },
            required=["file_path", "content"],
        ),
    )
    if types
    else None
)
