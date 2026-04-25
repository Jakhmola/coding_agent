from pathlib import Path

from coding_agent import constants
from coding_agent.workspace_policy import WorkspacePolicy


try:
    from google.genai import types
except ModuleNotFoundError:
    types = None


def _policy(working_directory: str | Path, policy: WorkspacePolicy | None) -> WorkspacePolicy:
    return policy or WorkspacePolicy(root=Path(working_directory))


def get_file_content(
    working_directory: str | Path,
    file_path: str,
    policy: WorkspacePolicy | None = None,
) -> str:
    workspace_policy = _policy(working_directory, policy)

    try:
        full_path = workspace_policy.resolve_path(file_path)

        if not full_path.is_file():
            return f'Error: File not found or is not a regular file: "{file_path}"'

        if full_path.stat().st_size > workspace_policy.max_file_size_bytes:
            return f'Error: File "{file_path}" exceeds the maximum allowed size'

        with full_path.open("r") as f:
            file_contents = f.read(workspace_policy.max_file_read_chars)

        if len(file_contents) == workspace_policy.max_file_read_chars:
            file_contents += (
                f'[...File "{file_path}" truncated at '
                f"{workspace_policy.max_file_read_chars} characters]"
            )

        return file_contents
    except ValueError:
        return f'Error: Cannot read "{file_path}" as it is outside the permitted working directory'
    except Exception as e:
        return f"Error: {e}"


schema_get_file_content = (
    types.FunctionDeclaration(
        name="get_file_content",
        description=f"Reads and returns the first {constants.DEFAULT_MAX_FILE_READ_CHARS} characters of the content from a specified file within the working directory.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(
                    type=types.Type.STRING,
                    description="The path to the file whose content should be read, relative to the working directory.",
                ),
            },
            required=["file_path"],
        ),
    )
    if types
    else None
)
