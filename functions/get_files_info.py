from pathlib import Path

from coding_agent.workspace_policy import WorkspacePolicy


try:
    from google.genai import types
except ModuleNotFoundError:
    types = None


def _policy(working_directory: str | Path, policy: WorkspacePolicy | None) -> WorkspacePolicy:
    return policy or WorkspacePolicy(root=Path(working_directory))


def get_files_info(
    working_directory: str | Path,
    directory: str,
    policy: WorkspacePolicy | None = None,
) -> str:
    workspace_policy = _policy(working_directory, policy)

    try:
        path = workspace_policy.resolve_path(directory)
        if not path.is_dir():
            return f'Error: "{directory}" is not a directory'

        dir_contents = list(path.iterdir())
        if len(dir_contents) == 0:
            return f'Error: "{directory}" is empty'

        files_info: list[str] = []
        for entry in dir_contents:
            file_size = entry.stat().st_size
            is_dir = entry.is_dir()
            files_info.append(f"- {entry.name}: file_size={file_size} bytes, is_dir={is_dir}")
        return "\n".join(files_info)
    except ValueError:
        return f'Error: Cannot list "{directory}" as it is outside the permitted working directory'
    except Exception as e:
        return f"Error listing files: {e}"


schema_get_files_info = (
    types.FunctionDeclaration(
        name="get_files_info",
        description="Lists files in the specified directory along with their sizes, constrained to the working directory.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "directory": types.Schema(
                    type=types.Type.STRING,
                    description='The directory to list files from, relative to the working directory. If not provided, lists files in the working directory itself (use ".").',
                ),
            },
        ),
    )
    if types
    else None
)
