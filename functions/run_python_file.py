from pathlib import Path
import subprocess

from coding_agent.workspace_policy import WorkspacePolicy


try:
    from google.genai import types
except ModuleNotFoundError:
    types = None


def _policy(working_directory: str | Path, policy: WorkspacePolicy | None) -> WorkspacePolicy:
    return policy or WorkspacePolicy(root=Path(working_directory))


def _truncate_output(output: str, max_chars: int) -> str:
    if len(output) <= max_chars:
        return output
    return output[:max_chars] + f"\n[...Output truncated at {max_chars} characters]"


def run_python_file(
    working_directory: str | Path,
    file_path: str,
    policy: WorkspacePolicy | None = None,
) -> str:
    workspace_policy = _policy(working_directory, policy)

    try:
        workspace_policy.ensure_command_allowed("python3")
        full_path = workspace_policy.resolve_path(file_path)

        if not full_path.exists():
            return f'Error: File "{file_path}" not found.'

        if not full_path.is_file() or full_path.suffix != ".py":
            return f'Error: "{file_path}" is not a Python file.'
    except ValueError:
        return f'Error: Cannot execute "{file_path}" as it is outside the permitted working directory'
    except PermissionError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"

    try:
        result = subprocess.run(
            ["python3", file_path],
            cwd=workspace_policy.resolved_root,
            capture_output=True,
            text=True,
            timeout=workspace_policy.tool_timeout_seconds,
        )

        output: list[str] = []

        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")

        if result.returncode != 0:
            output.append(f"Process exited with code {result.returncode}")

        if not output:
            return "No output produced."
        return _truncate_output(
            "\n".join(output), workspace_policy.max_tool_output_chars
        )
    except subprocess.TimeoutExpired:
        return (
            f"Error: executing Python file timed out after "
            f"{workspace_policy.tool_timeout_seconds} seconds"
        )
    except Exception as e:
        return f"Error: executing Python file: {e}"


schema_run_python_file = (
    types.FunctionDeclaration(
        name="run_python_file",
        description="Executes a Python 3 file within the working directory and returns the output from the interpreter.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "file_path": types.Schema(
                    type=types.Type.STRING,
                    description="Path to the Python file to execute, relative to the working directory.",
                ),
            },
            required=["file_path"],
        ),
    )
    if types
    else None
)
