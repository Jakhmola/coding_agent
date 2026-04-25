from google.genai import types

from coding_agent.config import load_settings
from functions.get_file_content import get_file_content, schema_get_file_content
from functions.get_files_info import get_files_info, schema_get_files_info
from functions.run_python_file import run_python_file, schema_run_python_file
from functions.write_file import schema_write_file, write_file

available_functions = types.Tool(
    function_declarations=[
        schema
        for schema in [
            schema_get_files_info,
            schema_get_file_content,
            schema_write_file,
            schema_run_python_file,
        ]
        if schema is not None
    ]
)

callable_functions = {
    "get_files_info": get_files_info,
    "get_file_content": get_file_content,
    "write_file": write_file,
    "run_python_file": run_python_file,
}

def call_function(function_call_part: types.FunctionCall, verbose: bool = False) -> None|types.Content:
    function_name = function_call_part.name
    if not function_name:
        print(f"Error: function has no name")
        return

    function_args = function_call_part.args
    if not function_args:
        print(f"Error: function \"{function_name}\" has no arguments")
        return

    if verbose:
        print(f" - Calling function: {function_name}({function_call_part.args})")
    else:
        print(f" - Calling function: {function_name}")

    try:
        func_to_run = callable_functions[function_name]
    except KeyError:
        return types.Content(
            role="tool",
            parts=[
                types.Part.from_function_response(
                    name=function_name,
                    response={"error": f"Unknown function: {function_name}"},
                )
            ],
        )

    settings = load_settings()
    function_result = func_to_run(
        settings.workspace_policy.root,
        policy=settings.workspace_policy,
        **function_args,
    )
    return types.Content(
        role="tool",
        parts=[
            types.Part.from_function_response(
                name=function_name,
                response={"result": function_result},
            )
        ],
    )
