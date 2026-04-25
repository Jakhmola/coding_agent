# MCP + A2A Coding Agent V1 Plan

This is the living implementation plan for the coding agent upgrade. Keep the checklist updated as each slice lands so future work can resume without reconstructing context.

## Target Architecture

- `ollama`: local OpenAI-compatible model endpoint on GPU.
- `mcp-server`: Streamable HTTP MCP server exposing prompts and current coding tools.
- `a2a-agent`: A2A HTTP agent backed by the MCP client and local model.
- `cli`: one-shot command wrapper that talks to the running A2A agent.
- `opik`: optional tracing for CLI requests, A2A tasks, model calls, and MCP tool calls.

Default model: `qwen2.5-coder:7b-instruct-q4_0`, wrapped as `coding-qwen-gpu`.

## Key Decisions

- Use Ollama instead of vLLM for v1 because it is simpler, OpenAI-compatible, Docker-friendly, and practical for Q4 models on a 6GB NVIDIA GPU.
- Require the model to run fully on GPU. If `num_ctx 4096` spills to CPU, reduce once to `2048`; if it still spills, fail loudly and do not change model automatically.
- Keep v1 tools limited to the current core set: list files, read file, write file, run Python file.
- Defer extra tools and multi-agent roles until after v1 is stable.
- Keep current direct Gemini path only as temporary compatibility until the OpenAI-compatible agent loop replaces it.
- Protect local `.env`; only `.env.example` belongs in git.

## Checklist

- [x] Slice 1: Config, constants, structured logging foundation, workspace policy config, `.env.example`.
- [x] Slice 2: Harden current tools with workspace policy and focused tests.
- [x] Slice 3: MCP server exposing current tools and system prompt.
- [x] Slice 4: Ollama Docker service, Modelfile, model pull/check workflow.
- [x] Slice 5: OpenAI-compatible MCP-backed agent loop with mocked model tests.
- [ ] Slice 6: A2A HTTP wrapper and CLI commands.
- [ ] Slice 7: Opik tracing hooks with clean disabled mode.
- [ ] Slice 8: Docker Compose and Makefile workflow.
- [ ] Slice 9: Remove obsolete direct-dispatch code and update README.

## Slice 3 Acceptance Criteria

- Add an MCP server module that exposes the current four tools and the coding-agent system prompt.
- Keep tool behavior backed by the already-hardened tool functions and `WorkspacePolicy`.
- Add a smoke command or test path that can list MCP tools and prompt metadata without starting the full future stack.
- Avoid model calls and avoid Docker work in this slice.
- Keep existing tests passing.

Status: complete in `coding_agent/mcp_server.py`. Smoke command:

```bash
.venv/bin/python -m coding_agent.mcp_server --list
```

## Slice 4 Acceptance Criteria

- Add an Ollama Docker Compose service with NVIDIA GPU access and persistent model storage.
- Add Modelfiles for `coding-qwen-gpu` using `qwen2.5-coder:7b-instruct-q4_0` at `num_ctx 4096`, plus one reduced-context `2048` fallback.
- Add Make targets for starting Ollama, pulling/creating the model, validating Compose config, and checking the model is loaded `100% GPU`.
- Enforce the strict fallback rule: if default context is not `100% GPU`, recreate once with reduced context; if still not `100% GPU`, fail loudly.
- Do not pull the model automatically during normal tests.

Status: complete in `docker-compose.yml`, `models/`, `Makefile`, and `scripts/model_check.py`.

## Slice 5 Acceptance Criteria

- Add an OpenAI-compatible chat completions client for Ollama without requiring model calls in unit tests.
- Add an MCP client adapter that can list tools, read the coding-agent system prompt, and call MCP tools over Streamable HTTP.
- Add an agent loop that converts MCP tool schemas to OpenAI tool definitions, handles model tool calls, sends tool results back to the model, and stops on final assistant content.
- Handle Qwen/Ollama's local text-form tool requests when it returns a JSON-like `{name, arguments}` payload instead of native OpenAI `tool_calls`.
- Cover the loop with mocked model and MCP clients, including direct answers, tool use, multiple tool calls, malformed tool arguments, and max-iteration failure.
- Keep the old direct Gemini CLI path intact until the later cleanup slice.

Status: complete in `coding_agent/model_client.py`, `coding_agent/mcp_client.py`, and `coding_agent/agent.py`. Programmatic entry point:

```bash
.venv/bin/python -m coding_agent.agent "your prompt here"
```

Use `--verbose` to print a sequential agent trace: user input, each model turn, what was sent to the model, model responses, tool names and arguments, tool results, and the final answer.

Local smoke validation passed with `coding-qwen-gpu` loaded `100% GPU`:

```bash
make model-check
.venv/bin/python -m coding_agent.mcp_server
.venv/bin/python -m coding_agent.agent --verbose "Use the available tool to list files in . and summarize what this project is."
.venv/bin/python -m coding_agent.agent --verbose "Use the available tool to read README.md. After reading it, answer with only the first Markdown heading from that file."
```

## Test Strategy

- Unit tests should avoid paid or remote model calls.
- Use mocked model responses for future agent-loop tests.
- Use MCP server/client smoke tests without involving Ollama.
- Keep one local model smoke test for the later Ollama slice, only after GPU verification passes.

## Current Validation Commands

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q coding_agent functions tests main.py call_function.py
python3 main.py
```

## Notes

- Existing local changes to `calculator/lorem.txt`, `pyproject.toml`, and `uv.lock` are user-owned unless a later slice explicitly adopts them.
- If dependencies need to change, prefer updating `pyproject.toml` intentionally in the slice that needs them and explain why.
