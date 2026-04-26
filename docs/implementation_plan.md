# MCP + A2A Coding Agent V1 Plan

This is the living implementation plan for the coding agent upgrade. Keep the checklist updated as each slice lands so future work can resume without reconstructing context.

## Target Architecture

- `llama.cpp`: local OpenAI-compatible model endpoint on GPU.
- `mcp-server`: Streamable HTTP MCP server exposing prompts and current coding tools.
- `a2a-agent`: A2A HTTP agent backed by the MCP client and local model.
- `cli`: one-shot command wrapper that talks to the running A2A agent.
- `opik`: optional tracing for CLI requests, A2A tasks, model calls, and MCP tool calls.

Default model: `WithinUsAI/Opus4.7-GODs.Ghost.Codex-4B.GGuF`, using `Opus4.7-Distill-GODsGhost-Codex-4B-Q5_K_M.gguf` and served as `opus-ghost-codex-4b-q5`.

## Key Decisions

- Use llama.cpp for v1 because it can serve GGUF models directly, remains OpenAI-compatible, is Docker-friendly, and gives explicit control over CUDA offload.
- Require the model to run with `--n-gpu-layers all`. If `ctx-size 4096` fails to start or fails the smoke checks, reduce once to `2048`; if it still fails, fail loudly and do not fall back silently.
- Keep v1 tools limited to the current core set: list files, read file, write file, run Python file.
- Defer extra tools and multi-agent roles until after v1 is stable.
- Keep current direct Gemini path only as temporary compatibility until the OpenAI-compatible agent loop replaces it.
- Protect local `.env`; only `.env.example` belongs in git.

## Checklist

- [x] Slice 1: Config, constants, structured logging foundation, workspace policy config, `.env.example`.
- [x] Slice 2: Harden current tools with workspace policy and focused tests.
- [x] Slice 3: MCP server exposing current tools and system prompt.
- [x] Slice 4: llama.cpp Docker service and model check workflow.
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

- Add a llama.cpp Docker Compose service with NVIDIA GPU access and persistent Hugging Face cache storage.
- Serve `WithinUsAI/Opus4.7-GODs.Ghost.Codex-4B.GGuF` with Q5_K_M quant, `ctx-size 4096`, `--n-gpu-layers all`, and `--jinja`.
- Add Make targets for starting llama.cpp, stopping it, validating Compose config, and checking the OpenAI-compatible model server.
- Enforce the strict fallback rule: if default context fails server/API smoke checks, recreate once with `ctx-size 2048`; if still failing, fail loudly.
- Do not download or start the model during normal unit tests.

Status: complete in `docker-compose.yml`, `Makefile`, and `scripts/model_check.py`.

## Slice 5 Acceptance Criteria

- Add an OpenAI-compatible chat completions client for the local model server without requiring model calls in unit tests.
- Add an MCP client adapter that can list tools, read the coding-agent system prompt, and call MCP tools over Streamable HTTP.
- Add an agent loop that converts MCP tool schemas to OpenAI tool definitions, handles model tool calls, sends tool results back to the model, and stops on final assistant content.
- Handle local model text-form tool requests when it returns a JSON-like `{name, arguments}` payload instead of native OpenAI `tool_calls`.
- Cover the loop with mocked model and MCP clients, including direct answers, tool use, multiple tool calls, malformed tool arguments, and max-iteration failure.
- Keep the old direct Gemini CLI path intact until the later cleanup slice.

Status: complete in `coding_agent/model_client.py`, `coding_agent/mcp_client.py`, and `coding_agent/agent.py`. Programmatic entry point:

```bash
.venv/bin/python -m coding_agent.agent "your prompt here"
```

Use `--verbose` to print a sequential agent trace: user input, each model turn, what was sent to the model, model responses, tool names and arguments, tool results, and the final answer.

Local smoke validation path:

```bash
make model-check
.venv/bin/python -m coding_agent.mcp_server
.venv/bin/python -m coding_agent.agent --verbose "Use the available tool to list files in . and summarize what this project is."
.venv/bin/python -m coding_agent.agent --verbose "Use the available tool to read README.md. After reading it, answer with only the first Markdown heading from that file."
```

## Test Strategy

- Unit tests should avoid paid or remote model calls.
- Use mocked model responses for future agent-loop tests.
- Use MCP server/client smoke tests without involving the model runtime.
- Keep one local model smoke test only after llama.cpp server verification passes.

## Current Validation Commands

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m compileall -q coding_agent functions tests main.py call_function.py
make llama-up
make model-check
```

## Notes

- Existing local changes to `calculator/lorem.txt`, `pyproject.toml`, and `uv.lock` are user-owned unless a later slice explicitly adopts them.
- If dependencies need to change, prefer updating `pyproject.toml` intentionally in the slice that needs them and explain why.
