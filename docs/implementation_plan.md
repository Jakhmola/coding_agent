# MCP Coding Agent V1 Plan

This is the living implementation plan for the coding agent upgrade. Keep the checklist updated as each slice lands so future work can resume without reconstructing context.

## Target Architecture

- `llama.cpp`: local OpenAI-compatible model endpoint on GPU.
- `mcp-server`: Streamable HTTP MCP server exposing prompts and current coding tools.
- `workflow`: LangGraph internal coding workflow backed by the MCP client and local model.
- `cli`: one-shot command wrapper that calls the internal workflow directly.
- `opik`: optional tracing for workflow runs, model calls, policy decisions, and MCP tool calls.
- `a2a-agent`: deferred future HTTP wrapper around the internal workflow.

Default model: `WithinUsAI/Opus4.7-GODs.Ghost.Codex-4B.GGuF`, using `Opus4.7-Distill-GODsGhost-Codex-4B-Q5_K_M.gguf` and served as `opus-ghost-codex-4b-q5`.

## Key Decisions

- Use llama.cpp for v1 because it can serve GGUF models directly, remains OpenAI-compatible, is Docker-friendly, and gives explicit control over CUDA offload.
- Require the model to run with `--n-gpu-layers all`. If `ctx-size 4096` fails to start or fails the smoke checks, reduce once to `2048`; if it still fails, fail loudly and do not fall back silently.
- Keep tool growth policy-driven: read/search tools are broadly safe, precise edit tools are preferred before full-file writes, and command tools remain tightly gated.
- Use a hybrid LangGraph state: explicit routing fields plus append-only events for trace/debug and future A2A streaming.
- Defer A2A and multi-agent roles until the direct CLI/workflow path is stable.
- Keep current direct Gemini path only as temporary compatibility until the OpenAI-compatible agent loop replaces it.
- Protect local `.env`; only `.env.example` belongs in git.

## Checklist

- [x] Slice 1: Config, constants, structured logging foundation, workspace policy config, `.env.example`.
- [x] Slice 2: Harden current tools with workspace policy and focused tests.
- [x] Slice 3: MCP server exposing current tools and system prompt.
- [x] Slice 4: llama.cpp Docker service and model check workflow.
- [x] Slice 5: OpenAI-compatible MCP-backed agent loop with mocked model tests.
- [x] Slice 6: Hybrid LangGraph internal workflow and tool policy.
- [x] Slice 7: Safer search/edit tools and node-specific executor prompting.
- [x] Slice 8: Opik tracing hooks with clean disabled mode.
- [ ] Slice 9: Docker Compose and Makefile workflow.
- [ ] Slice 10: Remove obsolete direct-dispatch code and update README.

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

## Slice 6 Acceptance Criteria

- Replace the single-loop agent internals with a raw LangGraph `StateGraph` while preserving the public `run_agent(...)` entry point.
- Use a hybrid state model with explicit routing fields and append-only events.
- Classify intent deterministically for read-only, write, run, git, dependency, and unknown requests.
- Enforce tool policy before MCP execution: read tools are allowed broadly, `write_file` requires write intent, and `run_python_file` requires run intent.
- Stop direct file-content requests after a successful `get_file_content` result instead of asking the model for more tool calls.
- Render verbose CLI output from structured workflow events.
- Do not add new tools, A2A, or human approval interrupts in this slice.

Status: complete in `coding_agent/workflow.py` and `coding_agent/agent.py`. Regression target:

```bash
.venv/bin/python -m coding_agent.agent --verbose "print lorem.txt file content"
```

Expected behavior: one `get_file_content` call, no `write_file`, and final answer is the file content.

## Slice 7 Acceptance Criteria

- Add safer workspace tools: `search_files`, `grep_files`, `append_file`, and `replace_in_file`.
- Register the new tools on the MCP server and keep direct tool behavior covered by unit tests.
- Expand workflow plans with allowed tools, forbidden tools, preferred tools, expected write paths, max tool calls, and completion signals.
- Filter tools sent to the executor model based on the current plan.
- Add executor-node instructions that tell the model which tools are preferred and which tools are forbidden for the current request.
- Prefer `append_file` for add/append/insert requests and `replace_in_file` for exact edit requests.
- Block broad `write_file` use when a narrower edit tool is appropriate.
- Block writes to a different path when the prompt names a specific expected write target.
- Keep read-only prompts able to use search and grep while still blocking writes and runs.

Status: complete in `coding_agent/workflow.py`, `coding_agent/mcp_server.py`, `functions/`, `prompts.py`, and tests.

## Slice 8 Acceptance Criteria

- Add optional Opik tracing without changing workflow behavior when `OPIK_ENABLED=false`.
- Keep the Opik SDK import lazy so disabled mode does not require credentials or an installed/configured Opik service.
- Record one top-level workflow trace with request id, intent, final status, iteration count, tool count, and blocked reason.
- Record spans for intake, intent classification, planning, model calls, policy checks, MCP tool execution, progress review, final response, and failure response.
- Mark model spans as `llm`, tool spans as `tool`, and policy block spans as `guardrail`.
- Sanitize trace payloads so full prompts, full files, full tool results, full message arrays, and secrets are not logged.
- Preserve OpenAI-compatible token usage on `ModelResponse` for tracing when the server returns it.
- Keep tracing at workflow orchestration boundaries so mocked tests and real clients share the same instrumentation.
- Do not add A2A in this slice.

Status: complete in `coding_agent/tracing.py`, `coding_agent/workflow.py`, and tests. Opik is enabled by setting `OPIK_ENABLED=true`; `OPIK_URL` is passed to the SDK as the client `host`.

## Test Strategy

- Unit tests should avoid paid or remote model calls.
- Use mocked model responses for future agent-loop tests.
- Use MCP server/client smoke tests without involving the model runtime.
- Keep one local model smoke test only after llama.cpp server verification passes.

## Current Validation Commands

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m compileall -q coding_agent functions tests main.py call_function.py scripts/model_check.py prompts.py
.venv/bin/python -m coding_agent.mcp_server --list
make compose-check
make llama-up
make model-check
```

## Notes

- Existing local changes to `calculator/lorem.txt` are user-owned.
- If dependencies need to change, prefer updating `pyproject.toml` intentionally in the slice that needs them and explain why.
- A2A remains deferred future work; keep CLI direct-to-workflow for the V1 path until tracing and policy behavior are stable.
