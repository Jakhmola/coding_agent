# Coding Agent

A local MCP-backed coding agent that runs against a llama.cpp OpenAI-compatible model server. It can inspect a workspace, search files, make narrow edits, run allowed Python files, and optionally send workflow traces to Opik.

This is a learning project. The agent can read, edit, and run code inside the configured workspace, so use it on disposable or version-controlled files.

## Quick Start

1. Install dependencies:

```bash
uv sync
```

2. Create your local environment file:

```bash
cp .env.example .env
```

3. Start the model server and MCP server, then ask a prompt:

```bash
make run PROMPT="what is the project even about"
```

For later prompts, use the faster path while the stack is already running:

```bash
make ask PROMPT="list files in the workspace"
```

Stop the local services with:

```bash
make dev-down
```

## Common Commands

```bash
make dev-up                 # start llama.cpp and MCP
make ask PROMPT="..."       # run the agent against a started stack
make run PROMPT="..."       # checked one-command run
make mcp-logs               # tail MCP server logs
make validate               # tests, compile check, MCP list, compose config
```

## Configuration

Important `.env` values:

```env
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=llama.cpp
WORKSPACE_DIR=./calculator
WORKSPACE_READ_ONLY=false
OPIK_ENABLED=false
OPIK_URL=https://www.comet.com/opik/api
```

Set `OPIK_ENABLED=true` plus `OPIK_API_KEY` and `OPIK_WORKSPACE` to log traces to Opik Cloud. For local Opik, point `OPIK_URL` at your local Opik API endpoint and leave workspace/API key empty if your local deployment does not use them.

## Architecture

- `llama.cpp` serves the local GGUF model through an OpenAI-compatible API.
- `coding_agent.mcp_server` exposes workspace tools over Streamable HTTP MCP.
- `coding_agent.agent` runs the LangGraph workflow, talks to the model, gates tool calls, calls MCP tools, and formats the final answer.
- `coding_agent.tracing` adds optional Opik traces around workflow, model, policy, and tool steps.

## Validation

```bash
make validate
```

For a model/API smoke check only:

```bash
make model-check
```
