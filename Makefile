SHELL := /usr/bin/env bash

COMPOSE ?= docker compose
PYTHON ?= .venv/bin/python
LLAMA_SERVICE ?= llama-cpp
LLAMA_URL ?= http://localhost:8080
LLAMA_MODEL ?= opus-ghost-codex-4b-q5
LLAMA_CTX_SIZE ?= 4096
LLAMA_LOWCTX_SIZE ?= 2048
RUN_DIR ?= .run
MCP_PID ?= $(RUN_DIR)/mcp-server.pid
MCP_LOG ?= $(RUN_DIR)/mcp-server.log
PROMPT ?= what is the project even about
AGENT_FLAGS ?= --verbose

.PHONY: help llama-up llama-down model-check mcp-up mcp-down mcp-restart mcp-logs mcp-check dev-up dev-down ask run smoke compose-check test compile validate

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make dev-up        Start llama.cpp container and local MCP server' \
		'  make dev-down      Stop local MCP server and llama.cpp container' \
		'  make ask PROMPT="..."  Run the agent against an already-started stack' \
		'  make run PROMPT="..."  One-command checked run: llama check + MCP + agent' \
		'  make smoke         Run the default project-overview agent smoke' \
		'  make mcp-up        Start local MCP server in the background' \
		'  make mcp-down      Stop local MCP server' \
		'  make mcp-logs      Tail local MCP server logs' \
		'  make llama-up      Start llama.cpp server detached' \
		'  make llama-down    Stop llama.cpp server' \
		'  make model-check   Verify llama.cpp model server and OpenAI-compatible API' \
		'  make compose-check Validate docker-compose.yml syntax' \
		'  make test          Run unit tests' \
		'  make validate      Run unit tests and compile checks'

llama-up:
	LLAMA_CTX_SIZE=$(LLAMA_CTX_SIZE) $(COMPOSE) up -d $(LLAMA_SERVICE)

llama-down:
	$(COMPOSE) stop $(LLAMA_SERVICE)

model-check:
	$(PYTHON) scripts/model_check.py \
		--compose "$(COMPOSE)" \
		--service "$(LLAMA_SERVICE)" \
		--model "$(LLAMA_MODEL)" \
		--base-url "$(LLAMA_URL)" \
		--ctx-size "$(LLAMA_CTX_SIZE)" \
		--lowctx-size "$(LLAMA_LOWCTX_SIZE)"

mcp-up:
	@mkdir -p "$(RUN_DIR)"
	@if [ -f "$(MCP_PID)" ] && kill -0 "$$(cat "$(MCP_PID)")" 2>/dev/null; then \
		printf '%s\n' "MCP server already running (pid $$(cat "$(MCP_PID)"))"; \
	elif $(PYTHON) scripts/wait_for_mcp.py --timeout 0.1 >/dev/null 2>&1; then \
		printf '%s\n' "MCP server already listening"; \
	else \
		printf '%s\n' "Starting MCP server; logs: $(MCP_LOG)"; \
		nohup $(PYTHON) -m coding_agent.mcp_server >"$(MCP_LOG)" 2>&1 & echo $$! >"$(MCP_PID)"; \
	fi
	@$(PYTHON) scripts/wait_for_mcp.py

mcp-down:
	@if [ -f "$(MCP_PID)" ] && kill -0 "$$(cat "$(MCP_PID)")" 2>/dev/null; then \
		printf '%s\n' "Stopping MCP server (pid $$(cat "$(MCP_PID)"))"; \
		kill "$$(cat "$(MCP_PID)")"; \
	else \
		printf '%s\n' "MCP server is not running"; \
	fi
	@rm -f "$(MCP_PID)"

mcp-restart: mcp-down mcp-up

mcp-logs:
	@mkdir -p "$(RUN_DIR)"
	@touch "$(MCP_LOG)"
	tail -f "$(MCP_LOG)"

mcp-check:
	$(PYTHON) -m coding_agent.mcp_server --list >/dev/null

dev-up: llama-up mcp-up

dev-down: mcp-down llama-down

ask: mcp-up
	$(PYTHON) -m coding_agent.agent $(AGENT_FLAGS) "$(PROMPT)"

run: model-check mcp-up
	$(PYTHON) -m coding_agent.agent $(AGENT_FLAGS) "$(PROMPT)"

smoke: run

compose-check:
	$(COMPOSE) config --quiet

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'

compile:
	$(PYTHON) -m compileall -q coding_agent functions tests main.py call_function.py scripts/model_check.py prompts.py

validate: test compile mcp-check compose-check
