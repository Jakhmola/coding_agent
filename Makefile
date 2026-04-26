SHELL := /usr/bin/env bash

COMPOSE ?= docker compose
LLAMA_SERVICE ?= llama-cpp
LLAMA_URL ?= http://localhost:8080
LLAMA_MODEL ?= opus-ghost-codex-4b-q5
LLAMA_CTX_SIZE ?= 4096
LLAMA_LOWCTX_SIZE ?= 2048

.PHONY: help llama-up llama-down model-check compose-check test

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make llama-up      Start llama.cpp server detached' \
		'  make llama-down    Stop llama.cpp server' \
		'  make model-check   Verify llama.cpp model server and OpenAI-compatible API' \
		'  make compose-check Validate docker-compose.yml syntax' \
		'  make test          Run unit tests'

llama-up:
	LLAMA_CTX_SIZE=$(LLAMA_CTX_SIZE) $(COMPOSE) up -d $(LLAMA_SERVICE)

llama-down:
	$(COMPOSE) stop $(LLAMA_SERVICE)

model-check:
	python3 scripts/model_check.py \
		--compose "$(COMPOSE)" \
		--service "$(LLAMA_SERVICE)" \
		--model "$(LLAMA_MODEL)" \
		--base-url "$(LLAMA_URL)" \
		--ctx-size "$(LLAMA_CTX_SIZE)" \
		--lowctx-size "$(LLAMA_LOWCTX_SIZE)"

compose-check:
	$(COMPOSE) config --quiet

test:
	python3 -m unittest discover -s tests -p 'test_*.py'
