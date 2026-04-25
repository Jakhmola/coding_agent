SHELL := /usr/bin/env bash

COMPOSE ?= docker compose
OLLAMA_SERVICE ?= ollama
OLLAMA_MODEL ?= coding-qwen-gpu
OLLAMA_BASE_MODEL ?= qwen2.5-coder:7b-instruct-q4_0
MODELFILE ?= models/Modelfile
LOWCTX_MODELFILE ?= models/Modelfile.lowctx

.PHONY: help ollama-up ollama-down model-pull model-check compose-check test

help:
	@printf '%s\n' \
		'Available targets:' \
		'  make ollama-up     Start Ollama container detached' \
		'  make ollama-down   Stop Ollama container' \
		'  make model-pull    Pull base Qwen model and create coding-qwen-gpu' \
		'  make model-check   Verify coding-qwen-gpu is loaded 100% on GPU' \
		'  make compose-check Validate docker-compose.yml syntax' \
		'  make test          Run unit tests'

ollama-up:
	$(COMPOSE) up -d $(OLLAMA_SERVICE)

ollama-down:
	$(COMPOSE) stop $(OLLAMA_SERVICE)

model-pull: ollama-up
	$(COMPOSE) exec $(OLLAMA_SERVICE) ollama pull $(OLLAMA_BASE_MODEL)
	$(COMPOSE) exec $(OLLAMA_SERVICE) ollama create $(OLLAMA_MODEL) -f /models/Modelfile

model-check: ollama-up
	python3 scripts/model_check.py \
		--compose "$(COMPOSE)" \
		--service "$(OLLAMA_SERVICE)" \
		--model "$(OLLAMA_MODEL)" \
		--model-file "/models/Modelfile" \
		--lowctx-model-file "/models/Modelfile.lowctx"

compose-check:
	$(COMPOSE) config --quiet

test:
	python3 -m unittest discover -s tests -p 'test_*.py'

