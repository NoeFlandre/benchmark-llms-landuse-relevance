UV ?= uv
RUN := $(UV) run --no-sync

.PHONY: help install baseline lint format types test property acceptance architecture \
	integration scripts crap mutation smoke check docs docs-build docker

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## Sync the pinned environment (gliner2 and gguf are node-only runtimes)
	$(UV) sync --extra inference --extra scoring --extra publish

baseline:  ## Run the existing suite before touching anything
	$(RUN) pytest

lint:  ## ruff format check + lint
	$(RUN) ruff format --check .
	$(RUN) ruff check .

format:  ## Apply ruff formatting and fixes
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

types:  ## Static type check
	$(RUN) ty check

test:  ## Unit and property tests with coverage (feeds the CRAP gate)
	$(RUN) pytest tests/unit tests/property --cov --cov-report=term-missing --cov-report=json

property:  ## Property-based tests alone
	$(RUN) pytest tests/property

acceptance:  ## Executable Gherkin scenarios
	$(RUN) pytest tests/acceptance

architecture:  ## Dependency boundaries
	$(RUN) pytest tests/architecture

scripts:  ## Grid'5000 shell script tests
	$(RUN) pytest tests/scripts

integration:  ## Real model runtime, downloads a tiny model
	$(RUN) pytest tests/integration -m integration

crap: test  ## CRAP score guardrail over the domain, from the coverage `test` wrote
	$(RUN) python scripts/crap.py --max 6

mutation:  ## Mutation testing over the domain, gated like CI
	$(RUN) mutmut run --max-children 4 || true
	$(RUN) python scripts/check_mutants.py --max-survivors 4

smoke:  ## End-to-end CLI smoke test against a stub-free tiny model
	$(RUN) lrb models
	$(RUN) lrb report --results-dir results || true

check: lint types crap acceptance architecture scripts  ## The deterministic gauntlet

docs:  ## Serve the documentation locally
	$(UV) run --with mkdocs-material mkdocs serve

docs-build:  ## Build the documentation strictly
	$(UV) run --with mkdocs-material mkdocs build --strict

docker:  ## Build the runtime image
	docker build -t landuse-relevance-bench .
