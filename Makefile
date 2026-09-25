UV ?= uv
RUN := $(UV) run --no-sync

.PHONY: help install baseline lint format types test property acceptance architecture integration crap mutation smoke check docs docs-build docker

help:
	@grep -E '^[a-z-]+:.*?##' $(MAKEFILE_LIST) | sed 's/:.*##/\t/'

install:  ## Sync the pinned environment
	$(UV) sync --extra inference --extra publish

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

test:  ## Unit tests with coverage
	$(RUN) pytest tests/unit --cov --cov-report=term-missing --cov-report=json

property:  ## Property-based tests
	$(RUN) pytest tests/property

acceptance:  ## Executable Gherkin scenarios
	$(RUN) pytest tests/acceptance

architecture:  ## Dependency boundaries
	$(RUN) pytest tests/architecture

integration:  ## Real model runtime, downloads a tiny model
	$(RUN) pytest tests/integration -m integration

crap:  ## CRAP score guardrail over the domain
	$(RUN) pytest tests/unit tests/property --cov --cov-report=json -q
	$(RUN) python scripts/crap.py --max 6

mutation:  ## Mutation testing over the domain
	$(RUN) mutmut run || true
	$(RUN) mutmut results

smoke:  ## End-to-end CLI smoke test against a stub-free tiny model
	$(RUN) lrb models
	$(RUN) lrb report --results-dir results || true

check: lint types test property acceptance architecture crap  ## The deterministic gauntlet

docs:  ## Serve the documentation locally
	$(UV) run --with mkdocs-material mkdocs serve

docs-build:
	$(UV) run --with mkdocs-material mkdocs build --strict

docker:  ## Build the runtime image
	docker build -t landuse-relevance-bench .
