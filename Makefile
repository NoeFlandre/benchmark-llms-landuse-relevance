UV ?= uv
SHELL := bash
.SHELLFLAGS := -eo pipefail -c
RUN := $(UV) run --no-sync

.PHONY: help install baseline lint format types test property acceptance architecture integration crap mutation smoke security lockfile check docs docs-build docker

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

test:  ## Unit and property tests with coverage (fails under the pyproject floor)
	$(RUN) pytest tests/unit tests/property --cov --cov-report=term-missing --cov-report=json

property:  ## Property-based tests
	$(RUN) pytest tests/property

acceptance:  ## Executable Gherkin scenarios
	$(RUN) pytest tests/acceptance

architecture:  ## Dependency boundaries
	$(RUN) pytest tests/architecture

integration:  ## Real model runtime, downloads a tiny model
	$(RUN) pytest tests/integration -m integration

crap: test  ## CRAP score guardrail over the domain (reads the coverage from `test`)
	$(RUN) python scripts/crap.py --max 6

mutation:  ## Mutation testing over the domain, gated on the reviewed survivor allowance
	$(RUN) mutmut run --max-children 4
	$(RUN) python scripts/check_mutants.py --max-survivors 4

smoke:  ## CLI smoke test: the entry point starts and lists the roster
	$(RUN) lrb --version
	$(RUN) lrb models

lockfile:  ## uv.lock matches pyproject.toml and the speculative extra resolves
	UV_FROZEN=0 $(UV) lock --check
	UV_FROZEN=0 $(UV) sync --locked --extra speculative --dry-run

# Known advisory with no fixed release, pulled in by sglang (speculative extra only).
AUDIT_IGNORES := --ignore-vuln PYSEC-2026-2447

security:  ## Audit the locked dependencies for known vulnerabilities
	$(UV) export --frozen --no-hashes --no-emit-project --extra inference --extra publish \
		| $(UV) tool run pip-audit --no-deps --disable-pip -r /dev/stdin
	$(UV) export --frozen --no-hashes --no-emit-project --extra speculative \
		| $(UV) tool run pip-audit --no-deps --disable-pip $(AUDIT_IGNORES) -r /dev/stdin

# Exactly the gates CI runs, with the same flags; CI calls these targets.
check: lint types test acceptance architecture crap mutation smoke docs-build lockfile security  ## The deterministic gauntlet

docs:  ## Serve the documentation locally
	$(UV) run --only-group docs mkdocs serve

docs-build:  ## Build the documentation site strictly
	$(UV) run --only-group docs mkdocs build --strict

docker:  ## Build the runtime image
	docker build -t landuse-relevance-bench .
