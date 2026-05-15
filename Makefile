.PHONY: install venv run run-server cli lint test clean help

VENV ?= .venv
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

venv:
	python3.11 -m venv $(VENV)

install: venv
	$(PIP) install -e .

run:
	$(PY) -m llm_relay.cli run --reload --port 8090

run-server:
	$(PY) -m llm_relay.cli run --port 8090

cli:
	$(PY) -m llm_relay.cli $(cmd)

lint:
	$(PY) -m py_compile llm_relay/**/*.py llm_relay/*.py

test:
	$(PY) -m pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info

help:
	@echo "llm-relay make targets:"
	@echo "  install     - Create .venv and pip install -e ."
	@echo "  run         - Start server with --reload on :8090"
	@echo "  run-server  - Start server (no reload) on :8090"
	@echo "  cli cmd='resolve fast'  - Run a CLI subcommand"
	@echo "  lint        - py_compile sanity check"
	@echo "  test        - Run pytest"
	@echo "  clean       - Remove build artifacts"
