# Python virtual environment
VENV_DIR = .venv
VENV_BIN = $(VENV_DIR)/bin
PYTHON = $(VENV_BIN)/python
UV = uv

# Default target
.PHONY: help
help:
	@echo "Available commands:"
	@echo "  make dev-setup  - Create virtualenv and install all dependencies with uv"
	@echo "  make py-fmt     - Format code with ruff"
	@echo "  make py-static  - Run static type checking with mypy"
	@echo "  make test       - Run test suite with coverage"
	@echo "  make lint       - Check formatting and linting without fixing"
	@echo "  make check      - Run pre-commit hooks on all files"
	@echo "  make clean      - Remove virtual environment and cache files"

# Ensure virtual environment exists
$(VENV_DIR):
	python3 -m venv $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install uv

# Install dependencies using uv
.PHONY: dev-setup
dev-setup: $(VENV_DIR)
	$(PYTHON) -m pip install --upgrade pip
	$(UV) pip install -e ".[development,docs]"

# Format code using ruff
.PHONY: py-fmt
py-fmt: $(VENV_DIR)
	$(VENV_BIN)/ruff format .
	$(VENV_BIN)/ruff check --fix .

# Run static type checking
.PHONY: py-static
py-static: $(VENV_DIR)
	$(VENV_BIN)/mypy .

# Run tests with coverage
.PHONY: test
test: $(VENV_DIR)
	$(VENV_BIN)/pytest

# Check formatting and linting (no auto-fix)
.PHONY: lint
lint: $(VENV_DIR)
	$(VENV_BIN)/ruff format --check .
	$(VENV_BIN)/ruff check .

# Run pre-commit hooks on all files
.PHONY: check
check: $(VENV_DIR)
	$(VENV_BIN)/pre-commit run --all-files

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 
