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
	@echo "  make mlflow-sync - Mirror every competition results/runs.jsonl into mlflow.db"
	@echo "  make mlflow-ui   - Browse the run registry in MLflow (http://127.0.0.1:5000)"
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

# Run registry -> local MLflow view (needs the [tracking] extra)
.PHONY: mlflow-sync
mlflow-sync: $(VENV_DIR)
	$(PYTHON) -m data_science_stuff.kaggle mlflow-sync playground-series-*/results playground_series_*/results

.PHONY: mlflow-ui
mlflow-ui: mlflow-sync
	$(VENV_BIN)/mlflow ui --backend-store-uri sqlite:///mlflow.db

# Clean up
.PHONY: clean
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete 
