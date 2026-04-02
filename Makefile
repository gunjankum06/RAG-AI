.PHONY: help install dev test lint format run docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -r requirements.txt

dev: ## Install dev dependencies and set up environment
	pip install -r requirements.txt -r requirements-dev.txt
	cp -n .env.example .env 2>/dev/null || true
	mkdir -p storage/chroma storage/faiss data

test: ## Run tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing --cov-report=html

lint: ## Run linters
	ruff check src/ tests/
	ruff format --check src/ tests/

format: ## Auto-format code
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck: ## Run type checking
	mypy src/ --ignore-missing-imports

run: ## Start the API server (development)
	uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

run-ui: ## Start the Streamlit UI
	streamlit run src/ui/app.py

docker-up: ## Start all services with Docker Compose
	docker compose up --build -d

docker-down: ## Stop all Docker Compose services
	docker compose down

docker-logs: ## Tail logs from all services
	docker compose logs -f

clean: ## Remove caches, build artifacts, and storage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	rm -rf storage/chroma/* storage/faiss/*
