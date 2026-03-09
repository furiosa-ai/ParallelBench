.PHONY: install test lint help

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (requires Python 3.12+)
	uv sync

test: ## Run tests
	pytest tests/ -v

lint: ## Run linters via pre-commit hooks
	pre-commit run --all-files
