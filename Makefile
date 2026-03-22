.PHONY: install install-java test lint help

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies (Python + Java)
	uv sync
	uv run scripts/install_java.sh

install-java: ## Install JDK 17 without sudo (for grammar-based evaluation)
	uv run scripts/install_java.sh

test: ## Run tests
	pytest tests/ -v

lint: ## Run linters via pre-commit hooks
	pre-commit run --all-files
