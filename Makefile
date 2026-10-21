.PHONY: install lint test data index eval serve docker

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check src tests
	ruff format --check src tests
	mypy

test:
	pytest -m "not network and not models"

data:
	cbrag ingest

index:
	cbrag index

eval:
	cbrag eval retrieval

serve:
	cbrag serve

docker:
	docker compose up --build -d
