.PHONY: format lint test

format:
	black app tests
	ruff check app tests --fix

lint:
	ruff check app tests
	black --check app tests

test:
	pytest -q
