install:
	pip3 install -e .[dev]

lint:
	flake8 few_state sos tests

test:
	pytest tests