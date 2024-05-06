.PHONY: quality style test

check_dirs := data tests scripts

quality:
	ruff check $(check_dirs)  # linter
	ruff format --check $(check_dirs) # formatter

style:
	ruff check --fix $(check_dirs) # linter
	ruff format $(check_dirs) # formatter
