.PHONY: .venv run

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = apricot-benchmark

.venv:
	poetry install

run: .venv
	poetry run streamlit run apps/streamlit_app.py

kernel: .venv
	poetry run python -m ipykernel install --user --name ${PROJECT_NAME}