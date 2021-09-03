.venv:
	poetry install

run: .venv
	poetry run streamlit run apps/streamlit_app.py
