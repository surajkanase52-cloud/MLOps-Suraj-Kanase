setup:
	python -m venv .venv
install:
	pip install -r requirements.txt
data:
	python -m src.download_data && python -c "from src.data import load_and_clean; load_and_clean()"
eda:
	python -m src.eda
train:
	python -m src.train
test:
	pytest -q
serve:
	uvicorn app.main:app --reload
