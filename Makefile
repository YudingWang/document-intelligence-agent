.PHONY: install test api web pdf

install:
	python3 -m venv .venv
	.venv/bin/pip install -r backend/requirements.txt
	cd frontend && npm install

test:
	.venv/bin/pytest -q

api:
	cd backend && PYTHONPATH=. ../.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

web:
	cd frontend && npm run dev

pdf:
	.venv/bin/python scripts/generate_sample_pdf.py
