local_start_dev:
	poetry run uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload