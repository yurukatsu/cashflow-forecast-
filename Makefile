format:
	uv run isort . && uv run ruff format

check:
	uv run ruff check --fix

test:
	uv run run_task.py --experiment-config-path configs/experiments/lgbm.yml --force-rerun

mlflow:
	uv run mlflow ui --port 15001 --host 0.0.0.0
