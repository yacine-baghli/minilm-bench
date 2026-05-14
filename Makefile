.PHONY: setup train eval viz test lint clean

# ── Setup ──────────────────────────────────────────────
setup:
	pip install -r requirements.txt

# ── Data ───────────────────────────────────────────────
data:
	python -m data.download --output_dir ./data/tokenized --max_shards 10

# ── Training ───────────────────────────────────────────
train:
	python scripts/train.py --config configs/$(CONFIG)

train-mha:
	python scripts/train.py --config configs/mha.yaml

train-gqa:
	python scripts/train.py --config configs/gqa.yaml

train-mqa:
	python scripts/train.py --config configs/mqa.yaml

train-swa:
	python scripts/train.py --config configs/swa.yaml

# Advanced sparse variants
train-diff:
	python scripts/train.py --config configs/diff.yaml

train-mla:
	python scripts/train.py --config configs/mla.yaml

train-moh:
	python scripts/train.py --config configs/moh.yaml

train-nsa:
	python scripts/train.py --config configs/nsa.yaml

# ── Evaluation ─────────────────────────────────────────
eval:
	python scripts/eval.py --config configs/$(CONFIG)

# ── Visualization ──────────────────────────────────────
viz:
	streamlit run viz/app.py

# ── Testing ────────────────────────────────────────────
test:
	python -m pytest tests/ -v --tb=short

test-cov:
	python -m pytest tests/ -v --cov=model --cov=training --cov-report=term-missing

# ── Code Quality ───────────────────────────────────────
lint:
	ruff check . --fix
	ruff format .

# ── Cleanup ────────────────────────────────────────────
clean:
	rm -rf checkpoints/ logs/ __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
