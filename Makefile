.PHONY: tests

tests:
	PYTHONPATH=. pytest tests

train:
	PYTHONPATH=. accelerate launch src/train.py