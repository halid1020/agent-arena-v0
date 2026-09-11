# actoris_harena -- task runner for the shared rig pipeline.
#
# The simulated arenas are NOT covered here: their tests are imperative
# __main__ scripts under test/, run individually, and they need the [sim]
# extra's dependency stack. These targets cover the rig pipeline, which is
# what the robot repos import.

PY ?= python3

.PHONY: test test-unit lint

test: test-unit

test-unit:
	$(PY) -m unittest discover -s test/rig -t .

# black/isort/flake8/mypy, scoped to the rig pipeline -- see the comment at
# the head of .pre-commit-config.yaml for why the sim tree is excluded.
lint:
	pre-commit run --all-files
