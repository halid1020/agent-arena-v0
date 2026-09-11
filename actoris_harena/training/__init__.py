"""Launching a training run: where it may go, and whether it can work there.

The two halves are deliberately separate. ``destinations`` knows about machines
-- the SSH alias, the scratch layout, what that GPU has been measured to carry
-- and ``matrix`` knows about runs, as the same seven-plus-one columns
``hpc/runs.tsv`` has always had. Neither imports aiohttp or opens a connection,
so both are unit-tested directly; ``tool/train_launch.py`` and the console's
Training tab are two front ends over the same rules.
"""
