.PHONY: all

all: lineage_sims.html

lineage_sims.html: sims.csv scripts/visualizer.py
	uv run scripts/visualizer.py

sims.csv: scripts/lhs.py
	uv run python scripts/lhs.py
