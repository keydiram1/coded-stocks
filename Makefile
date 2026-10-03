.PHONY: install test demo lint

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

demo:
	python -m quant_edge_lab init-sample-data
	python -m quant_edge_lab validate-hypothesis hypotheses/gap_rvol_continuation_v1.yaml
	python -m quant_edge_lab run hypotheses/gap_rvol_continuation_v1.yaml
	python -m quant_edge_lab list-experiments
