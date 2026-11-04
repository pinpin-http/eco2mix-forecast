# Calcul limité à 2 threads pour cohabiter avec d'autres tâches sur la machine.
export OMP_NUM_THREADS ?= 2
export ECO2MIX_N_JOBS ?= 2
export MPLBACKEND = Agg

RUN = uv run python -m eco2mix_forecast

.PHONY: install data backtest figures test all clean

install:
	uv sync --frozen

data:
	$(RUN) data

backtest:
	$(RUN) backtest

figures:
	$(RUN) figures

test:
	uv run pytest

all: data backtest figures

clean:
	rm -rf data/processed
