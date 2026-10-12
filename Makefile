# Reproduction pipeline.  Every step is resumable: results already in the
# CSVs are not recomputed.  TabPFN-3, Orion and LimiX run from their own environments
# (see environments/), so a full sweep is one call per environment:
#
#   make sweep                                  # base models
#   make sweep PY=.venv-tabpfn3/bin/python MODELS=tabpfn3
#   make sweep PY=.venv-orion/bin/python   MODELS=orion_msp,orion_bix
#   make sweep PY=.venv-limix/bin/python   MODELS=limix_16m_pf
#   make select

PY     ?= python
DEVICE ?= cuda
MODELS ?= tabpfn,tabicl,tabdpt,tabfm
WIDTHS ?= 3,4,half,max-1,max
# TabICL orders its ensemble members from a Python set, so the batch the
# translator is fitted on depends on hash randomisation; 0 is the value used for
# all released results.
export PYTHONHASHSEED ?= 0

.PHONY: smoke sweep select test

## One small end-to-end run, to check the install (~2 min on a GPU).
smoke:
	$(PY) scripts/sweeps/run_skip_eval.py --models tabicl --datasets diabetes --seeds 0 --skip_widths 3 --translator ridge,identity --device $(DEVICE)
	$(PY) scripts/sweeps/compute_window_kl.py

## Main classification sweep (eight clinical datasets + the 163 suite datasets) and the regression sweep.
sweep:
	$(PY) scripts/sweeps/run_skip_eval.py --models $(MODELS) --skip_widths $(WIDTHS) --translator ridge,identity --device $(DEVICE)
	$(PY) scripts/sweeps/run_skip_eval.py --models $(MODELS) --skip_widths $(WIDTHS) --translator ridge,identity --suites all --device $(DEVICE)
	$(PY) scripts/sweeps/run_regression_skip_eval.py --models $(MODELS) --skip_widths $(WIDTHS) --device $(DEVICE)

## Label-free window selection (calibration KL), needed by every table and figure.
select:
	$(PY) scripts/sweeps/compute_window_kl.py

test:
	$(PY) -m pytest tests -q
