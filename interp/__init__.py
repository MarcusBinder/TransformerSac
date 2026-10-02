"""Mechanistic-interpretability pipeline for derating agents.

Capture -> probe -> steer, on the trained power_max_derate (delmax) and
limit-conditioned (randlim) Transformer-SAC checkpoints:

- common.py               checkpoint load, delmax env rebuild, actor rebuild,
                          batch-prep closure (the shared plumbing)
- hooks.py                ActivationCapture + SteeringHook on the residual
                          stream of the 2-layer TransformerEncoder
- capture_activations.py  Stage 1: closed-loop rollouts -> per-condition npz
                          of activations + actor inputs + behavior labels
- analyze_directions.py   Stage 2: PCA / linear probes / difference-of-means
                          "conservativeness" direction per site
- steer_eval.py           Stage 3: closed-loop alpha-sweep under SteeringHook,
                          power/DEL trade-off vs. the natural limit knob

Everything here is run from TransformerSac/ (or anywhere -- the bootstrap
below pins sys.path), e.g.:

    ../.venv/bin/python interp/capture_activations.py --checkpoint runs/...

Imports are the repo's usual bare style (``from networks import ...``,
``from del_surrogate import ...``): TransformerSac/ and the repo root are
inserted on sys.path when this package (or any script in it) loads.
"""

import os
import sys

_INTERP_DIR = os.path.dirname(os.path.abspath(__file__))
TSAC_DIR = os.path.dirname(_INTERP_DIR)
REPO_ROOT = os.path.dirname(TSAC_DIR)
for _p in (REPO_ROOT, TSAC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
