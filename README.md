# risk-tools

This repository contains a small portfolio optimisation toolkit used in
examples and tests.  Functionality is split across several modules
under `opt/src/opt/`.

## Synthetic instruments

The latest release adds support for *synthetic instruments* such as index
futures or ETFs.  Synthetic instruments are represented via their underlying
compositions which allows risk to be modelled on the constituents while the
synthetic itself remains a separate tradeable instrument.

To define a synthetic instrument supply a mapping of underlying instrument
weights and (optionally) a cost model or alpha overlay:

```python
from opt.synthetic import SyntheticInstrument

syn = SyntheticInstrument(
    name="ES1",
    weights={"SPY": 1.0},
    alpha_overlay=0.001,
)
```

Use :func:`opt.synthetic.apply_synthetics` to extend an existing
``ProblemConfig`` with a collection of synthetic instruments.  Individual
synthetic cost models are combined using
:class:`opt.synthetic.PerInstrumentCostModel` and automatically placed in the
resulting config.

Raw instrument definitions can be loaded from CSV using
:func:`opt.synthetic.load_synthetics_csv`.

## Problem setup

Optimisation problems are described by :class:`opt.config.ProblemConfig`.  The
main fields are:

* ``instrument_map`` (often called ``E``) — matrix mapping decision weights to
  risk exposures.
* ``alpha_dec`` — expected returns for each decision variable.
* ``alpha_exp`` — optional expected returns per risk exposure.
* ``start_dec`` — starting portfolio weights used for turnover and cost
  calculations.
* ``risk_model`` — provides systematic and specific risk measures.
* ``utility`` — risk aversion and transaction cost settings.
* ``constraints`` — list of additional :class:`opt.constraints.ConstraintSpec`
  instances.

Within the optimiser the decision weights are referred to as ``w_dec`` while the
corresponding risk exposures are ``w_exp``.
