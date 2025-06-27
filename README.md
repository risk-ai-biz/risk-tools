# risk-tools

This repository contains a small portfolio optimisation toolkit used in
examples and tests.  The main functionality lives in
`opt/src/opt/portfolio_core.py`.

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

To simplify integration the :class:`~opt.portfolio_core.ProblemConfig` class
accepts a ``synthetic_instruments`` field.  During validation the base
``InstrumentMap`` and alphas are automatically extended using
:func:`opt.synthetic.extend_instrument_map`.  Any cost models defined for the
synthetics are collected into a :class:`PerInstrumentCostModel` and assigned to
``ProblemConfig.utility.cost_model``.

A simple CSV loader (:func:`load_synthetics_csv`) is also provided for
convenience.
