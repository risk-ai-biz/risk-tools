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

Use :func:`opt.portfolio_core.apply_synthetics` to extend an existing
``ProblemConfig`` with a collection of synthetic instruments.  Individual
synthetic cost models are combined using
:class:`opt.synthetic.PerInstrumentCostModel` and automatically placed in the
resulting config.

Raw instrument definitions can be loaded from CSV using
:func:`opt.synthetic.load_synthetics_csv`.
