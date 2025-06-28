# Synthetic instrument support

from __future__ import annotations

import csv
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from .core import OptBaseModel
from .instruments import InstrumentMap
from .costs import TransactionCostModel
from .config import ProblemConfig




class SyntheticInstrument(OptBaseModel):
    """Definition of a synthetic instrument.

    Parameters
    ----------
    name:
        Name of the synthetic instrument.
    weights:
        Mapping from underlying instrument name to the weight of that
        instrument in the synthetic.
    cost_model:
        Optional cost model used when trading this synthetic.
    alpha_overlay:
        Extra alpha applied on top of the implied alpha from the
        underlying instruments.
    """

    name: str
    weights: Mapping[str, float]
    cost_model: Optional[Any] = None
    alpha_overlay: float = 0.0

    def implied_exposures(self, base_map: InstrumentMap) -> NDArray[np.floating]:
        """Return synthetic exposure vector implied by *base_map*."""
        idx = {n: i for i, n in enumerate(base_map.names_decision)}
        exp = np.zeros(base_map.E.shape[0])
        for inst, w in self.weights.items():
            j = idx[inst]
            exp += w * base_map.E[:, j]
        return exp

    def implied_alpha(self, alpha_map: Mapping[str, float]) -> float:
        """Return synthetic alpha from underlying alphas."""
        a = sum(alpha_map[n] * w for n, w in self.weights.items())
        return a + self.alpha_overlay


def load_synthetics_csv(path: str) -> List[SyntheticInstrument]:
    """Load :class:`SyntheticInstrument` objects from ``CSV`` file.

    The CSV file must contain the columns ``name``, ``underlying`` and
    ``weight``.  An optional column ``alpha_overlay`` may be supplied to
    specify per-instrument overlays.  Multiple rows with the same ``name``
    are aggregated into a single synthetic instrument.
    """

    by_name: Dict[str, Dict[str, float]] = {}
    overlay: Dict[str, float] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            u = row["underlying"]
            w = float(row["weight"])
            by_name.setdefault(name, {})[u] = w
            if "alpha_overlay" in row and row["alpha_overlay"]:
                overlay[name] = float(row["alpha_overlay"])
    out = []
    for name, weights in by_name.items():
        out.append(
            SyntheticInstrument(
                name=name,
                weights=weights,
                alpha_overlay=overlay.get(name, 0.0),
            )
        )
    return out




class PerInstrumentCostModel(OptBaseModel):
    """Wrapper allowing different cost models per decision variable."""

    models: Sequence[Optional[Any]]

    def cost_soc(self, M: "mf.Model", delta_dec: "mf.Expr") -> "mf.Expr":  # noqa: F821
        import mosek.fusion as mf

        exprs = []
        for i, cm in enumerate(self.models):
            if cm is None:
                continue
            slice_i = delta_dec.pick([i])
            exprs.append(cm.cost_soc(M, slice_i))
        if not exprs:
            return 0.0
        return mf.Expr.sum(mf.Expr.vstack(exprs))


def extend_instrument_map(
    base_map: InstrumentMap,
    base_alpha: NDArray[np.floating],
    synthetics: Iterable[SyntheticInstrument],
    base_cost_model: Optional[TransactionCostModel] = None,
) -> Tuple[InstrumentMap, NDArray[np.floating], List[Optional[TransactionCostModel]]]:
    """Extend ``base_map`` with ``synthetics``.

    Returns new instrument map, new ``alpha_dec`` vector and list of cost
    models aligned with the decision variables.
    """

    n_risk, n_dec = base_map.E.shape
    E_parts = [base_map.E]
    alpha_list = list(base_alpha)
    cost_models: List[Optional[TransactionCostModel]] = []
    for _ in range(n_dec):
        cost_models.append(base_cost_model)

    alpha_map = {n: a for n, a in zip(base_map.names_decision, base_alpha)}
    names_decision = list(base_map.names_decision)

    for s in synthetics:
        E_parts.append(s.implied_exposures(base_map)[:, None])
        alpha_list.append(s.implied_alpha(alpha_map))
        cost_models.append(s.cost_model)
        names_decision.append(s.name)

    E_new = np.hstack(E_parts)
    imap = InstrumentMap(E=E_new, names_risk=base_map.names_risk, names_decision=names_decision)
    return imap, np.array(alpha_list, dtype=float), cost_models


def apply_synthetics(
    cfg: ProblemConfig,
    synthetics: Sequence[SyntheticInstrument],
) -> ProblemConfig:
    """Return new config with ``synthetics`` added to ``cfg``."""

    imap, alpha_dec, cost_models = extend_instrument_map(
        cfg.instrument_map,
        cfg.alpha_dec,
        synthetics,
        cfg.utility.cost_model,
    )

    util = cfg.utility.model_copy()
    util.cost_model = PerInstrumentCostModel(models=cost_models)
    start_dec = np.concatenate([cfg.start_dec, np.zeros(len(synthetics))])
    return cfg.model_copy(
        update={
            "instrument_map": imap,
            "alpha_dec": alpha_dec,
            "utility": util,
            "start_dec": start_dec,
        }
    )
