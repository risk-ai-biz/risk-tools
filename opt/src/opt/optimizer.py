import numpy as np

from .config import ProblemConfig
from .result import PortfolioResult

try:
    import mosek.fusion as mf
    import mosek.fusion.pythonic  # noqa: F401 – enable operator overloads
except ImportError:  # pragma: no cover – unit-test builds without MOSEK
    mf = None  # type: ignore


class ConvexFactorOptimizer:
    """Conic optimiser with synthetic-instrument mapping."""

    def solve(self, cfg: ProblemConfig) -> PortfolioResult:  # noqa: C901
        if mf is None:
            raise ImportError("MOSEK not available in this environment.")

        E = cfg.instrument_map.E
        n_dec = E.shape[1]

        with mf.Model("synthetic_portfolio") as M:
            # Decision vars --------------------------------------------------
            w_dec = M.variable("w_dec", n_dec)
            w_exp = E @ w_dec  # computed expression (risk exposures)

            # Risk cones -----------------------------------------------------
            rm = cfg.risk_model
            B, F = rm.data.loadings, rm.data.factor_cov
            A_sys = B @ np.linalg.cholesky(F)  # (n_risk, n_factors)
            D_sqrt = np.sqrt(rm.data.specific_var)
            A_spec = np.diag(D_sqrt)

            t_sys = M.variable("t_sys")
            M.constraint(mf.vstack(0.5, t_sys, A_sys @ w_exp) == mf.Domain.inRotatedQCone())
            t_spec = M.variable("t_spec")
            M.constraint(mf.vstack(0.5, t_spec, A_spec @ w_exp) == mf.Domain.inRotatedQCone())

            # Transaction cost ---------------------------------------------
            if cfg.utility.cost_model is not None:
                delta_dec = w_dec - cfg.start_dec
                cost_expr = cfg.utility.cost_model.cost_soc(M, delta_dec)
            else:
                cost_expr = 0.0

            # Plug-in constraints ------------------------------------------
            for c in cfg.constraints:
                c.apply(M, w_dec, w_exp)

            # Objective -----------------------------------------------------
            reward = cfg.alpha_dec @ w_dec + cfg.alpha_exp @ w_exp  # type: ignore[arg-type]
            penalty = (
                cfg.utility.risk_aversion_sys * t_sys
                + cfg.utility.risk_aversion_spec * t_spec
                + cost_expr
            )
            M.maximize(reward - penalty)
            M.setLogHandler(None)
            M.solve()

            return PortfolioResult(
                decision_weights=w_dec.level(),
                exposure_weights=(E @ w_dec).level(),
                obj_value=M.primalObjValue(),
                risk_sys=t_sys.level()[0],
                risk_spec=t_spec.level()[0],
                cost=float(cost_expr.level()[0]) if cfg.utility.cost_model else 0.0,
            )


def optimize_portfolio(cfg: ProblemConfig) -> PortfolioResult:
    """Solve *cfg* with the default optimiser instance."""
    return ConvexFactorOptimizer().solve(cfg)
