"""Utility functions for optimisation components."""


def abs_expr(M: "mf.Model", x: "mf.Expr", name: str) -> "mf.Var":  # noqa: F821
    """Return non-negative *u* with u \u2265 |x| element-wise."""
    u = M.variable(name, x.shape[0])
    M.constraint(x <= u)
    M.constraint(-x <= u)
    return u
