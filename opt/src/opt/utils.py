"""Utility functions for optimisation components."""

from __future__ import annotations

from typing import Any

import numpy as np


def abs_expr(M: "mf.Model", x: "mf.Expr", name: str) -> "mf.Var":  # noqa: F821
    """Return non-negative *u* with u \u2265 |x| element-wise."""
    u = M.variable(name, x.shape[0])
    M.constraint(x <= u)
    M.constraint(-x <= u)
    return u


def value_of(obj: Any) -> Any:
    """Return numeric data from MOSEK expressions or Python objects.

    Scalars are returned as ``float`` while larger results are returned as
    :class:`numpy.ndarray`.  The helper attempts the following strategies:

    #. Call ``obj.level()`` if available (for variables and parameters).
    #. If MOSEK is present and ``obj`` is an expression, evaluate it using a
       :class:`mosek.fusion.WorkStack`.
    #. Fallback to ``numpy.asarray`` for anything else.
    """

    if hasattr(obj, "level"):
        try:
            obj = obj.level()
        except Exception:
            pass

    try:
        import mosek.fusion as mf
    except Exception:  # pragma: no cover - MOSEK not installed
        mf = None

    if mf is not None and isinstance(obj, mf.Expr):
        from mosek.fusion import WorkStack

        rs, ws, xs = WorkStack(), WorkStack(), WorkStack()
        n = int(obj.getSize())
        rs.allocf64(n)
        obj.eval(rs, ws, xs)
        vec = [rs.popf64() for _ in range(n)]
        arr = np.array(vec, dtype=float).reshape(obj.getShape())
        return arr if arr.ndim else float(arr)

    arr = np.asarray(obj, dtype=float)
    return arr if arr.ndim else float(arr)
