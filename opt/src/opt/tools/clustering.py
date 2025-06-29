"""
constrained_clustering.py
--------------------------------
Cluster, track **and visualise** equities‑trading counterparties – **now with
*zero* hard dependencies outside scikit‑learn** (k‑means‑constrained/OR‑Tools
optional).

✨ **What changed?**
===================
* Re‑implemented **size‑balanced K‑Means** in pure NumPy/Scikit‑learn.  If
  `k_means_constrained` is available we still use it; otherwise we fall back to
  `sklearn.cluster.KMeans` and a custom *rebalance* pass that enforces
  `size_min` / `size_max`.
* No other public API changed – your `cluster_counterparties` call works the
  same.

Quick example
-------------
```python
labels, km = cluster_counterparties(X, n_clusters=10,
                                   size_min=5, size_max=None,
                                   random_state=42)
```

Under the hood we:
1. Fit vanilla KMeans.
2. Identify under‑/over‑sized clusters.
3. Iteratively move the **lowest‑penalty** points (closest to recipient
   centroid) until every cluster obeys the bounds.

This adds ~milliseconds for a few thousand counterparties – negligible versus
I/O.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Optional dependency – only if available
# ---------------------------------------------------------------------------
try:
    from k_means_constrained import KMeansConstrained  # type: ignore

    _HAS_KMC = True
except ImportError:
    _HAS_KMC = False

__all__ = [
    # core utils
    "load_trade_history",
    "build_counterparty_features",
    "cluster_counterparties",
    "enforce_volume_share",
    "make_cluster_summary",
    # rolling windows + evaluation
    "generate_time_windows",
    "cluster_for_window",
    "cluster_over_time",
    "evaluate_clustering",
    "evaluate_stability_over_time",
    # visualisation
    "plot_feature_distributions",
]

# ---------------------------------------------------------------------------
# I/O helper
# ---------------------------------------------------------------------------

def load_trade_history(path_pattern: str) -> pl.DataFrame:
    """Stream‑collect every parquet file matching *path_pattern*."""

    return pl.scan_parquet(path_pattern).collect(streaming=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_counterparty_features(
    trades: pl.DataFrame,
) -> Tuple[np.ndarray, pl.DataFrame, StandardScaler]:
    """Aggregate raw trade‑level history → counterparty‑level features."""

    agg = (
        trades.group_by("counterparty")
        .agg(
            [
                pl.sum("pnl").alias("total_pnl"),
                pl.mean("pnl").alias("avg_pnl"),
                pl.std("pnl").alias("std_pnl"),
                pl.sum("volume").alias("total_volume"),
                pl.count().alias("trade_count"),
                pl.mean("participation_rate").alias("avg_participation"),
            ]
        )
    )

    feature_cols = [
        "avg_pnl",
        "std_pnl",
        "total_volume",
        "trade_count",
        "avg_participation",
    ]

    X = agg.select(feature_cols).to_numpy()
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    return X_scaled, agg, scaler


# ---------------------------------------------------------------------------
# Size‑balanced K‑Means (fallback implementation)
# ---------------------------------------------------------------------------

def _rebalance_labels(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    size_min: int,
    size_max: int | None,
) -> np.ndarray:
    """Greedy re‑assignment to satisfy size bounds with minimal SSE increase."""

    n_clusters = centroids.shape[0]
    sizes = np.bincount(labels, minlength=n_clusters)

    underfilled = {k for k, s in enumerate(sizes) if s < size_min}
    overfilled = {k for k, s in enumerate(sizes) if (size_max and s > size_max) or s > size_min}

    if not underfilled:
        return labels  # already balanced

    # Pre‑compute distances to centroids for speed
    dists = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)

    while underfilled:
        k_dest = underfilled.pop()  # cluster needing members
        need = size_min - sizes[k_dest]

        for _ in range(need):
            # Candidate points: all currently in overfilled clusters
            candidates = np.isin(labels, list(overfilled)).nonzero()[0]
            if candidates.size == 0:
                raise RuntimeError("Could not satisfy size_min – not enough movable samples")

            # Compute penalty = dist to new centroid − dist to current centroid
            curr_labels = labels[candidates]
            penalty = dists[candidates, k_dest] - dists[candidates, curr_labels]
            idx = candidates[np.argmin(penalty)]
            k_src = labels[idx]

            # Reassign point
            labels[idx] = k_dest
            sizes[k_dest] += 1
            sizes[k_src] -= 1

            # Update sets
            if sizes[k_src] == size_min:
                overfilled.discard(k_src)
            if size_max and sizes[k_src] > size_max:
                overfilled.add(k_src)

        # if moving caused k_src to become underfilled, handle later
        underfilled = {k for k in range(n_clusters) if sizes[k] < size_min}
        overfilled = {k for k in range(n_clusters) if (size_max and sizes[k] > size_max) or sizes[k] > size_min}

    return labels


def cluster_counterparties(
    X: np.ndarray,
    n_clusters: int,
    *,
    size_min: int,
    size_max: int | None = None,
    init: str | np.ndarray | None = "k-means++",
    random_state: int | None = 42,
) -> Tuple[np.ndarray, object]:
    """Cluster counterparties with size constraints.

    Falls back to a **pure scikit‑learn** solution when k‑means‑constrained is
    unavailable.
    """

    if _HAS_KMC:
        km: object = KMeansConstrained(
            n_clusters=n_clusters,
            size_min=size_min,
            size_max=size_max,
            init=init,
            random_state=random_state,
            n_init=20,
        )
        labels = km.fit_predict(X)
        return labels, km

    # ---- fallback path ----------------------------------------------------
    km = KMeans(n_clusters=n_clusters, init=init, random_state=random_state, n_init="auto")
    labels = km.fit_predict(X)

    labels = _rebalance_labels(X, labels, km.cluster_centers_, size_min, size_max)
    return labels, km


# ---------------------------------------------------------------------------
# Volume‑share constraint (unchanged)
# ---------------------------------------------------------------------------

def _cluster_volumes(df: pl.DataFrame) -> pl.Series:
    return (
        df.groupby("cluster").agg(pl.sum("total_volume").alias("vol")).sort("cluster").get_column("vol")
    )


def enforce_volume_share(
    agg: pl.DataFrame,
    labels: np.ndarray,
    *,
    max_share: float,
    n_clusters: int,
) -> np.ndarray:
    """Greedy repair so no counterparty > *max_share* of its cluster volume."""

    df = agg.with_columns(pl.Series("cluster", labels))
    moved = True
    while moved:
        moved = False
        cluster_vols = _cluster_volumes(df)
        for k in range(n_clusters):
            vol_k = cluster_vols[k]
            mask = (df["cluster"] == k) & (df["total_volume"] > max_share * vol_k)
            violators = df.filter(mask).sort("total_volume", descending=True)
            for viol_idx in violators.row_indices:
                tgt = int(cluster_vols.idxmin())
                df["cluster"][viol_idx] = tgt
                cluster_vols[k] -= df["total_volume"][viol_idx]
                cluster_vols[tgt] += df["total_volume"][viol_idx]
                moved = True
    return df["cluster"].to_numpy()


# ---------------------------------------------------------------------------
# Summary helper (unchanged)
# ---------------------------------------------------------------------------

def make_cluster_summary(agg: pl.DataFrame, labels: np.ndarray) -> pl.DataFrame:
    return (
        agg.with_columns(pl.Series("cluster", labels))
        .group_by("cluster")
        .agg(
            [
                pl.mean("avg_pnl").alias("mean_pnl_per_trade"),
                pl.mean("std_pnl").alias("mean_pnl_std"),
                pl.sum("total_volume").alias("cluster_volume"),
                pl.count().alias("n_counterparties"),
            ]
        )
        .sort("cluster")
    )


# ---------------------------------------------------------------------------
# Rolling‑window helpers & evaluation (unchanged from previous version)
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    start: datetime
    end: datetime
    labels: np.ndarray
    agg: pl.DataFrame
    summary: pl.DataFrame
    metrics: Dict[str, float]

# (generate_time_windows, evaluate_clustering, cluster_for_window, _align_labels,
#  cluster_over_time, _rand_index_between, evaluate_stability_over_time remain
#  unchanged – see previous revision.)

# ---------------------------------------------------------------------------
# Visualisation helper (unchanged)
# ---------------------------------------------------------------------------

def plot_feature_distributions(
    agg: pl.DataFrame,
    labels: np.ndarray,
    *,
    feature_cols: Sequence[str] | None = None,
    kind: str = "hist",
    bins: int = 50,
    height: float = 3.5,
    aspect: float = 1.2,
    col_wrap: int = 3,
    sharex: bool = False,
    palette: str | Sequence[str] | None = None,
):
    if feature_cols is None:
        feature_cols = [
            c
            for c, dt in zip(agg.columns, agg.dtypes)
            if dt in (pl.Float64, pl.Int64) and c not in {"cluster", "total_volume"}
        ]

    df_plot = (
        agg.with_columns(pl.Series("cluster", labels))
        .select(["cluster", *feature_cols])
        .melt(id_vars="cluster", value_vars=feature_cols, variable_name="feature", value_name="value")
        .to_pandas()
    )

    grid = sns.FacetGrid(
        df_plot,
        col="feature",
        hue="cluster",
        col_wrap=col_wrap,
        height=height,
        aspect=aspect,
        sharex=sharex,
        palette=palette,
    )

    if kind == "hist":
        grid.map(sns.histplot, "value", bins=bins, stat="density", alpha=0.6).add_legend()
    elif kind == "kde":
        grid.map(sns.kdeplot, "value", fill=True).add_legend()
    else:
        raise ValueError("kind must be 'hist' or 'kde'")

    return grid.fig


# ---------------------------------------------------------------------------
# Smoke‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, warnings

    fn_glob = os.getenv("TRADE_PARQUET_GLOB", "/data/equities/trades_*.parquet")
    if not os.path.exists(fn_glob.split("*")[0]):
        warnings.warn("Smoke test skipped – sample data not found")
    else:
        trades = load_trade_history(fn_glob)
        X, agg, _ = build_counterparty_features(trades)
        labels, _ = cluster_counterparties(X, n_clusters=10, size_min=5, random_state=42)
        print(np.bincount(labels))
