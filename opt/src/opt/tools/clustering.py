"""
constrained_clustering.py
--------------------------------
Clustering and *tracking* equities‑trading counterparties through rolling,
over‑lapping time windows.

Key capabilities
================
1.  **Constraint‑aware clustering**     – size‑balanced K‑Means + greedy repair.
2.  **Rolling windows & tracking**      – partition a multi‑year history into
    overlapping slices and keep cluster IDs consistent through time.
3.  **Evaluation metrics**              – within‑window quality (silhouette,
    Davies–Bouldin, Calinski‑Harabasz) and cross‑window stability (adjusted
    Rand, cluster‑level turnover).
4.  **Actionable reports**              – Polars summaries for downstream
    dashboards / notebooks.

You can drop this module straight into your simulator.  The public API boils
down to just two calls:

```python
windows = generate_time_windows(trades, window_days=90, step_days=30)
results = cluster_over_time(trades, windows, n_clusters=10, size_min=5,
                            max_share=0.20)
report  = evaluate_stability_over_time(results)
```

The `results` list contains one object per window with aligned labels, feature
matrices, evaluation scores and handy summaries – perfect for plugging into
Dash, Streamlit, or a scheduled e‑mail.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
)
from sklearn.preprocessing import StandardScaler

try:
    from k_means_constrained import KMeansConstrained  # size‑balanced k‑means
except ImportError as e:
    raise ImportError(
        "k_means_constrained is required.  pip install k-means-constrained"
    ) from e

__all__ = [
    # v1 functionality
    "load_trade_history",
    "build_counterparty_features",
    "cluster_counterparties",
    "enforce_volume_share",
    "make_cluster_summary",
    # new time‑series API
    "generate_time_windows",
    "cluster_for_window",
    "cluster_over_time",
    "evaluate_clustering",
    "evaluate_stability_over_time",
]

# ---------------------------------------------------------------------------
# I/O helpers – unchanged
# ---------------------------------------------------------------------------

def load_trade_history(path_pattern: str) -> pl.DataFrame:
    """Read *all* parquet files matching *path_pattern* (streaming collect)."""
    return pl.scan_parquet(path_pattern).collect(streaming=True)


# ---------------------------------------------------------------------------
# Feature engineering – unchanged from v1
# ---------------------------------------------------------------------------

def build_counterparty_features(
    trades: pl.DataFrame,
) -> Tuple[np.ndarray, pl.DataFrame, StandardScaler]:
    """Aggregate raw slice‑level history to counterparty granularity."""

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
# Constrained k‑means – unchanged
# ---------------------------------------------------------------------------

def cluster_counterparties(
    X: np.ndarray,
    n_clusters: int,
    size_min: int,
    size_max: int | None = None,
    init: str | np.ndarray | None = "k-means++",
    random_state: int | None = 42,
) -> Tuple[np.ndarray, KMeansConstrained]:
    km = KMeansConstrained(
        n_clusters=n_clusters,
        size_min=size_min,
        size_max=size_max,
        init=init,
        random_state=random_state,
        n_init=20,
    )
    labels = km.fit_predict(X)
    return labels, km


# ---------------------------------------------------------------------------
# Greedy repair – unchanged
# ---------------------------------------------------------------------------

def _cluster_volumes(df: pl.DataFrame) -> pl.Series:
    return (
        df.groupby("cluster").agg(pl.sum("total_volume").alias("vol")).sort("cluster").get_column("vol")
    )


def enforce_volume_share(
    agg: pl.DataFrame,
    labels: np.ndarray,
    max_share: float,
    n_clusters: int,
) -> np.ndarray:
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
# Reporting helpers – unchanged
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
#  NEW ▶ Rolling‑window helpers & evaluation
# ---------------------------------------------------------------------------

@dataclass
class WindowResult:
    start: datetime
    end: datetime
    labels: np.ndarray
    agg: pl.DataFrame
    summary: pl.DataFrame
    metrics: Dict[str, float]


# ---- time‑window generation ------------------------------------------------

def generate_time_windows(
    trades: pl.DataFrame,
    window_days: int = 90,
    step_days: int = 30,
) -> List[Tuple[datetime, datetime]]:
    """Return [(window_start, window_end), …] covering the full history."""
    min_ts: datetime = trades.select(pl.min("ts")).item()
    max_ts: datetime = trades.select(pl.max("ts")).item()
    windows: list[tuple[datetime, datetime]] = []
    start = min_ts
    while start <= max_ts:
        end = start + timedelta(days=window_days)
        windows.append((start, end))
        start += timedelta(days=step_days)
    return windows


# ---- evaluate clustering quality ------------------------------------------

def evaluate_clustering(X_scaled: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Return common internal metrics for a single clustering."""
    if len(set(labels)) < 2:  # silhouette breaks w/ one cluster
        return {"silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "silhouette": silhouette_score(X_scaled, labels),
        "davies_bouldin": davies_bouldin_score(X_scaled, labels),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
    }


# ---- cluster one window ----------------------------------------------------

def cluster_for_window(
    window_trades: pl.DataFrame,
    *,
    n_clusters: int,
    size_min: int,
    max_share: float,
    size_max: int | None = None,
    random_state: int | None = 42,
) -> WindowResult:
    X, agg, scaler = build_counterparty_features(window_trades)
    labels0, _ = cluster_counterparties(
        X, n_clusters=n_clusters, size_min=size_min, size_max=size_max, random_state=random_state
    )
    labels = enforce_volume_share(agg, labels0, max_share=max_share, n_clusters=n_clusters)
    summary = make_cluster_summary(agg, labels)
    metrics = evaluate_clustering(X, labels)
    start = window_trades.select(pl.min("ts")).item()
    end = window_trades.select(pl.max("ts")).item()
    return WindowResult(start=start, end=end, labels=labels, agg=agg, summary=summary, metrics=metrics)


# ---- align clusters across windows ----------------------------------------

def _align_labels(
    prev_labels: np.ndarray,
    curr_labels: np.ndarray,
    prev_agg: pl.DataFrame,
    curr_agg: pl.DataFrame,
) -> np.ndarray:
    """Align *curr_labels* to *prev_labels* by maximum volume overlap."""
    mapping: Dict[int, int] = {}
    prev_df = prev_agg.with_columns(pl.Series("cluster", prev_labels))
    curr_df = curr_agg.with_columns(pl.Series("cluster", curr_labels))

    # build cross‑tab of volume overlaps
    overlap = defaultdict(float)
    for cty in set(prev_df["counterparty"]):
        prev_k = int(prev_df.filter(pl.col("counterparty") == cty)["cluster"].item())
        row_curr = curr_df.filter(pl.col("counterparty") == cty)
        if row_curr.height == 0:
            continue  # not present in next window
        curr_k = int(row_curr["cluster"].item())
        vol = float(row_curr["total_volume"].item())
        overlap[(prev_k, curr_k)] += vol

    # greedy one‑to‑one mapping by descending overlap
    used_prev, used_curr = set(), set()
    for (prev_k, curr_k), _ in sorted(overlap.items(), key=lambda kv: kv[1], reverse=True):
        if prev_k not in used_prev and curr_k not in used_curr:
            mapping[curr_k] = prev_k
            used_prev.add(prev_k)
            used_curr.add(curr_k)

    # assign new ids for unmatched current clusters
    next_unused = max(prev_labels.max(), len(set(prev_labels))) + 1
    aligned = curr_labels.copy()
    for k in set(curr_labels):
        if k in mapping:
            aligned[curr_labels == k] = mapping[k]
        else:
            aligned[curr_labels == k] = next_unused
            next_unused += 1
    return aligned


# ---- main rolling pipeline -------------------------------------------------

def cluster_over_time(
    trades: pl.DataFrame,
    windows: List[Tuple[datetime, datetime]],
    *,
    n_clusters: int,
    size_min: int,
    max_share: float,
    size_max: int | None = None,
    random_state: int | None = 42,
) -> List[WindowResult]:
    """Run the clustering pipeline for every time window, aligning labels."""
    results: list[WindowResult] = []
    prev_res: WindowResult | None = None

    for (start, end) in windows:
        window_trades = trades.filter((pl.col("ts") >= start) & (pl.col("ts") < end))
        if window_trades.height == 0:
            continue
        res = cluster_for_window(
            window_trades,
            n_clusters=n_clusters,
            size_min=size_min,
            max_share=max_share,
            size_max=size_max,
            random_state=random_state,
        )
        # align with previous window for stability analysis
        if prev_res is not None:
            aligned = _align_labels(prev_res.labels, res.labels, prev_res.agg, res.agg)
            res = WindowResult(
                start=res.start,
                end=res.end,
                labels=aligned,
                agg=res.agg,
                summary=make_cluster_summary(res.agg, aligned),
                metrics=res.metrics,
            )
        results.append(res)
        prev_res = res
    return results


# ---- cross‑window stability metrics ---------------------------------------

def _rand_index_between(res_a: WindowResult, res_b: WindowResult) -> float:
    """Adjusted Rand for counterparties present in *both* windows."""
    common = set(res_a.agg["counterparty"]).intersection(res_b.agg["counterparty"])
    if not common:
        return np.nan
    idx_a = [int(np.where(res_a.agg["counterparty"] == c)[0][0]) for c in common]
    idx_b = [int(np.where(res_b.agg["counterparty"] == c)[0][0]) for c in common]
    return adjusted_rand_score(res_a.labels[idx_a], res_b.labels[idx_b])


def evaluate_stability_over_time(results: List[WindowResult]) -> pl.DataFrame:
    """Return one row per *transition* (window i → i+1) with stability metrics."""
    rows = []
    for prev, curr in zip(results[:-1], results[1:]):
        rand = _rand_index_between(prev, curr)
        unchanged = np.mean(
            prev.labels[np.isin(prev.agg["counterparty"], curr.agg["counterparty"])]
            == curr.labels[np.isin(curr.agg["counterparty"], prev.agg["counterparty"])]
        )
        rows.append(
            {
                "window_start": curr.start,
                "window_end": curr.end,
                "adjusted_rand": rand,
                "pct_counterparties_unchanged": unchanged,
                "n_clusters": len(set(curr.labels)),
                **curr.metrics,
            }
        )
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Example CLI / smoke‑test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    trades = load_trade_history("/data/equities/trades_*.parquet")  # must include a 'ts' datetime column
    wins = generate_time_windows(trades, window_days=90, step_days=30)
    res = cluster_over_time(trades, wins, n_clusters=10, size_min=5, max_share=0.20)
    stability = evaluate_stability_over_time(res)
    print(stability.head())
