"""
make_datasets.py
-----------------
Generates synthetic benchmark datasets for testing Fair-PCM.

Each dataset is a 2D (or higher-D) point cloud with a categorical
"sensitive attribute" column S, saved as CSV with columns:
    x1, x2, ..., xd, S

Datasets
--------
1. synthetic_blobs_2group.csv   - 3 imbalanced Gaussian blobs, 2 groups,
                                   group proportions skewed per-blob
                                   (classic fair-clustering stress test).
2. synthetic_blobs_3group.csv   - 4 blobs, 3 groups.
3. synthetic_overlap.csv        - 2 overlapping blobs, 2 groups, to test
                                   possibilistic (noise-robust) behavior
                                   with an added outlier cluster.

These are provided so the repo is runnable end-to-end without requiring
network access to external dataset hosts (e.g. UCI Adult/Bank). If you
want to benchmark against real-world fairness datasets, drop UCI's
"Adult" or "Bank Marketing" CSVs into this folder and adapt
`load_real_dataset()` at the bottom of this file.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).parent


def _blob(n, center, cov, group_probs, group_labels, rng):
    X = rng.multivariate_normal(center, cov, size=n)
    S = rng.choice(group_labels, size=n, p=group_probs)
    return X, S


def make_blobs_2group():
    """3 blobs, 2 groups ('A', 'B'), skewed proportions per blob."""
    specs = [
        # (n, center, cov, [p_A, p_B])
        (300, [0, 0], [[1, 0], [0, 1]], [0.9, 0.1]),
        (300, [8, 0], [[1, 0], [0, 1]], [0.1, 0.9]),
        (300, [4, 7], [[1, 0], [0, 1]], [0.5, 0.5]),
    ]
    Xs, Ss = [], []
    for n, c, cov, probs in specs:
        X, S = _blob(n, c, cov, probs, ['A', 'B'], RNG)
        Xs.append(X); Ss.append(S)
    X = np.vstack(Xs)
    S = np.concatenate(Ss)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['S'] = S
    df.to_csv(OUT_DIR / 'synthetic_blobs_2group.csv', index=False)
    return df


def make_blobs_3group():
    """4 blobs, 3 groups ('A', 'B', 'C'), skewed proportions per blob."""
    specs = [
        (250, [0, 0],  [[1, 0], [0, 1]], [0.7, 0.2, 0.1]),
        (250, [9, 0],  [[1, 0], [0, 1]], [0.1, 0.7, 0.2]),
        (250, [4, 8],  [[1, 0], [0, 1]], [0.2, 0.1, 0.7]),
        (250, [9, 9],  [[1, 0], [0, 1]], [0.34, 0.33, 0.33]),
    ]
    Xs, Ss = [], []
    for n, c, cov, probs in specs:
        X, S = _blob(n, c, cov, probs, ['A', 'B', 'C'], RNG)
        Xs.append(X); Ss.append(S)
    X = np.vstack(Xs)
    S = np.concatenate(Ss)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['S'] = S
    df.to_csv(OUT_DIR / 'synthetic_blobs_3group.csv', index=False)
    return df


def make_overlap_with_outliers():
    """2 overlapping blobs + sparse outlier cluster, to exercise the
    possibilistic (noise-robust) term of Fair-PCM."""
    specs = [
        (400, [0, 0], [[2, 0.5], [0.5, 2]], [0.8, 0.2]),
        (400, [3, 3], [[2, 0.5], [0.5, 2]], [0.2, 0.8]),
    ]
    Xs, Ss = [], []
    for n, c, cov, probs in specs:
        X, S = _blob(n, c, cov, probs, ['A', 'B'], RNG)
        Xs.append(X); Ss.append(S)

    # sparse outliers, evenly split by group
    n_out = 40
    X_out = RNG.uniform(-15, 15, size=(n_out, 2))
    S_out = RNG.choice(['A', 'B'], size=n_out, p=[0.5, 0.5])
    Xs.append(X_out); Ss.append(S_out)

    X = np.vstack(Xs)
    S = np.concatenate(Ss)
    df = pd.DataFrame(X, columns=['x1', 'x2'])
    df['S'] = S
    df.to_csv(OUT_DIR / 'synthetic_overlap.csv', index=False)
    return df


def load_real_dataset(csv_path: str, sensitive_col: str, feature_cols=None):
    """
    Helper to load a real-world fairness-clustering dataset (e.g. UCI
    Adult, Bank Marketing, COMPAS) that you've placed in this folder.

    Parameters
    ----------
    csv_path       : path to the CSV file
    sensitive_col  : name of the column to use as the sensitive attribute S
    feature_cols   : list of numeric feature columns to use as X
                      (defaults to all numeric columns except sensitive_col)

    Returns
    -------
    X : (N, D) ndarray
    S : (N,)   ndarray
    """
    df = pd.read_csv(csv_path)
    if feature_cols is None:
        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                         if c != sensitive_col]
    X = df[feature_cols].to_numpy(dtype=float)
    S = df[sensitive_col].to_numpy()
    return X, S


if __name__ == '__main__':
    d1 = make_blobs_2group()
    d2 = make_blobs_3group()
    d3 = make_overlap_with_outliers()
    print(f"synthetic_blobs_2group.csv : {d1.shape}")
    print(f"synthetic_blobs_3group.csv : {d2.shape}")
    print(f"synthetic_overlap.csv      : {d3.shape}")
    print(f"Written to: {OUT_DIR}")
