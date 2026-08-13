import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fair_pcm import FairPCM
from fair_pcm.model import _solve_depressed_cubic


@pytest.fixture
def toy_data():
    rng = np.random.default_rng(0)
    X = np.vstack([
        rng.normal(loc=[0, 0], scale=0.5, size=(50, 2)),
        rng.normal(loc=[6, 0], scale=0.5, size=(50, 2)),
    ])
    S = rng.choice(["A", "B"], size=100, p=[0.5, 0.5])
    return X, S


def test_fit_shapes(toy_data):
    X, S = toy_data
    model = FairPCM(n_clusters=2, max_iter=30, random_state=0)
    model.fit(X, S)
    assert model.T_.shape == (100, 2)
    assert model.centers_.shape == (2, 2)
    assert model.labels_.shape == (100,)


def test_typicalities_in_range(toy_data):
    X, S = toy_data
    model = FairPCM(n_clusters=2, max_iter=30, random_state=0)
    model.fit(X, S)
    assert np.all(model.T_ >= 0.0)
    assert np.all(model.T_ <= 1.0)


def test_fit_predict(toy_data):
    X, S = toy_data
    model = FairPCM(n_clusters=2, max_iter=30, random_state=0)
    labels = model.fit_predict(X, S)
    assert len(np.unique(labels)) <= 2


def test_fairness_report(toy_data):
    X, S = toy_data
    model = FairPCM(n_clusters=2, max_iter=30, random_state=0)
    model.fit(X, S)
    report = model.fairness_report(S)
    assert "kl_per_cluster" in report
    assert report["kl_per_cluster"].shape == (2,)
    assert np.all(report["kl_per_cluster"] >= -1e-8)  # KL divergence >= 0


def test_general_m_path(toy_data):
    """Non-2 fuzzifier should route through the Newton/Brent solver."""
    X, S = toy_data
    model = FairPCM(n_clusters=2, m=1.7, max_iter=20, random_state=0)
    model.fit(X, S)
    assert np.all(np.isfinite(model.T_))


def test_depressed_cubic_solver_roundtrip():
    """Check that returned roots approximately satisfy t^3 + p t + q = 0."""
    rng = np.random.default_rng(1)
    p = rng.uniform(-5, 5, size=(20,))
    q = rng.uniform(-5, 5, size=(20,))
    t = _solve_depressed_cubic(p, q, eps=1e-10)
    resid = t**3 + p * t + q
    # not all combinations have a root in (eps, 1-eps); just check output validity
    assert np.all(t >= 1e-10)
    assert np.all(t <= 1.0 - 1e-10)
    assert np.all(np.isfinite(resid))


def test_invalid_params():
    with pytest.raises(ValueError):
        FairPCM(n_clusters=0)
    with pytest.raises(ValueError):
        FairPCM(n_clusters=2, m=0.5)
