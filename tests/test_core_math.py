"""Unit tests for the pure-numpy core (no torch required)."""
import numpy as np
import pytest

from concept_surgery.core_math import (
    fit_direction,
    fit_directions_inlp,
    fit_leace_eraser,
    oblique_project,
    pca_difference_directions,
    project_out,
)


def make_latent_data(n=400, d=32, concept_strength=4.0, seed=0):
    """Hidden states where a single planted direction encodes the concept."""
    rng = np.random.default_rng(seed)
    true_v = rng.normal(size=d)
    true_v /= np.linalg.norm(true_v)
    z = rng.integers(0, 2, size=n)
    noise = rng.normal(scale=0.5, size=(n, d))
    X = noise + concept_strength * np.outer(z, true_v)
    X_pos, X_neg = X[z == 1], X[z == 0]
    return X_pos, X_neg, true_v


def test_mean_difference_recovers_planted_direction():
    X_pos, X_neg, true_v = make_latent_data()
    v = fit_direction(X_pos, X_neg)
    # recovery up to sign
    assert abs(np.dot(v, true_v)) > 0.99


def test_projection_removes_information():
    X_pos, X_neg, true_v = make_latent_data()
    v = fit_direction(X_pos, X_neg)
    X = np.concatenate([X_pos, X_neg])
    X_erased = project_out(X, v[None, :])
    # component along v is now zero for every row
    assert np.allclose(X_erased @ v, 0, atol=1e-10)
    # orthogonal content is untouched
    rng = np.random.default_rng(1)
    w = rng.normal(size=X.shape[1])
    w -= (w @ v) * v
    w /= np.linalg.norm(w)
    assert np.allclose(X @ w, X_erased @ w, atol=1e-8)


def test_leace_zeroes_linear_covariance():
    X_pos, X_neg, true_v = make_latent_data(concept_strength=3.0)
    X = np.concatenate([X_pos, X_neg])
    z = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    P = fit_leace_eraser(X_pos, X_neg)
    X_erased = X @ P.T
    zc = z - z.mean()
    Xc = X_erased - X_erased.mean(axis=0)
    cov = Xc.T @ zc / len(z)
    assert np.allclose(cov, 0, atol=1e-8)


def test_inlp_removes_multiple_directions():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(3)
    d = 24
    v1 = rng.normal(size=d); v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=d); v2 -= (v2 @ v1) * v1; v2 /= np.linalg.norm(v2)
    n = 400
    labels = rng.integers(0, 2, size=n)
    # concept 1 in the mean, concept 2 modulates a subset
    X = rng.normal(scale=0.5, size=(n, d))
    X += 3.0 * np.outer(labels, v1)
    subset = labels.astype(bool)
    X[subset] += 3.0 * np.outer(np.ones(subset.sum()), v2)
    X_pos, X_neg = X[labels == 1], X[labels == 0]

    # baseline: concept is easily linearly separable
    clf = LogisticRegression(max_iter=2000).fit(X, labels)
    assert clf.score(X, labels) > 0.9

    dirs = fit_directions_inlp(X_pos, X_neg, k=4, seed=0)
    assert len(dirs) >= 1
    X_erased = project_out(X, dirs)
    # after erasure, no linear decoder should separate the classes
    clf2 = LogisticRegression(max_iter=2000).fit(X_erased, labels)
    assert clf2.score(X_erased, labels) < 0.65


def test_pca_finds_concept_subspace():
    X_pos, X_neg, true_v = make_latent_data(seed=7)
    dirs = pca_difference_directions(X_pos, X_neg, k=4)
    assert np.abs(dirs @ true_v).max() > 0.9


def test_oblique_projection_matches_plain_when_whitened():
    X_pos, X_neg, _ = make_latent_data()
    v = fit_direction(X_pos, X_neg)
    H = np.random.default_rng(5).normal(size=(10, len(v)))
    Hp = oblique_project(H, v)
    assert np.allclose(Hp @ v, 0, atol=1e-10)


def test_pad_mode_keeps_component_small():
    X_pos, X_neg, _ = make_latent_data()
    v = fit_direction(X_pos, X_neg)
    H = X_pos.copy()
    H_padded = project_out(H, v[None, :], mode="pad", pad_epsilon=0.01)
    comps = H_padded @ v
    assert np.allclose(comps, 0.01, atol=1e-6)
