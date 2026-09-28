"""Core linear-algebra layer: finding and erasing concept directions.

Everything here operates on plain numpy arrays so the mathematics is
independent of any deep-learning framework and fully unit-testable.

Conventions
-----------
* ``X_pos`` : activations for prompts containing the concept, shape
  ``(n_pos, d)`` where ``d`` is the hidden dimension.
* ``X_neg`` : activations for neutral prompts, shape ``(n_neg, d)``.
* Directions are returned as unit vectors, shape ``(d,)``.
* Erasers are returned as ``(d, d)`` matrices ``P`` such that ``h @ P.T``
  is the erased hidden state.

A note on centering that cost us a debugging session: the concept signal
lives in the *between-group* mean difference. Any step that centers each
group by its own mean (or centers the difference distribution) destroys the
signal before the math runs. Always center against the *pooled* mean, or
not at all when the shared offset is the quantity of interest.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import pinv


# --------------------------------------------------------------------------
# Direction fitting
# --------------------------------------------------------------------------

def mean_difference(X_pos: np.ndarray, X_neg: np.ndarray) -> np.ndarray:
    """The classic difference-in-means concept direction.

    v = mean(X_pos) - mean(X_neg), normalized to unit length.
    """
    v = X_pos.mean(axis=0) - X_neg.mean(axis=0)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Difference in means is ~zero; the two prompt sets are indistinguishable.")
    return v / n


# alias used across the pipeline and tests
fit_direction = mean_difference


def pca_difference_directions(
    X_pos: np.ndarray, X_neg: np.ndarray, k: int = 4
) -> np.ndarray:
    """Top-k principal components of the paired difference distribution.

    Rows of the difference matrix share a constant component (the concept
    offset); the leading right-singular vector recovers it. Do NOT center
    the differences first — the shared offset *is* the signal.
    """
    diffs = X_pos.mean(axis=0, keepdims=True) - X_neg  # broadcast: (n_neg, d)
    _, _, Vt = np.linalg.svd(diffs, full_matrices=False)
    return Vt[:k]


def _balance_classes(X_pos: np.ndarray, X_neg: np.ndarray, rng: np.random.Generator):
    n = min(len(X_pos), len(X_neg))
    return (
        X_pos[rng.choice(len(X_pos), n, replace=False)],
        X_neg[rng.choice(len(X_neg), n, replace=False)],
    )


def fit_directions_inlp(
    X_pos: np.ndarray, X_neg: np.ndarray, k: int = 4, seed: int = 0, max_iter: int = 10
) -> np.ndarray:
    """Iterative Nullspace Projection (INLP; Ravfogel et al., 2020).

    Repeatedly fits a linear regressor for the concept label, projects the
    data into the nullspace of its weight vector, and repeats, yielding up
    to ``k`` directions that collectively remove all linearly separable
    concept information.
    """
    from sklearn.linear_model import LinearRegression

    rng = np.random.default_rng(seed)
    P, N = _balance_classes(X_pos.copy(), X_neg.copy(), rng)
    X = np.concatenate([P, N], axis=0)
    y = np.concatenate([np.ones(len(P)), np.zeros(len(N))])

    directions: list[np.ndarray] = []
    for _ in range(max_iter):
        clf = LinearRegression()
        clf.fit(X, y)
        w = clf.coef_
        if np.linalg.norm(w) < 1e-10:
            break
        w = w / np.linalg.norm(w)
        directions.append(w)
        X = X - np.outer(X @ w, w)  # project onto w's nullspace
        # note: std(X @ w) is ~0 by construction now — the stopping signal
        # is the *next* iteration's regressor finding no signal (coef ~ 0).
    return np.array(directions) if directions else np.zeros((0, X_pos.shape[1]))


# --------------------------------------------------------------------------
# LEACE-style oblique eraser (Belrose et al., 2023; binary-label closed form)
# --------------------------------------------------------------------------

def fit_leace_eraser(
    X_pos: np.ndarray, X_neg: np.ndarray, reg: float = 1e-6
) -> np.ndarray:
    """Return the oblique projection ``P`` that zeroes out all linear
    covariance between hidden states and the (binary) concept indicator.

    With ``z in {0,1}`` the concept indicator, ``Sigma`` the *pooled*
    feature covariance (both groups centered by the pooled mean — the
    between-group difference must survive centering), and
    ``b = Sigma^{-1} Cov(x, z)`` the optimal linear predictor direction,
    the eraser

        P = I - Sigma b b^T / (b^T Sigma b)

    satisfies ``Cov(P x, z) = 0`` exactly: no linear statistic of the
    hidden state carries information about the concept. This is the
    closed-form LEACE eraser for a binary label (without the leakage
    rectification refinement for degenerate labels).

    Returns ``(d, d)``; apply as ``h_erased = h @ P.T``.
    """
    X = np.concatenate([X_pos, X_neg], axis=0)
    n = len(X)
    Xc = X - X.mean(axis=0)
    Sigma = (Xc.T @ Xc) / n
    Sigma += reg * np.eye(Sigma.shape[0]) * np.trace(Sigma) / Sigma.shape[0]

    z = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    zc = z - z.mean()
    cov_xz = (Xc.T @ zc) / n  # (d,)

    b = pinv(Sigma) @ cov_xz
    denom = b @ Sigma @ b
    if abs(denom) < 1e-14:
        return np.eye(Sigma.shape[0])  # nothing to erase
    return np.eye(Sigma.shape[0]) - Sigma @ np.outer(b, b) / denom


def oblique_project(H: np.ndarray, v: np.ndarray, Sigma: np.ndarray | None = None) -> np.ndarray:
    """Project rows of ``H`` onto the nullspace of ``v`` (optionally using a
    covariance-weighted dual vector). Plain orthogonal projection when
    ``Sigma`` is None."""
    v = v / np.linalg.norm(v)
    if Sigma is None:
        return H - np.outer(H @ v, v)
    u = pinv(Sigma) @ v
    u = u / (u @ v)  # ensures u^T v = 1
    return H - np.outer(H @ v, u)


# --------------------------------------------------------------------------
# Applying erasers
# --------------------------------------------------------------------------

def project_out(
    H: np.ndarray,
    directions: np.ndarray,
    mode: str = "project",
    pad_epsilon: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Remove every direction in ``directions`` from each row of ``H``.

    mode="project" : h <- h - sum_i (h . v_i) v_i   (orthogonal projection)
    mode="pad"     : same, but the removed component is replaced by
                     ``pad_epsilon * v_i`` (ablative ablation keeps the
                     representation near the data manifold instead of
                     pinning it to the nullspace hyperplane).
    """
    H = H.copy()
    for v in directions:
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv
        comp = H @ v
        H = H - np.outer(comp, v)
        if mode == "pad":
            H = H + pad_epsilon * np.outer(np.ones(len(H)), v)
    return H
