# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Fair-PCM algorithm                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
Fair Possibilistic C-Means (Fair-PCM)
--------------------------------------
Objective
---------
    F(T, V) = Σ_{ik} t_{ik}^m ‖x_i − v_k‖²
            + Σ_k η_k Σ_i (1 − t_{ik})^m
            + λ Σ_k KL( q̂_k ‖ p )
"""

from __future__ import annotations
import warnings
from typing import Optional
import numpy as np
from scipy.optimize import brentq
from sklearn.cluster import KMeans


# ── Solver A: m = 2, exact depressed-cubic via Cardano / trigonometric ────────

def _solve_depressed_cubic(p: np.ndarray, q: np.ndarray, eps: float) -> np.ndarray:
    """
    Solve  t³ + p·t + q = 0  (vectorised, any input shape).

    Discriminant Δ = -4p³ - 27q²:
      Δ ≤ 0  → one real root  (Cardano)
      Δ > 0  → three real roots  (trigonometric form)
    Returns root in (eps, 1-eps) with smallest |residual|.
    """
    shape = p.shape
    p, q  = p.ravel(), q.ravel()

    disc = -4.0 * p**3 - 27.0 * q**2
    out  = np.empty_like(p)

    # one real root (Cardano)
    m1 = disc <= 0
    if m1.any():
        p1, q1  = p[m1], q[m1]
        inner   = np.maximum(q1**2 / 4.0 + p1**3 / 27.0, 0.0)
        s       = np.sqrt(inner)
        out[m1] = np.cbrt(-q1 / 2.0 + s) + np.cbrt(-q1 / 2.0 - s)

    # three real roots (trigonometric)
    m2 = ~m1
    if m2.any():
        p2, q2  = p[m2], q[m2]
        r       = np.sqrt(np.maximum(-(p2 / 3.0)**3, 0.0))
        cos_arg = np.clip(-q2 / (2.0 * np.maximum(r, eps)), -1.0, 1.0)
        theta   = np.arccos(cos_arg)
        r_cbrt  = np.cbrt(r)

        cands = np.stack(
            [2.0 * r_cbrt * np.cos((theta - 2.0 * np.pi * j) / 3.0)
             for j in range(3)],
            axis=-1,
        )  # (n, 3)

        valid  = (cands > eps) & (cands < 1.0 - eps)
        resid  = np.abs(cands**3 + p2[:, None] * cands + q2[:, None])
        masked = np.where(valid, resid, np.inf)
        best   = np.argmin(masked, axis=-1)
        t_best = cands[np.arange(len(best)), best]

        no_valid = ~valid.any(axis=-1)
        if no_valid.any():
            t_best[no_valid] = np.clip(cands[no_valid, 0], eps, 1.0 - eps)

        out[m2] = t_best

    return np.clip(out, eps, 1.0 - eps).reshape(shape)


# ── Solver B: general m, Newton + Brent fallback ──────────────────────────────

def _solve_general_m(
    T_old: np.ndarray,
    A: np.ndarray,
    B_k: np.ndarray,
    eta: np.ndarray,
    m: float,
    eps: float,
    newton_iters: int,
) -> np.ndarray:
    """
    Solve  r(t) = t^(m-1)·A + B·t^(2m-1) - η·(1-t)^(m-1) = 0
    via vectorised Newton; Brent fallback per cell when Newton stalls.
    """
    t = np.clip(T_old.copy(), eps, 1.0 - eps)

    for _ in range(newton_iters):
        r  = (t**(m-1) * A
              + B_k[None, :] * t**(2*m-1)
              - eta[None, :] * (1.0 - t)**(m-1))
        rp = ((m-1) * t**(m-2) * A
              + (2*m-1) * B_k[None, :] * t**(2*m-2)
              + eta[None, :] * (m-1) * (1.0 - t)**(m-2))

        safe = np.abs(rp) > eps
        t    = np.clip(t - np.where(safe, r / np.where(safe, rp, 1.0), 0.0),
                       eps, 1.0 - eps)
        if np.max(np.abs(r)) < 1e-12:
            break

    # Brent fallback for cells where |r| > 1e-6
    r_final  = (t**(m-1) * A
                + B_k[None, :] * t**(2*m-1)
                - eta[None, :] * (1.0 - t)**(m-1))
    rows, cols = np.where(np.abs(r_final) > 1e-6)

    for i, k in zip(rows, cols):
        A_ik, B, eta_k = float(A[i, k]), float(B_k[k]), float(eta[k])

        def _f(tv, _A=A_ik, _B=B, _e=eta_k):
            return _A*tv**(m-1) + _B*tv**(2*m-1) - _e*(1.0-tv)**(m-1)

        fa, fb = _f(eps), _f(1.0 - eps)
        if fa * fb < 0:
            try:
                t[i, k] = brentq(_f, eps, 1.0 - eps, xtol=1e-12, rtol=1e-12)
            except Exception:
                pass
        else:
            t[i, k] = eps if abs(fa) < abs(fb) else 1.0 - eps

    return np.clip(t, eps, 1.0 - eps)


# ── Main class ────────────────────────────────────────────────────────────────

class FairPCM:
    """
    Fair Possibilistic C-Means.

    Parameters
    ----------
    n_clusters   : int    – number of clusters K
    m            : float  – fuzzifier (≥1); m=2 uses closed-form cubic solver
    lam          : float  – fairness regularisation weight λ
    tol          : float  – relative objective change for convergence
    max_iter     : int    – maximum iterations
    newton_iters : int    – Newton steps per update (general-m path only)
    eps          : float  – numerical floor
    verbose      : bool   – print objective each iteration
    random_state : int    – seed for KMeans++ init

    Attributes (after fit)
    ----------------------
    T_         : (N, K) typicality matrix
    centers_   : (K, D) cluster centres
    labels_    : (N,)   hard labels = argmax_k t_{ik}
    objective_ : float  final objective value
    n_iter_    : int    iterations run
    """

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        lam: float = 1.0,
        tol: float = 1e-4,
        max_iter: int = 150,
        newton_iters: int = 20,
        eps: float = 1e-10,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        if n_clusters < 1: raise ValueError("n_clusters must be ≥ 1")
        if m < 1:          raise ValueError("fuzzifier m must be ≥ 1")
        self.K            = n_clusters
        self.m            = float(m)
        self.lam          = lam
        self.tol          = tol
        self.max_iter     = max_iter
        self.newton_iters = newton_iters
        self.eps          = eps
        self.verbose      = verbose
        self.random_state = random_state

    def _initialise(self, X, S):
        N, _      = X.shape
        groups    = np.unique(S)
        group_idx = {g: np.where(S == g)[0] for g in groups}
        p_g       = np.array([len(group_idx[g]) / N for g in groups])

        km = KMeans(n_clusters=self.K, init='k-means++', n_init=1,
                    random_state=self.random_state)
        km.fit(X)
        V, km_labels = km.cluster_centers_.copy(), km.labels_

        eta = np.array([
            np.mean(np.sum((X[km_labels == k] - V[k])**2, axis=1))
            if (km_labels == k).any() else 1.0
            for k in range(self.K)
        ])
        eta = np.maximum(eta, self.eps)
        T   = 0.5 * np.ones((N, self.K))
        return groups, group_idx, p_g, V, eta, T

    def fit(self, X: np.ndarray, S: np.ndarray) -> "FairPCM":
        X = np.asarray(X, dtype=float)
        S = np.asarray(S)
        N, D      = X.shape
        K, m      = self.K, self.m
        lam, eps  = self.lam, self.eps

        groups, group_idx, p_g, V, eta, T = self._initialise(X, S)
        F_prev, F_new, rel = np.inf, np.inf, np.inf

        for it in range(self.max_iter):
            # Step 1: update centres
            u     = T**m
            u_sum = np.maximum(u.sum(axis=0), eps)
            V     = (u.T @ X) / u_sum[:, None]
            d2    = np.sum((X[:, None, :] - V[None, :, :])**2, axis=2)

            # Step 2: fairness statistics
            n_hat_kg = np.stack(
                [u[group_idx[g]].sum(axis=0) for g in groups], axis=1)   # (K,G)
            n_hat_k  = np.maximum(n_hat_kg.sum(axis=1), eps)             # (K,)

            q_hat_kg  = n_hat_kg / n_hat_k[:, None]
            q_hat_kg  = np.maximum(q_hat_kg, eps)
            q_hat_kg /= q_hat_kg.sum(axis=1, keepdims=True)

            KL_k = np.sum(q_hat_kg * np.log(q_hat_kg / p_g[None, :]), axis=1)

            psi = np.empty((N, K))
            for gi, g in enumerate(groups):
                psi[group_idx[g]] = (
                    (np.log(q_hat_kg[:, gi] / p_g[gi]) - KL_k) / n_hat_k
                )

            # Step 3: Lipschitz constant + modified distance
            delta_k = np.max(np.abs(np.log(q_hat_kg / p_g[None, :])), axis=1)
            min_q   = np.maximum(q_hat_kg.min(axis=1), eps)
            C_k     = (N / n_hat_k**2) * (1.0 / min_q + 1.0 + 4.0 * delta_k)
            B_k     = np.maximum(lam * C_k, eps)

            A = d2 + lam * psi - B_k[None, :] * T   # modified effective distance

            # Step 4: update typicalities
            if m == 2.0:
                p_c = (A + eta[None, :]) / B_k[None, :]
                q_c = np.broadcast_to(-eta / B_k, (N, K)).copy()
                T   = _solve_depressed_cubic(p_c, q_c, eps)
            else:
                T   = _solve_general_m(T, A, B_k, eta, m, eps, self.newton_iters)

            # Step 5: objective + convergence
            F_new = (float(np.sum(T**m * d2))
                     + float(np.dot(eta, np.sum((1.0 - T)**m, axis=0)))
                     + lam * float(KL_k.sum()))
            rel   = abs(F_new - F_prev) / (abs(F_prev) + eps)

            if self.verbose:
                print(f"  iter {it+1:4d} | F={F_new:14.4f} | Δrel={rel:.3e} | "
                      f"KL_mean={KL_k.mean():.5f}")

            if rel < self.tol:
                if self.verbose:
                    print(f"  ✓ Converged after {it+1} iterations.")
                break
            F_prev = F_new
        else:
            warnings.warn(
                f"FairPCM did not converge in {self.max_iter} iterations "
                f"(last Δrel={rel:.3e}).", stacklevel=2)

        self.T_, self.centers_, self.labels_ = T, V, np.argmax(T, axis=1)
        self.objective_, self.n_iter_        = F_new, it + 1
        self._groups, self._p_g              = groups, p_g
        return self

    def fit_predict(self, X, S):
        return self.fit(X, S).labels_

    def fairness_report(self, S):
        """Per-cluster KL divergences and soft proportions q̂_{kg}."""
        if not hasattr(self, 'T_'):
            raise RuntimeError("Call .fit() first.")
        groups    = self._groups
        group_idx = {g: np.where(S == g)[0] for g in groups}
        u         = self.T_**self.m
        n_hat_kg  = np.stack([u[group_idx[g]].sum(axis=0) for g in groups], axis=1)
        n_hat_k   = np.maximum(n_hat_kg.sum(axis=1), self.eps)
        q_hat_kg  = n_hat_kg / n_hat_k[:, None]
        q_hat_kg  = np.maximum(q_hat_kg, self.eps)
        q_hat_kg /= q_hat_kg.sum(axis=1, keepdims=True)
        kl_k      = np.sum(q_hat_kg * np.log(q_hat_kg / self._p_g[None, :]), axis=1)
        return {
            'group_priors'   : dict(zip(groups, self._p_g)),
            'soft_props'     : q_hat_kg,
            'kl_per_cluster' : kl_k,
        }
