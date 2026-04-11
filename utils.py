"""Helpers for SF2943 Part A cleaning pipeline.

Functions for harmonic-regression seasonality estimation and residual
diagnostics. Referenced by cleaning.ipynb.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def harmonic_design(t: np.ndarray, period: float, k: int) -> np.ndarray:
    """Design matrix [cos(2 pi j t / d), sin(2 pi j t / d)] for j=1..k.

    Intercept column NOT included — caller adds it (or not) explicitly.
    """
    cols = []
    for j in range(1, k + 1):
        lam = 2.0 * np.pi * j / period
        cols.append(np.cos(lam * t))
        cols.append(np.sin(lam * t))
    return np.column_stack(cols)


def fit_harmonic(y: np.ndarray, period: float, k: int):
    """Fit y = a0 + sum_{j=1}^k [a_j cos + b_j sin] by OLS.

    Returns (fitted_values, coefficients_dict). The fitted component is
    returned **without** the intercept a0, so it is mean-zero over the
    sample — this matches the B&D convention that s_t sums to zero over
    one period and the level is absorbed by the trend.
    """
    t = np.arange(len(y), dtype=float)
    X_harm = harmonic_design(t, period, k)
    X = np.column_stack([np.ones_like(t), X_harm])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    a0 = beta[0]
    harm_coefs = beta[1:]
    fitted_harm = X_harm @ harm_coefs
    coefs = {"a0": float(a0)}
    for j in range(1, k + 1):
        coefs[f"a{j}"] = float(harm_coefs[2 * (j - 1)])
        coefs[f"b{j}"] = float(harm_coefs[2 * (j - 1) + 1])
    return fitted_harm, coefs


def fit_poly_trend(y: np.ndarray, degree: int):
    """Fit polynomial trend of given degree by OLS. Returns (fitted, coefs)."""
    t = np.arange(len(y), dtype=float)
    X = np.column_stack([t ** d for d in range(degree + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    coefs = {f"beta{d}": float(beta[d]) for d in range(degree + 1)}
    return fitted, coefs


def ljung_box(residuals: np.ndarray, lags=(20, 40)) -> pd.DataFrame:
    """Ljung-Box statistics at the requested lags."""
    return acorr_ljungbox(residuals, lags=list(lags), return_df=True)


def diagnostic_plots(residuals: pd.Series, max_lag: int = 50, title_prefix: str = ""):
    """Three-panel diagnostic: residual time series, ACF, PACF."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    axes[0].plot(residuals.index, residuals.values, lw=0.6)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title(f"{title_prefix}Residuals $\\hat Y_t$")
    axes[0].set_xlabel("date")
    axes[0].set_ylabel("residual (log MW)")

    plot_acf(residuals.values, lags=max_lag, ax=axes[1])
    axes[1].set_title(f"{title_prefix}Sample ACF (dashed: $\\pm 1.96/\\sqrt{{n}}$)")
    axes[1].set_xlabel("lag (days)")

    plot_pacf(residuals.values, lags=max_lag, ax=axes[2], method="ywm")
    axes[2].set_title(f"{title_prefix}Sample PACF")
    axes[2].set_xlabel("lag (days)")

    fig.tight_layout()
    return fig
