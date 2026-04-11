"""Helpers for SF2943 Part A: cleaning, ARMA fitting, forecasting.

Used by project_part_a.ipynb. Method references are Brockwell & Davis,
*Introduction to Time Series and Forecasting*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


# --- Harmonic seasonality (B&D §1.3) --------------------------------------

def harmonic_design(t: np.ndarray, period: float, k: int) -> np.ndarray:
    """Columns cos(2 pi j t / d), sin(2 pi j t / d) for j=1..k."""
    cols = []
    for j in range(1, k + 1):
        lam = 2.0 * np.pi * j / period
        cols.append(np.cos(lam * t))
        cols.append(np.sin(lam * t))
    return np.column_stack(cols)


def fit_harmonic(y: np.ndarray, period: float, k: int):
    """Fit y = a0 + sum_j (a_j cos + b_j sin) by OLS.

    Returns (seasonal_component, amplitude_coefs). The intercept a0 is
    absorbed by the downstream trend fit, so the returned component is
    only the cos/sin sum and the coefs dict contains only a_j, b_j.
    """
    t = np.arange(len(y), dtype=float)
    X_harm = harmonic_design(t, period, k)
    X = np.column_stack([np.ones_like(t), X_harm])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    harm_coefs = beta[1:]
    fitted_harm = X_harm @ harm_coefs
    coefs = {}
    for j in range(1, k + 1):
        coefs[f"a{j}"] = float(harm_coefs[2 * (j - 1)])
        coefs[f"b{j}"] = float(harm_coefs[2 * (j - 1) + 1])
    return fitted_harm, coefs


def harmonic_at(t: np.ndarray, period: float, coefs: dict) -> np.ndarray:
    """Evaluate a fitted harmonic series at arbitrary integer time indices."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    j = 1
    while f"a{j}" in coefs:
        lam = 2.0 * np.pi * j / period
        out = out + coefs[f"a{j}"] * np.cos(lam * t) + coefs[f"b{j}"] * np.sin(lam * t)
        j += 1
    return out


# --- Polynomial trend -----------------------------------------------------

def fit_poly_trend(y: np.ndarray, degree: int):
    t = np.arange(len(y), dtype=float)
    X = np.column_stack([t ** d for d in range(degree + 1)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    coefs = {f"beta{d}": float(beta[d]) for d in range(degree + 1)}
    return fitted, coefs


def trend_at(t: np.ndarray, coefs: dict) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    d = 0
    while f"beta{d}" in coefs:
        out = out + coefs[f"beta{d}"] * (t ** d)
        d += 1
    return out


# --- Inverse transform: residual (log MW) -> MW ---------------------------

def reconstruct_mw(residuals, t_idx, trend_coefs, year_coefs, week_coefs,
                   d_year: float = 365, d_week: float = 7) -> np.ndarray:
    """residual + trend(t) + yearly(t) + weekly(t) -> exp -> MW."""
    log_load = (
        np.asarray(residuals, dtype=float)
        + trend_at(t_idx, trend_coefs)
        + harmonic_at(t_idx, d_year, year_coefs)
        + harmonic_at(t_idx, d_week, week_coefs)
    )
    return np.exp(log_load)


# --- ARMA helpers ---------------------------------------------------------

def aicc_from_aic(aic: float, k_params: int, n_obs: int) -> float:
    """AICC = AIC + 2 k (k+1) / (n - k - 1). B&D §5.5.2."""
    return float(aic + (2 * k_params * (k_params + 1)) / (n_obs - k_params - 1))


# --- Diagnostics (B&D §1.4, §1.6) -----------------------------------------

def ljung_box(residuals, lags=(20, 40), model_df: int = 0) -> pd.DataFrame:
    return acorr_ljungbox(
        np.asarray(residuals), lags=list(lags), return_df=True, model_df=model_df
    )


def diagnostic_plots(residuals, max_lag: int = 50, title_prefix: str = ""):
    fig, axes = plt.subplots(3, 1, figsize=(11, 9))
    y = np.asarray(residuals)
    idx = getattr(residuals, "index", np.arange(len(y)))
    axes[0].plot(idx, y, lw=0.6)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title(f"{title_prefix}Residuals")
    axes[0].set_xlabel("date")
    axes[0].set_ylabel("residual")

    plot_acf(y, lags=max_lag, ax=axes[1])
    axes[1].set_title(rf"{title_prefix}Sample ACF ($\pm 1.96/\sqrt{{n}}$)")
    axes[1].set_xlabel("lag (days)")

    plot_pacf(y, lags=max_lag, ax=axes[2], method="ywm")
    axes[2].set_title(f"{title_prefix}Sample PACF")
    axes[2].set_xlabel("lag (days)")

    fig.tight_layout()
    return fig
