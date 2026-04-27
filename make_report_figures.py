"""Regenerate the figures used in report_part_a.tex.

Reads time_series_60min_singleindex.csv, reruns the cleaning + ARMA(2,1)
fit + forecast (same as project_part_a.ipynb) and writes 3 PNGs to
figures/.
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

import utils

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})

CSV_PATH = "time_series_60min_singleindex.csv"
COL = "SE_3_load_actual_entsoe_transparency"

# ---- recreate the analysis ----------------------------------------------
raw = pd.read_csv(CSV_PATH, usecols=["utc_timestamp", COL], parse_dates=["utc_timestamp"])
raw = raw.set_index("utc_timestamp")[COL]
raw = raw.loc[raw.first_valid_index(): raw.last_valid_index()]

grp = raw.groupby(raw.index.floor("D"))
daily = grp.mean()
daily[grp.count() < 20] = np.nan
daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
daily = daily.interpolate(method="linear", limit_direction="both")
n = len(daily)

log_load = np.log(daily)
s_year_vals, year_coefs = utils.fit_harmonic(log_load.values, 365, 2)
s_year = pd.Series(s_year_vals, index=log_load.index)
log_des_year = log_load - s_year
s_week_vals, week_coefs = utils.fit_harmonic(log_des_year.values, 7, 3)
s_week = pd.Series(s_week_vals, index=log_load.index)
log_deseason = log_des_year - s_week

trend_vals, trend_coefs = utils.fit_poly_trend(log_deseason.values, degree=2)
trend = pd.Series(trend_vals, index=log_load.index)
residuals = log_deseason - trend

# ARMA(2,1) on full and on training window
H = 30
train_resid = residuals.iloc[:-H]
test_resid = residuals.iloc[-H:]
train_daily = daily.iloc[:-H]
test_daily = daily.iloc[-H:]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit_full = ARIMA(residuals.values, order=(2, 0, 1), trend="c").fit()
    refit = ARIMA(train_resid.values, order=(2, 0, 1), trend="c").fit()

arma_resid = pd.Series(fit_full.resid, index=residuals.index)

fc = refit.get_forecast(steps=H)
fc_mean = np.asarray(fc.predicted_mean)
fc_ci = np.asarray(fc.conf_int(alpha=0.05))

t_future = np.arange(len(train_resid), n, dtype=float)
add_future = (
    utils.trend_at(t_future, trend_coefs)
    + utils.harmonic_at(t_future, 365, year_coefs)
    + utils.harmonic_at(t_future, 7, week_coefs)
)
fc_mw = np.exp(fc_mean + add_future)
fc_lo = np.exp(fc_ci[:, 0] + add_future)
fc_hi = np.exp(fc_ci[:, 1] + add_future)

# ---- Figure 1: cleaning / decomposition ---------------------------------
fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.6), sharex=True)
axes[0].plot(daily.index, daily.values, lw=0.6, color="C0")
axes[0].set_ylabel("MW")
axes[0].set_title(r"(a) Daily mean load $X_t$ for SE\_3 (n = %d)" % n)

deterministic = trend + s_year + s_week
axes[1].plot(log_load.index, log_load.values, lw=0.5, color="0.55", label=r"$\log X_t$")
axes[1].plot(deterministic.index, deterministic.values, lw=1.2, color="C3",
             label=r"$\hat m_t + \hat s_t^{(\mathrm{year})} + \hat s_t^{(\mathrm{week})}$")
axes[1].set_ylabel(r"log MW")
axes[1].set_title(r"(b) Log load and fitted deterministic component")
axes[1].legend(loc="lower left", framealpha=0.9)

axes[2].plot(residuals.index, residuals.values, lw=0.5, color="C2")
axes[2].axhline(0, color="k", lw=0.4)
axes[2].set_ylabel("residual")
axes[2].set_xlabel("date")
axes[2].set_title(r"(c) Cleaning residuals $\hat Y_t = \log X_t - \hat m_t - \hat s_t^{(\mathrm{year})} - \hat s_t^{(\mathrm{week})}$")

fig.tight_layout()
fig.savefig("figures/fig_decomposition.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 2: diagnostics ----------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
plot_acf(residuals.values, lags=50, ax=axes[0, 0])
axes[0, 0].set_title(r"(a) ACF of cleaning residuals $\hat Y_t$")
axes[0, 0].set_xlabel("lag (days)")

plot_pacf(residuals.values, lags=50, ax=axes[0, 1], method="ywm")
axes[0, 1].set_title(r"(b) PACF of cleaning residuals")
axes[0, 1].set_xlabel("lag (days)")

plot_acf(arma_resid.values, lags=50, ax=axes[1, 0])
axes[1, 0].set_title(r"(c) ACF of ARMA(2,1) residuals")
axes[1, 0].set_xlabel("lag (days)")

axes[1, 1].plot(arma_resid.index, arma_resid.values, lw=0.4, color="C2")
axes[1, 1].axhline(0, color="k", lw=0.4)
axes[1, 1].set_title(r"(d) ARMA(2,1) residual time series")
axes[1, 1].set_xlabel("date")
axes[1, 1].set_ylabel("residual")

fig.tight_layout()
fig.savefig("figures/fig_diagnostics.png", bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: forecast --------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 3.2))
last_90 = train_daily.iloc[-90:]
ax.plot(last_90.index, last_90.values, lw=0.8, color="C0", label="training (last 90 d)")
ax.plot(test_daily.index, test_daily.values, lw=0.8, color="k",
        marker="o", markersize=3, label="held-out actual")
ax.plot(test_daily.index, fc_mw, lw=1.4, color="C3", label="ARMA(2,1) forecast")
ax.fill_between(test_daily.index, fc_lo, fc_hi, color="C3", alpha=0.18, label="95% PI")
ax.set_xlabel("date")
ax.set_ylabel("load (MW)")
ax.set_title("30-day forecast for SE_3, ARMA(2,1) on cleaning residuals")
ax.legend(loc="lower left", framealpha=0.9)
fig.tight_layout()
fig.savefig("figures/fig_forecast.png", bbox_inches="tight")
plt.close(fig)

# ---- Console summary so we know the report numbers ---------------------
rmse = float(np.sqrt(np.mean((fc_mw - test_daily.values) ** 2)))
mae = float(np.mean(np.abs(fc_mw - test_daily.values)))
cov = float(np.mean((test_daily.values >= fc_lo) & (test_daily.values <= fc_hi)))
print(f"RMSE={rmse:.1f}  MAE={mae:.1f}  coverage={cov:.2f}")
print("Figures written to figures/")
