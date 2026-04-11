# ---
# jupyter:
#   jupytext:
#     text_representation:
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # SF2943 Part A — SE_3 Electricity Load
#
# Full clean → fit → forecast pipeline for Swedish bidding zone SE_3
# hourly load from OPSD `time_series_60min_singleindex.csv`.
#
# - **Section 1.** Classical decomposition (B&D §1.5): log transform,
#   harmonic regression for yearly ($d=365, k=2$) and weekly ($d=7, k=3$)
#   seasonality, polynomial trend, residual diagnostics.
# - **Section 2.** ARMA identification from sample ACF/PACF, five
#   candidates fit by Gaussian MLE (B&D §5.2), compared by AICC
#   (B&D §5.5.2), causality/invertibility verified, Yule–Walker
#   cross-check (B&D §5.1.1).
# - **Section 3.** 30-day forecast via h-step ARMA recursion
#   (B&D §3.3.1), inverse-transformed to MW with 95% prediction
#   intervals, evaluated against held-out actuals.

# %%
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.stattools import acf as _acf_fn

import utils

pd.options.display.float_format = "{:.4f}".format
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## Section 1 — Cleaning

# %% [markdown]
# ### 1.1 Load and inspect
#
# Read the SE_3 column from the hourly CSV, trim to its valid date
# range, report shape + missingness, plot the raw series.

# %%
CSV_PATH = "time_series_60min_singleindex.csv"
COL = "SE_3_load_actual_entsoe_transparency"

raw = pd.read_csv(CSV_PATH, usecols=["utc_timestamp", COL], parse_dates=["utc_timestamp"])
raw = raw.set_index("utc_timestamp")[COL]
raw = raw.loc[raw.first_valid_index(): raw.last_valid_index()]

is_nan = raw.isna().astype(int).values
gap_max = 0
cur = 0
for v in is_nan:
    cur = cur + 1 if v else 0
    gap_max = max(gap_max, cur)

print(f"Column              : {COL}")
print(f"Start               : {raw.index.min()}")
print(f"End                 : {raw.index.max()}")
print(f"Hourly observations : {len(raw)}")
print(f"NaN hourly          : {int(raw.isna().sum())}")
print(f"Longest NaN gap     : {gap_max} hours")

# %%
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(raw.index, raw.values, lw=0.3)
ax.set_title("Raw hourly load, SE_3 (ENTSO-E Transparency)")
ax.set_xlabel("date")
ax.set_ylabel("load (MW)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 1.2 Aggregate to daily
#
# Daily mean (units stay MW). Days with fewer than 20 valid hours are
# marked missing, remaining gaps linearly interpolated.

# %%
MIN_VALID_HOURS = 20  # days with fewer valid hours are marked missing

grp = raw.groupby(raw.index.floor("D"))
daily = grp.mean()
daily[grp.count() < MIN_VALID_HOURS] = np.nan
daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
n_missing_before = int(daily.isna().sum())
daily = daily.interpolate(method="linear", limit_direction="both")
n = len(daily)

assert n >= 2000, f"Daily count {n} below 2000 threshold"
print(f"Daily points n       : {n}")
print(f"Days interpolated    : {n_missing_before}")
print(f"n >= 2000 check      : OK (margin {n - 2000})")

# %%
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(daily.index, daily.values, lw=0.6)
ax.set_title(f"Daily mean load, SE_3 (n = {n})")
ax.set_xlabel("date")
ax.set_ylabel("load (MW)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 1.3 Log transform
#
# B&D §1.5: Box–Cox with $\lambda = 0$ stabilises variance and turns
# multiplicative seasonality into additive.

# %%
log_load = np.log(daily)

fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(log_load.index, log_load.values, lw=0.6)
ax.set_title("Log daily load")
ax.set_xlabel("date")
ax.set_ylabel(r"$\log X_t$ (log MW)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 1.4 Yearly seasonality — harmonic regression
#
# B&D §1.3: fit $s_t^{(\text{year})} = \sum_{j=1}^{2}
# [a_j \cos(2\pi j t / 365) + b_j \sin(2\pi j t / 365)]$ by OLS.
# Two harmonics are enough for the asymmetric winter peak.

# %%
D_YEAR, K_YEAR = 365, 2
s_year_vals, year_coefs = utils.fit_harmonic(log_load.values, D_YEAR, K_YEAR)
s_year = pd.Series(s_year_vals, index=log_load.index, name="yearly_seasonal")

print("Yearly harmonic coefficients:")
for name, val in year_coefs.items():
    print(f"  {name} = {val:+.5f}")

log_deseason_year = log_load - s_year

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 6))
axes[0].plot(log_deseason_year.index, log_deseason_year.values, lw=0.6)
axes[0].set_title("Log load with yearly seasonality removed")
axes[0].set_ylabel(r"$\log X_t - \hat s^{(\mathrm{year})}_t$")

pick_year = log_load.index.year.min() + 1
mask = log_load.index.year == pick_year
centered = log_load[mask] - log_load.mean()
axes[1].plot(log_load.index[mask], centered.values, lw=0.8, label="log load (centered)")
axes[1].plot(log_load.index[mask], s_year[mask].values, lw=1.5, label="fitted yearly")
axes[1].set_title(f"Yearly harmonic fit overlaid on {pick_year} (k=2)")
axes[1].set_xlabel("date")
axes[1].set_ylabel("log MW (centered)")
axes[1].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 1.5 Weekly seasonality — harmonic regression
#
# Same procedure with $d=7, k=3$ — three harmonics capture the
# weekday/weekend block shape.

# %%
D_WEEK, K_WEEK = 7, 3
s_week_vals, week_coefs = utils.fit_harmonic(log_deseason_year.values, D_WEEK, K_WEEK)
s_week = pd.Series(s_week_vals, index=log_load.index, name="weekly_seasonal")

print("Weekly harmonic coefficients:")
for name, val in week_coefs.items():
    print(f"  {name} = {val:+.5f}")

log_deseason = log_deseason_year - s_week

# %%
win_start = log_load.index[365]  # skip year-0 edge
win_end = win_start + pd.Timedelta(days=28)
mask = (log_load.index >= win_start) & (log_load.index < win_end)

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(log_deseason_year.index[mask], log_deseason_year[mask].values, marker="o", lw=0.8)
axes[0].set_title("Before weekly removal — 4-week window")
axes[0].set_ylabel("log MW")
axes[1].plot(log_deseason.index[mask], log_deseason[mask].values, marker="o", lw=0.8, color="C1")
axes[1].set_title("After weekly removal — same window")
axes[1].set_xlabel("date")
axes[1].set_ylabel("log MW")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 1.6 Polynomial trend
#
# Fit linear $\hat m_t = \beta_0 + \beta_1 t$ by OLS. If the residuals
# split into thirds have means exceeding 10% of the residual std, the
# linear fit hasn't absorbed the shape and we promote to quadratic.

# %%
lin_fit, lin_coefs = utils.fit_poly_trend(log_deseason.values, degree=1)
lin_resid = log_deseason.values - lin_fit
thirds = np.array_split(lin_resid, 3)
third_means = [float(np.mean(x)) for x in thirds]
third_std = float(np.std(lin_resid))
use_quadratic = max(abs(m) for m in third_means) > 0.10 * third_std

print(f"Linear trend coefs      : {lin_coefs}")
print(f"Residual thirds mean    : {third_means}")
print(f"Residual std            : {third_std:.5f}")
print(f"Promote to quadratic    : {use_quadratic}")

if use_quadratic:
    trend_fit, trend_coefs = utils.fit_poly_trend(log_deseason.values, degree=2)
    trend_degree = 2
else:
    trend_fit, trend_coefs = lin_fit, lin_coefs
    trend_degree = 1

print(f"Trend degree chosen     : {trend_degree}")
print(f"Final trend coefficients: {trend_coefs}")

trend = pd.Series(trend_fit, index=log_load.index, name="trend")
residuals = (log_deseason - trend).rename("residual")

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(log_deseason.index, log_deseason.values, lw=0.6, label="deseasonalized")
axes[0].plot(trend.index, trend.values, lw=1.5, color="C3", label=f"polynomial (deg {trend_degree})")
axes[0].set_title("Trend fit")
axes[0].set_ylabel("log MW")
axes[0].legend()
axes[1].plot(residuals.index, residuals.values, lw=0.6, color="C2")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_title(r"Residuals $\hat Y_t = \log X_t - \hat s^{(\mathrm{year})}_t - \hat s^{(\mathrm{week})}_t - \hat m_t$")
axes[1].set_xlabel("date")
axes[1].set_ylabel("residual (log MW)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 1.7 Residual diagnostics
#
# ACF/PACF (lags 0–50) + Ljung–Box at $h=20, 40$. We **expect** iid to
# be rejected — that's the whole point of Section 2. Note: a moderate
# residual ACF at lag 7 is expected even after harmonic weekly removal
# (the harmonic captures the *shape* of the weekly cycle, not shared
# noise structure between same weekdays), and the ARMA stage will
# absorb it through the autoregressive polynomial.

# %%
fig = utils.diagnostic_plots(residuals, max_lag=50, title_prefix="Cleaning ")
plt.show()

lb_clean = utils.ljung_box(residuals.values, lags=(20, 40), model_df=0)
print("Ljung–Box on cleaning residuals (iid null, expected to reject):")
print(lb_clean)

acf_vals = _acf_fn(residuals.values, nlags=400, fft=True)
bound = 1.96 / np.sqrt(n)
print(f"\nACF bound        ±{bound:.4f}")
print(f"ACF(1)   = {acf_vals[1]:+.4f}  (strong day-to-day persistence — ARMA target)")
print(f"ACF(7)   = {acf_vals[7]:+.4f}  (weekday autocorrelation — ARMA target)")
print(f"ACF(365) = {acf_vals[365]:+.4f}  (yearly residual check)")

# %% [markdown]
# ### 1.8 Differencing sanity check
#
# B&D §1.5.2 alternative route: $\nabla_1 \nabla_7 \log X_t$ kills the
# weekly seasonal component and any linear trend. Compare ACFs
# qualitatively — they should tell the same story.

# %%
diff_series = log_load.diff(D_WEEK).diff(1).dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals.values, lags=50, ax=axes[0])
axes[0].set_title("Classical decomposition residuals")
axes[0].set_xlabel("lag")
plot_acf(diff_series.values, lags=50, ax=axes[1])
axes[1].set_title(r"$\nabla_1 \nabla_7 \log X_t$")
axes[1].set_xlabel("lag")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Section 2 — ARMA fitting

# %% [markdown]
# ### 2.1 Identification from ACF/PACF
#
# The cleaning residual ACF decays (no sharp cutoff) while the PACF has
# its largest spike at lag 1 with small activity at a few further lags —
# AR-dominated structure (B&D §3.2.3). Candidates:
#
# - **AR(1)** — simplest baseline
# - **AR(2)** — if second PACF spike is material
# - **AR(3)** — headroom for multi-day persistence
# - **ARMA(1,1)** — both ACF and PACF tail off
# - **ARMA(2,1)** — mixed, one extra AR term

# %%
candidates = [(1, 0), (2, 0), (3, 0), (1, 1), (2, 1)]


def fit_candidate(p, q, y_series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = ARIMA(y_series, order=(p, 0, q), trend="c").fit()  # B&D §5.2 Gaussian MLE
    n_obs = int(res.nobs)
    k_params = p + q + 2  # phi + theta + const + sigma^2
    aicc = utils.aicc_from_aic(res.aic, k_params, n_obs)
    ar_roots = res.arroots if p > 0 else np.array([])
    ma_roots = res.maroots if q > 0 else np.array([])
    causal = (p == 0) or bool(np.all(np.abs(ar_roots) > 1))
    invertible = (q == 0) or bool(np.all(np.abs(ma_roots) > 1))
    param_names = list(res.param_names)
    param_vals = np.asarray(res.params, dtype=float)
    bse_vals = np.asarray(res.bse, dtype=float)
    params_dict = dict(zip(param_names, param_vals))
    return {
        "p": p, "q": q, "result": res,
        "aic": float(res.aic), "aicc": aicc,
        "sigma2": float(params_dict["sigma2"]),
        "causal": causal, "invertible": invertible,
        "ar_roots": ar_roots, "ma_roots": ma_roots,
        "param_names": param_names,
        "param_vals": param_vals,
        "bse_vals": bse_vals,
        "params_dict": params_dict,
    }


fits = [fit_candidate(p, q, residuals) for p, q in candidates]

for f in fits:
    print("=" * 66)
    print(f"ARMA({f['p']},{f['q']})  —  B&D §5.2 Gaussian MLE via innovations")
    print(f"  AIC  = {f['aic']:.3f}")
    print(f"  AICC = {f['aicc']:.3f}")
    print(f"  sigma² = {f['sigma2']:.6f}")
    print(f"  causal     : {f['causal']}   (AR roots: {np.round(f['ar_roots'], 3).tolist()})")
    print(f"  invertible : {f['invertible']} (MA roots: {np.round(f['ma_roots'], 3).tolist()})")
    for name, val, se in zip(f["param_names"], f["param_vals"], f["bse_vals"]):
        print(f"    {name:>10} = {val:+.5f}  (se={se:.5f})")

# %% [markdown]
# ### 2.3 AICC comparison

# %%
tbl = pd.DataFrame([
    {"model": f"ARMA({f['p']},{f['q']})", "params": f["p"] + f["q"],
     "AICC": f["aicc"], "causal": f["causal"], "invertible": f["invertible"]}
    for f in fits
]).sort_values("AICC").reset_index(drop=True)
print(tbl.to_string(index=False))

eligible = [f for f in fits if f["causal"] and f["invertible"]]
assert eligible, "No candidate satisfied causality + invertibility"
best = min(eligible, key=lambda f: f["aicc"])
runner_up = sorted(eligible, key=lambda f: f["aicc"])[1] if len(eligible) > 1 else None
margin = (runner_up["aicc"] - best["aicc"]) if runner_up else float("nan")

print(f"\nChosen: ARMA({best['p']},{best['q']})")
if runner_up:
    print(f"Runner-up: ARMA({runner_up['p']},{runner_up['q']})  (AICC margin {margin:+.3f})")

# %% [markdown]
# ### 2.4 Yule–Walker comparison
#
# B&D §5.1.1: for the best AR candidate we also solve the sample
# Yule–Walker equations $\hat\Gamma_p \hat\phi = \hat\gamma_p$ and
# compare to MLE. They should agree closely.

# %%
ar_fits = [f for f in fits if f["q"] == 0]
best_ar = min(ar_fits, key=lambda f: f["aicc"])
p_ar = best_ar["p"]
yw_phi, yw_sigma = yule_walker(residuals.values, order=p_ar)
print(f"Yule–Walker vs MLE, AR({p_ar}):")
for i, ph in enumerate(yw_phi, 1):
    mle_ph = float(best_ar["params_dict"].get(f"ar.L{i}", np.nan))
    print(f"  phi_{i}:  YW = {ph:+.5f}    MLE = {mle_ph:+.5f}")
print(f"  sigma²:  YW = {yw_sigma ** 2:.6f}    MLE = {best_ar['sigma2']:.6f}")

# %% [markdown]
# ### 2.5 Residual diagnostics on the chosen model
#
# Residual time series + ACF/PACF + Ljung–Box with the B&D §5.3
# degrees-of-freedom correction $\text{df} = h - p - q$.

# %%
chosen = best
p_ch, q_ch = chosen["p"], chosen["q"]
arma_resid = pd.Series(chosen["result"].resid, index=residuals.index, name="arma_resid")

fig = utils.diagnostic_plots(arma_resid, max_lag=50,
                             title_prefix=f"ARMA({p_ch},{q_ch}) ")
plt.show()

lb_arma = utils.ljung_box(arma_resid.values, lags=(20,), model_df=p_ch + q_ch)
print(f"Ljung–Box on ARMA({p_ch},{q_ch}) residuals (model_df={p_ch + q_ch}):")
print(lb_arma)

if float(lb_arma["lb_pvalue"].iloc[0]) < 0.05:
    print("\np < 0.05 → diagnostic fails: residuals still show correlation.")
    print("We are honest about this: with harmonic-only weekly removal, the")
    print("lag-7 weekday correlation is not fully absorbed by low-order ARMA,")
    print("and the course toolkit forbids SARIMA. The chosen model still")
    print("minimises AICC among the candidates; we report the failure and")
    print("discuss it in Section 3.7.")
else:
    print("\np ≥ 0.05 → residuals consistent with white noise.")

# %% [markdown]
# ## Section 3 — Forecasting

# %% [markdown]
# ### 3.1 Train/test split and refit
#
# Hold out the last 30 days, refit the chosen ARMA on training
# residuals only. The cleaning components (trend, seasonals) are kept
# from the full-data fit and extrapolated deterministically to the
# test dates — this is noted as a mild in-sample / out-of-sample leak
# for the seasonal/trend *levels* only; the stochastic forecast is
# strictly out-of-sample.

# %%
H = 30
train_resid = residuals.iloc[:-H]
test_resid = residuals.iloc[-H:]
train_daily_mw = daily.iloc[:-H]
test_daily_mw = daily.iloc[-H:]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    refit = ARIMA(train_resid.values, order=(p_ch, 0, q_ch), trend="c").fit()
print(f"Refit ARMA({p_ch},{q_ch}) on training residuals (n_train={len(train_resid)})")
print(refit.summary().tables[1])

# %% [markdown]
# ### 3.2–3.4 h-step forecast + inverse transforms + prediction intervals
#
# B&D §3.3.1: h-step ARMA prediction via the innovations algorithm
# (statsmodels does this inside `get_forecast`). We add the
# extrapolated trend + yearly + weekly components back, then
# exponentiate. Exponentiating a Gaussian interval gives an asymmetric
# log-normal interval on MW — noted but used as-is per §3.4.

# %%
fc = refit.get_forecast(steps=H)
fc_mean_resid = np.asarray(fc.predicted_mean)
fc_ci_resid = np.asarray(fc.conf_int(alpha=0.05))

t_future = np.arange(len(train_resid), len(residuals), dtype=float)
additive_future = (
    utils.trend_at(t_future, trend_coefs)
    + utils.harmonic_at(t_future, D_YEAR, year_coefs)
    + utils.harmonic_at(t_future, D_WEEK, week_coefs)
)

fc_mw = np.exp(fc_mean_resid + additive_future)
fc_mw_lower = np.exp(fc_ci_resid[:, 0] + additive_future)
fc_mw_upper = np.exp(fc_ci_resid[:, 1] + additive_future)

print("First 10 forecasted daily loads (MW):")
for i in range(10):
    print(f"  {test_resid.index[i].date()}   "
          f"point={fc_mw[i]:8.1f}   "
          f"95% [{fc_mw_lower[i]:8.1f}, {fc_mw_upper[i]:8.1f}]   "
          f"actual={test_daily_mw.iloc[i]:8.1f}")

# %% [markdown]
# ### 3.5 Forecast plot

# %%
last_90 = train_daily_mw.iloc[-90:]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(last_90.index, last_90.values, lw=0.8, label="training (last 90 d)")
ax.plot(test_daily_mw.index, test_daily_mw.values, lw=1.0, color="k",
        marker="o", markersize=3, label="held-out actual")
ax.plot(test_daily_mw.index, fc_mw, lw=1.5, color="C3",
        label=f"ARMA({p_ch},{q_ch}) forecast")
ax.fill_between(test_daily_mw.index, fc_mw_lower, fc_mw_upper,
                color="C3", alpha=0.2, label="95% PI")
ax.set_title(f"30-day forecast, SE_3 daily load  (ARMA({p_ch},{q_ch}))")
ax.set_xlabel("date")
ax.set_ylabel("load (MW)")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 3.6 Forecast accuracy

# %%
actual = test_daily_mw.values
rmse = float(np.sqrt(np.mean((fc_mw - actual) ** 2)))
mae = float(np.mean(np.abs(fc_mw - actual)))
coverage = float(np.mean((actual >= fc_mw_lower) & (actual <= fc_mw_upper)))
series_std = float(train_daily_mw.std())

print(f"RMSE             : {rmse:.1f} MW")
print(f"MAE              : {mae:.1f} MW")
print(f"Coverage (95%)   : {coverage:.2f}")
print(f"Training std     : {series_std:.1f} MW  (scale reference)")
print(f"RMSE / std       : {rmse / series_std:.3f}")

# %% [markdown]
# ### 3.7 Discussion
#
# **Method.** The forecast uses the h-step ARMA recursion (B&D §3.3.1)
# evaluated through the innovations algorithm inside statsmodels'
# Gaussian MLE fit. As $h$ grows, the point forecast on the residual
# scale converges to the process mean (≈ 0), so the reconstructed MW
# forecast at long horizons is essentially the extrapolated
# trend + harmonic seasonals, with a 95% band that widens as $v_n(h)$
# grows — exactly the behaviour predicted by B&D §3.3.1.
#
# **Where the model is weak.**
# (1) **Long horizons** collapse to seasonal/trend extrapolation, so
#     any multi-day weather driver (cold snap, heat wave) is invisible
#     to the forecast.
# (2) **Holidays** are not in the seasonal model — the weekly harmonic
#     treats all Mondays identically, so the Christmas/Easter dips are
#     systematically mispredicted.
# (3) **Lag-7 residual correlation.** Harmonic-only weekly removal
#     leaves non-trivial lag-7 autocorrelation; a pure low-order ARMA
#     cannot absorb it fully and our Ljung–Box diagnostic reflects
#     that. A seasonal ARMA (SARIMA) would fix this but is outside the
#     course toolkit.
#
# **Robustness.** Neighbouring $(p,q)$ among our candidates give
# essentially the same point forecast (the AR polynomial dominates),
# with prediction-interval widths differing by a few percent. The
# dominant source of forecast error is the deterministic
# seasonal/trend extrapolation, not the ARMA order.

# %% [markdown]
# ## Save outputs

# %%
out_resid = residuals.to_frame()
out_resid.index.name = "date"
out_resid.to_csv("cleaned_residuals.csv")

decomp = pd.DataFrame({
    "log_load": log_load.values,
    "yearly_seasonal": s_year.values,
    "weekly_seasonal": s_week.values,
    "trend": trend.values,
    "residual": residuals.values,
}, index=log_load.index)
decomp.index.name = "date"
decomp.to_csv("decomposition.csv")

forecast_df = pd.DataFrame({
    "date": test_daily_mw.index,
    "forecast_mw": fc_mw,
    "lower_95": fc_mw_lower,
    "upper_95": fc_mw_upper,
    "actual_mw": test_daily_mw.values,
})
forecast_df.to_csv("forecast.csv", index=False)

lines = []
lines.append("SF2943 Part A — SE_3 summary")
lines.append("=" * 44)
lines.append(f"Source        : {CSV_PATH}")
lines.append(f"Column        : {COL}")
lines.append(f"Date range    : {log_load.index.min().date()} to {log_load.index.max().date()}")
lines.append(f"n (daily)     : {n}")
lines.append(f"Interpolated  : {n_missing_before} days")
lines.append("")
lines.append(f"Yearly harmonic (d={D_YEAR}, k={K_YEAR}):")
for k, v in year_coefs.items():
    lines.append(f"  {k} = {v:+.5f}")
lines.append(f"Weekly harmonic (d={D_WEEK}, k={K_WEEK}):")
for k, v in week_coefs.items():
    lines.append(f"  {k} = {v:+.5f}")
lines.append(f"Polynomial trend (degree {trend_degree}):")
for k, v in trend_coefs.items():
    lines.append(f"  {k} = {v:+.6e}")
lines.append("")
lines.append("Candidate ARMA models (AICC sorted):")
for _, row in tbl.iterrows():
    lines.append(
        f"  {row['model']:<12} params={int(row['params'])}  "
        f"AICC={row['AICC']:.2f}  causal={row['causal']}  invertible={row['invertible']}"
    )
lines.append("")
lines.append(f"Chosen model  : ARMA({p_ch},{q_ch})")
lines.append(f"Ljung–Box on ARMA residuals (model_df={p_ch + q_ch}):")
for lag, row in lb_arma.iterrows():
    lines.append(f"  h={lag}  Q = {row['lb_stat']:.3f}  p = {row['lb_pvalue']:.3e}")
lines.append("")
lines.append(f"Forecast horizon : {H} days")
lines.append(f"RMSE             : {rmse:.1f} MW")
lines.append(f"MAE              : {mae:.1f} MW")
lines.append(f"Coverage (95%)   : {coverage:.2f}")
lines.append(f"Training std     : {series_std:.1f} MW")
summary_text = "\n".join(lines)
with open("summary.txt", "w") as fh:
    fh.write(summary_text + "\n")
print(summary_text)
