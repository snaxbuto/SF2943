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
# # SF2943 Part A — Data Cleaning
#
# Cleaning pipeline for the Swedish bidding zone **SE_3** hourly load from
# OPSD `time_series_60min_singleindex.csv`. Goal: produce a stationary
# residual series $\hat Y_t$ and diagnostic plots. No ARMA fitting here —
# that's the next stage.
#
# Pipeline (all steps reference the math in `cleaning_spec.md`):
#
# 0. Load & inspect raw hourly data.
# 1. Aggregate hourly → daily (mean). Confirm $n \geq 2000$.
# 2. Log transform (Box–Cox $\lambda=0$, §3).
# 3. Remove yearly seasonality by harmonic regression, $d=365, k=2$ (§4).
# 4. Remove weekly seasonality by harmonic regression, $d=7, k=3$ (§4).
# 5. Remove trend by polynomial OLS (§5).
# 6. Diagnostics: residual plot, ACF, PACF, Ljung–Box (§7–8).
# 7. Differencing sanity check ($\nabla_7 \nabla_1$, §6).
# 8. Save CSVs + report.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

import utils

pd.options.display.float_format = "{:.4f}".format
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## Step 0 — Load and inspect
#
# Read the SE_3 column, parse the `utc_timestamp` index, and report basic
# shape, missing-value counts, and the longest contiguous NaN gap.

# %%
CSV_PATH = "time_series_60min_singleindex.csv"
COL = "SE_3_load_actual_entsoe_transparency"

raw = pd.read_csv(
    CSV_PATH,
    usecols=["utc_timestamp", COL],
    parse_dates=["utc_timestamp"],
)
raw = raw.set_index("utc_timestamp")[COL]
# Trim to the valid range (avoids leading/trailing NaNs from other cols in the file)
raw = raw.loc[raw.first_valid_index() : raw.last_valid_index()]

print(f"Column           : {COL}")
print(f"Start            : {raw.index.min()}")
print(f"End              : {raw.index.max()}")
print(f"Hourly rows      : {len(raw)}")
print(f"NaN hourly rows  : {int(raw.isna().sum())}")

# Longest run of consecutive NaNs in hourly data.
is_nan = raw.isna().astype(int).values
gap_max = 0
cur = 0
for v in is_nan:
    cur = cur + 1 if v else 0
    gap_max = max(gap_max, cur)
print(f"Longest NaN gap  : {gap_max} hours")

# %%
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(raw.index, raw.values, lw=0.3)
ax.set_title("Raw hourly load, SE_3 (ENTSO-E Transparency)")
ax.set_xlabel("date")
ax.set_ylabel("load (MW)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step 1 — Aggregate to daily
#
# Aggregate hourly → daily by **mean**, so units stay MW (an energy sum
# would be MWh and grow with day length). Days with fewer than 20 valid
# hours are marked missing, then linearly interpolated. Finally confirm
# $n \geq 2000$ as the spec requires.

# %%
MIN_VALID_HOURS = 20  # days with fewer valid hours are treated as missing

def daily_mean_strict(hourly: pd.Series, min_hours: int) -> pd.Series:
    grp = hourly.groupby(hourly.index.floor("D"))
    means = grp.mean()
    counts = grp.count()
    means[counts < min_hours] = np.nan
    return means

daily = daily_mean_strict(raw, MIN_VALID_HOURS)
daily.index = pd.DatetimeIndex(daily.index).tz_localize(None)
n_missing_before = int(daily.isna().sum())
daily = daily.interpolate(method="linear", limit_direction="both")
n_total = len(daily)

print(f"Daily points n   : {n_total}")
print(f"Days interpolated: {n_missing_before}")
assert n_total >= 2000, f"Daily count {n_total} below required floor of 2000"
print(f"n >= 2000 check  : OK (margin {n_total - 2000})")

# %%
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(daily.index, daily.values, lw=0.6)
ax.set_title(f"Daily mean load, SE_3 (n = {n_total})")
ax.set_xlabel("date")
ax.set_ylabel("load (MW)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step 2 — Log transform
#
# Apply natural log (Box–Cox with $\lambda=0$, §3). This stabilises the
# variance and turns the multiplicative winter-summer amplitude into an
# additive seasonal component, which is what classical decomposition
# assumes.

# %%
log_load = np.log(daily)
log_load.name = "log_load"

fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(log_load.index, log_load.values, lw=0.6)
ax.set_title("Log daily load (log MW)")
ax.set_xlabel("date")
ax.set_ylabel(r"$\log X_t$")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step 3 — Yearly seasonality (harmonic regression)
#
# Fit $\log X_t$ to $\cos(2\pi j t / 365) + \sin(2\pi j t / 365)$ for
# $j = 1, 2$ by OLS. Two harmonics capture the asymmetric winter peak
# without over-fitting. Subtract the fitted mean-zero seasonal component.

# %%
D_YEAR = 365
K_YEAR = 2  # fundamental + first harmonic

y_hat_year, coefs_year = utils.fit_harmonic(log_load.values, D_YEAR, K_YEAR)
s_year = pd.Series(y_hat_year, index=log_load.index, name="yearly_seasonal")

print("Yearly harmonic coefficients (log MW):")
for k, v in coefs_year.items():
    print(f"  {k:>3} = {v:+.5f}")

log_deseason_year = log_load - s_year

# %%
fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(log_deseason_year.index, log_deseason_year.values, lw=0.6)
ax.set_title("Log load after removing yearly seasonality")
ax.set_xlabel("date")
ax.set_ylabel(r"$\log X_t - \hat s^{(\mathrm{year})}_t$")
fig.tight_layout()
plt.show()

# %%
# Overlay fit on a single representative year to verify shape.
year_pick = log_load.index.year.min() + 1  # skip year-0 edge effects
mask = log_load.index.year == year_pick
fig, ax = plt.subplots(figsize=(11, 3))
# Plot the mean-removed log series (so the harmonic fit, also mean-removed,
# overlays cleanly on the same vertical scale).
centered = log_load[mask] - log_load.mean()
ax.plot(log_load.index[mask], centered.values, lw=0.8, label="log load (centered)")
ax.plot(log_load.index[mask], s_year[mask].values, lw=1.5, label="fitted yearly")
ax.set_title(f"Yearly harmonic fit overlaid on {year_pick} (k={K_YEAR})")
ax.set_xlabel("date")
ax.set_ylabel("log MW (centered)")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step 4 — Weekly seasonality (harmonic regression)
#
# Same procedure with $d = 7$, $k = 3$ — three harmonics give enough
# resolution for the weekday/weekend block shape. Fitted against the
# yearly-deseasonalised series and subtracted.

# %%
D_WEEK = 7
K_WEEK = 3

y_hat_week, coefs_week = utils.fit_harmonic(log_deseason_year.values, D_WEEK, K_WEEK)
s_week = pd.Series(y_hat_week, index=log_load.index, name="weekly_seasonal")

print("Weekly harmonic coefficients (log MW):")
for k, v in coefs_week.items():
    print(f"  {k:>3} = {v:+.5f}")

log_deseason = log_deseason_year - s_week

# %%
# 4-week window comparison, before vs after weekly removal.
win_start = log_load.index[365]  # skip first year to dodge any edge artefacts
win_end = win_start + pd.Timedelta(days=28)
mask = (log_load.index >= win_start) & (log_load.index < win_end)

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(log_deseason_year.index[mask], log_deseason_year[mask].values, marker="o", lw=0.8)
axes[0].set_title("Before weekly removal (yearly already gone) — 4-week window")
axes[0].set_ylabel("log MW")
axes[1].plot(log_deseason.index[mask], log_deseason[mask].values, marker="o", lw=0.8, color="C1")
axes[1].set_title("After weekly removal — same window")
axes[1].set_xlabel("date")
axes[1].set_ylabel("log MW")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step 5 — Trend (polynomial regression)
#
# Start with a **linear** trend $\hat m_t = \beta_0 + \beta_1 t$. If the
# post-detrend residuals visibly curve, promote to quadratic and use that
# instead. The choice is printed.

# %%
lin_fit, lin_coefs = utils.fit_poly_trend(log_deseason.values, degree=1)
lin_resid = log_deseason.values - lin_fit

# Curvature check: split residuals into thirds; if means deviate, prefer quadratic.
thirds = np.array_split(lin_resid, 3)
third_means = [float(np.mean(x)) for x in thirds]
third_std = float(np.std(lin_resid))
print(f"Linear trend coefs   : {lin_coefs}")
print(f"Residual thirds mean : {third_means}")
print(f"Residual std         : {third_std:.5f}")

# Heuristic: if the absolute mean of any third exceeds ~10% of the overall std,
# the linear fit hasn't captured the shape well enough — go quadratic.
CURVATURE_TOL = 0.10
use_quadratic = max(abs(m) for m in third_means) > CURVATURE_TOL * third_std
print(f"Curvature exceeds {CURVATURE_TOL:.0%} of std : {use_quadratic}")

if use_quadratic:
    trend_fit, trend_coefs = utils.fit_poly_trend(log_deseason.values, degree=2)
    trend_degree = 2
else:
    trend_fit = lin_fit
    trend_coefs = lin_coefs
    trend_degree = 1

print(f"Using polynomial degree: {trend_degree}")
print(f"Final trend coefficients: {trend_coefs}")

trend = pd.Series(trend_fit, index=log_load.index, name="trend")
residuals = log_deseason - trend
residuals.name = "residual"

# %%
fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(log_deseason.index, log_deseason.values, lw=0.6, label="deseasonalized")
axes[0].plot(trend.index, trend.values, lw=1.5, color="C3", label=f"polynomial fit (deg {trend_degree})")
axes[0].set_title("Trend fit on deseasonalized log load")
axes[0].set_ylabel("log MW")
axes[0].legend()
axes[1].plot(residuals.index, residuals.values, lw=0.6, color="C2")
axes[1].axhline(0, color="k", lw=0.5)
axes[1].set_title(r"Residuals $\hat Y_t = \log X_t - \hat s^{(year)}_t - \hat s^{(week)}_t - \hat m_t$")
axes[1].set_xlabel("date")
axes[1].set_ylabel("residual (log MW)")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Step 6 — Diagnostics on $\hat Y_t$
#
# Residual time series, sample ACF, sample PACF (all up to lag 50), plus
# the Ljung–Box statistic at $h \in \{20, 40\}$. Per §8 we **expect** iid
# to be rejected — that is the right outcome, it means we have stripped
# the deterministic structure and what's left is correlated stationary
# noise ready for ARMA fitting.

# %%
fig = utils.diagnostic_plots(residuals, max_lag=50, title_prefix="")
plt.show()

# %%
lb = utils.ljung_box(residuals.values, lags=(20, 40))
print("Ljung–Box test (iid null):")
print(lb)
print()
print("Interpretation: p-values are expected to be << 0.05 — we reject iid,")
print("which is the correct outcome. The deterministic structure has been")
print("removed; what remains is correlated stationary noise for ARMA to model.")

# %%
# Explicit check for residual trend / seasonality: look for slow decay
# and spikes at lag 7 and 365.
from statsmodels.tsa.stattools import acf as _acf

acf_vals = _acf(residuals.values, nlags=400, fft=True)
bound = 1.96 / np.sqrt(len(residuals))
print(f"ACF bound       : +/- {bound:.4f}")
print(f"ACF at lag 1    : {acf_vals[1]:+.4f}")
print(f"ACF at lag 7    : {acf_vals[7]:+.4f}   (weekly residual check)")
print(f"ACF at lag 14   : {acf_vals[14]:+.4f}")
print(f"ACF at lag 30   : {acf_vals[30]:+.4f}")
print(f"ACF at lag 365  : {acf_vals[365]:+.4f}  (yearly residual check)")
if abs(acf_vals[7]) > 3 * bound:
    print("WARNING: large lag-7 ACF — possible residual weekly seasonality.")
if abs(acf_vals[365]) > 3 * bound:
    print("WARNING: large lag-365 ACF — possible residual yearly seasonality.")

# %% [markdown]
# ## Step 7 — Differencing sanity check
#
# Parallel route from §6: compute $\nabla_7 \log X_t$ to kill weekly
# seasonality, then $\nabla_1$ to kill any linear trend. Compare its ACF
# to the ACF of $\hat Y_t$ from Step 6. The two should look qualitatively
# similar.

# %%
diff_series = log_load.diff(D_WEEK).diff(1).dropna()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
plot_acf(residuals.values, lags=50, ax=axes[0])
axes[0].set_title("ACF: classical decomposition residuals")
axes[0].set_xlabel("lag")
plot_acf(diff_series.values, lags=50, ax=axes[1])
axes[1].set_title(r"ACF: $\nabla_1 \nabla_7 \log X_t$")
axes[1].set_xlabel("lag")
fig.tight_layout()
plt.show()

print("Both ACFs should decay quickly and stay mostly within the +/-1.96/sqrt(n)")
print("bounds. The differencing route is *not* what we hand off to the ARMA stage")
print("(we hand off the classical residuals), but qualitative agreement here")
print("cross-validates the classical decomposition.")

# %% [markdown]
# ## Step 8 — Save outputs
#
# Three files for the downstream ARMA stage:
# - `cleaned_residuals.csv` — date + residual, the ARMA input.
# - `decomposition.csv` — every component, for reproducing plots.
# - `cleaning_report.txt` — one-screen summary of parameters & diagnostics.

# %%
out_residuals = residuals.rename("residual").to_frame()
out_residuals.index.name = "date"
out_residuals.to_csv("cleaned_residuals.csv")

decomposition = pd.DataFrame(
    {
        "log_load": log_load.values,
        "yearly_seasonal": s_year.values,
        "weekly_seasonal": s_week.values,
        "trend": trend.values,
        "residual": residuals.values,
    },
    index=log_load.index,
)
decomposition.index.name = "date"
decomposition.to_csv("decomposition.csv")

report_lines = []
report_lines.append("SF2943 Part A — SE_3 cleaning report")
report_lines.append("=" * 44)
report_lines.append(f"Source file          : {CSV_PATH}")
report_lines.append(f"Column               : {COL}")
report_lines.append(f"Date range           : {log_load.index.min().date()} to {log_load.index.max().date()}")
report_lines.append(f"Daily observations n : {n_total}")
report_lines.append(f"Days interpolated    : {n_missing_before}")
report_lines.append(f"Longest hourly NaN   : {gap_max} hours")
report_lines.append("")
report_lines.append(f"Yearly harmonic  (d={D_YEAR}, k={K_YEAR}):")
for k, v in coefs_year.items():
    report_lines.append(f"  {k:>3} = {v:+.5f}")
report_lines.append("")
report_lines.append(f"Weekly harmonic  (d={D_WEEK}, k={K_WEEK}):")
for k, v in coefs_week.items():
    report_lines.append(f"  {k:>3} = {v:+.5f}")
report_lines.append("")
report_lines.append(f"Polynomial trend (degree {trend_degree}):")
for k, v in trend_coefs.items():
    report_lines.append(f"  {k:>5} = {v:+.6e}")
report_lines.append("")
report_lines.append("Ljung-Box (iid null):")
for lag, row in lb.iterrows():
    report_lines.append(
        f"  h={lag:>2}  Q = {row['lb_stat']:9.3f}  p = {row['lb_pvalue']:.3e}"
    )
report_lines.append("")
report_lines.append(f"ACF bound +/- 1.96/sqrt(n) = +/- {bound:.4f}")
report_lines.append(f"ACF(1)   = {acf_vals[1]:+.4f}")
report_lines.append(f"ACF(7)   = {acf_vals[7]:+.4f}")
report_lines.append(f"ACF(365) = {acf_vals[365]:+.4f}")
report_lines.append("")
report_lines.append("Note: rejection of iid by Ljung-Box is expected and correct;")
report_lines.append("it confirms correlated stationary structure remains for ARMA fitting.")
report_text = "\n".join(report_lines)
with open("cleaning_report.txt", "w") as fh:
    fh.write(report_text + "\n")
print(report_text)

# %% [markdown]
# Outputs written to the project directory. The next stage (ARMA fitting)
# reads `cleaned_residuals.csv`.
