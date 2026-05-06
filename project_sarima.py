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
# # SF2943 Part A — SARIMA Extension
#
# Extends the Part A ARMA(2,1) baseline by replacing the non-seasonal ARMA
# step with a seasonal ARMA model at period $s = 7$ days.  The cleaning
# pipeline (log transform, yearly harmonic, weekly harmonic, polynomial trend)
# is unchanged; only the stochastic model fitted on the cleaning residuals
# $\hat Y_t$ is upgraded.
#
# **Motivation.** The ARMA(2,1) residuals failed the Ljung–Box whiteness
# test ($Q(20)=43.7$, $p=4\times10^{-4}$) with surviving correlation at
# multiples of lag 7.  A SARIMA$(p,0,q)(P,0,Q)_{[7]}$ model augments the
# ARMA polynomial with seasonal AR/MA factors at the weekly frequency and
# directly targets this structure.
#
# **Requires.** Run `project_part_a.ipynb` (or `project_part_a.py`) first so
# that `cleaned_residuals.csv`, `decomposition.csv`, and `forecast.csv` are
# present.

# %%
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

import utils

pd.options.display.float_format = "{:.4f}".format
plt.rcParams["figure.dpi"] = 100

# %% [markdown]
# ## Section 1 — Load Part A outputs

# %%
decomp = pd.read_csv("decomposition.csv", index_col="date", parse_dates=True)
cleaned = pd.read_csv("cleaned_residuals.csv", index_col="date", parse_dates=True)
arma_fc_df = pd.read_csv("forecast.csv", parse_dates=["date"])

residuals = cleaned["residual"]
log_load   = decomp["log_load"]
s_year     = decomp["yearly_seasonal"]
s_week     = decomp["weekly_seasonal"]
trend      = decomp["trend"]
daily_mw   = np.exp(log_load)
n          = len(residuals)

print(f"Loaded {n} daily observations, {residuals.index.min().date()} – {residuals.index.max().date()}")
print(f"Residual mean  : {residuals.mean():+.6f}  (zero by OLS construction)")
print(f"Residual std   : {residuals.std():.6f}")

# %% [markdown]
# ## Section 2 — Train / test split
#
# Identical 30-day hold-out as Part A so forecasts are directly comparable.

# %%
H = 30

train_resid  = residuals.iloc[:-H]
test_resid   = residuals.iloc[-H:]
train_mw     = daily_mw.iloc[:-H]
test_mw      = daily_mw.iloc[-H:]

# Deterministic additive term for the test window (log scale); used to
# invert the stochastic forecast back to MW.
additive_test = (trend + s_year + s_week).iloc[-H:].values

print(f"Training residuals : {len(train_resid)} obs  "
      f"({train_resid.index.min().date()} – {train_resid.index.max().date()})")
print(f"Test window        : {len(test_resid)} obs  "
      f"({test_resid.index.min().date()} – {test_resid.index.max().date()})")

# %% [markdown]
# ## Section 3 — SARIMA candidate fitting
#
# We try $\text{SARIMA}(p,0,q)(P,0,Q)_{[7]}$ for a grid of small orders,
# keeping the non-seasonal block anchored near the Part A ARMA(2,1) choice
# and varying the seasonal block.  The seasonal period $s=7$ was chosen
# because the residual ACF has its dominant surviving spike at lag 7.
#
# Model selection criterion: AICC (B&D §5.5.2) with
# $k = p + q + P + Q + 2$ counting intercept and $\sigma^2$.

# %%
S = 7

# (p, q, P, Q) — non-seasonal and seasonal ARMA orders
candidate_orders = [
    (2, 1, 1, 0),   # Part A winner + seasonal AR(1)
    (2, 1, 0, 1),   # Part A winner + seasonal MA(1)
    (2, 1, 1, 1),   # Part A winner + both seasonal terms
    (1, 1, 1, 1),   # smaller non-seasonal, both seasonal
    (2, 1, 2, 0),   # Part A winner + seasonal AR(2)
]


def fit_sarima(p, q, P, Q, y, s=S):
    label = f"SARIMA({p},0,{q})({P},0,{Q})[{s}]"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(
            y,
            order=(p, 0, q),
            seasonal_order=(P, 0, Q, s),
            trend="c",
        ).fit(disp=False)

    n_obs   = int(res.nobs)
    k       = p + q + P + Q + 2          # intercept + sigma^2 + AR/MA params
    aicc    = utils.aicc_from_aic(res.aic, k, n_obs)
    sigma2  = float(res.params.get("sigma2", np.nan))

    # Roots of the combined (regular * seasonal) AR and MA polynomials.
    try:
        ar_roots = np.atleast_1d(res.arroots)
    except Exception:
        ar_roots = np.array([])
    try:
        ma_roots = np.atleast_1d(res.maroots)
    except Exception:
        ma_roots = np.array([])

    causal     = bool(np.all(np.abs(ar_roots) > 1)) if len(ar_roots) else True
    invertible = bool(np.all(np.abs(ma_roots) > 1)) if len(ma_roots) else True

    return {
        "label": label, "p": p, "q": q, "P": P, "Q": Q,
        "result": res, "aic": float(res.aic), "aicc": aicc,
        "sigma2": sigma2, "causal": causal, "invertible": invertible,
        "ar_roots": ar_roots, "ma_roots": ma_roots,
        "model_df": p + q + P + Q,       # for Ljung–Box df correction
    }


fits = []
for p, q, P, Q in candidate_orders:
    print(f"Fitting SARIMA({p},0,{q})({P},0,{Q})[{S}] ...", end="  ", flush=True)
    f = fit_sarima(p, q, P, Q, train_resid)
    fits.append(f)
    print(f"AICC = {f['aicc']:.2f}  causal={f['causal']}  invertible={f['invertible']}")

# %% [markdown]
# ### 3.1 AICC comparison table

# %%
tbl = pd.DataFrame([
    {
        "model": f["label"],
        "params (p+q+P+Q)": f["p"] + f["q"] + f["P"] + f["Q"],
        "AICC": f["aicc"],
        "sigma2": f["sigma2"],
        "causal": f["causal"],
        "invertible": f["invertible"],
    }
    for f in fits
]).sort_values("AICC").reset_index(drop=True)

print(tbl.to_string(index=False))

eligible = [f for f in fits if f["causal"] and f["invertible"]]
assert eligible, "No candidate is both causal and invertible."
best = min(eligible, key=lambda f: f["aicc"])
others = sorted(eligible, key=lambda f: f["aicc"])[1:]
margin = others[0]["aicc"] - best["aicc"] if others else float("nan")

print(f"\nChosen : {best['label']}")
if others:
    print(f"Runner-up: {others[0]['label']}  (ΔAICC = {margin:+.2f})")

# %% [markdown]
# ### 3.2 Chosen model — parameter estimates

# %%
print(best["result"].summary().tables[1])
print(f"\nAR roots : {np.round(best['ar_roots'], 3).tolist()}")
print(f"MA roots : {np.round(best['ma_roots'], 3).tolist()}")
print(f"All AR roots outside unit disc: {best['causal']}")
print(f"All MA roots outside unit disc: {best['invertible']}")

# %% [markdown]
# ## Section 4 — Residual diagnostics on chosen model
#
# Ljung–Box with B&D §5.3 df correction: $\text{df} = h - p - q - P - Q$.
# We check whether the seasonal ARMA has absorbed the lag-7 spikes that
# the ARMA(2,1) left behind.

# %%
sarima_resid = pd.Series(
    best["result"].resid, index=train_resid.index, name="sarima_resid"
)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

axes[0, 0].plot(sarima_resid.index, sarima_resid.values, lw=0.5, color="C2")
axes[0, 0].axhline(0, color="k", lw=0.5)
axes[0, 0].set_title(f"{best['label']} residuals")
axes[0, 0].set_xlabel("date")
axes[0, 0].set_ylabel("residual")

plot_acf(sarima_resid.values, lags=50, ax=axes[0, 1])
axes[0, 1].set_title(f"ACF of {best['label']} residuals")
axes[0, 1].set_xlabel("lag (days)")

plot_pacf(sarima_resid.values, lags=50, ax=axes[1, 0], method="ywm")
axes[1, 0].set_title("PACF")
axes[1, 0].set_xlabel("lag (days)")

# Ljung–Box p-value profile
lags_range = list(range(1, 41))
lb_all = acorr_ljungbox(
    sarima_resid.values, lags=lags_range, return_df=True, model_df=best["model_df"]
)
axes[1, 1].bar(lb_all.index, lb_all["lb_pvalue"], color="C2", alpha=0.7)
axes[1, 1].axhline(0.05, color="r", lw=1.2, linestyle="--", label="α = 0.05")
axes[1, 1].set_title("Ljung–Box p-values by lag h")
axes[1, 1].set_xlabel("lag h")
axes[1, 1].set_ylabel("p-value")
axes[1, 1].legend()

fig.tight_layout()
plt.show()

# Summary Ljung–Box at h = 20 and h = 40
for h in (20, 40):
    lb_h = acorr_ljungbox(
        sarima_resid.values, lags=[h], return_df=True, model_df=best["model_df"]
    )
    stat  = float(lb_h["lb_stat"].iloc[0])
    pval  = float(lb_h["lb_pvalue"].iloc[0])
    df_h  = h - best["model_df"]
    flag  = "PASS" if pval >= 0.05 else "FAIL"
    print(f"Ljung–Box Q({h:2d})  df={df_h:2d}  stat={stat:.2f}  p={pval:.4f}  [{flag}]")

# %% [markdown]
# ## Section 5 — 30-day forecast and comparison with ARMA(2,1)

# %%
fc       = best["result"].get_forecast(steps=H)
fc_mean  = np.asarray(fc.predicted_mean)
fc_ci    = np.asarray(fc.conf_int(alpha=0.05))

fc_mw_sarima = np.exp(fc_mean          + additive_test)
fc_mw_lo     = np.exp(fc_ci[:, 0]     + additive_test)
fc_mw_hi     = np.exp(fc_ci[:, 1]     + additive_test)

# ARMA(2,1) baseline loaded from Part A CSV
arma_idx     = arma_fc_df["date"].values
fc_mw_arma   = arma_fc_df["forecast_mw"].values
arma_lo      = arma_fc_df["lower_95"].values
arma_hi      = arma_fc_df["upper_95"].values
actual_mw    = test_mw.values


def eval_metrics(fc_pt, actual, lo, hi):
    rmse     = float(np.sqrt(np.mean((fc_pt - actual) ** 2)))
    mae      = float(np.mean(np.abs(fc_pt - actual)))
    coverage = float(np.mean((actual >= lo) & (actual <= hi)))
    return rmse, mae, coverage


rmse_a, mae_a, cov_a = eval_metrics(fc_mw_arma,   actual_mw, arma_lo,    arma_hi)
rmse_s, mae_s, cov_s = eval_metrics(fc_mw_sarima, actual_mw, fc_mw_lo,   fc_mw_hi)

print(f"\n{'Model':<40} {'RMSE (MW)':>10} {'MAE (MW)':>10} {'Coverage':>10}")
print("-" * 72)
print(f"{'ARMA(2,1) — Part A baseline':<40} {rmse_a:>10.1f} {mae_a:>10.1f} {cov_a:>10.2f}")
print(f"{best['label']:<40} {rmse_s:>10.1f} {mae_s:>10.1f} {cov_s:>10.2f}")

print(f"\nRMSE improvement : {rmse_a - rmse_s:+.1f} MW  ({100*(rmse_a - rmse_s)/rmse_a:+.1f}%)")
print(f"MAE  improvement : {mae_a  - mae_s:+.1f} MW  ({100*(mae_a  - mae_s)/mae_a:+.1f}%)")

# %% [markdown]
# ### 5.1 Forecast plot

# %%
last_90    = daily_mw.iloc[-H - 90:-H]
test_dates = test_mw.index

fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)

for ax in axes:
    ax.plot(last_90.index, last_90.values, lw=0.8, color="C0", label="training (last 90 d)")
    ax.plot(test_dates, actual_mw, lw=1.0, color="k",
            marker="o", markersize=3, label="held-out actual")

# Top panel — ARMA(2,1) baseline
axes[0].plot(test_dates, fc_mw_arma, lw=1.5, color="C3", label="ARMA(2,1) forecast")
axes[0].fill_between(test_dates, arma_lo, arma_hi, color="C3", alpha=0.2, label="95% PI")
axes[0].set_title(
    f"ARMA(2,1) baseline — RMSE={rmse_a:.1f} MW, MAE={mae_a:.1f} MW, Coverage={cov_a:.2f}"
)
axes[0].set_ylabel("load (MW)")
axes[0].legend(loc="upper left")

# Bottom panel — best SARIMA
axes[1].plot(test_dates, fc_mw_sarima, lw=1.5, color="C1", label=f"{best['label']} forecast")
axes[1].fill_between(test_dates, fc_mw_lo, fc_mw_hi, color="C1", alpha=0.2, label="95% PI")
axes[1].set_title(
    f"{best['label']} — RMSE={rmse_s:.1f} MW, MAE={mae_s:.1f} MW, Coverage={cov_s:.2f}"
)
axes[1].set_ylabel("load (MW)")
axes[1].set_xlabel("date")
axes[1].legend(loc="upper left")

fig.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2 Point-forecast comparison per day

# %%
cmp = pd.DataFrame({
    "date":         test_dates,
    "actual_mw":    actual_mw,
    "arma21_mw":    fc_mw_arma,
    "sarima_mw":    fc_mw_sarima,
    "arma21_err":   fc_mw_arma   - actual_mw,
    "sarima_err":   fc_mw_sarima - actual_mw,
})
cmp["winner"] = np.where(np.abs(cmp["sarima_err"]) < np.abs(cmp["arma21_err"]), "SARIMA", "ARMA")
print(cmp[["date", "actual_mw", "arma21_mw", "sarima_mw", "arma21_err", "sarima_err", "winner"]]
      .to_string(index=False))
print(f"\nSARIMA wins on {(cmp['winner']=='SARIMA').sum()} / {H} days")

# %% [markdown]
# ## Section 6 — Save outputs

# %%
sarima_fc_out = pd.DataFrame({
    "date":        test_dates,
    "forecast_mw": fc_mw_sarima,
    "lower_95":    fc_mw_lo,
    "upper_95":    fc_mw_hi,
    "actual_mw":   actual_mw,
})
sarima_fc_out.to_csv("sarima_forecast.csv", index=False)

lines = []
lines.append("SF2943 Part A — SARIMA extension summary")
lines.append("=" * 50)
lines.append(f"Seasonal period           : s = {S}")
lines.append(f"Chosen model              : {best['label']}")
lines.append(f"AICC                      : {best['aicc']:.2f}")
lines.append(f"sigma2                    : {best['sigma2']:.6f}")
lines.append(f"Causal                    : {best['causal']}")
lines.append(f"Invertible                : {best['invertible']}")
lines.append("")
lines.append("Candidate AICC values:")
for _, row in tbl.iterrows():
    lines.append(f"  {row['model']:<38}  AICC={row['AICC']:.2f}")
lines.append("")
lines.append("Forecast metrics (30-day hold-out):")
lines.append(f"  {'ARMA(2,1) baseline':<38}  RMSE={rmse_a:.1f} MW  MAE={mae_a:.1f} MW  Coverage={cov_a:.2f}")
lines.append(f"  {best['label']:<38}  RMSE={rmse_s:.1f} MW  MAE={mae_s:.1f} MW  Coverage={cov_s:.2f}")
lines.append(f"  RMSE improvement  : {rmse_a - rmse_s:+.1f} MW ({100*(rmse_a-rmse_s)/rmse_a:+.1f}%)")
lines.append(f"  MAE  improvement  : {mae_a  - mae_s:+.1f} MW ({100*(mae_a-mae_s)/mae_a:+.1f}%)")

summary_text = "\n".join(lines)
with open("sarima_summary.txt", "w") as fh:
    fh.write(summary_text + "\n")

print(summary_text)
print("\nSaved: sarima_forecast.csv, sarima_summary.txt")
