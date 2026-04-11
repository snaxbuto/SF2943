# SF2943 Project — Part A — Full Spec

## Context for Claude Code

This is the **complete Part A** of an SF2943 (Time Series Analysis) project at KTH. The dataset is hourly electricity load for the Swedish bidding zone SE3, downloaded from the Open Power System Data (OPSD) `time_series_60min_singleindex.csv` release. The deliverable is a Jupyter notebook that walks through three stages — **clean → fit → forecast** — with course-grounded justification at each step.

This is a **student project**, not a research paper. The goal is "comfortably above pass," which means: do every step the project description asks for, motivate every choice with course material, and stop. Do not chase optimal performance.

---

## Hard guardrails

These are non-negotiable. Every one of these is a way Claude Code could go off the rails, and each one would make the project worse, not better.

1. **Stick to the course toolkit.** The course is Brockwell & Davis, *Introduction to Time Series and Forecasting*, chapters 1–6. Allowed methods: classical decomposition, harmonic regression, differencing, sample ACF/PACF, Yule–Walker, innovations algorithm, Durbin–Levinson, Gaussian MLE, AICC, Ljung–Box. **Forbidden**: `auto_arima` from `pmdarima`, SARIMA, VAR, state-space / Kalman, GARCH, neural networks, Prophet, exponential smoothing, anything from `sklearn`. If a method isn't in the course book, don't use it.

2. **Show your work.** Every model fit prints its parameter estimates, the white noise variance, and the AICC. Every forecast prints the predicted values and the prediction interval bounds. No silent choices, no black boxes.

3. **Try a small number of models, motivated by ACF/PACF.** Inspect the sample ACF and PACF, *write down* what they suggest about $(p, q)$, then fit 3–5 candidate models. Do not grid-search over $(p, q) \in [0, 10]^2$. The course teaches identification by inspection followed by AICC comparison among a handful of candidates — do that.

4. **Cite the chapter.** Every method used has a comment naming the Brockwell & Davis section it comes from. Examples: `# B&D §1.5 classical decomposition`, `# B&D §5.2 Gaussian MLE via innovations algorithm`, `# B&D §3.3 h-step prediction for ARMA`.

5. **Markdown cells are 1–3 sentences.** Not paragraphs. The notebook is a story, not a textbook. State *what* the cell does and *why* in two sentences and move on.

6. **No fancy plotting.** `matplotlib` only. No seaborn, no plotly. Functional plots: title, axis labels with units, legend if needed, dashed lines for $\pm 1.96/\sqrt{n}$ bounds. That's it.

7. **Be honest about limitations.** Forecasts will be imperfect. The discussion should say so, with reasons grounded in the data and the model. Do not oversell.

8. **Do not edit files outside the project directory.** All outputs go in the working directory.

---

## Mathematical foundations

This section is the math you need to justify what the code does. Reference it from markdown cells in the notebook.

### Part I — Stationarity and decomposition

A time series $\{X_t\}_{t \in \mathbb{Z}}$ is **weakly stationary** (B&D §1.4) if

$$\mu_X(t) = \mathbb{E}[X_t] = \mu \quad \text{and} \quad \gamma_X(t, t+h) = \mathrm{Cov}(X_t, X_{t+h}) = \gamma_X(h)$$

i.e., the mean is constant and the autocovariance depends only on the lag $h$. Variable meanings: $\mu_X(t)$ is the mean function, $\gamma_X(h)$ is the autocovariance at lag $h$, and $\rho_X(h) = \gamma_X(h)/\gamma_X(0)$ is the **autocorrelation function (ACF)**, with $\gamma_X(0) = \mathrm{Var}(X_t)$.

Cleaning's job is to transform raw load into something where stationarity is a defensible assumption. The course model (B&D §1.5) is **classical decomposition**:

$$X_t = m_t + s_t + Y_t$$

where $X_t$ is the observation, $m_t$ is a deterministic trend, $s_t$ is a deterministic seasonal component with known period $d$ satisfying $s_{t-d} = s_t$ and $\sum_{j=1}^d s_j = 0$ (zero-mean over one period), and $Y_t$ is the residual we want to be stationary. The procedure estimates $\hat m_t$ and $\hat s_t$ and forms

$$\hat Y_t = X_t - \hat m_t - \hat s_t$$

For electricity load there are two seasonal components, so we generalize to

$$X_t = m_t + s_t^{(\text{week})} + s_t^{(\text{year})} + Y_t$$

with $d_{\text{week}} = 7$ and $d_{\text{year}} = 365$ after aggregating to daily resolution.

**Variance stabilization.** Load fluctuations grow with the level. The Box–Cox transform with $\lambda = 0$ (B&D §1.5) is the natural log: $X_t \mapsto \log X_t$. Apply this *first*; multiplicative seasonality becomes additive, which is what the decomposition model assumes.

**Harmonic regression for $\hat s_t$ (B&D §1.3).** For period $d$,

$$s_t = a_0 + \sum_{j=1}^k \big( a_j \cos(\lambda_j t) + b_j \sin(\lambda_j t) \big), \qquad \lambda_j = \frac{2\pi j}{d}$$

Variable meanings: $d$ is the known period (7 or 365), $k$ is the number of harmonics, $\lambda_j$ is the angular frequency in radians per time step, and $\{a_j, b_j\}$ are the unknown coefficients found by ordinary least squares. We use $k = 2$ for yearly (captures asymmetric winter peak) and $k = 3$ for weekly (captures the weekday/weekend block).

**Polynomial trend.** After removing seasonality, fit $\hat m_t = \beta_0 + \beta_1 t$ (linear) by OLS. Only escalate to quadratic if visible curvature remains.

**Differencing as alternative (B&D §1.5.2).** With backshift operator $B X_t = X_{t-1}$ and lag-$d$ difference $\nabla_d = 1 - B^d$, applying $\nabla_d$ to $X_t = m_t + s_t + Y_t$ kills the period-$d$ seasonal component exactly. We use this only as a sanity check.

### Part II — Identification and estimation of ARMA models

A causal $\mathrm{ARMA}(p, q)$ process satisfies (B&D §3.1)

$$\phi(B) X_t = \theta(B) Z_t, \qquad \{Z_t\} \sim \mathrm{WN}(0, \sigma^2)$$

where $\phi(z) = 1 - \phi_1 z - \cdots - \phi_p z^p$ is the autoregressive polynomial, $\theta(z) = 1 + \theta_1 z + \cdots + \theta_q z^q$ is the moving-average polynomial, and $\{Z_t\}$ is white noise with variance $\sigma^2$. **Causality** requires all roots of $\phi(z)$ to lie outside the unit circle; **invertibility** requires the same of $\theta(z)$. Both must hold for estimation and prediction to be well-defined.

**Identification from sample ACF and PACF (B&D §3.2.3, §6.2).** The shapes are diagnostic:

- $\mathrm{AR}(p)$: ACF tails off (geometric decay), **PACF cuts off after lag $p$**.
- $\mathrm{MA}(q)$: **ACF cuts off after lag $q$**, PACF tails off.
- $\mathrm{ARMA}(p, q)$: both tail off.

"Cuts off" means: drops inside the $\pm 1.96/\sqrt{n}$ bounds and stays there. This is the first-pass tool for picking $(p, q)$ candidates.

**Yule–Walker estimation for AR($p$) (B&D §5.1.1).** From multiplying the AR equations by $X_{t-j}$ and taking expectations,

$$\Gamma_p \boldsymbol{\phi} = \boldsymbol{\gamma}_p, \qquad \sigma^2 = \gamma(0) - \boldsymbol{\phi}^\top \boldsymbol{\gamma}_p$$

Variable meanings: $\Gamma_p = [\gamma(i-j)]_{i,j=1}^p$ is the $p \times p$ autocovariance matrix, $\boldsymbol{\gamma}_p = (\gamma(1), \ldots, \gamma(p))^\top$, and $\boldsymbol{\phi} = (\phi_1, \ldots, \phi_p)^\top$. Sample Yule–Walker replaces $\gamma$ with $\hat\gamma$:

$$\hat{\boldsymbol{\phi}} = \hat\Gamma_p^{-1} \hat{\boldsymbol{\gamma}}_p, \qquad \hat\sigma^2 = \hat\gamma(0) - \hat{\boldsymbol{\phi}}^\top \hat{\boldsymbol{\gamma}}_p$$

**Maximum likelihood estimation (B&D §5.2).** For models with $q > 0$, use Gaussian MLE. The innovations algorithm gives the one-step prediction errors and their variances, which lets the joint Gaussian likelihood be written as a product over uncorrelated terms — that's the trick that makes likelihood evaluation feasible. In practice, `statsmodels.tsa.arima.model.ARIMA` does this under the hood; the comment in the notebook should say so explicitly: `# B&D §5.2: Gaussian MLE, computed via innovations algorithm`.

**Order selection via AICC (B&D §5.5.2).** Among candidate models, choose the one minimizing

$$\text{AICC} = -2 \ln L(\hat\beta, \hat\sigma^2) + \frac{2(p + q + 1) n}{n - p - q - 2}$$

where $L$ is the Gaussian likelihood, $n$ is the number of observations, and $p + q + 1$ is the number of estimated parameters (the $+1$ is $\sigma^2$). **AICC, not AIC** — the course explicitly uses the bias-corrected version because AIC underpenalizes overfitting in finite samples.

**Diagnostic checking on residuals (B&D §5.3).** Once a model is fit, compute its residuals $\hat R_t$ and check whether they look like white noise:
- Plot $\hat R_t$ — should look unstructured.
- Sample ACF of $\hat R_t$ — most values inside $\pm 1.96/\sqrt n$.
- Ljung–Box statistic $Q_{\text{LB}} = n(n+2) \sum_{j=1}^h \hat\rho^2(j) / (n-j)$, distributed as $\chi^2(h - p - q)$ under the null that residuals are white noise. **Note the degrees-of-freedom correction $h - p - q$**: when applied to ARMA residuals, $p + q$ degrees of freedom are lost from estimating the parameters. `statsmodels.stats.diagnostic.acorr_ljungbox` has a `model_df` argument for this.

If the model passes diagnostics, accept it. If it fails, the residuals still have structure and we should try a different $(p, q)$.

### Part III — Linear prediction and forecasting

**Best linear predictor (B&D §2.5).** For zero-mean stationary $\{X_t\}$, the best linear predictor of $X_{n+h}$ in terms of $X_1, \ldots, X_n$ is

$$P_n X_{n+h} = \boldsymbol{\phi}_n^\top \mathbf{X}_n = \phi_{n1} X_n + \cdots + \phi_{nn} X_1$$

where the coefficients satisfy the prediction equations $\Gamma_n \boldsymbol{\phi}_n = \boldsymbol{\gamma}_n(h)$ with $\boldsymbol{\gamma}_n(h) = (\gamma(h), \gamma(h+1), \ldots, \gamma(h + n - 1))^\top$. The mean squared prediction error is

$$v_n(h) = \mathbb{E}[(X_{n+h} - P_n X_{n+h})^2] = \gamma(0) - \boldsymbol{\phi}_n^\top \boldsymbol{\gamma}_n(h)$$

For nonzero-mean series, predict $X_t - \mu$ then add $\mu$ back.

**Durbin–Levinson recursion (B&D §2.5.3).** Recursively computes the coefficients $\phi_{n1}, \ldots, \phi_{nn}$ and the one-step prediction errors $v_n$ without inverting $\Gamma_n$ directly. Particularly natural for AR processes.

**Innovations algorithm (B&D §2.5.4).** Expresses the predictor as a linear combination of the **innovations** $X_j - \hat X_j$ rather than the raw observations:

$$\hat X_{n+1} = \sum_{j=1}^n \theta_{nj} (X_{n+1-j} - \hat X_{n+1-j})$$

Because innovations are uncorrelated by construction, the algebra is much cleaner — this is what makes it the natural prediction tool for MA and ARMA processes.

**$h$-step prediction for ARMA (B&D §3.3.1).** For $n > \max(p, q)$ and $h \geq 1$,

$$P_n X_{n+h} = \sum_{i=1}^p \phi_i \, P_n X_{n+h-i} + \sum_{j=h}^q \theta_{n+h-1, j} (X_{n+h-j} - \hat X_{n+h-j})$$

The second sum vanishes for $h > q$, so the long-horizon forecast follows the AR recursion alone. As $h$ grows, $P_n X_{n+h} \to \mu$ (the process mean) and $v_n(h) \to \mathrm{Var}(X_t)$.

**Prediction intervals.** Under Gaussian innovations, an approximate 95% prediction interval is

$$\hat X_{n+h} \pm 1.96 \sqrt{v_n(h)}$$

Intervals widen with $h$, reflecting the increasing uncertainty. This widening is the most important visual feature of the forecast plot.

---

## Pipeline

The notebook is `project_part_a.ipynb` with helper functions in `utils.py`. Three sections, one per project requirement.

### Section 1 — Cleaning

**1.1 Load and inspect.** Read the CSV, extract `SE_3_load_actual_entsoe_transparency`, set the index from `utc_timestamp`. Report start date, end date, hourly count, NaN count, longest gap. Plot the raw hourly series.

**1.2 Aggregate to daily.** Daily mean (units stay in MW). Days with fewer than 20 valid hours are marked missing; remaining gaps linearly interpolated. Report $n$ and confirm $n \geq 2000$. Plot.

**1.3 Log transform.** Apply $\log$. One-line markdown: "Box–Cox $\lambda=0$, B&D §1.5, stabilizes variance." Plot.

**1.4 Yearly seasonality (harmonic regression, $d = 365$, $k = 2$).** Build the design matrix $[\cos(2\pi j t / 365), \sin(2\pi j t / 365)]_{j=1,2}$, fit by OLS, **print the coefficients** (these go in the report — the project description requires explicit expressions for seasonal components). Subtract the fit. Plot the deseasonalized series and the fitted yearly component overlaid on one year of raw log data.

**1.5 Weekly seasonality (harmonic regression, $d = 7$, $k = 3$).** Same procedure. Print coefficients. Plot a 4-week window before-vs-after.

**1.6 Linear trend.** Fit $\hat m_t = \beta_0 + \beta_1 t$ by OLS, print coefficients, subtract. Plot residuals. **Decision point**: if residuals visibly curve, refit with quadratic; print which choice was made and why.

**1.7 Diagnostics.** Plot residuals as a time series. Plot sample ACF (lags 0–50) and sample PACF (lags 0–50) with $\pm 1.96/\sqrt n$ bounds. Compute Ljung–Box at $h = 20$ and $h = 40$, print the statistic and $p$-value. Markdown: "We expect to reject iid; the rejection means there is correlated structure left for ARMA to model." Verify no slow decay and no spike at lag 7 or 365.

**1.8 Sanity check via differencing.** Compute $\nabla_7 \log X_t$ followed by $\nabla_1$. Plot its sample ACF next to the harmonic-regression residual ACF. One sentence: do they look qualitatively similar?

### Section 2 — ARMA model fitting

**2.1 Identification from sample ACF and PACF.** Look at the residual ACF/PACF from Section 1. **Write down in markdown** what the shapes suggest. Examples of valid reasoning:
- "PACF cuts off after lag 2 and ACF tails off → AR(2) candidate."
- "Both tail off → mixed ARMA, try ARMA(1,1) and ARMA(2,1)."
- "ACF cuts off after lag 1 → MA(1) candidate."

Pick **3 to 5 candidate models** based on this inspection. List them.

**2.2 Fit candidates.** For each candidate $(p, q)$:
- Fit using `statsmodels.tsa.arima.model.ARIMA(residuals, order=(p, 0, q))` with default Gaussian MLE. Comment: `# B&D §5.2: Gaussian MLE via innovations algorithm`.
- Print parameter estimates $\hat\phi_1, \ldots, \hat\phi_p, \hat\theta_1, \ldots, \hat\theta_q, \hat\sigma^2$ with their standard errors.
- **Verify causality and invertibility**: compute the roots of $\hat\phi(z)$ and $\hat\theta(z)$ and confirm all lie outside the unit circle. If any model fails, flag it and exclude from further consideration.
- Compute and store the AICC.

**2.3 AICC comparison.** Print a table: model $|$ parameters $|$ AICC. Pick the minimum-AICC model that also satisfies causality/invertibility. Markdown: state the choice and the AICC margin over the runner-up.

**2.4 Yule–Walker comparison.** For one of the AR candidates, *also* fit it using sample Yule–Walker (`statsmodels.regression.linear_model.yule_walker` or hand-coded from the equations $\hat\Gamma_p \hat{\boldsymbol\phi} = \hat{\boldsymbol\gamma}_p$). Print both the YW and MLE estimates side by side. They should be close. Markdown: one sentence noting they agree (or, if not, why MLE differs).

**2.5 Residual diagnostics on the chosen model.** Compute $\hat R_t = $ residuals from the fitted ARMA. Plot. Sample ACF with $\pm 1.96/\sqrt n$ bounds. Ljung–Box at $h = 20$ with `model_df = p + q` to correct degrees of freedom. Print statistic and $p$-value. **If the $p$-value is below 0.05**: state that the diagnostic fails, and either accept the failure with explicit discussion of what structure is left, or go back to 2.2 and try a different model. Don't quietly pretend it passed.

### Section 3 — Forecasting

**3.1 Train/test split.** Hold out the last 30 days as a test set. Refit the chosen ARMA on the training portion only.

**3.2 Compute $h$-step forecasts.** For $h = 1, \ldots, 30$, compute $P_n X_{n+h}$ on the **residual scale** using the recursion from B&D §3.3.1 (or `model.get_forecast(steps=30)`, which implements it). Comment naming the formula.

**3.3 Invert the cleaning transforms.** To get back to MW, **add back** the trend $\hat m_{n+h}$ (extrapolated from the linear/quadratic fit), the weekly seasonal $\hat s_{n+h}^{(\text{week})}$, and the yearly seasonal $\hat s_{n+h}^{(\text{year})}$, then exponentiate. The order matters: residual → add trend → add seasonals → exp. Print the first few forecasted values in MW.

**3.4 Prediction intervals.** Compute the prediction MSE $v_n(h)$ from the model (statsmodels returns it via `get_forecast().conf_int()`). Form 95% intervals on the residual scale, then push them through the same inverse transforms. Note in a markdown cell: the resulting intervals on the MW scale are *approximate* because exponentiating a Gaussian interval gives a log-normal interval, which is asymmetric — that's fine, just acknowledge it.

**3.5 Plot the forecast.** Single plot:
- Last 90 days of observed load (MW)
- Forecast for the next 30 days as a line
- 95% prediction interval as a shaded band
- The held-out test data overlaid for comparison

**3.6 Forecast accuracy.** Compute RMSE and MAE between the 30-day forecast and the held-out test data. Print them. Compute the empirical coverage: what fraction of held-out points fall inside the 95% prediction band? Should be near 0.95.

**3.7 Discussion (markdown, ~one paragraph).** Address:
- Which prediction method was used (h-step ARMA recursion via innovations algorithm, B&D §3.3.1).
- How accurate is the forecast in absolute terms and relative to the variance of the series.
- Where the model is likely to fail: long horizons (forecast collapses to mean), unusual weather events, holidays not captured by simple weekly seasonality.
- One sentence on robustness: would a different $(p, q)$ near the chosen one give a noticeably different forecast?

---

## Outputs to save to disk

- `project_part_a.ipynb` — the notebook itself.
- `utils.py` — harmonic regression, plotting helpers, anything reused.
- `cleaned_residuals.csv` — `date, residual` after Section 1.
- `decomposition.csv` — `date, log_load, yearly_seasonal, weekly_seasonal, trend, residual`.
- `forecast.csv` — `date, forecast_mw, lower_95, upper_95, actual_mw` for the 30-day test horizon.
- `summary.txt` — plain text: $n$, harmonic coefficients, trend coefficients, candidate models with AICC, chosen model, Ljung–Box on ARMA residuals, RMSE/MAE/coverage on the test set. One screen of text.

---

## What success looks like

A notebook that reads top to bottom as a coherent story:

1. Raw hourly load → daily → log → yearly seasonal removed → weekly seasonal removed → detrended → residual that looks like noise.
2. ACF and PACF of the residuals, with a written diagnosis of what $(p, q)$ to try.
3. A small table of 3–5 fitted ARMA models with parameter estimates and AICC values.
4. A chosen model, with causality/invertibility verified and residual diagnostics shown.
5. A 30-day forecast plot with prediction interval, overlaid on held-out test data, with RMSE and coverage printed.
6. A short, honest discussion paragraph.
7. CSV files on disk for the rest of the group / report-writers to consume.

That's the project. Don't add more.
