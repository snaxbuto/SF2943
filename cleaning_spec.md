# SF2943 Project — Part A: Data Cleaning Spec

## Context for Claude Code

This is the **cleaning stage only** of an SF2943 (Time Series Analysis) project. The dataset is hourly electricity load for the Swedish bidding zone SE3, downloaded from the Open Power System Data (OPSD) `time_series_60min_singleindex.csv` release. The downstream task — ARMA model fitting, forecasting, diagnostics — will be done by other group members. **Your job is to produce a clean, stationary residual series and the figures/diagnostics that justify each step.**

Do **not** fit ARMA models. Do **not** do MLE or innovations algorithms. Stop at the residual series + diagnostic plots.

---

## Underlying mathematics (course material)

### 1. Weak stationarity

A time series $\{X_t\}_{t \in \mathbb{Z}}$ is **weakly stationary** if:

$$\mu_X(t) = \mathbb{E}[X_t] = \mu \quad \text{(constant in } t\text{)}$$

$$\gamma_X(t, t+h) = \mathrm{Cov}(X_t, X_{t+h}) = \gamma_X(h) \quad \text{(depends only on lag } h\text{)}$$

Here:
- $\mu_X(t)$ is the mean function — what we need to be constant.
- $\gamma_X(h)$ is the **autocovariance function** at lag $h$.
- $\rho_X(h) = \gamma_X(h) / \gamma_X(0)$ is the **autocorrelation function (ACF)**, with $\gamma_X(0) = \mathrm{Var}(X_t)$.

**The goal of cleaning is to transform the raw load series into a series for which weak stationarity is a defensible assumption.** Everything below is in service of that goal.

### 2. Classical decomposition

Course model (Brockwell & Davis, §1.5):

$$X_t = m_t + s_t + Y_t$$

where:
- $X_t$ is the observed value at time $t$ (here: daily mean load in MW).
- $m_t$ is a deterministic **trend component** (slowly varying).
- $s_t$ is a deterministic **seasonal component** with known period $d$, satisfying $s_{t-d} = s_t$ and $\sum_{j=1}^{d} s_j = 0$ (zero-mean over one period, by convention, so the trend absorbs the level).
- $Y_t$ is the **residual** (estimated noise) — this is what we want to be stationary.

The cleaning procedure estimates $\hat m_t$ and $\hat s_t$ and produces

$$\hat Y_t = X_t - \hat m_t - \hat s_t$$

For electricity load there are **two seasonal components** of different periods, so we generalize:

$$X_t = m_t + s_t^{(\mathrm{week})} + s_t^{(\mathrm{year})} + Y_t$$

with periods $d_{\mathrm{week}} = 7$ and $d_{\mathrm{year}} = 365$ (working in daily data, after aggregation from hourly).

### 3. Variance stabilization (Box–Cox, $\lambda = 0$)

Electricity load fluctuations grow with the level (winters are bigger in absolute terms). The course remedy is the Box–Cox transform; with $\lambda = 0$ this is the log:

$$X_t \mapsto \log X_t$$

After log-transform, multiplicative seasonality becomes additive seasonality, which is what the classical decomposition model assumes. Apply this **first**, before any decomposition.

### 4. Estimating the seasonal component — harmonic regression

For a seasonal component with period $d$, the course (B&D §1.3) fits

$$s_t = a_0 + \sum_{j=1}^{k} \left( a_j \cos(\lambda_j t) + b_j \sin(\lambda_j t) \right)$$

where $\lambda_j = 2\pi j / d$ are integer multiples of the fundamental frequency $2\pi/d$, and $k$ is the number of harmonics. The coefficients $\{a_j, b_j\}$ are obtained by **ordinary least squares** regression of $X_t$ (or, here, $\log X_t$) on the design matrix of cosines and sines.

Variable meanings:
- $d$: known seasonal period (7 for weekly, 365 for yearly).
- $k$: how many harmonics to include — $k=1$ is a pure sinusoid, larger $k$ captures sharper peaks.
- $\lambda_j = 2\pi j / d$: angular frequency of the $j$-th harmonic, in radians per time step.
- $a_0$: the constant level (will be absorbed by the trend later).

**Concrete recipe used here:** fit yearly seasonality with $d = 365$, $k = 2$ (fundamental + first harmonic — captures the asymmetric winter peak); fit weekly seasonality with $d = 7$, $k = 3$ (covers up to the third harmonic, enough resolution for the weekday/weekend block structure). These choices should be reported and justified in the deliverable.

**Alternative (also course material):** instead of harmonic regression, estimate the seasonal component as the **average over each phase** of the period — i.e., the mean of $\log X_t$ for each day-of-week (period 7) or each day-of-year (period 365), centered to have mean zero. Both are course-sanctioned. Harmonic regression gives a smoother $\hat s_t$ and fewer parameters.

### 5. Trend estimation

After removing the seasonal components, fit a low-order polynomial trend by least squares:

$$\hat m_t = \beta_0 + \beta_1 t + \beta_2 t^2 + \cdots$$

Start with **linear** ($\beta_2 = 0$). Only go to quadratic if the residuals after detrending still show visible curvature.

### 6. Differencing as an alternative (B&D §1.5.2)

The course offers a second route. Define the **lag-$d$ difference operator** $\nabla_d$ by

$$\nabla_d X_t = X_t - X_{t-d} = (1 - B^d) X_t$$

where $B$ is the **backshift (lag) operator**, $B X_t = X_{t-1}$. Applied to $X_t = m_t + s_t + Y_t$ with $s_t$ of period $d$:

$$\nabla_d X_t = (m_t - m_{t-d}) + (Y_t - Y_{t-d})$$

so the seasonal component is killed exactly. A linear trend is killed by $\nabla_1 = 1 - B$; a quadratic trend by $\nabla_1^2$.

**For this project we use classical decomposition (harmonic regression + polynomial trend) as the primary method**, and report differencing as a sanity check / alternative in a single comparison plot. The classical approach gives explicit $\hat m_t$ and $\hat s_t$ that can be reported in the writeup, which the project description asks for ("provide explicit expressions for the trend and seasonal components if present").

### 7. Diagnostic: sample ACF and the $\pm 1.96/\sqrt{n}$ bounds

For observations $x_1, \ldots, x_n$, the **sample autocovariance** is

$$\hat\gamma(h) = \frac{1}{n} \sum_{t=1}^{n - |h|} (x_{t+|h|} - \bar x)(x_t - \bar x)$$

and the **sample ACF** is $\hat\rho(h) = \hat\gamma(h)/\hat\gamma(0)$. Variable meanings: $\bar x$ is the sample mean, $h$ is the lag, $n$ is the number of observations. The divisor $n$ (rather than $n-h$) is used to keep the sample covariance matrix nonnegative definite.

For an iid sequence with finite variance, the course result (B&D §1.4.1) is that for large $n$,

$$\hat\rho(h) \;\overset{\text{approx}}{\sim}\; \mathcal{N}\!\left(0, \tfrac{1}{n}\right), \quad h \neq 0$$

so approximately 95% of the $\hat\rho(h)$ should fall inside

$$\pm \frac{1.96}{\sqrt{n}}$$

These are the standard "blue dashed lines" on an ACF plot. **The cleaned residual series should have a sample ACF that mostly stays inside these bounds**, with no slow decay (which would indicate residual trend) and no spikes at multiples of 7 or 365 (which would indicate residual seasonality).

### 8. Diagnostic: Ljung–Box portmanteau test

Instead of eyeballing each $\hat\rho(h)$, compute the joint statistic (B&D §1.6):

$$Q_{\mathrm{LB}} = n(n+2) \sum_{j=1}^{h} \frac{\hat\rho^2(j)}{n - j}$$

Under the iid null hypothesis, $Q_{\mathrm{LB}} \overset{\text{approx}}{\sim} \chi^2(h)$. Reject iid at level $\alpha = 0.05$ if $Q_{\mathrm{LB}} > \chi^2_{0.95}(h)$, equivalently if the $p$-value is below $0.05$.

For the cleaned residuals, report $Q_{\mathrm{LB}}$ and its $p$-value at $h = 20$ and $h = 40$. The residuals will almost certainly **fail** strict iid (real data has dependence), and that's fine — failing iid is what *justifies* fitting an ARMA model in the next stage. The point is to confirm that we have removed the deterministic structure (trend, seasonality) and what's left is stationary correlated noise, not nonstationarity.

---

## Cleaning pipeline — what Claude Code should produce

The deliverable is a single Jupyter notebook `cleaning.ipynb` plus a small `utils.py` next to it for the heavy functions. Use `pandas`, `numpy`, `matplotlib`, `statsmodels`. No seaborn, no plotly.

The notebook should read like a story: load → look → transform → look → transform → look. Each transform is a **separate cell or small group of cells**, each followed immediately by a plot showing the effect.

### Step 0: Load and inspect

- Load `time_series_60min_singleindex.csv` from the project directory.
- Extract the column `SE_3_load_actual_entsoe_transparency`.
- Set the index to the `utc_timestamp` column, parsed as datetime.
- Report: start date, end date, number of hourly points, number of NaNs, longest gap.
- **Stop and plot the raw hourly series.** (Just the plot — no transforms yet.)

### Step 1: Aggregate to daily

- Aggregate hourly → daily by **mean** (so the units stay MW, not MWh). Justify this in a one-line comment.
- Handle NaNs: if a day has fewer than, say, 20 valid hours, mark the whole day as missing; otherwise the mean over the available hours is fine.
- After aggregation, linearly interpolate any remaining missing days and report how many.
- Report: number of daily points $n$. Confirm $n \geq 2000$.
- Plot the daily series.

### Step 2: Log transform

- Apply $\log$ (natural log). Justify with one sentence referencing variance stabilization (Box–Cox $\lambda = 0$).
- Plot. Visually confirm that the amplitude of fluctuations no longer grows with the level.

### Step 3: Remove yearly seasonality (harmonic regression)

- Build a design matrix with columns $\cos(2\pi j t / 365)$ and $\sin(2\pi j t / 365)$ for $j = 1, 2$.
- Fit $\log X_t$ against this design matrix by OLS (use `numpy.linalg.lstsq` or `statsmodels.OLS`).
- Save the fitted $\hat s_t^{(\mathrm{year})}$ as a series. **Print the coefficients** $a_1, b_1, a_2, b_2$ — these go in the report.
- Subtract $\hat s_t^{(\mathrm{year})}$ from the log series.
- Plot: (a) the deseasonalized series, (b) the fitted yearly component overlaid on a single year of raw log data, so we can see the fit visually.

### Step 4: Remove weekly seasonality (harmonic regression)

- Same procedure with $d = 7$, $j = 1, 2, 3$.
- Print the coefficients.
- Subtract from the (already yearly-deseasonalized) series.
- Plot: a 4-week window before vs after, so the weekday/weekend pattern is visibly gone.

### Step 5: Remove trend (polynomial regression)

- Fit a **linear** trend $\hat m_t = \beta_0 + \beta_1 t$ to the deseasonalized series by OLS. Print $\beta_0, \beta_1$.
- Subtract.
- Plot residuals.
- **Decision point:** if residuals visibly curve, refit with a quadratic and use that instead. Print the choice and the coefficients you went with.

### Step 6: Diagnostics on the residual series $\hat Y_t$

- Plot the residuals as a time series. They should look like noise — no trend, no obvious periodicity, roughly constant variance.
- Plot the **sample ACF** of $\hat Y_t$ for lags $0$ to $50$, with the $\pm 1.96/\sqrt{n}$ bounds shown as dashed horizontal lines. Use `statsmodels.graphics.tsaplots.plot_acf`.
- Also plot the **sample PACF** for the same range — the next stage of the project will need it for ARMA order selection, so producing it now is a courtesy to the rest of the group.
- Compute and print the **Ljung–Box statistic** $Q_{\mathrm{LB}}$ and its $p$-value at $h = 20$ and $h = 40$ (use `statsmodels.stats.diagnostic.acorr_ljungbox`). State explicitly in a markdown cell that we *expect* to reject iid, and that this is the correct outcome — it means there is correlated structure left for ARMA to model.
- Verify there is **no slow decay** in the ACF (would indicate residual trend) and **no spike at lag 7 or 365** (would indicate residual seasonality). If either appears, flag it loudly.

### Step 7: Sanity check via differencing (brief)

- As a parallel sanity check, compute $\nabla_7 \log X_t = (1 - B^7) \log X_t$ followed by $\nabla_1$ to kill any linear trend.
- Plot its sample ACF on the same axes (or side-by-side) with the ACF of $\hat Y_t$ from Step 6.
- One sentence of commentary: do the two methods give qualitatively similar residuals? They should.

### Step 8: Save outputs

Save to disk:
- `cleaned_residuals.csv` — two columns: `date`, `residual`. This is the input for the rest of the group's ARMA fitting.
- `decomposition.csv` — columns: `date`, `log_load`, `yearly_seasonal`, `weekly_seasonal`, `trend`, `residual`. Lets the report-writers reproduce any plot.
- `cleaning_report.txt` — a plain-text summary: $n$, missing-day count, harmonic coefficients, trend coefficients, Ljung–Box statistics. One screen of text.

---

## Style requirements

- Every transformation step has a **markdown cell above it** stating, in one or two sentences, *what* it does and *why* (referencing the math section above by section number).
- Every plot has a title and axis labels with units (MW, log MW, day-of-year, lag).
- No magic numbers in the code without a comment naming what they are.
- No seaborn. No plotly. `matplotlib` only.
- Function definitions for the harmonic-regression and the diagnostic-plot routines go in `utils.py`. The notebook imports from it. Keeps the notebook readable.
- **Do not fit any ARMA / AR / MA model.** That is the next person's job.

---

## What success looks like

A notebook where someone in the group can scroll from top to bottom and see:
1. Raw hourly load.
2. Daily aggregated load.
3. Log-transformed load.
4. The fitted yearly seasonal component, with coefficients printed.
5. Log load with yearly seasonality removed.
6. The fitted weekly seasonal component, with coefficients printed.
7. Log load with both seasonalities removed.
8. The fitted trend, with coefficients printed.
9. The residual series — visually noise-like.
10. The residual ACF inside the $\pm 1.96/\sqrt n$ bounds at most lags, with no slow decay and no period-7 or period-365 spikes.
11. Ljung–Box numbers + an honest statement that iid is rejected, which is fine.
12. CSV files written to disk for the next stage.

That output is enough for the rest of the group to start fitting ARMA models without ever opening the raw CSV.
