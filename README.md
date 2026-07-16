# US House-Price Index: Hierarchical Model & Adverse-Scenario Stress Forecast

A state-level model of the US housing market built on the **FHFA House Price Index**
(2000–2025, 51 regions) and macroeconomic drivers, with a **stress-test forecast**
of how prices would move under an adverse economic scenario.

> **Context.** Coursework for **Math 574 (Bayesian Computational Statistics)** at
> Illinois Institute of Technology, cleaned up and extended for portfolio use. The
> "Bayesian" element is a **normal (Laplace) approximation** to the posterior of the
> model coefficients — not full MCMC — used to propagate parameter uncertainty into
> the forecast.

## Data

Quarterly panel combining regional **FHFA HPI** with national **mortgage rates**
(Freddie Mac PMMS), **unemployment**, **home sales**, **listing inventory**, and
**negative-equity share**. 51 regions (50 states + DC), 2000–2025. Sources are
listed in [`SOURCES.md`](SOURCES.md). The derived, analysis-ready panel
(`data/cleaned_housing_data.csv`) is included; the large raw files are not — see
[`preprocessing.py`](preprocessing.py) to rebuild them.

## Part 1 — Hierarchical model of price levels (`model.py`)

A mixed-effects regression of `log(HPI)` on lagged HPI and macro covariates, with a
**random intercept per state**, fit with statsmodels `MixedLM`. Standardised
fixed-effect coefficients:

| Covariate | Coef. | Reading |
|---|---|---|
| lagged HPI | **+0.319** | strong price momentum (autocorrelation) |
| mortgage rate | −0.040 | higher rates → lower prices |
| unemployment | −0.020 | weaker labour market → lower prices |
| inventory | −0.015 | more supply → lower prices |
| negative equity | −0.011 | distress → lower prices |
| home sales | −0.002 | small |

**R² = 0.957** — but that is largely the lagged term: with the lag removed, the
macro covariates alone explain **R² = 0.47**. The high headline number is mostly
price momentum, and the README/code say so rather than overselling it.

## Part 2 — Adverse-scenario stress forecast (`forecast.py`)

**What "adverse" means here:** mortgage rate 8%, unemployment 6%, home sales 500k,
inventory 1.2M, negative-equity share 15%, held fixed for eight quarters.

![Adverse-scenario forecast](figures/adverse_forecast.png)

Under this scenario the model projects house prices to **fall about 12% over eight
quarters** (two years):

| Quarter | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| Cumulative % change | −1.4% | −2.9% | −6.0% | −9.0% | **−11.9%** |
| 95% band (Q8) | | | | | [−13.1%, −10.6%] |

**A modeling note (and fix).** HPI is close to a random walk, so a model of the
price *level* cannot be recursed forward without blowing up numerically. The
forecast therefore models the quarterly **growth rate** (log return) as a function
of the macro scenario and cumulates it — the standard, stable way to build a
multi-step stress projection. The growth model's R² is a realistic **0.28**
(quarter-to-quarter growth is hard to predict), but it yields a sensible, bounded
decline rather than a runaway path.

## Run it

```bash
pip install -r requirements.txt
python model.py       # hierarchical level model: coefficients + R²
python forecast.py    # adverse-scenario projection + figures/… chart
```

## Methodology notes & limitations
- **"Bayesian" = normal approximation.** Coefficient uncertainty comes from the
  asymptotic normal (Laplace) approximation of the fitted model, sampled 1,000
  times — not MCMC with explicit priors.
- **The forecast band is parameter uncertainty only.** It reflects uncertainty in
  the coefficients, not residual/observation noise or state-level random-effect
  draws, so it is narrower than a full predictive interval.
- The mixed model reports a boundary/convergence warning (the region random-effect
  variance is small relative to residual variance); the fixed-effect estimates are
  stable.
- Single adverse scenario, national average across states; not a substitute for a
  full regulatory stress model (e.g. CCAR/DFAST).

## Repository layout

```
hpi-hierarchical-forecast/
├── preprocessing.py     # build the quarterly panel from raw sources (raw data not committed)
├── model.py             # hierarchical (mixed-effects) level model
├── forecast.py          # adverse-scenario growth forecast + chart
├── data/
│   └── cleaned_housing_data.csv   # derived, analysis-ready panel
├── figures/adverse_forecast.png
├── SOURCES.md
├── requirements.txt
└── LICENSE
```

## License
[MIT](LICENSE). Data belongs to its original publishers (FHFA, Freddie Mac, FRED,
and the cited Kaggle datasets); see SOURCES.md.
